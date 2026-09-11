"""Training entry point for the multi-agent NxN grid extension."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import TypedDict

from constants import DEFAULT_MODEL_PATH, DEFAULT_SETTINGS_PATH
from agent.model import EarlyStopping
from grid.config import build_grid_config
from grid.coordinator import MultiAgentCoordinator
from grid.grid_env import GridEnvStats
from grid.grid_episode import GridRecord
from grid.network_gen import generate_grid_network, generate_grid_sumocfg
from grid.parallel_worker import WorkerArgs, run_episode_worker
from logger import get_logger
from metrics import (
    AVG_QUEUE_LABEL,
    NEG_REWARD_LABEL,
    TOTAL_DELAY_LABEL,
    avg_queue,
    sum_negative_rewards,
    total_delay,
)
from plots import save_data_and_plot
from settings import load_grid_training_settings

logger = get_logger(__name__)

_GRID_TRAINING_SETTINGS_FILE = Path("grid_training_settings.yaml")


class GridTrainingStats(TypedDict):
    """Aggregated per-episode statistics for grid training.

    Same definitions as the single-intersection ``TrainingStats`` in
    ``train.py``, summed across all junctions. Both come from ``metrics.py``.
    """

    neg_reward: list[float]   # training diagnostic only, all junctions
    total_delay: list[int]    # vehicle-seconds, all junctions
    avg_queue: list[float]    # vehicles, all junctions


def _add_experiences(
    coordinator: MultiAgentCoordinator,
    history: dict[str, list[GridRecord]],
) -> None:
    """Push consecutive-pair transitions into each junction's replay buffer."""
    for tl, records in history.items():
        for i in range(len(records) - 1):
            coordinator.add_experience(
                tl_id=tl,
                state=records[i].state,
                action=records[i].action,
                reward=records[i].reward,
                next_state=records[i + 1].state,
            )


def _update_stats(
    history: dict[str, list[GridRecord]],
    env_stats: list[GridEnvStats],
    max_steps: int,
    stats: GridTrainingStats,
) -> GridTrainingStats:
    """Update *stats* in-place with one episode's data and return it."""
    rewards = [rec.reward for records in history.values() for rec in records]
    stats["neg_reward"].append(sum_negative_rewards(rewards))

    # One scalar per simulation step: the queue summed over every junction.
    queue_per_step = [sum(s.queue_lengths.values()) for s in env_stats]
    stats["total_delay"].append(total_delay(queue_per_step))
    stats["avg_queue"].append(avg_queue(queue_per_step, max_steps))

    return stats


def grid_training_session(settings_file: Path, out_path: Path) -> None:
    """Run a full multi-agent training session and save results.

    Episodes are dispatched in parallel batches of ``num_parallel_episodes``
    SUMO processes.  Each batch runs data collection simultaneously; the main
    process then adds all experiences to the replay buffers and trains.

    Args:
        settings_file: Path to the grid training settings YAML.
        out_path: Directory for model weights and plots.
    """
    settings = load_grid_training_settings(settings_file)

    # Ensure network files exist
    grid_dir = settings.grid_net_file.parent
    if not settings.grid_net_file.exists():
        logger.info(f"Generating grid network: {settings.grid_net_file}")
        generate_grid_network(settings.grid_n, grid_dir, settings.junction_spacing)
    if not settings.grid_sumocfg_file.exists():
        logger.info(f"Generating grid sumocfg: {settings.grid_sumocfg_file}")
        generate_grid_sumocfg(settings.grid_n, grid_dir)

    grid_cfg = build_grid_config(
        n=settings.grid_n,
        spacing=settings.junction_spacing,
        net_file=settings.grid_net_file,
        sumocfg_file=settings.grid_sumocfg_file,
        routes_file=settings.grid_routes_file,
    )

    coordinator = MultiAgentCoordinator(
        tl_ids=grid_cfg.tl_ids,
        settings=settings,
        epsilon=1.0,
    )

    early_stopping = EarlyStopping(patience=settings.early_stopping_patience)

    timestamp_start = datetime.now()
    tot_episodes = settings.total_episodes
    n_parallel = settings.num_parallel_episodes
    routes_dir = settings.grid_routes_file.parent
    project_root = str(Path.cwd())
    src_path = str(Path(__file__).resolve().parent)

    training_stats: GridTrainingStats = {
        "neg_reward": [],
        "total_delay": [],
        "avg_queue": [],
    }

    should_stop = False
    episode_idx = 0

    with ProcessPoolExecutor(max_workers=n_parallel) as pool:
        while episode_idx < tot_episodes and not should_stop:
            batch_count = min(n_parallel, tot_episodes - episode_idx)
            new_epsilon = round(1.0 - (episode_idx / tot_episodes), 2)
            coordinator.set_epsilon(new_epsilon)
            weights = coordinator.get_weights()

            logger.info(
                f"Episodes {episode_idx + 1}–{episode_idx + batch_count} of {tot_episodes} "
                f"(ε={new_epsilon}, {batch_count} parallel workers)"
            )

            # Build one WorkerArgs per parallel episode with a unique routes file
            worker_args = [
                WorkerArgs(
                    seed=episode_idx + i,
                    epsilon=new_epsilon,
                    weights_np=weights,
                    settings=settings,
                    grid_cfg=grid_cfg,
                    routes_path=routes_dir / f"routes_w{i}.rou.xml",
                    project_root=project_root,
                    src_path=src_path,
                )
                for i in range(batch_count)
            ]

            # Run all episodes in parallel, collect results
            results = list(pool.map(run_episode_worker, worker_args))

            # Add experiences from every episode and train once per batch
            for history, env_stats in results:
                _add_experiences(coordinator, history)
                _update_stats(history, env_stats, settings.max_steps, training_stats)

            coordinator.replay_all(
                gamma=settings.gamma,
                batch_size=settings.batch_size,
                training_epochs=settings.training_epochs,
            )

            # Log and check early stopping for each episode in the batch
            for i in range(batch_count):
                ep_num = episode_idx + i + 1
                ep_reward = training_stats["neg_reward"][episode_idx + i]
                ep_queue = training_stats["avg_queue"][episode_idx + i]
                logger.info(
                    f"\tEp {ep_num}: reward={ep_reward:.1f}  avg_queue={ep_queue:.1f}"
                )

                if settings.checkpoint_interval > 0 and ep_num % settings.checkpoint_interval == 0:
                    out_path.mkdir(parents=True, exist_ok=True)
                    coordinator.save_models(out_path / f"checkpoint_ep{ep_num}")
                    logger.info(f"\tCheckpoint saved at episode {ep_num}")

                if ep_num >= settings.early_stopping_min_episode:
                    if early_stopping.step(ep_reward):
                        logger.info(
                            f"\tEarly stopping triggered after {ep_num} episodes "
                            f"(no improvement for {settings.early_stopping_patience} episodes, "
                            f"best reward: {early_stopping.best:.1f})"
                        )
                        should_stop = True
                        break

                    if early_stopping.improved:
                        out_path.mkdir(parents=True, exist_ok=True)
                        coordinator.save_models(out_path / "best")
                        logger.info(f"\tNew best reward {early_stopping.best:.1f} — saved best/ models")

            episode_idx += batch_count

    out_path.mkdir(parents=True, exist_ok=True)
    coordinator.save_models(out_path)

    logger.info(f"Start time: {timestamp_start}")
    logger.info(f"End time: {datetime.now()}")
    logger.info(f"Session info saved at: {out_path}")

    copyfile(src=settings_file, dst=out_path / _GRID_TRAINING_SETTINGS_FILE)

    save_data_and_plot(
        data=training_stats["neg_reward"],
        filename="grid_reward",
        xlabel="Episode",
        ylabel=f"{NEG_REWARD_LABEL} (all junctions)",
        out_folder=out_path,
    )
    save_data_and_plot(
        data=training_stats["total_delay"],
        filename="grid_delay",
        xlabel="Episode",
        ylabel=f"{TOTAL_DELAY_LABEL} (all junctions)",
        out_folder=out_path,
    )
    save_data_and_plot(
        data=training_stats["avg_queue"],
        filename="grid_queue",
        xlabel="Episode",
        ylabel=f"{AVG_QUEUE_LABEL} (all junctions)",
        out_folder=out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multi-agent grid TLCS.")
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH / _GRID_TRAINING_SETTINGS_FILE,
        help="Path to grid_training_settings.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Output directory for model weights and plots (default: model/)",
    )
    args = parser.parse_args()
    grid_training_session(settings_file=args.settings, out_path=args.out)
