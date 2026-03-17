"""Training entry point for the multi-agent NxN grid extension."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import TypedDict

from constants import DEFAULT_MODEL_PATH, DEFAULT_SETTINGS_PATH
from agent.model import EarlyStopping
from grid.config import build_grid_config
from grid.coordinator import AgentMode, MultiAgentCoordinator
from grid.grid_env import GridEnvStats, GridEnvironment
from grid.grid_episode import GridRecord, run_grid_episode
from grid.network_gen import generate_grid_network, generate_grid_sumocfg
from logger import get_logger
from plots import save_data_and_plot
from settings import load_grid_training_settings

logger = get_logger(__name__)

_GRID_TRAINING_SETTINGS_FILE = Path("grid_training_settings.yaml")


class GridTrainingStats(TypedDict):
    """Aggregated per-episode statistics for grid training."""

    sum_neg_reward: list[float]      # summed negative rewards (all junctions)
    cumulative_wait: list[int]       # summed queue-length steps (all junctions)
    avg_queue_length: list[float]    # average queue length per step


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
    # Sum negative rewards across all junctions and all steps
    neg_reward = sum(
        rec.reward
        for records in history.values()
        for rec in records
        if rec.reward < 0
    )
    stats["sum_neg_reward"].append(neg_reward)

    # Sum queue lengths across all junctions and all steps
    total_queue = sum(
        sum(s.queue_lengths.values()) for s in env_stats
    )
    stats["cumulative_wait"].append(total_queue)
    stats["avg_queue_length"].append(round(total_queue / max(max_steps, 1), 1))

    return stats


def grid_training_session(settings_file: Path, out_path: Path) -> None:
    """Run a full multi-agent training session and save results.

    Generates the SUMO network and sumocfg if they do not already exist,
    then runs *total_episodes* training episodes with linear epsilon decay.

    Args:
        settings_file: Path to the grid training settings YAML.
        out_path: Directory for model weights and plots.
    """
    settings = load_grid_training_settings(settings_file)
    mode: AgentMode = settings.mode  # type: ignore[assignment]

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
        mode=mode,
        epsilon=1.0,
    )

    early_stopping = EarlyStopping(patience=settings.early_stopping_patience)

    timestamp_start = datetime.now()
    tot_episodes = settings.total_episodes

    training_stats: GridTrainingStats = {
        "sum_neg_reward": [],
        "cumulative_wait": [],
        "avg_queue_length": [],
    }

    for episode in range(tot_episodes):
        logger.info(f"Episode {episode + 1} of {tot_episodes}")

        new_epsilon = round(1.0 - (episode / tot_episodes), 2)
        coordinator.set_epsilon(new_epsilon)

        env = GridEnvironment(
            grid_cfg=grid_cfg,
            n_cars_generated=settings.n_cars_generated,
            max_steps=settings.max_steps,
            yellow_duration=settings.yellow_duration,
            green_duration=settings.green_duration,
            turn_chance=settings.turn_chance,
            gui=settings.gui,
        )

        history, env_stats = run_grid_episode(
            env=env,
            coordinator=coordinator,
            mode=mode,
            seed=episode,
        )

        _add_experiences(coordinator, history)

        coordinator.replay_all(
            gamma=settings.gamma,
            batch_size=settings.batch_size,
            training_epochs=settings.training_epochs,
        )

        training_stats = _update_stats(history, env_stats, settings.max_steps, training_stats)

        logger.info(f"\tEpsilon: {new_epsilon}")
        logger.info(f"\tReward: {training_stats['sum_neg_reward'][-1]}")
        logger.info(f"\tCumulative wait: {training_stats['cumulative_wait'][-1]}")
        logger.info(f"\tAvg queue: {training_stats['avg_queue_length'][-1]}")

        if settings.checkpoint_interval > 0 and (episode + 1) % settings.checkpoint_interval == 0:
            out_path.mkdir(parents=True, exist_ok=True)
            for tl in grid_cfg.tl_ids:
                safe = tl.replace("_", "") + f"_ep{episode + 1}.pt"
                coordinator.agents[tl].save_checkpoint(out_path, episode + 1)
            logger.info(f"\tCheckpoint saved at episode {episode + 1}")

        if early_stopping.step(training_stats["sum_neg_reward"][-1]):
            logger.info(
                f"\tEarly stopping triggered after {episode + 1} episodes "
                f"(no improvement for {settings.early_stopping_patience} episodes, "
                f"best reward: {early_stopping.best:.1f})"
            )
            break
        if early_stopping.improved:
            out_path.mkdir(parents=True, exist_ok=True)
            coordinator.save_models(out_path / "best")
            logger.info(f"\tNew best reward {early_stopping.best:.1f} — saved best/ models")

    out_path.mkdir(parents=True, exist_ok=True)
    coordinator.save_models(out_path)

    logger.info(f"Start time: {timestamp_start}")
    logger.info(f"End time: {datetime.now()}")
    logger.info(f"Session info saved at: {out_path}")

    copyfile(src=settings_file, dst=out_path / _GRID_TRAINING_SETTINGS_FILE)

    save_data_and_plot(
        data=training_stats["sum_neg_reward"],
        filename="grid_reward",
        xlabel="Episode",
        ylabel="Cumulative negative reward (all junctions)",
        out_folder=out_path,
    )
    save_data_and_plot(
        data=training_stats["cumulative_wait"],
        filename="grid_delay",
        xlabel="Episode",
        ylabel="Cumulative delay (all junctions)",
        out_folder=out_path,
    )
    save_data_and_plot(
        data=training_stats["avg_queue_length"],
        filename="grid_queue",
        xlabel="Episode",
        ylabel="Average queue length (vehicles)",
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
