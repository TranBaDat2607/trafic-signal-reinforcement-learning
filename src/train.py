"""Training entry point for the TLCS agent."""

import argparse
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import TypedDict

from agent import Agent, Memory, Sample
from agent.model import EarlyStopping
from constants import DEFAULT_MODEL_PATH, DEFAULT_SETTINGS_PATH, TRAINING_SETTINGS_FILE
from environment import Environment, EnvStats
from episode import Record, run_episode
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
from settings import load_training_settings

logger = get_logger(__name__)


class TrainingStats(TypedDict):
    """Aggregated per-episode statistics collected during training.

    Metric definitions live in ``metrics.py`` so that the agent and the
    fixed-time baseline are scored by the same function.
    """

    neg_reward: list[float]   # training diagnostic only, never a benchmark
    total_delay: list[int]    # vehicle-seconds
    avg_queue: list[float]    # vehicles


def add_experience_to_memory(memory: Memory, history: list[Record]) -> None:
    """Add transitions from an episode history to replay memory.

    Each pair of consecutive records is converted into a (s, a, r, s') sample.
    """
    for i in range(len(history) - 1):
        sample = Sample(
            state=history[i].state,
            action=history[i].action,
            reward=history[i].reward,
            next_state=history[i + 1].state,
        )
        memory.add_sample(sample)


def update_training_stats(
    episode_history: list[Record],
    env_stats: list[EnvStats],
    max_steps: int,
    training_stats: TrainingStats,
) -> TrainingStats:
    """Update cumulative training statistics with metrics from one episode."""
    queue_per_step = [stats.queue_length for stats in env_stats]

    training_stats["total_delay"].append(total_delay(queue_per_step))
    training_stats["avg_queue"].append(avg_queue(queue_per_step, max_steps))
    training_stats["neg_reward"].append(
        sum_negative_rewards([record.reward for record in episode_history])
    )

    return training_stats


def training_session(settings_file: Path, out_path: Path) -> None:
    """Run a full training session and save the trained model and statistics."""
    settings = load_training_settings(settings_file)

    memory = Memory(size_max=settings.memory_size_max, size_min=settings.memory_size_min)
    agent = Agent(settings=settings)

    early_stopping = EarlyStopping(patience=settings.early_stopping_patience)

    timestamp_start = datetime.now()
    tot_episodes = settings.total_episodes

    training_stats: TrainingStats = {
        "neg_reward": [],
        "total_delay": [],
        "avg_queue": [],
    }

    for episode in range(tot_episodes):
        logger.info(f"Episode {episode + 1} of {tot_episodes}")

        new_epsilon = round(1.0 - (episode / tot_episodes), 2)
        agent.set_epsilon(new_epsilon)

        env = Environment(
            n_cars_generated=settings.n_cars_generated,
            max_steps=settings.max_steps,
            yellow_duration=settings.yellow_duration,
            green_duration=settings.green_duration,
            turn_chance=settings.turn_chance,
            gui=settings.gui,
            sumocfg_file=settings.sumocfg_file,
        )

        def on_step() -> None:
            for _ in range(settings.training_epochs):
                agent.replay(
                    memory=memory,
                    gamma=settings.gamma,
                    batch_size=settings.batch_size,
                )

        episode_history, env_stats = run_episode(env=env, agent=agent, seed=episode, on_step=on_step)

        add_experience_to_memory(memory=memory, history=episode_history)

        training_stats = update_training_stats(
            episode_history=episode_history,
            env_stats=env_stats,
            max_steps=settings.max_steps,
            training_stats=training_stats,
        )

        logger.info(f"\tEpsilon: {agent.epsilon}")
        logger.info(f"\tTotal delay (vehicle-seconds): {training_stats['total_delay'][-1]}")
        logger.info(f"\tAvg queue length: {training_stats['avg_queue'][-1]}")
        logger.info(f"\tNeg-reward diagnostic: {training_stats['neg_reward'][-1]:.1f}")

        if settings.checkpoint_interval > 0 and (episode + 1) % settings.checkpoint_interval == 0:
            out_path.mkdir(parents=True, exist_ok=True)
            agent.save_checkpoint(out_path, episode + 1)
            logger.info(f"\tCheckpoint saved: checkpoint_ep{episode + 1}.pt")

        if early_stopping.step(training_stats["neg_reward"][-1]):
            logger.info(
                f"\tEarly stopping triggered after {episode + 1} episodes "
                f"(no improvement for {settings.early_stopping_patience} episodes, "
                f"best reward: {early_stopping.best:.1f})"
            )
            break
        if early_stopping.improved:
            out_path.mkdir(parents=True, exist_ok=True)
            agent.model.save_weights(out_path / "best_model.pt")
            logger.info(f"\tNew best reward {early_stopping.best:.1f} — saved best_model.pt")

    out_path.mkdir(parents=True, exist_ok=True)
    agent.save_model(out_path)

    logger.info(f"Start time: {timestamp_start}")
    logger.info(f"End time: {datetime.now()}")
    logger.info(f"Session info saved at: {out_path}")

    copyfile(src=settings_file, dst=out_path / TRAINING_SETTINGS_FILE)

    save_data_and_plot(
        data=training_stats["neg_reward"],
        filename="reward",
        xlabel="Episode",
        ylabel=NEG_REWARD_LABEL,
        out_folder=out_path,
    )
    save_data_and_plot(
        data=training_stats["total_delay"],
        filename="delay",
        xlabel="Episode",
        ylabel=TOTAL_DELAY_LABEL,
        out_folder=out_path,
    )
    save_data_and_plot(
        data=training_stats["avg_queue"],
        filename="queue",
        xlabel="Episode",
        ylabel=AVG_QUEUE_LABEL,
        out_folder=out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TLCS agent.")
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH / TRAINING_SETTINGS_FILE,
        help="Path to training_settings.yaml (default: settings/training_settings.yaml)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Output directory for model and plots (default: model/)",
    )
    args = parser.parse_args()

    training_session(settings_file=args.settings, out_path=args.out)
