"""Evaluation entry point for the TLCS agent."""

import argparse
from pathlib import Path
from typing import TypedDict

from agent import Agent
from constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SETTINGS_PATH,
    TESTING_SETTINGS_FILE,
    TRAINING_SETTINGS_FILE,
)
from environment import Environment
from episode import run_episode
from logger import get_logger
from metrics import avg_queue, total_delay
from settings import load_testing_settings, load_training_settings

logger = get_logger(__name__)


class TestStats(TypedDict):
    """Aggregated per-episode statistics collected during evaluation.

    Metric definitions live in ``metrics.py``; see ``TrainingStats`` in
    ``train.py``. No reward diagnostic here — evaluation runs greedily and the
    reward signal is a property of training, not of controller quality.
    """

    total_delay: list[int]   # vehicle-seconds
    avg_queue: list[float]   # vehicles


def testing_session(model_path: Path, settings_file: Path) -> None:
    """Evaluate a trained model over multiple episodes and report statistics.

    Loads architecture from the training settings saved alongside the model,
    then runs each test episode with epsilon=0 (fully greedy policy).

    Args:
        model_path: Directory containing ``trained_model.pt`` and
            ``training_settings.yaml``.
        settings_file: Path to ``testing_settings.yaml``.
    """
    settings = load_testing_settings(settings_file)

    # Reconstruct agent architecture from the saved training settings.
    train_settings_file = model_path / TRAINING_SETTINGS_FILE
    train_settings = load_training_settings(train_settings_file)

    agent = Agent(settings=train_settings, epsilon=0.0, model_path=model_path)

    test_stats: TestStats = {
        "total_delay": [],
        "avg_queue": [],
    }

    for episode in range(settings.total_episodes):
        seed = settings.episode_seed + episode
        logger.info(f"Test episode {episode + 1} of {settings.total_episodes} (seed={seed})")

        env = Environment(
            n_cars_generated=settings.n_cars_generated,
            max_steps=settings.max_steps,
            yellow_duration=settings.yellow_duration,
            green_duration=settings.green_duration,
            turn_chance=settings.turn_chance,
            gui=settings.gui,
            sumocfg_file=settings.sumocfg_file,
        )

        _, env_stats = run_episode(env=env, agent=agent, seed=seed)

        queue_per_step = [s.queue_length for s in env_stats]
        episode_delay = total_delay(queue_per_step)
        episode_queue = avg_queue(queue_per_step, settings.max_steps)

        test_stats["total_delay"].append(episode_delay)
        test_stats["avg_queue"].append(episode_queue)

        logger.info(f"\tTotal delay (vehicle-seconds): {episode_delay}")
        logger.info(f"\tAvg queue length: {episode_queue}")

    mean_delay = sum(test_stats["total_delay"]) / settings.total_episodes
    mean_queue = sum(test_stats["avg_queue"]) / settings.total_episodes

    logger.info("--- Test Summary ---")
    logger.info(f"Episodes:                           {settings.total_episodes}")
    logger.info(f"Mean total delay (vehicle-seconds): {mean_delay:.1f}")
    logger.info(f"Mean queue length:                  {mean_queue:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained TLCS agent.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Directory containing trained_model.pt and training_settings.yaml (default: model/)",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH / TESTING_SETTINGS_FILE,
        help="Path to testing_settings.yaml (default: settings/testing_settings.yaml)",
    )
    args = parser.parse_args()

    testing_session(model_path=args.model, settings_file=args.settings)
