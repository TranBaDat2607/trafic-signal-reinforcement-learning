"""Worker function for parallel SUMO episode data collection.

This module must be a top-level importable module (not a nested function) so
that ProcessPoolExecutor can pickle and transfer it to spawned workers on
Windows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WorkerArgs:
    """Arguments passed to a parallel episode worker.

    Attributes:
        seed: Random seed for route file generation.
        epsilon: Exploration probability used during this episode.
        weights_np: Current model weights as numpy arrays (picklable).
            Mapping: tl_id -> {param_name: NDArray}.
        settings: GridTrainingSettings instance (Pydantic model, picklable).
        grid_cfg: GridConfig for the network (dataclass, picklable).
        routes_path: Unique route file path for this worker, preventing
            write conflicts when multiple workers run simultaneously.
        project_root: Absolute path to the project root directory.
        src_path: Absolute path to the ``src/`` directory.  Added to
            ``sys.path`` in the worker so that spawned subprocesses can
            import project modules regardless of how ``sys.path`` was set
            in the parent process (e.g. a Jupyter notebook).
    """

    seed: int
    epsilon: float
    weights_np: dict[str, dict[str, Any]]
    settings: Any
    grid_cfg: Any
    routes_path: Path
    project_root: str
    src_path: str


def run_episode_worker(args: WorkerArgs) -> tuple[dict, list]:
    """Run one SUMO episode in a subprocess and return collected experiences.

    Imports are deferred to the body of this function so that the module can
    be imported in the main process without triggering SUMO/TraCI initialisation.

    Each worker:
    1. Changes to the project root so that relative SUMO paths resolve.
    2. Builds a coordinator initialised with the provided model weights.
    3. Runs a full episode (no training — data collection only).
    4. Returns (history, env_stats) back to the main process.

    Args:
        args: Worker configuration and current model weights.

    Returns:
        Tuple of ``(history, env_stats)`` as returned by
        :func:`~grid.grid_episode.run_grid_episode`.
    """
    os.chdir(args.project_root)

    # Ensure src/ is importable — spawned processes do not inherit runtime
    # sys.path modifications made in the parent (e.g. from a Jupyter notebook).
    import sys
    if args.src_path not in sys.path:
        sys.path.insert(0, args.src_path)

    from grid.coordinator import MultiAgentCoordinator
    from grid.grid_env import GridEnvironment
    from grid.grid_episode import run_grid_episode

    coordinator = MultiAgentCoordinator(
        tl_ids=args.grid_cfg.tl_ids,
        settings=args.settings,
        epsilon=args.epsilon,
    )
    coordinator.load_weights(args.weights_np)

    env = GridEnvironment(
        grid_cfg=args.grid_cfg,
        n_cars_generated=args.settings.n_cars_generated,
        max_steps=args.settings.max_steps,
        yellow_duration=args.settings.yellow_duration,
        green_duration=args.settings.green_duration,
        turn_chance=args.settings.turn_chance,
        gui=False,
        routes_override=args.routes_path,
    )

    history, env_stats = run_grid_episode(
        env=env,
        coordinator=coordinator,
        seed=args.seed,
    )

    return history, env_stats
