"""Single-episode runner for multi-agent NxN grid training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from numpy.typing import NDArray

from grid.coordinator import MultiAgentCoordinator
from grid.grid_env import GridEnvStats, GridEnvironment


@dataclass
class GridRecord:
    """One decision-step transition for a single junction.

    Attributes:
        tl_id: Junction ID.
        state: State observed before the action.
        action: Action chosen by the agent.
        reward: Reward received after the action.
    """

    tl_id: str
    state: NDArray
    action: int
    reward: float


def run_grid_episode(
    env: GridEnvironment,
    coordinator: MultiAgentCoordinator,
    seed: int,
    on_step: Callable[[], None] | None = None,
) -> tuple[dict[str, list[GridRecord]], list[GridEnvStats]]:
    """Run one training episode in the grid environment.

    Args:
        env: Grid environment to interact with.
        coordinator: Multi-agent coordinator that holds all agents.
        seed: Seed for route file generation.
        on_step: Optional callback invoked at the end of every decision step
            (e.g. to trigger an interleaved replay).

    Returns:
        A tuple ``(history, env_stats)`` where:

        * ``history`` maps each TL ID to the list of per-step
          :class:`GridRecord` objects collected during the episode.
        * ``env_stats`` is the flat list of :class:`GridEnvStats` collected
          from every simulation step.
    """
    env.generate_routefile(seed=seed)
    env.activate()

    history: dict[str, list[GridRecord]] = {tl: [] for tl in env.grid_cfg.tl_ids}
    env_stats: list[GridEnvStats] = []
    previous_wait: dict[str, float] = {tl: 0.0 for tl in env.grid_cfg.tl_ids}

    while not env.is_over():
        states = env.get_states()

        actions = coordinator.choose_actions(states)
        step_stats = env.execute(actions)
        env_stats.extend(step_stats)

        # Compute per-junction rewards
        for tl in env.grid_cfg.tl_ids:
            current_wait = env.get_cumulated_waiting_time(tl)
            reward = previous_wait[tl] - current_wait
            previous_wait[tl] = current_wait
            history[tl].append(GridRecord(
                tl_id=tl,
                state=states[tl],
                action=actions[tl],
                reward=reward,
            ))

        if on_step is not None:
            on_step()

    env.deactivate()
    return history, env_stats
