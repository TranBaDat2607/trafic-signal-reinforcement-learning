from collections.abc import Callable
from dataclasses import dataclass

from numpy.typing import NDArray

from agent import Agent
from environment import Environment, EnvStats


@dataclass
class Record:
    """Single time-step transition experienced during an episode.

    Attributes:
        state: Environment state observed before taking the action.
        action: Action chosen by the agent.
        reward: Reward obtained after executing the action.
    """

    state: NDArray
    action: int
    reward: float


def run_episode(
    env: Environment,
    agent: Agent,
    seed: int,
    on_step: Callable[[], None] | None = None,
) -> tuple[list[Record], list[EnvStats]]:
    """Runs one episode and returns per-step records and environment statistics.

    Args:
        env: Environment to interact with.
        agent: Agent used to select actions from states.
        seed: Seed used to generate the route file for this episode.
        on_step: Optional callback invoked after every decision step.
            Used to interleave training with simulation so that GPU work
            is distributed evenly instead of batched after the episode.

    Returns:
        A tuple (history, env_stats) where:
            history is the list of per-step records.
            env_stats is the list of environment statistics for each executed action.
    """
    env.generate_routefile(seed=seed)

    previous_total_wait = 0.0
    history: list[Record] = []
    env_stats: list[EnvStats] = []

    env.activate()

    while not env.is_over():
        state = env.get_state()
        action = agent.choose_action(state)

        action_stats = env.execute(action)
        env_stats.extend(action_stats)

        current_total_wait = env.get_cumulated_waiting_time()
        reward = previous_total_wait - current_total_wait
        previous_total_wait = current_total_wait

        record = Record(state=state, action=action, reward=reward)
        history.append(record)

        if on_step is not None:
            on_step()

    env.deactivate()

    return history, env_stats
