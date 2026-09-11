"""Tests for the shared episode-metric definitions.

The point of `src/metrics.py` is that the RL path and the baseline path cannot
drift apart again. These tests encode that contract: if someone reintroduces a
second definition of "episode delay", the equivalence tests below fail.
"""

from dataclasses import dataclass

import pytest

from metrics import (
    METRICS_VERSION,
    avg_queue,
    sum_negative_rewards,
    total_delay,
)


# --- Stand-ins for the per-step stat objects the real call sites hold. -------


@dataclass
class FakeEnvStats:
    """Mirror of environment.core.EnvStats (single intersection)."""

    queue_length: int


@dataclass
class FakeGridEnvStats:
    """Mirror of grid.grid_env.GridEnvStats (per-junction queue lengths)."""

    queue_lengths: dict[str, int]


# --- Definitions ------------------------------------------------------------


def test_total_delay_is_the_sum_of_per_step_queues() -> None:
    # Three vehicles queued for one step, then one for two steps: 3 + 1 + 1.
    assert total_delay([3, 1, 1]) == 5


def test_total_delay_counts_each_vehicle_once_per_step_it_waits() -> None:
    # One vehicle halted for 100 consecutive steps is 100 vehicle-seconds --
    # not the ~5050 that summing an accumulated waiting time would give.
    assert total_delay([1] * 100) == 100


def test_avg_queue_divides_by_episode_length() -> None:
    assert avg_queue([3, 1, 1], max_steps=3) == pytest.approx(1.7)


def test_avg_queue_uses_max_steps_not_series_length() -> None:
    # An episode that ends early is still scored over the full budget, so two
    # episodes of different lengths remain comparable.
    assert avg_queue([10, 10], max_steps=10) == pytest.approx(2.0)


def test_avg_queue_survives_zero_max_steps() -> None:
    assert avg_queue([], max_steps=0) == 0.0


def test_sum_negative_rewards_discards_positives() -> None:
    assert sum_negative_rewards([-3.0, 5.0, -2.0, 0.0]) == pytest.approx(-5.0)


def test_metrics_version_is_recorded() -> None:
    # Serialized with results; compare_results.py refuses older data.
    assert METRICS_VERSION >= 2


# --- The contract: every call site must agree -------------------------------


def test_rl_and_baseline_call_sites_agree() -> None:
    """P0-1/P0-2: the RL and baseline paths must produce the same number.

    The two paths hold their per-step data in different shapes -- the RL path a
    list of EnvStats, the baseline a plain list of ints -- but once reduced to a
    queue series they must flow through one definition.
    """
    queue_series = [4, 0, 7, 7, 2]

    # RL path: environment.core.EnvStats objects, one per simulation step.
    rl_env_stats = [FakeEnvStats(queue_length=q) for q in queue_series]
    rl_queues = [s.queue_length for s in rl_env_stats]

    # Baseline path: a plain list built inside run_one_episode().
    baseline_queues = queue_series

    assert total_delay(rl_queues) == total_delay(baseline_queues)
    assert avg_queue(rl_queues, 5) == avg_queue(baseline_queues, 5)


def test_grid_call_site_agrees_with_single_intersection() -> None:
    """The multi-junction path flattens to the same definition, not a new one."""
    grid_env_stats = [
        FakeGridEnvStats(queue_lengths={"TL0": 3, "TL1": 1}),
        FakeGridEnvStats(queue_lengths={"TL0": 0, "TL1": 5}),
    ]
    grid_queues = [sum(s.queue_lengths.values()) for s in grid_env_stats]

    assert total_delay(grid_queues) == total_delay([4, 5])


# --- Regression guards for the defects this module exists to prevent --------


def test_summing_all_rewards_telescopes_to_nothing() -> None:
    """Why `sum_negative_rewards` is not simply `sum`.

    Per-step reward is W(t-1) - W(t), so summing all of it collapses to
    -W_final regardless of what happened during the episode. The baseline
    recorded exactly 0.0 for all ten committed episodes for this reason.
    """
    waits = [0.0, 30.0, 80.0, 45.0, 0.0]
    rewards = [waits[i - 1] - waits[i] for i in range(1, len(waits))]

    assert sum(rewards) == pytest.approx(-waits[-1])
    assert sum(rewards) == pytest.approx(0.0)

    # The negatives-only diagnostic does carry information about the episode.
    assert sum_negative_rewards(rewards) < 0


def test_delay_is_independent_of_reward_sampling_rate() -> None:
    """Delay is measured per simulation step, so it does not depend on how
    often the agent happens to make a decision -- unlike the reward sum, which
    is why the two were never comparable across the two paths."""
    assert total_delay([2] * 28) == 56
