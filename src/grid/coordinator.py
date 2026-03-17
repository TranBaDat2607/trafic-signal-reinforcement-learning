"""Multi-agent coordinator for NxN intersection grids.

Supports three operating modes:

* **Independent** — each intersection has its own model and memory.
* **SharedParameters** — all intersections share one model object; gradients
  accumulate across all N² replay calls per epoch.
* **NeighborAware** — like Independent but the state vector is the 5×240
  concatenation of the target junction and its four neighbours.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from agent import Agent, Memory, Sample
from agent.model import Model
from constants import NUM_ACTIONS, STATE_SIZE
from settings import GridTrainingSettings

AgentMode = Literal["Independent", "SharedParameters", "NeighborAware"]

_NEIGHBOR_AWARE_DIM = STATE_SIZE * 5  # 1200


class MultiAgentCoordinator:
    """Coordinates a collection of DQN agents for multi-intersection control.

    Args:
        tl_ids: Ordered list of junction IDs (row-major).
        settings: Grid training settings (model architecture, memory sizes, …).
        mode: Operating mode — one of ``"Independent"``, ``"SharedParameters"``,
            or ``"NeighborAware"``.
        epsilon: Initial exploration probability for all agents.
    """

    def __init__(
        self,
        tl_ids: list[str],
        settings: GridTrainingSettings,
        mode: AgentMode,
        epsilon: float = 1.0,
    ) -> None:
        self.tl_ids = tl_ids
        self.settings = settings
        self.mode = mode

        # Build memories (one per junction for all modes)
        self.memories: dict[str, Memory] = {
            tl: Memory(
                size_max=settings.memory_size_max,
                size_min=settings.memory_size_min,
            )
            for tl in tl_ids
        }

        # Build agents
        self.agents: dict[str, Agent] = {}

        if mode == "SharedParameters":
            # One shared model; all agents point to the same object.
            shared_model = Model(
                num_layers=settings.num_layers,
                width=settings.width_layers,
                learning_rate=settings.learning_rate,
                input_dim=STATE_SIZE,
                output_dim=NUM_ACTIONS,
            )
            for tl in tl_ids:
                self.agents[tl] = Agent(settings, epsilon=epsilon, model=shared_model)

        elif mode == "NeighborAware":
            for tl in tl_ids:
                self.agents[tl] = Agent(
                    settings,
                    epsilon=epsilon,
                    input_dim=_NEIGHBOR_AWARE_DIM,
                )

        else:  # Independent
            for tl in tl_ids:
                self.agents[tl] = Agent(settings, epsilon=epsilon)

    # ------------------------------------------------------------------
    # Episode helpers
    # ------------------------------------------------------------------

    def choose_actions(self, states: dict[str, NDArray]) -> dict[str, int]:
        """Select actions for all junctions.

        Args:
            states: Mapping from TL ID to its current state vector.

        Returns:
            Mapping from TL ID to chosen action index.
        """
        return {tl: self.agents[tl].choose_action(states[tl]) for tl in self.tl_ids}

    def add_experience(
        self,
        tl_id: str,
        state: NDArray,
        action: int,
        reward: float,
        next_state: NDArray,
    ) -> None:
        """Add a transition to the replay buffer for *tl_id*.

        Args:
            tl_id: Junction whose buffer receives the transition.
            state: State observed before the action.
            action: Action taken.
            reward: Reward received.
            next_state: State observed after the action.
        """
        self.memories[tl_id].add_sample(
            Sample(state=state, action=action, reward=reward, next_state=next_state)
        )

    def replay_all(self, gamma: float, batch_size: int, training_epochs: int) -> None:
        """Perform *training_epochs* replay steps for every junction.

        Args:
            gamma: Discount factor.
            batch_size: Number of samples per replay call.
            training_epochs: Number of gradient steps per junction per call.
        """
        for _ in range(training_epochs):
            for tl in self.tl_ids:
                self.agents[tl].replay(
                    memory=self.memories[tl],
                    gamma=gamma,
                    batch_size=batch_size,
                )

    def set_epsilon(self, epsilon: float) -> None:
        """Update the exploration probability for all agents.

        Args:
            epsilon: New exploration probability in [0, 1].
        """
        for agent in self.agents.values():
            agent.set_epsilon(epsilon)

    def save_models(self, out_path: Path) -> None:
        """Save model weights to *out_path*.

        In SharedParameters mode only one file is written (all agents share
        the same weights).  In other modes one file per junction is written.

        Args:
            out_path: Directory to write model files into.
        """
        out_path.mkdir(parents=True, exist_ok=True)

        if self.mode == "SharedParameters":
            # Save the single shared model once
            self.agents[self.tl_ids[0]].model.save_weights(out_path / "shared_model.pt")
        else:
            for tl in self.tl_ids:
                safe_name = tl.replace("_", "") + "_model.pt"
                self.agents[tl].model.save_weights(out_path / safe_name)
