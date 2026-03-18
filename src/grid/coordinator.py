"""Multi-agent coordinator for NxN intersection grids.

All intersections share one model object (SharedParameters mode); gradients
accumulate across all N² replay calls per epoch.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from agent import Agent, Memory, Sample
from agent.model import Model
from constants import NUM_ACTIONS, STATE_SIZE
from settings import GridTrainingSettings


class MultiAgentCoordinator:
    """Coordinates a collection of DQN agents for multi-intersection control.

    All agents share a single model (SharedParameters mode).

    Args:
        tl_ids: Ordered list of junction IDs (row-major).
        settings: Grid training settings (model architecture, memory sizes, …).
        epsilon: Initial exploration probability for all agents.
    """

    def __init__(
        self,
        tl_ids: list[str],
        settings: GridTrainingSettings,
        epsilon: float = 1.0,
    ) -> None:
        self.tl_ids = tl_ids
        self.settings = settings

        # Build memories (one per junction)
        self.memories: dict[str, Memory] = {
            tl: Memory(
                size_max=settings.memory_size_max,
                size_min=settings.memory_size_min,
            )
            for tl in tl_ids
        }

        # One shared model; all agents point to the same object.
        shared_model = Model(
            num_layers=settings.num_layers,
            width=settings.width_layers,
            learning_rate=settings.learning_rate,
            input_dim=STATE_SIZE,
            output_dim=NUM_ACTIONS,
        )
        self.agents: dict[str, Agent] = {
            tl: Agent(settings, epsilon=epsilon, model=shared_model)
            for tl in tl_ids
        }

    # ------------------------------------------------------------------
    # Episode helpers
    # ------------------------------------------------------------------

    def choose_actions(self, states: dict[str, NDArray]) -> dict[str, int]:
        """Select actions for all junctions.

        Exploring junctions pick a random action.  Exploiting junctions are
        batched into a single CPU forward pass (SharedParameters mode) or
        handled individually via the CPU inference shadow (other modes).

        Args:
            states: Mapping from TL ID to its current state vector.

        Returns:
            Mapping from TL ID to chosen action index.
        """
        actions: dict[str, int] = {}
        exploit_ids: list[str] = []
        exploit_states: list[NDArray] = []

        for tl in self.tl_ids:
            if random.random() < self.agents[tl].epsilon:
                actions[tl] = random.randrange(NUM_ACTIONS)
            else:
                exploit_ids.append(tl)
                exploit_states.append(states[tl])

        if exploit_ids:
            arr = np.array(exploit_states, dtype=np.float32)
            t = torch.from_numpy(arr)
            ref_model = self.agents[self.tl_ids[0]].model
            with torch.no_grad():
                q = ref_model.inference_model(t).numpy()
            for i, tl in enumerate(exploit_ids):
                actions[tl] = int(np.argmax(q[i]))

        return actions

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

    def get_weights(self) -> dict[str, dict[str, NDArray]]:
        """Return current online model weights as picklable numpy arrays.

        Returns:
            Mapping from TL ID to a state-dict of numpy arrays.
        """
        shared = {
            k: v.cpu().numpy()
            for k, v in self.agents[self.tl_ids[0]].model.model.state_dict().items()
        }
        return {tl: shared for tl in self.tl_ids}

    def load_weights(self, weights: dict[str, dict[str, NDArray]]) -> None:
        """Load online model weights from numpy arrays.

        Only one model object is updated (all agents share it).

        Args:
            weights: Mapping from TL ID to state-dict of numpy arrays.
        """
        self.agents[self.tl_ids[0]].model.load_state_dict_np(weights[self.tl_ids[0]])

    def save_models(self, out_path: Path) -> None:
        """Save model weights to *out_path*.

        Writes a single ``shared_model.pt`` (all agents share the same weights).

        Args:
            out_path: Directory to write model files into.
        """
        out_path.mkdir(parents=True, exist_ok=True)
        self.agents[self.tl_ids[0]].model.save_weights(out_path / "shared_model.pt")
