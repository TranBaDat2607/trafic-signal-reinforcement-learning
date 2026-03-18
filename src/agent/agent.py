from pathlib import Path

import numpy as np
import torch

from agent.memory import Memory
from agent.model import Model
from constants import NUM_ACTIONS, STATE_SIZE
from policy import EpsilonGreedyPolicy
from settings import TrainingSettings


class Agent:
    """Reinforcement-learning agent using an epsilon-greedy policy."""

    def __init__(
        self,
        settings: TrainingSettings,
        epsilon: float = 1.0,
        model_path: Path | None = None,
        model: "Model | None" = None,
    ) -> None:
        """Initialize the agent, its model, and the epsilon-greedy policy.

        Args:
            settings: Training settings with model architecture and hyperparameters.
            epsilon: Initial exploration probability in [0, 1].
            model_path: Optional path to load pre-trained model weights from.
            model: Optional pre-built :class:`~agent.model.Model` to share
                (used for SharedParameters multi-agent mode).  When provided,
                *model_path* is ignored.
        """
        if model is not None:
            self.model = model
        else:
            self.model = Model(
                num_layers=settings.num_layers,
                width=settings.width_layers,
                learning_rate=settings.learning_rate,
                input_dim=STATE_SIZE,
                output_dim=NUM_ACTIONS,
                model_path=model_path,
            )
        self.policy = EpsilonGreedyPolicy(self.model, NUM_ACTIONS, epsilon)
        self.tau = settings.tau

    @property
    def epsilon(self) -> float:
        """Current exploration probability (delegated to the policy)."""
        return self.policy.epsilon

    def set_epsilon(self, epsilon: float) -> None:
        """Update the exploration probability.

        Args:
            epsilon: New exploration probability in [0, 1].
        """
        self.policy.set_epsilon(epsilon)

    def choose_action(self, state: np.ndarray) -> int:
        """Choose an action for the given state.

        Args:
            state: Current state representation as a 1-D array.

        Returns:
            Index of the selected action.
        """
        return self.policy.select_action(state)

    def replay(self, memory: Memory, gamma: float, batch_size: int) -> None:
        """Sample from replay memory and perform a Q-learning update.

        Args:
            memory: Experience replay buffer.
            gamma: Discount factor for future rewards.
            batch_size: Number of samples to draw.
        """
        batch = memory.get_samples(batch_size)

        if not batch:
            return

        device = self.model.device

        # Transfer states to GPU once — reused for both inference and training.
        states_t = torch.from_numpy(
            np.array([s.state for s in batch], dtype=np.float32)
        ).to(device, non_blocking=True)
        next_states_t = torch.from_numpy(
            np.array([s.next_state for s in batch], dtype=np.float32)
        ).to(device, non_blocking=True)

        # Both forward passes stay on GPU — no round-trip to CPU.
        q_values_t = self.model.forward_online(states_t)
        next_q_values_t = self.model.forward_target(next_states_t)

        # Bellman targets computed entirely on GPU (vectorised, no Python loop).
        rewards_t = torch.tensor(
            [s.reward for s in batch], dtype=torch.float32, device=device
        )
        actions = [s.action for s in batch]
        targets_t = q_values_t.clone()
        targets_t[torch.arange(len(batch), device=device), actions] = (
            rewards_t + gamma * next_q_values_t.max(dim=1).values
        )

        self.model.train_on_tensors(states_t, targets_t)
        self.model.update_target_network(self.tau)

    def save_model(self, out_path: Path) -> None:
        """Save the underlying model weights to disk.

        Args:
            out_path: Destination directory for the saved model.
        """
        self.model.save_model(out_path)

    def save_checkpoint(self, out_path: Path, episode: int) -> None:
        """Save a checkpoint of the model weights for a specific episode.

        Args:
            out_path: Directory to write the checkpoint file into.
            episode: Episode number used in the checkpoint filename.
        """
        self.model.save_weights(out_path / f"checkpoint_ep{episode}.pt")
