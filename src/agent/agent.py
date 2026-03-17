from pathlib import Path

import numpy as np

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
        input_dim: int | None = None,
    ) -> None:
        """Initialize the agent, its model, and the epsilon-greedy policy.

        Args:
            settings: Training settings with model architecture and hyperparameters.
            epsilon: Initial exploration probability in [0, 1].
            model_path: Optional path to load pre-trained model weights from.
            model: Optional pre-built :class:`~agent.model.Model` to share
                (used for SharedParameters multi-agent mode).  When provided,
                *model_path* and *input_dim* are ignored.
            input_dim: Override the default ``STATE_SIZE`` input dimension
                (used for NeighborAware multi-agent mode).
        """
        effective_dim = input_dim if input_dim is not None else STATE_SIZE
        if model is not None:
            self.model = model
        else:
            self.model = Model(
                num_layers=settings.num_layers,
                width=settings.width_layers,
                learning_rate=settings.learning_rate,
                input_dim=effective_dim,
                output_dim=NUM_ACTIONS,
                model_path=model_path,
            )
        self.policy = EpsilonGreedyPolicy(self.model, NUM_ACTIONS, epsilon)
        self.target_update_interval = settings.target_update_interval
        self._replay_count = 0

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

        states = np.array([sample.state for sample in batch])
        next_states = np.array([sample.next_state for sample in batch])

        q_values = self.model.predict_batch(states)
        # Use frozen target network to compute stable bootstrap targets.
        next_q_values = self.model.predict_batch_target(next_states)

        x = states
        y = q_values.copy()

        for i, sample in enumerate(batch):
            # Q-learning target: r + gamma * max_a' Q_target(s', a')
            target = sample.reward + gamma * np.max(next_q_values[i])
            y[i, sample.action] = target

        self.model.train_batch(x, y)

        self._replay_count += 1
        if self._replay_count % self.target_update_interval == 0:
            self.model.update_target_network()

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
