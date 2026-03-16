from pathlib import Path

import numpy as np

from constants import NUM_ACTIONS, STATE_SIZE
from agent.memory import Memory
from agent.model import Model
from policy import EpsilonGreedyPolicy


class Agent:
    """Reinforcement-learning agent using an epsilon-greedy policy."""

    def __init__(
        self,
        settings: object,
        epsilon: float = 1.0,
        model_path: Path | None = None,
    ) -> None:
        """Initialize the agent, its model, and the epsilon-greedy policy.

        Args:
            settings: Training settings with ``num_layers``, ``width_layers``,
                and ``learning_rate`` attributes.
            epsilon: Initial exploration probability in [0, 1].
            model_path: Optional path to load pre-trained model weights from.
        """
        self.model = Model(
            num_layers=settings.num_layers,
            width=settings.width_layers,
            learning_rate=settings.learning_rate,
            input_dim=STATE_SIZE,
            output_dim=NUM_ACTIONS,
            model_path=model_path,
        )
        self.policy = EpsilonGreedyPolicy(self.model, NUM_ACTIONS, epsilon)

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
        next_q_values = self.model.predict_batch(next_states)

        x = states
        y = q_values.copy()

        for i, sample in enumerate(batch):
            # Q-learning target: r + gamma * max_a' Q(s', a')
            target = sample.reward + gamma * np.max(next_q_values[i])
            y[i, sample.action] = target

        self.model.train_batch(x, y)

    def save_model(self, out_path: Path) -> None:
        """Save the underlying model weights to disk.

        Args:
            out_path: Destination directory for the saved model.
        """
        self.model.save_model(out_path)
