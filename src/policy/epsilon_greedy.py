import random

import numpy as np

from agent.model import Model


class EpsilonGreedyPolicy:
    """Epsilon-greedy action-selection policy backed by a Q-value model."""

    def __init__(self, model: Model, num_actions: int, epsilon: float = 1.0) -> None:
        """Initialize the policy.

        Args:
            model: Trained (or untrained) Q-value model used for exploitation.
            num_actions: Number of discrete actions available.
            epsilon: Initial exploration probability in [0, 1].
        """
        self.model = model
        self.num_actions = num_actions
        self.set_epsilon(epsilon)

    def set_epsilon(self, epsilon: float) -> None:
        """Update the exploration probability.

        Args:
            epsilon: New exploration probability; must be in [0, 1].

        Raises:
            ValueError: If epsilon is outside [0, 1].
        """
        if not 0.0 <= epsilon <= 1.0:
            msg = f"Epsilon must be in [0, 1], got {epsilon}."
            raise ValueError(msg)
        self.epsilon = epsilon

    def select_action(self, state: np.ndarray) -> int:
        """Choose an action using the epsilon-greedy rule.

        With probability epsilon a random action is returned (explore);
        otherwise the action with the highest predicted Q-value is returned
        (exploit).

        Args:
            state: Current state representation as a 1-D array.

        Returns:
            Index of the selected action.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        q_values = self.model.predict_one(state)
        return int(np.argmax(q_values))
