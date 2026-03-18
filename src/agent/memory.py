import random
from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass(slots=True)
class Sample:
    """Single transition sample stored in replay memory."""

    state: NDArray
    action: int
    reward: float
    next_state: NDArray


class Memory:
    """Replay memory with a bounded size and a minimum warmup threshold.

    Uses a pre-allocated list as a circular ring buffer so that
    random.sample runs in O(k) instead of O(k * n) as with deque.
    """

    def __init__(self, size_max: int, size_min: int) -> None:
        """Initialize the replay memory.

        Args:
            size_max: Maximum number of samples to store.
            size_min: Minimum number of samples required before sampling.
        """
        self._buf: list[Sample | None] = [None] * size_max
        self._head: int = 0
        self._count: int = 0
        self.size_max = size_max
        self.size_min = size_min

    @property
    def is_ready(self) -> bool:
        """Return True when the buffer has at least size_min samples."""
        return self._count >= self.size_min

    def add_sample(self, sample: Sample) -> None:
        """Add a sample to memory.

        Args:
            sample: The sample to store.
        """
        self._buf[self._head] = sample
        self._head = (self._head + 1) % self.size_max
        self._count = min(self._count + 1, self.size_max)

    def get_samples(self, n: int) -> list[Sample]:
        """Return up to n random samples from memory.

        If the buffer is not yet ready or n is not positive, returns an empty list.

        Args:
            n: Number of samples to draw.

        Returns:
            A list of randomly drawn samples.
        """
        if n <= 0 or not self.is_ready:
            return []

        actual_n = min(n, self._count)
        indices = random.sample(range(self._count), actual_n)
        return [self._buf[i] for i in indices]  # type: ignore[return-value]

    def __len__(self) -> int:
        """Return the current number of stored samples."""
        return self._count
