
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Transition:

    __slots__ = ["obs", "actions", "rewards", "next_obs", "dones"]

    def __init__(
        self,
        obs: Dict[int, np.ndarray],
        actions: Dict[int, Any],
        rewards: Dict[int, float],
        next_obs: Dict[int, np.ndarray],
        dones: Dict[int, bool],
    ):
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.next_obs = next_obs
        self.dones = dones


class PrioritizedReplayBuffer:

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self._buffer: List[Optional[Transition]] = [None] * capacity
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, transition: Transition, priority: Optional[float] = None) -> None:
        if priority is None:
            priority = self._priorities[: self._size].max() if self._size > 0 else 1.0

        self._buffer[self._position] = transition
        self._priorities[self._position] = (priority + self.epsilon) ** self.alpha
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        if self._size < batch_size:
            batch_size = self._size

        probs = self._priorities[: self._size]
        probs = probs / probs.sum()

        indices = np.random.choice(self._size, size=batch_size, p=probs, replace=False)

        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self._size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        transitions = [self._buffer[i] for i in indices]
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, prio in zip(indices, priorities):
            self._priorities[idx] = (abs(prio) + self.epsilon) ** self.alpha

    def clear(self) -> None:
        self._buffer = [None] * self.capacity
        self._priorities = np.zeros(self.capacity, dtype=np.float64)
        self._position = 0
        self._size = 0

