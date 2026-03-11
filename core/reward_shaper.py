
from enum import Enum
from typing import Any, Callable, Dict, Optional

import numpy as np


class ShapingStrategy(str, Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    CURIOSITY = "curiosity"
    POTENTIAL = "potential"


class RewardShaper:

    def __init__(
        self,
        strategy: str = "dense",
        cooperative_weight: float = 0.5,
        potential_fn: Optional[Callable] = None,
    ):
        self.strategy = ShapingStrategy(strategy)
        self.cooperative_weight = cooperative_weight
        self.potential_fn = potential_fn
        self._prev_potentials: Dict[int, float] = {}
        self._visit_counts: Dict[str, int] = {}

    def shape(
        self,
        rewards: Dict[int, float],
        observations: Dict[int, np.ndarray],
        actions: Dict[int, Any],
        next_observations: Dict[int, np.ndarray],
        step: int,
    ) -> Dict[int, float]:
        if self.strategy == ShapingStrategy.SPARSE:
            return self._sparse(rewards)
        elif self.strategy == ShapingStrategy.DENSE:
            return self._dense(rewards)
        elif self.strategy == ShapingStrategy.CURIOSITY:
            return self._curiosity(rewards, next_observations)
        elif self.strategy == ShapingStrategy.POTENTIAL:
            return self._potential_based(rewards, observations, next_observations)
        return rewards

    def cooperative_decompose(
        self, rewards: Dict[int, float]
    ) -> Dict[int, float]:
        team_reward = np.mean(list(rewards.values()))
        return {
            agent_id: (
                self.cooperative_weight * team_reward
                + (1 - self.cooperative_weight) * individual
            )
            for agent_id, individual in rewards.items()
        }

    def competitive_decompose(
        self, rewards: Dict[int, float]
    ) -> Dict[int, float]:
        mean_r = np.mean(list(rewards.values()))
        return {aid: r - mean_r for aid, r in rewards.items()}


    def _sparse(self, rewards: Dict[int, float]) -> Dict[int, float]:
        threshold = 0.1
        return {
            aid: r if abs(r) > threshold else 0.0
            for aid, r in rewards.items()
        }

    def _dense(self, rewards: Dict[int, float]) -> Dict[int, float]:
        return dict(rewards)

    def _curiosity(
        self,
        rewards: Dict[int, float],
        next_obs: Dict[int, np.ndarray],
    ) -> Dict[int, float]:
        shaped = {}
        for aid, r in rewards.items():
            obs_key = str(np.round(next_obs[aid], 2).tobytes())
            self._visit_counts[obs_key] = self._visit_counts.get(obs_key, 0) + 1
            curiosity_bonus = 1.0 / np.sqrt(self._visit_counts[obs_key])
            shaped[aid] = r + 0.1 * curiosity_bonus
        return shaped

    def _potential_based(
        self,
        rewards: Dict[int, float],
        obs: Dict[int, np.ndarray],
        next_obs: Dict[int, np.ndarray],
    ) -> Dict[int, float]:
        if self.potential_fn is None:
            return dict(rewards)

        shaped = {}
        gamma = 0.99
        for aid, r in rewards.items():
            phi_next = self.potential_fn(next_obs[aid])
            phi_curr = self._prev_potentials.get(aid, 0.0)
            shaped[aid] = r + gamma * phi_next - phi_curr
            self._prev_potentials[aid] = phi_next
        return shaped

    def reset(self) -> None:
        self._prev_potentials.clear()
        self._visit_counts.clear()

