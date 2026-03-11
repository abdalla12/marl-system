
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch


class BaseAgent(ABC):

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.training_step = 0
        self.episode_count = 0
        self._metrics: Dict[str, list] = {
            "loss": [],
            "reward": [],
            "epsilon": [],
        }

    @abstractmethod
    def select_action(
        self, observation: np.ndarray, explore: bool = True
    ) -> int | np.ndarray:
        pass

    @abstractmethod
    def learn(self, experiences: Any) -> Dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    def reset(self) -> None:
        pass

    def log_metric(self, key: str, value: float) -> None:
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(value)

    def get_metrics(self) -> Dict[str, list]:
        return dict(self._metrics)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.agent_id}, "
            f"obs={self.obs_dim}, act={self.action_dim})"
        )

