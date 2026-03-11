
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class MultiAgentEnv(ABC):

    metadata: Dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_agents: int,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.current_step = 0
        self._agent_ids = list(range(num_agents))
        self.history: List[Dict[str, Any]] = []

    @property
    def agent_ids(self) -> List[int]:
        return list(self._agent_ids)

    @abstractmethod
    def observation_space(self, agent_id: int) -> gym.Space:
        pass

    @abstractmethod
    def action_space(self, agent_id: int) -> gym.Space:
        pass

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        pass

    @abstractmethod
    def step(
        self, actions: Dict[int, Any]
    ) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[int, bool],
        Dict[int, Dict],
    ]:
        pass

    def render(self) -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        pass

    def get_state_snapshot(self) -> Dict[str, Any]:
        return {"step": self.current_step}

    def _record_step(self, actions, rewards, snapshot):
        self.history.append(
            {
                "step": self.current_step,
                "actions": {k: int(v) if np.isscalar(v) else v.tolist()
                            for k, v in actions.items()},
                "rewards": {k: float(v) for k, v in rewards.items()},
                "snapshot": snapshot,
            }
        )

