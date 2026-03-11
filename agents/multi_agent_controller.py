
import os
from typing import Any, Dict, List, Optional, Type

import numpy as np

from core.base_agent import BaseAgent
from core.base_environment import MultiAgentEnv
from core.replay_buffer import Transition


class MultiAgentController:

    MODES = ("independent", "shared", "competitive")

    def __init__(
        self,
        env: MultiAgentEnv,
        agent_class: Type[BaseAgent],
        agent_config: Dict[str, Any],
        mode: str = "independent",
        device: Optional[str] = None,
    ):
        assert mode in self.MODES, f"Unknown mode '{mode}', pick from {self.MODES}"
        self.env = env
        self.mode = mode
        self.agent_config = agent_config

        sample_obs_space = env.observation_space(0)
        sample_act_space = env.action_space(0)
        obs_dim = int(np.prod(sample_obs_space.shape))
        action_dim = int(sample_act_space.n)

        if mode == "shared":
            shared_agent = agent_class(
                agent_id=0,
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=agent_config,
                device=device,
            )
            self.agents: Dict[int, BaseAgent] = {
                aid: shared_agent for aid in env.agent_ids
            }
        else:
            self.agents = {
                aid: agent_class(
                    agent_id=aid,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    config=agent_config,
                    device=device,
                )
                for aid in env.agent_ids
            }

    def collect_actions(
        self, observations: Dict[int, np.ndarray], explore: bool = True
    ) -> Dict[int, Any]:
        return {
            aid: self.agents[aid].select_action(observations[aid], explore)
            for aid in self.env.agent_ids
        }

    def store_transitions(
        self,
        obs: Dict[int, np.ndarray],
        actions: Dict[int, Any],
        rewards: Dict[int, float],
        next_obs: Dict[int, np.ndarray],
        dones: Dict[int, bool],
    ) -> None:
        for aid in self.env.agent_ids:
            agent = self.agents[aid]

            if hasattr(agent, "store_transition"):
                if hasattr(agent, "buffer"):
                    agent.store_transition(
                        obs[aid], actions[aid],
                        rewards[aid], dones[aid],
                    )
                elif hasattr(agent, "replay_buffer"):
                    transition = Transition(
                        obs=obs, actions=actions,
                        rewards=rewards, next_obs=next_obs,
                        dones=dones,
                    )
                    agent.store_transition(transition)

    def learn_all(self) -> Dict[int, Dict[str, float]]:
        results = {}
        seen = set()
        for aid in self.env.agent_ids:
            agent = self.agents[aid]
            if id(agent) in seen:
                continue
            seen.add(id(agent))
            results[aid] = agent.learn()
        return results

    def reset_all(self) -> None:
        seen = set()
        for aid in self.env.agent_ids:
            agent = self.agents[aid]
            if id(agent) not in seen:
                agent.reset()
                seen.add(id(agent))

    def save_all(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        seen = set()
        for aid in self.env.agent_ids:
            agent = self.agents[aid]
            if id(agent) in seen:
                continue
            seen.add(id(agent))
            path = os.path.join(directory, f"agent_{aid}.pt")
            agent.save(path)

    def load_all(self, directory: str) -> None:
        seen = set()
        for aid in self.env.agent_ids:
            agent = self.agents[aid]
            if id(agent) in seen:
                continue
            seen.add(id(agent))
            path = os.path.join(directory, f"agent_{aid}.pt")
            if os.path.exists(path):
                agent.load(path)

    def get_all_metrics(self) -> Dict[int, Dict[str, list]]:
        return {aid: self.agents[aid].get_metrics() for aid in self.env.agent_ids}

