
import os
from typing import Any, Dict, Optional, Type

import numpy as np

from core.base_agent import BaseAgent
from core.base_environment import MultiAgentEnv


def is_ray_available() -> bool:
    try:
        import ray
        return True
    except ImportError:
        return False


class DistributedTrainer:

    def __init__(
        self,
        env_class: Type[MultiAgentEnv],
        env_config: Dict[str, Any],
        agent_class: Type[BaseAgent],
        agent_config: Dict[str, Any],
        num_workers: int = 4,
        num_gpus: int = 0,
        episodes_per_worker: int = 10,
    ):
        self.env_class = env_class
        self.env_config = env_config
        self.agent_class = agent_class
        self.agent_config = agent_config
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.episodes_per_worker = episodes_per_worker

    def train(self, total_iterations: int = 100) -> Dict[str, Any]:
        if not is_ray_available():
            print("  [DistributedTrainer] Ray not available — "
                  "falling back to single-process training.")
            return self._train_single(total_iterations)

        import ray

        if not ray.is_initialized():
            ray.init(num_gpus=self.num_gpus, ignore_reinit_error=True)

        RolloutWorker = ray.remote(self._RolloutWorkerClass)

        workers = [
            RolloutWorker.remote(
                self.env_class, self.env_config,
                self.agent_class, self.agent_config,
                worker_id=i,
            )
            for i in range(self.num_workers)
        ]

        all_rewards = []

        for iteration in range(1, total_iterations + 1):
            futures = [
                w.collect_rollouts.remote(self.episodes_per_worker)
                for w in workers
            ]
            results = ray.get(futures)

            iter_rewards = []
            for res in results:
                iter_rewards.extend(res["mean_rewards"])

            mean_r = float(np.mean(iter_rewards))
            all_rewards.append(mean_r)

            if iteration % 10 == 0:
                print(f"  [Distributed] Iter {iteration}: "
                      f"mean_reward={mean_r:.2f}")

        ray.shutdown()
        return {"rewards": all_rewards, "iterations": total_iterations}

    def _train_single(self, total_iterations: int) -> Dict[str, Any]:
        from agents.multi_agent_controller import MultiAgentController
        from training.trainer import Trainer

        env = self.env_class(**self.env_config)
        controller = MultiAgentController(
            env, self.agent_class, self.agent_config, mode="independent"
        )
        trainer = Trainer(env, controller, {
            "episodes": total_iterations * self.episodes_per_worker,
            "eval_interval": 50,
            "save_interval": 100,
        })
        return trainer.train()

    class _RolloutWorkerClass:

        def __init__(
            self,
            env_class, env_config,
            agent_class, agent_config,
            worker_id: int = 0,
        ):
            self.env = env_class(**env_config)
            self.worker_id = worker_id

            obs_dim = int(np.prod(self.env.observation_space(0).shape))
            act_dim = int(self.env.action_space(0).n)

            self.agents = {
                aid: agent_class(aid, obs_dim, act_dim, agent_config)
                for aid in self.env.agent_ids
            }

        def collect_rollouts(self, num_episodes: int) -> Dict[str, Any]:
            mean_rewards = []

            for _ in range(num_episodes):
                obs, _ = self.env.reset()
                total_r = {aid: 0.0 for aid in self.env.agent_ids}

                while True:
                    actions = {
                        aid: self.agents[aid].select_action(obs[aid])
                        for aid in self.env.agent_ids
                    }
                    obs, rewards, terminated, truncated, _ = self.env.step(actions)
                    for aid in self.env.agent_ids:
                        total_r[aid] += rewards[aid]

                    if any(terminated.values()) or any(truncated.values()):
                        break

                mean_rewards.append(float(np.mean(list(total_r.values()))))

            return {"mean_rewards": mean_rewards, "worker_id": self.worker_id}

