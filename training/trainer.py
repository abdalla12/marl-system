
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from agents.multi_agent_controller import MultiAgentController
from core.base_environment import MultiAgentEnv
from core.reward_shaper import RewardShaper


class Trainer:

    def __init__(
        self,
        env: MultiAgentEnv,
        controller: MultiAgentController,
        config: Dict[str, Any],
        reward_shaper: Optional[RewardShaper] = None,
        experiment_name: Optional[str] = None,
    ):
        self.env = env
        self.controller = controller
        self.config = config
        self.reward_shaper = reward_shaper or RewardShaper(strategy="dense")

        self.episodes = config.get("episodes", 1000)
        self.eval_interval = config.get("eval_interval", 50)
        self.eval_episodes = config.get("eval_episodes", 10)
        self.save_interval = config.get("save_interval", 100)
        self.patience = config.get("early_stopping_patience", 200)
        self.seed = config.get("seed", 42)

        self.experiment_name = experiment_name or f"exp_{int(time.time())}"

        log_dir = config.get("log_dir", "logs")
        self.result_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.result_dir, exist_ok=True)

        self.checkpoint_dir = os.path.join(
            config.get("checkpoint_dir", "checkpoints"), self.experiment_name
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.training_log: List[Dict[str, Any]] = []
        self.best_eval_reward = -float("inf")
        self.episodes_without_improvement = 0

    def train(self) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"  MARL Training: {self.experiment_name}")
        print(f"  Environment: {self.env.__class__.__name__}")
        print(f"  Agents: {len(self.env.agent_ids)} × "
              f"{list(self.controller.agents.values())[0].__class__.__name__}")
        print(f"  Episodes: {self.episodes}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for episode in tqdm(range(1, self.episodes + 1), desc="Training"):
            episode_data = self._run_episode(explore=True)

            learn_metrics = self.controller.learn_all()

            record = {
                "episode": episode,
                "rewards": {str(k): v for k, v in episode_data["total_rewards"].items()},
                "mean_reward": float(np.mean(list(episode_data["total_rewards"].values()))),
                "steps": episode_data["steps"],
                "learn_metrics": {
                    str(k): v for k, v in learn_metrics.items()
                },
            }
            self.training_log.append(record)

            if episode % self.eval_interval == 0:
                eval_result = self.evaluate()
                record["eval"] = eval_result
                mean_eval = eval_result["mean_reward"]

                tqdm.write(
                    f"  [Ep {episode}] train={record['mean_reward']:.2f}  "
                    f"eval={mean_eval:.2f}  best={self.best_eval_reward:.2f}"
                )

                if mean_eval > self.best_eval_reward:
                    self.best_eval_reward = mean_eval
                    self.episodes_without_improvement = 0
                    self.controller.save_all(
                        os.path.join(self.checkpoint_dir, "best")
                    )
                else:
                    self.episodes_without_improvement += self.eval_interval

                if self.episodes_without_improvement >= self.patience:
                    tqdm.write(f"  Early stopping at episode {episode}")
                    break

            if episode % self.save_interval == 0:
                self.controller.save_all(
                    os.path.join(self.checkpoint_dir, f"ep_{episode}")
                )

        elapsed = time.time() - start_time

        summary = {
            "experiment": self.experiment_name,
            "total_episodes": len(self.training_log),
            "best_eval_reward": self.best_eval_reward,
            "elapsed_seconds": elapsed,
            "final_mean_reward": self.training_log[-1]["mean_reward"],
        }

        self._save_results(summary)
        print(f"\n  Training complete in {elapsed:.1f}s")
        print(f"  Best eval reward: {self.best_eval_reward:.2f}")
        print(f"  Results saved to: {self.result_dir}\n")

        return summary

    def evaluate(self) -> Dict[str, Any]:
        eval_rewards = []
        for _ in range(self.eval_episodes):
            data = self._run_episode(explore=False)
            eval_rewards.append(
                np.mean(list(data["total_rewards"].values()))
            )
        return {
            "mean_reward": float(np.mean(eval_rewards)),
            "std_reward": float(np.std(eval_rewards)),
            "min_reward": float(np.min(eval_rewards)),
            "max_reward": float(np.max(eval_rewards)),
        }

    def _run_episode(self, explore: bool = True) -> Dict[str, Any]:
        obs, infos = self.env.reset(seed=None)
        self.controller.reset_all()
        if self.reward_shaper:
            self.reward_shaper.reset()

        total_rewards = {aid: 0.0 for aid in self.env.agent_ids}
        steps = 0

        while True:
            actions = self.controller.collect_actions(obs, explore=explore)
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

            if self.reward_shaper:
                rewards = self.reward_shaper.shape(
                    rewards, obs, actions, next_obs, steps
                )

            dones = {
                aid: terminated[aid] or truncated[aid]
                for aid in self.env.agent_ids
            }

            if explore:
                self.controller.store_transitions(
                    obs, actions, rewards, next_obs, dones
                )

            for aid in self.env.agent_ids:
                total_rewards[aid] += rewards[aid]

            steps += 1
            obs = next_obs

            if any(terminated.values()) or any(truncated.values()):
                break

        return {"total_rewards": total_rewards, "steps": steps}

    def _save_results(self, summary: Dict[str, Any]) -> None:
        log_path = os.path.join(self.result_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)

        summary_path = os.path.join(self.result_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

