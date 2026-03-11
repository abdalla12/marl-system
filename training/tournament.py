
import json
import os
import time
from itertools import combinations
from typing import Any, Dict, List, Type

import numpy as np

from agents.multi_agent_controller import MultiAgentController
from core.base_agent import BaseAgent
from core.base_environment import MultiAgentEnv


class Tournament:

    def __init__(
        self,
        env: MultiAgentEnv,
        agent_classes: Dict[str, Type[BaseAgent]],
        agent_configs: Dict[str, Dict[str, Any]],
        rounds: int = 10,
        episodes_per_match: int = 50,
        initial_elo: float = 1000.0,
        k_factor: float = 32.0,
        result_dir: str = "results/tournament",
    ):
        self.env = env
        self.agent_classes = agent_classes
        self.agent_configs = agent_configs
        self.rounds = rounds
        self.eps_per_match = episodes_per_match
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)

        self.elo_ratings: Dict[str, float] = {
            name: initial_elo for name in agent_classes
        }
        self.match_history: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"  MARL Tournament")
        print(f"  Environment: {self.env.__class__.__name__}")
        print(f"  Contestants: {list(self.agent_classes.keys())}")
        print(f"  Rounds: {self.rounds}")
        print(f"{'='*60}\n")

        matchups = list(combinations(self.agent_classes.keys(), 2))

        for round_num in range(1, self.rounds + 1):
            print(f"  Round {round_num}/{self.rounds}")

            for name_a, name_b in matchups:
                result = self._run_match(name_a, name_b)
                self._update_elo(name_a, name_b, result)
                self.match_history.append({
                    "round": round_num,
                    "agent_a": name_a,
                    "agent_b": name_b,
                    "mean_reward_a": result["mean_a"],
                    "mean_reward_b": result["mean_b"],
                    "winner": result["winner"],
                })

                print(f"    {name_a} vs {name_b}: "
                      f"{result['mean_a']:.2f} vs {result['mean_b']:.2f} "
                      f"→ {result['winner']}")

        self._print_leaderboard()
        summary = self._save_results()
        return summary

    def _run_match(
        self, name_a: str, name_b: str
    ) -> Dict[str, Any]:
        rewards_a = []
        rewards_b = []

        for _ in range(self.eps_per_match):
            reward_a = self._evaluate_agent(name_a)
            reward_b = self._evaluate_agent(name_b)
            rewards_a.append(reward_a)
            rewards_b.append(reward_b)

        mean_a = float(np.mean(rewards_a))
        mean_b = float(np.mean(rewards_b))

        if mean_a > mean_b:
            winner = name_a
        elif mean_b > mean_a:
            winner = name_b
        else:
            winner = "draw"

        return {"mean_a": mean_a, "mean_b": mean_b, "winner": winner}

    def _evaluate_agent(self, name: str) -> float:
        agent_class = self.agent_classes[name]
        config = self.agent_configs[name]

        obs_dim = int(np.prod(self.env.observation_space(0).shape))
        act_dim = int(self.env.action_space(0).n)

        agents = {
            aid: agent_class(aid, obs_dim, act_dim, config)
            for aid in self.env.agent_ids
        }

        obs, _ = self.env.reset()
        total_reward = 0.0

        while True:
            actions = {
                aid: agents[aid].select_action(obs[aid], explore=False)
                for aid in self.env.agent_ids
            }
            obs, rewards, terminated, truncated, _ = self.env.step(actions)
            total_reward += np.mean(list(rewards.values()))

            if any(terminated.values()) or any(truncated.values()):
                break

        return total_reward

    def _update_elo(
        self, name_a: str, name_b: str, result: Dict[str, Any]
    ) -> None:
        ra = self.elo_ratings[name_a]
        rb = self.elo_ratings[name_b]

        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 / (1 + 10 ** ((ra - rb) / 400))

        if result["winner"] == name_a:
            sa, sb = 1.0, 0.0
        elif result["winner"] == name_b:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.elo_ratings[name_a] = ra + self.k_factor * (sa - ea)
        self.elo_ratings[name_b] = rb + self.k_factor * (sb - eb)

    def _print_leaderboard(self) -> None:
        print(f"\n{'='*40}")
        print("  LEADERBOARD")
        print(f"{'='*40}")
        sorted_agents = sorted(
            self.elo_ratings.items(), key=lambda x: x[1], reverse=True
        )
        for rank, (name, elo) in enumerate(sorted_agents, 1):
            print(f"  #{rank}  {name:>10s}  ELO: {elo:.1f}")
        print()

    def _save_results(self) -> Dict[str, Any]:
        summary = {
            "elo_ratings": self.elo_ratings,
            "match_history": self.match_history,
            "total_matches": len(self.match_history),
        }
        path = os.path.join(self.result_dir, "tournament_results.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Results saved to: {path}")
        return summary

