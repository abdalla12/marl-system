
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import AGENT_REGISTRY
from envs import ENV_REGISTRY
from training.tournament import Tournament


def parse_args():
    parser = argparse.ArgumentParser(description="MARL Tournament")
    parser.add_argument("--env", type=str, default="traffic",
                        choices=list(ENV_REGISTRY.keys()))
    parser.add_argument("--agents", type=str, default="dqn,ppo",
                        help="Comma-separated agent types to compete")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--episodes-per-match", type=int, default=50)
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_config = config.get("environments", {}).get(args.env, {})
    env_class = ENV_REGISTRY[args.env]
    env = env_class(**env_config)

    agent_names = [a.strip() for a in args.agents.split(",")]
    agent_classes = {name: AGENT_REGISTRY[name] for name in agent_names}
    agent_configs = {
        name: config.get("agents", {}).get(name, {})
        for name in agent_names
    }

    tournament = Tournament(
        env=env,
        agent_classes=agent_classes,
        agent_configs=agent_configs,
        rounds=args.rounds,
        episodes_per_match=args.episodes_per_match,
    )

    results = tournament.run()
    print(f"\nTournament Results: {results['elo_ratings']}")


if __name__ == "__main__":
    main()

