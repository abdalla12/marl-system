
import argparse
import sys
import os

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import AGENT_REGISTRY
from agents.multi_agent_controller import MultiAgentController
from core.reward_shaper import RewardShaper
from envs import ENV_REGISTRY
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="MARL Training System")
    parser.add_argument("--env", type=str, default="traffic",
                        choices=list(ENV_REGISTRY.keys()),
                        help="Environment name")
    parser.add_argument("--agent", type=str, default="dqn",
                        choices=list(AGENT_REGISTRY.keys()),
                        help="Agent algorithm")
    parser.add_argument("--num-agents", type=int, default=4,
                        help="Number of agents")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Training episodes (overrides config)")
    parser.add_argument("--mode", type=str, default="independent",
                        choices=["independent", "shared", "competitive"],
                        help="Multi-agent mode")
    parser.add_argument("--reward-strategy", type=str, default="dense",
                        choices=["sparse", "dense", "curiosity", "potential"],
                        help="Reward shaping strategy")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_config = config.get("environments", {}).get(args.env, {})
    env_config["num_agents"] = args.num_agents
    env_class = ENV_REGISTRY[args.env]
    env = env_class(**env_config)

    agent_config = config.get("agents", {}).get(args.agent, {})
    agent_class = AGENT_REGISTRY[args.agent]

    controller = MultiAgentController(
        env=env,
        agent_class=agent_class,
        agent_config=agent_config,
        mode=args.mode,
    )

    reward_shaper = RewardShaper(strategy=args.reward_strategy)

    training_config = config.get("training", {})
    if args.episodes:
        training_config["episodes"] = args.episodes

    experiment_name = args.experiment or f"{args.env}_{args.agent}_{args.mode}"

    trainer = Trainer(
        env=env,
        controller=controller,
        config=training_config,
        reward_shaper=reward_shaper,
        experiment_name=experiment_name,
    )

    summary = trainer.train()
    print(f"\nFinal Summary: {summary}")


if __name__ == "__main__":
    main()

