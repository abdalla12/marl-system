
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.multi_agent_controller import MultiAgentController

AGENT_REGISTRY = {
    "dqn": DQNAgent,
    "ppo": PPOAgent,
}

__all__ = ["DQNAgent", "PPOAgent", "MultiAgentController", "AGENT_REGISTRY"]

