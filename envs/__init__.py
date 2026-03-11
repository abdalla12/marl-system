
from envs.traffic_env import TrafficGridEnv
from envs.trading_env import TradingFloorEnv
from envs.supply_chain_env import SupplyChainEnv

ENV_REGISTRY = {
    "traffic": TrafficGridEnv,
    "trading": TradingFloorEnv,
    "supply_chain": SupplyChainEnv,
}

__all__ = ["TrafficGridEnv", "TradingFloorEnv", "SupplyChainEnv", "ENV_REGISTRY"]

