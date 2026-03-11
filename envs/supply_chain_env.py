
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from core.base_environment import MultiAgentEnv


ROLE_NAMES = ["Supplier", "Manufacturer", "Distributor", "Retailer"]


class SupplyChainEnv(MultiAgentEnv):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_agents: int = 4,
        max_steps: int = 200,
        max_inventory: int = 500,
        holding_cost: float = 1.0,
        backorder_cost: float = 5.0,
        ordering_cost: float = 2.0,
        demand_mean: float = 50.0,
        demand_std: float = 15.0,
        lead_time: int = 2,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(num_agents=num_agents, max_steps=max_steps,
                         render_mode=render_mode)
        self.max_inventory = max_inventory
        self.holding_cost = holding_cost
        self.backorder_cost = backorder_cost
        self.ordering_cost = ordering_cost
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.lead_time = lead_time

        self._obs_size = 6

        self.inventory: Optional[np.ndarray] = None
        self.backorders: Optional[np.ndarray] = None
        self.incoming_orders: Optional[list] = None
        self.in_transit: Optional[list] = None
        self.demand_history: list = []

    def observation_space(self, agent_id: int) -> gym.Space:
        return gym.spaces.Box(low=-1.0, high=1.0,
                              shape=(self._obs_size,), dtype=np.float32)

    def action_space(self, agent_id: int) -> gym.Space:
        return gym.spaces.Discrete(11)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.history.clear()
        self.demand_history.clear()

        self.inventory = np.full(self.num_agents, 100.0, dtype=np.float64)
        self.backorders = np.zeros(self.num_agents, dtype=np.float64)

        self.incoming_orders = [
            [0.0] * self.lead_time for _ in range(self.num_agents)
        ]
        self.in_transit = [
            [0.0] * self.lead_time for _ in range(self.num_agents)
        ]

        obs = {aid: self._get_obs(aid) for aid in self.agent_ids}
        infos = {aid: {"role": ROLE_NAMES[aid % len(ROLE_NAMES)]}
                 for aid in self.agent_ids}
        return obs, infos

    def step(
        self, actions: Dict[int, Any]
    ) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[int, bool],
        Dict[int, Dict],
    ]:
        self.current_step += 1

        order_quantities = {}
        for aid in self.agent_ids:
            action = int(actions[aid])
            order_qty = action * (self.max_inventory / 10)
            order_quantities[aid] = order_qty

        customer_demand = max(0, np.random.normal(self.demand_mean, self.demand_std))
        self.demand_history.append(customer_demand)

        rewards = {}
        for aid in self.agent_ids:
            arrived = self.in_transit[aid].pop(0)
            self.inventory[aid] += arrived

            if aid == self.num_agents - 1:
                demand = customer_demand
            else:
                demand = order_quantities.get(aid + 1, 0)

            self.incoming_orders[aid].append(demand)
            _ = self.incoming_orders[aid].pop(0)

            fulfilled = min(self.inventory[aid], demand + self.backorders[aid])
            self.inventory[aid] -= fulfilled

            unfulfilled = (demand + self.backorders[aid]) - fulfilled
            self.backorders[aid] = max(0, unfulfilled)

            order_qty = order_quantities[aid]
            if aid == 0:
                shipped = order_qty
            else:
                upstream_inv = self.inventory[aid - 1]
                shipped = min(order_qty, upstream_inv)

            self.in_transit[aid].append(shipped)

            h_cost = self.holding_cost * max(0, self.inventory[aid])
            b_cost = self.backorder_cost * self.backorders[aid]
            o_cost = self.ordering_cost * (order_qty / self.max_inventory)

            rewards[aid] = float(-(h_cost + b_cost + o_cost))

            self.inventory[aid] = np.clip(self.inventory[aid], -self.max_inventory,
                                          self.max_inventory)

        obs = {aid: self._get_obs(aid) for aid in self.agent_ids}
        done = self.current_step >= self.max_steps
        terminated = {aid: done for aid in self.agent_ids}
        truncated = {aid: False for aid in self.agent_ids}
        infos = {
            aid: {
                "inventory": float(self.inventory[aid]),
                "backorders": float(self.backorders[aid]),
                "role": ROLE_NAMES[aid % len(ROLE_NAMES)],
            }
            for aid in self.agent_ids
        }

        snapshot = self.get_state_snapshot()
        self._record_step(actions, rewards, snapshot)

        return obs, rewards, terminated, truncated, infos

    def get_state_snapshot(self) -> Dict[str, Any]:
        return {
            "step": self.current_step,
            "inventory": self.inventory.tolist(),
            "backorders": self.backorders.tolist(),
            "demand_history": list(self.demand_history[-20:]),
        }

    def _get_obs(self, agent_id: int) -> np.ndarray:
        inv_norm = self.inventory[agent_id] / self.max_inventory
        bo_norm = self.backorders[agent_id] / self.max_inventory

        last_order = (self.incoming_orders[agent_id][-1] / self.max_inventory
                      if self.incoming_orders[agent_id] else 0.0)

        if len(self.demand_history) >= 5:
            recent = self.demand_history[-5:]
            trend = (recent[-1] - recent[0]) / (self.demand_mean + 1e-8)
        else:
            trend = 0.0

        transit_sum = sum(self.in_transit[agent_id]) / self.max_inventory

        chain_pos = agent_id / max(1, self.num_agents - 1)

        return np.array([inv_norm, bo_norm, last_order, trend, transit_sum,
                         chain_pos], dtype=np.float32)

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            print(f"\n=== Step {self.current_step} ===")
            if self.demand_history:
                print(f"  Customer demand: {self.demand_history[-1]:.1f}")
            for aid in self.agent_ids:
                role = ROLE_NAMES[aid % len(ROLE_NAMES)]
                print(f"  {role} (Agent {aid}): "
                      f"inv={self.inventory[aid]:.1f}, "
                      f"backorders={self.backorders[aid]:.1f}, "
                      f"in_transit={sum(self.in_transit[aid]):.1f}")
        return None

