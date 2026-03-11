
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from core.base_environment import MultiAgentEnv


class TradingFloorEnv(MultiAgentEnv):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_agents: int = 4,
        max_steps: int = 1000,
        initial_balance: float = 100_000.0,
        num_assets: int = 3,
        lookback_window: int = 20,
        transaction_cost: float = 0.001,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(num_agents=num_agents, max_steps=max_steps,
                         render_mode=render_mode)
        self.initial_balance = initial_balance
        self.num_assets = num_assets
        self.lookback = lookback_window
        self.tx_cost = transaction_cost

        self._obs_size = num_assets * 4 + num_assets + 1

        self.prices: Optional[np.ndarray] = None
        self.price_history: list = []
        self.balances: Optional[np.ndarray] = None
        self.holdings: Optional[np.ndarray] = None
        self.portfolio_values: list = []

        self._mu = np.array([0.0005, 0.0003, 0.0007])[:num_assets]
        self._sigma = np.array([0.02, 0.015, 0.025])[:num_assets]

    def observation_space(self, agent_id: int) -> gym.Space:
        return gym.spaces.Box(low=-np.inf, high=np.inf,
                              shape=(self._obs_size,), dtype=np.float32)

    def action_space(self, agent_id: int) -> gym.Space:
        return gym.spaces.Discrete(self.num_assets * 3)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.history.clear()

        self.prices = np.array([100.0] * self.num_assets, dtype=np.float64)
        self.price_history = [self.prices.copy()]

        self.balances = np.full(self.num_agents, self.initial_balance, dtype=np.float64)
        self.holdings = np.zeros((self.num_agents, self.num_assets), dtype=np.float64)
        self.portfolio_values = []

        obs = {aid: self._get_obs(aid) for aid in self.agent_ids}
        infos = {aid: {"portfolio_value": float(self.initial_balance)}
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
        prev_values = {
            aid: self._portfolio_value(aid) for aid in self.agent_ids
        }

        self._evolve_prices()

        rewards = {}
        for aid in self.agent_ids:
            action = int(actions[aid])
            asset_idx = action // 3
            trade_type = action % 3

            trade_amount = self.balances[aid] * 0.1

            if trade_type == 0:
                shares = trade_amount / self.prices[asset_idx]
                cost = trade_amount * (1 + self.tx_cost)
                if cost <= self.balances[aid]:
                    self.holdings[aid, asset_idx] += shares
                    self.balances[aid] -= cost
            elif trade_type == 1:
                if self.holdings[aid, asset_idx] > 0:
                    shares_to_sell = self.holdings[aid, asset_idx] * 0.5
                    revenue = shares_to_sell * self.prices[asset_idx] * (1 - self.tx_cost)
                    self.holdings[aid, asset_idx] -= shares_to_sell
                    self.balances[aid] += revenue

            curr_value = self._portfolio_value(aid)
            ret = (curr_value - prev_values[aid]) / (prev_values[aid] + 1e-8)
            rewards[aid] = float(ret * 100)

        self.current_step += 1

        step_values = {aid: self._portfolio_value(aid) for aid in self.agent_ids}
        self.portfolio_values.append(step_values)

        obs = {aid: self._get_obs(aid) for aid in self.agent_ids}
        done = self.current_step >= self.max_steps
        terminated = {aid: done for aid in self.agent_ids}
        truncated = {aid: False for aid in self.agent_ids}
        infos = {
            aid: {"portfolio_value": step_values[aid]} for aid in self.agent_ids
        }

        snapshot = self.get_state_snapshot()
        self._record_step(actions, rewards, snapshot)

        return obs, rewards, terminated, truncated, infos

    def get_state_snapshot(self) -> Dict[str, Any]:
        return {
            "step": self.current_step,
            "prices": self.prices.tolist(),
            "balances": self.balances.tolist(),
            "holdings": self.holdings.tolist(),
        }

    def _evolve_prices(self) -> None:
        dt = 1.0 / 252
        z = np.random.standard_normal(self.num_assets)
        self.prices = self.prices * np.exp(
            (self._mu - 0.5 * self._sigma ** 2) * dt + self._sigma * np.sqrt(dt) * z
        )
        self.prices = np.maximum(self.prices, 0.01)
        self.price_history.append(self.prices.copy())

    def _portfolio_value(self, agent_id: int) -> float:
        return float(
            self.balances[agent_id]
            + np.dot(self.holdings[agent_id], self.prices)
        )

    def _get_obs(self, agent_id: int) -> np.ndarray:
        history = np.array(self.price_history[-self.lookback:])

        if len(history) < 2:
            returns = np.zeros(self.num_assets)
            volatility = np.zeros(self.num_assets)
            momentum = np.zeros(self.num_assets)
        else:
            log_returns = np.diff(np.log(history), axis=0)
            returns = log_returns[-1] if len(log_returns) > 0 else np.zeros(self.num_assets)
            volatility = log_returns.std(axis=0) if len(log_returns) > 1 else np.zeros(self.num_assets)
            momentum = (history[-1] / history[0] - 1) if len(history) > 1 else np.zeros(self.num_assets)

        price_level = self.prices / 100.0

        holdings_norm = self.holdings[agent_id] / 1000.0
        balance_norm = np.array([self.balances[agent_id] / self.initial_balance])

        obs = np.concatenate([
            returns, volatility, momentum, price_level,
            holdings_norm, balance_norm,
        ]).astype(np.float32)

        return obs

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            print(f"\n=== Step {self.current_step} ===")
            print(f"  Prices: {self.prices.round(2).tolist()}")
            for aid in self.agent_ids:
                pv = self._portfolio_value(aid)
                print(f"  Agent {aid}: balance=${self.balances[aid]:.2f}, "
                      f"holdings={self.holdings[aid].round(2).tolist()}, "
                      f"value=${pv:.2f}")
        return None

