
import os
import sys
import json
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.traffic_env import TrafficGridEnv
from envs.trading_env import TradingFloorEnv
from envs.supply_chain_env import SupplyChainEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.multi_agent_controller import MultiAgentController
from core.reward_shaper import RewardShaper
from core.replay_buffer import PrioritizedReplayBuffer, Transition
from training.trainer import Trainer


class TestTrafficEnv:
    def test_reset_returns_correct_shapes(self):
        env = TrafficGridEnv(grid_size=2, num_agents=2, max_steps=100)
        obs, infos = env.reset(seed=42)
        assert len(obs) == 2
        for aid in env.agent_ids:
            assert obs[aid].shape == (12,)
            assert obs[aid].dtype == np.float32

    def test_step_returns_correct_structure(self):
        env = TrafficGridEnv(num_agents=2, max_steps=50)
        obs, _ = env.reset(seed=42)
        actions = {aid: env.action_space(aid).sample() for aid in env.agent_ids}
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        assert len(next_obs) == 2
        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards.values())

    def test_episode_terminates(self):
        env = TrafficGridEnv(num_agents=2, max_steps=10)
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            actions = {aid: env.action_space(aid).sample() for aid in env.agent_ids}
            obs, rewards, terminated, truncated, infos = env.step(actions)
            done = any(terminated.values())
            steps += 1
        assert steps == 10

    def test_history_recorded(self):
        env = TrafficGridEnv(num_agents=2, max_steps=5)
        env.reset(seed=42)
        for _ in range(5):
            actions = {aid: env.action_space(aid).sample() for aid in env.agent_ids}
            env.step(actions)
        assert len(env.history) == 5


class TestTradingEnv:
    def test_reset_and_step(self):
        env = TradingFloorEnv(num_agents=3, max_steps=100, num_assets=2)
        obs, infos = env.reset(seed=42)
        assert len(obs) == 3
        actions = {aid: env.action_space(aid).sample() for aid in env.agent_ids}
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        assert all("portfolio_value" in infos[aid] for aid in env.agent_ids)

    def test_portfolio_value_positive(self):
        env = TradingFloorEnv(num_agents=2, max_steps=10)
        env.reset(seed=42)
        for _ in range(10):
            actions = {aid: 2 for aid in env.agent_ids}
            env.step(actions)
        for aid in env.agent_ids:
            assert env._portfolio_value(aid) > 0


class TestSupplyChainEnv:
    def test_reset_and_step(self):
        env = SupplyChainEnv(num_agents=4, max_steps=50)
        obs, infos = env.reset(seed=42)
        assert len(obs) == 4
        for aid in env.agent_ids:
            assert obs[aid].shape == (6,)
        actions = {aid: env.action_space(aid).sample() for aid in env.agent_ids}
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        assert all("inventory" in infos[aid] for aid in env.agent_ids)

    def test_demand_history_grows(self):
        env = SupplyChainEnv(num_agents=4, max_steps=20)
        env.reset(seed=42)
        for _ in range(10):
            actions = {aid: 5 for aid in env.agent_ids}
            env.step(actions)
        assert len(env.demand_history) == 10


class TestReplayBuffer:
    def test_push_and_sample(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(50):
            t = Transition(
                obs={0: np.zeros(4)}, actions={0: 0},
                rewards={0: float(i)}, next_obs={0: np.ones(4)},
                dones={0: False},
            )
            buf.push(t)
        assert len(buf) == 50
        transitions, indices, weights = buf.sample(16)
        assert len(transitions) == 16
        assert len(indices) == 16
        assert weights.shape == (16,)


class TestRewardShaper:
    def test_dense_passthrough(self):
        shaper = RewardShaper(strategy="dense")
        rewards = {0: 1.0, 1: -0.5}
        shaped = shaper.shape(rewards, {}, {}, {}, 0)
        assert shaped == rewards

    def test_cooperative_decompose(self):
        shaper = RewardShaper(cooperative_weight=0.5)
        rewards = {0: 10.0, 1: 0.0}
        decomposed = shaper.cooperative_decompose(rewards)
        assert decomposed[0] == pytest.approx(7.5)
        assert decomposed[1] == pytest.approx(2.5)


class TestDQNAgent:
    def test_select_action(self):
        agent = DQNAgent(0, 12, 4, {"epsilon_start": 0.0})
        obs = np.random.randn(12).astype(np.float32)
        action = agent.select_action(obs, explore=False)
        assert 0 <= action < 4

    def test_save_load(self):
        agent = DQNAgent(0, 12, 4, {})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "agent.pt")
            agent.save(path)
            agent.load(path)


class TestPPOAgent:
    def test_select_action(self):
        agent = PPOAgent(0, 12, 4, {})
        obs = np.random.randn(12).astype(np.float32)
        action = agent.select_action(obs, explore=True)
        assert 0 <= action < 4

    def test_save_load(self):
        agent = PPOAgent(0, 12, 4, {})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "agent.pt")
            agent.save(path)
            agent.load(path)


class TestMultiAgentController:
    def test_collect_actions(self):
        env = TrafficGridEnv(num_agents=2, max_steps=10)
        controller = MultiAgentController(env, DQNAgent, {}, mode="independent")
        obs, _ = env.reset(seed=42)
        actions = controller.collect_actions(obs)
        assert len(actions) == 2

    def test_shared_mode(self):
        env = TrafficGridEnv(num_agents=3, max_steps=10)
        controller = MultiAgentController(env, DQNAgent, {}, mode="shared")
        assert id(controller.agents[0]) == id(controller.agents[1])
        assert id(controller.agents[1]) == id(controller.agents[2])


class TestTrainer:
    def test_short_training_run(self):
        env = TrafficGridEnv(num_agents=2, max_steps=20)
        controller = MultiAgentController(env, DQNAgent, {}, mode="independent")
        with tempfile.TemporaryDirectory() as d:
            trainer = Trainer(
                env, controller,
                config={
                    "episodes": 5,
                    "eval_interval": 5,
                    "eval_episodes": 2,
                    "save_interval": 5,
                    "early_stopping_patience": 100,
                    "log_dir": os.path.join(d, "logs"),
                    "checkpoint_dir": os.path.join(d, "ckpt"),
                },
                experiment_name="test_run",
            )
            summary = trainer.train()
            assert "total_episodes" in summary
            assert os.path.isfile(
                os.path.join(d, "logs", "test_run", "training_log.json")
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

