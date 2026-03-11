
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.base_agent import BaseAgent
from core.replay_buffer import PrioritizedReplayBuffer, Transition


class DuelingQNetwork(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.LayerNorm(h)])
            in_dim = h
        self.features = nn.Sequential(*layers)

        self.value_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DQNAgent(BaseAgent):

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(agent_id, obs_dim, action_dim, config, device)

        hidden = config.get("hidden_dims", [256, 256])
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 0.001)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.batch_size = config.get("batch_size", 64)
        self.target_update_freq = config.get("target_update_freq", 100)

        self.q_network = DuelingQNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.target_network = DuelingQNetwork(obs_dim, action_dim, hidden).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        buf_size = config.get("buffer_size", 100_000)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buf_size)

    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            obs_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, transition: Transition) -> None:
        self.replay_buffer.push(transition)

    def learn(self, experiences: Any = None) -> Dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}

        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        weights_t = torch.FloatTensor(weights).to(self.device)

        obs_batch = torch.FloatTensor(
            np.array([t.obs[self.agent_id] for t in transitions])
        ).to(self.device)
        action_batch = torch.LongTensor(
            [t.actions[self.agent_id] for t in transitions]
        ).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(
            [t.rewards[self.agent_id] for t in transitions]
        ).to(self.device)
        next_obs_batch = torch.FloatTensor(
            np.array([t.next_obs[self.agent_id] for t in transitions])
        ).to(self.device)
        done_batch = torch.FloatTensor(
            [float(t.dones[self.agent_id]) for t in transitions]
        ).to(self.device)

        current_q = self.q_network(obs_batch).gather(1, action_batch).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_network(next_obs_batch).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_obs_batch).gather(1, next_actions).squeeze(1)
            target_q = reward_batch + self.gamma * next_q * (1 - done_batch)

        td_errors = (current_q - target_q).abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        loss = (weights_t * nn.functional.smooth_l1_loss(
            current_q, target_q, reduction="none"
        )).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        loss_val = loss.item()
        self.log_metric("loss", loss_val)
        self.log_metric("epsilon", self.epsilon)

        return {"loss": loss_val, "epsilon": self.epsilon}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_step": self.training_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]

