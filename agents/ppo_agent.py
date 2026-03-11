
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.base_agent import BaseAgent


class ActorCritic(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        logits = self.actor(feat)
        value = self.critic(feat)
        return logits, value

    def get_action(self, x: torch.Tensor, explore: bool = True):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        if explore:
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value.squeeze(-1), entropy


class RolloutBuffer:

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def push(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.obs)


class PPOAgent(BaseAgent):

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(agent_id, obs_dim, action_dim, config, device)

        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_range = config.get("clip_range", 0.2)
        self.n_epochs = config.get("n_epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.ent_coef = config.get("ent_coef", 0.01)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.lr = config.get("learning_rate", 0.0003)
        self.n_steps = config.get("n_steps", 2048)

        hidden = config.get("hidden_dims", [256, 256])
        h = hidden[0] if hidden else 256

        self.network = ActorCritic(obs_dim, action_dim, hidden=h).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        self.buffer = RolloutBuffer()

    def select_action(
        self, observation: np.ndarray, explore: bool = True
    ) -> int:
        with torch.no_grad():
            obs_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action, log_prob, value, _ = self.network.get_action(obs_t, explore)
        self._last_log_prob = log_prob.item()
        self._last_value = value.item()
        return int(action.item())

    def store_transition(
        self, obs: np.ndarray, action: int, reward: float, done: bool
    ) -> None:
        self.buffer.push(
            obs, action,
            self._last_log_prob, reward,
            self._last_value, done,
        )

    def learn(self, experiences: Any = None) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0}

        advantages, returns = self._compute_gae()

        obs_t = torch.FloatTensor(np.array(self.buffer.obs)).to(self.device)
        actions_t = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_loss = 0.0
        n = len(self.buffer)

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]

                logits, values = self.network(obs_t[batch_idx])
                values = values.squeeze(-1)

                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - old_log_probs_t[batch_idx]).exp()

                adv = advantages_t[batch_idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_range,
                                    1 + self.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, returns_t[batch_idx])

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / max(1, self.n_epochs * (n // self.batch_size))
        self.log_metric("loss", avg_loss)
        self.training_step += 1
        self.buffer.clear()

        return {"loss": avg_loss}

    def _compute_gae(self):
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        n = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_step": self.training_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_step = checkpoint["training_step"]

