
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from core.base_environment import MultiAgentEnv


class TrafficGridEnv(MultiAgentEnv):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        grid_size: int = 4,
        num_agents: int = 4,
        max_steps: int = 500,
        vehicle_spawn_rate: float = 0.3,
        yellow_phase_duration: int = 3,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(num_agents=num_agents, max_steps=max_steps,
                         render_mode=render_mode)
        self.grid_size = grid_size
        self.spawn_rate = vehicle_spawn_rate
        self.yellow_duration = yellow_phase_duration

        self.num_phases = 4
        self._obs_size = 12

        self.queues: Optional[np.ndarray] = None
        self.wait_times: Optional[np.ndarray] = None
        self.phases: Optional[np.ndarray] = None
        self.yellow_timers: Optional[np.ndarray] = None
        self.throughput: Optional[np.ndarray] = None

    def observation_space(self, agent_id: int) -> gym.Space:
        return gym.spaces.Box(low=0.0, high=1.0, shape=(self._obs_size,),
                              dtype=np.float32)

    def action_space(self, agent_id: int) -> gym.Space:
        return gym.spaces.Discrete(self.num_phases)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.history.clear()

        self.queues = np.random.randint(0, 5, size=(self.num_agents, 4)).astype(np.float32)
        self.wait_times = np.zeros((self.num_agents, 4), dtype=np.float32)
        self.phases = np.zeros(self.num_agents, dtype=np.int32)
        self.yellow_timers = np.zeros(self.num_agents, dtype=np.int32)
        self.throughput = np.zeros(self.num_agents, dtype=np.float32)

        obs = {aid: self._get_obs(aid) for aid in self.agent_ids}
        infos = {aid: {} for aid in self.agent_ids}
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
        rewards = {}

        for aid in self.agent_ids:
            action = int(actions[aid])

            if action != self.phases[aid]:
                self.yellow_timers[aid] = self.yellow_duration

            if self.yellow_timers[aid] > 0:
                self.yellow_timers[aid] -= 1
                if self.yellow_timers[aid] == 0:
                    self.phases[aid] = action
            else:
                self.phases[aid] = action

            green_dir = self.phases[aid]
            served = min(self.queues[aid, green_dir], 3)
            self.queues[aid, green_dir] -= served
            self.throughput[aid] += served

            for d in range(4):
                if self.queues[aid, d] > 0:
                    self.wait_times[aid, d] += 1
                if np.random.random() < self.spawn_rate:
                    self.queues[aid, d] += 1

            self.queues[aid] = np.clip(self.queues[aid], 0, 20)

            avg_wait = self.wait_times[aid].mean()
            total_queue = self.queues[aid].sum()
            rewards[aid] = float(-0.1 * avg_wait - 0.05 * total_queue + 0.2 * served)

        obs = {aid: self._get_obs(aid) for aid in self.agent_ids}
        done = self.current_step >= self.max_steps
        terminated = {aid: done for aid in self.agent_ids}
        truncated = {aid: False for aid in self.agent_ids}
        infos = {
            aid: {
                "throughput": float(self.throughput[aid]),
                "total_queue": float(self.queues[aid].sum()),
            }
            for aid in self.agent_ids
        }

        snapshot = self.get_state_snapshot()
        self._record_step(actions, rewards, snapshot)

        return obs, rewards, terminated, truncated, infos

    def get_state_snapshot(self) -> Dict[str, Any]:
        return {
            "step": self.current_step,
            "queues": self.queues.tolist(),
            "phases": self.phases.tolist(),
            "wait_times": self.wait_times.tolist(),
            "throughput": self.throughput.tolist(),
        }

    def _get_obs(self, agent_id: int) -> np.ndarray:
        queue_norm = self.queues[agent_id] / 20.0
        wait_norm = np.clip(self.wait_times[agent_id] / 50.0, 0, 1)
        phase_onehot = np.zeros(self.num_phases, dtype=np.float32)
        phase_onehot[self.phases[agent_id]] = 1.0
        return np.concatenate([queue_norm, wait_norm, phase_onehot]).astype(np.float32)

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            print(f"\n=== Step {self.current_step} ===")
            for aid in self.agent_ids:
                q = self.queues[aid]
                p = self.phases[aid]
                print(f"  Intersection {aid}: queues={q.tolist()}, "
                      f"phase={p}, throughput={self.throughput[aid]:.0f}")
        return None

