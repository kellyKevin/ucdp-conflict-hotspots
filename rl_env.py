import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ConflictSupplyChainEnv(gym.Env):
    def __init__(self, risk_series):
        super(ConflictSupplyChainEnv, self).__init__()
        self.risk_series = risk_series.values
        self.current_step = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.max_steps = len(risk_series)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        observation = np.array([self.risk_series[self.current_step]], dtype=np.float32)
        return observation, {}

    def step(self, action):
        risk = self.risk_series[self.current_step]
        reward = 0
        if action == 0: # Normal
            reward = -100 if np.random.random() < risk else 10
        elif action == 1: # Reroute
            reward = 2
        elif action == 2: # Halt
            reward = -5

        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        observation = np.array([self.risk_series[self.current_step if not done else 0]], dtype=np.float32)
        return observation, reward, done, False, {}
