import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch
import bps


class GridEnvCNN(gym.Env):
    """Custom Environment for Grid World"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, grid, start, goal, max_steps, bps_ = False):
        super().__init__()
        self.grid = np.array(grid, dtype=np.int8)
        self.size = self.grid.shape[0]
        self.start = start
        self.goal  = goal
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_position = list(self.start)
        self.action_space = spaces.Discrete(4) # up, down, left, right
        self.agent_position_buffer = [] # stores positions of the agent for early stopping if it does not move for 5 steps
        # self.observation_space = spaces.Box(low=0, high=255, shape=(3, self.size, self.size), dtype=np.uint8)
        self.n_channels = 3
        self.bps_ = bps_
        if self.bps_:
            basis = bps.generate_bps_ngrid(32, 2)
            full, empty = bps.create_scene_point_cloud(self.grid, create_empty_cloud=True)

            self.bps_encoding = bps.encode_scene(full, basis, "scalar", "none", empty, (32, 32))
            self.size = self.bps_encoding.shape[0]
            self.n_channels = 1

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.n_channels, self.size, self.size), dtype=np.uint8),
            "coords": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
        })
        


    def step(self, action):
        self.current_step += 1
        next_position = list(self.agent_position)
        
        if action == 0: next_position[0] -= 1 # up
        elif action == 1: next_position[0] += 1 # down
        elif action == 2: next_position[1] -= 1 # left
        elif action == 3: next_position[1] += 1 # right
        
        collision = False
        if 0 <= next_position[0] < self.size and 0 <= next_position[1] < self.size and self.grid[next_position[0], next_position[1]] == 0:
            self.agent_position = np.array(next_position, copy=True)
        else:
            collision = True

        self.agent_position_buffer.append(self.agent_position)

        if self.bps_:
            obs = (self.bps_encoding+1)/2 * 255
            obs = obs[np.newaxis, :, :]
        else:
            obs = np.zeros((3, self.size, self.size), dtype=np.uint8)
            obs[0] = (self.grid == 1).astype(np.uint8)        # obstacles
            obs[1, self.goal[0], self.goal[1]] = 1              # goal
            obs[2, self.agent_position[0], self.agent_position[1]] = 1  # agent
            obs *= 255

        
        # early stopping if the agent stays at the same position for 5 steps
        last5 = self.agent_position_buffer[-5:]
        stuck = all(np.array_equal(a, last5[0]) for a in last5) and len(last5) == 5
        
        terminated = np.array_equal(self.agent_position, self.goal)
        truncated = self.current_step >= self.max_steps or stuck


        reward = 0
        if terminated:
            reward += 10
        elif collision:
            reward -= 0.1
        elif stuck: # negative reward for beeing stuck
            reward -= 0.1
        else:
            reward -= 0.01
        
        info = {}
        ay, ax = self.agent_position
        gy, gx = self.goal
        coords = np.array([ay, ax, gy, gx], dtype=np.float32) / self.size # normalize

        return {'image': obs, 'coords': coords}, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        if self.bps_:
            obs = (self.bps_encoding+1)/2 * 255
            obs = obs[np.newaxis, :, :]
        else:
            obs = np.zeros((3, self.size, self.size), dtype=np.uint8)
            obs[0] = (self.grid == 1).astype(np.uint8)        # obstacles
            obs[1, self.goal[0], self.goal[1]] = 1              # goal
            obs[2, self.start[0], self.start[1]] = 1  # agent  # agent
            obs *= 255

        # observation = obs[np.newaxis, :, :] # shape to (1, Size, Size)
        self.agent_position = list(self.start)
        self.current_step = 0
        info = {}
        
        ay, ax = self.agent_position
        gy, gx = self.goal
        coords = np.array([ay, ax, gy, gx], dtype=np.float32) / self.size # normalize
        
        return {'image': obs, 'coords': coords}, info

    def render(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.uint8)
        obs[0] = (self.grid == 1).astype(np.uint8)        # obstacles
        obs[1, self.goal[0], self.goal[1]] = 1              # goal
        obs[2, self.agent_position[0], self.agent_position[1]] = 1  # agent
        obs *= 255

        observation = obs

        symbols = {0: '.', 1: '#', 2: 'A', 3: 'G'}
        print("\n".join(" ".join(symbols.get(v, '?') for v in row) for row in observation))
        print()




class GridEnvMLP(gym.Env):
    """Custom Environment for Grid World"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, grid, start, goal, max_steps):
        super().__init__()
        self.grid = np.array(grid, dtype=np.int8)
        self.size = self.grid.shape[0]
        self.start = start
        self.goal  = goal
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_position = list(self.start)
        self.action_space = spaces.Discrete(4) # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=3, shape=(1, self.size, self.size), dtype=np.uint8) # 0: free space, 1: obstacle, 2: agent, 3: goal
        


    def step(self, action):
        self.current_step += 1
        next_position = list(self.agent_position)
        prev_distance = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal))
        
        if action == 0: next_position[0] -= 1 # up
        elif action == 1: next_position[0] += 1 # down
        elif action == 2: next_position[1] -= 1 # left
        elif action == 3: next_position[1] += 1 # right
        
        collision = False
        if 0 <= next_position[0] < self.size and 0 <= next_position[1] < self.size and self.grid[next_position[0], next_position[1]] == 0:
            self.agent_position = np.array(next_position, copy=True)
        else:
            collision = True

        obs = np.zeros((3, self.size, self.size), dtype=np.uint8)
        obs = (self.grid == 1).astype(np.uint8)        # obstacles
        obs[self.goal[0], self.goal[1]] = 3              # goal
        obs[self.agent_position[0], self.agent_position[1]] = 2  # agent
        obs = obs[np.newaxis, :, :]

        # observation = obs[np.newaxis, :, :] # shape to (1, Size, Size)
        terminated = np.array_equal(self.agent_position, self.goal)
        truncated = self.current_step >= self.max_steps

        reward = 0
        if terminated:
            reward += 10
        elif collision:
            reward -= 0.1
        else:
            reward -= 0.01

        next_distance = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal))
        distance = prev_distance - next_distance
        # reward += distance * 0.01  # reward getting closer
        
        info = {}

        return obs, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        obs = np.zeros((3, self.size, self.size), dtype=np.uint8)
        obs = (self.grid == 1).astype(np.uint8)        # obstacles
        obs[self.goal[0], self.goal[1]] = 3              # goal
        obs[self.start[0], self.start[1]] = 2  # agent  # agent
        obs = obs[np.newaxis, :, :]

        # observation = obs[np.newaxis, :, :] # shape to (1, Size, Size)
        self.agent_position = list(self.start)
        self.current_step = 0
        info = {}
        
        return obs, info

    def render(self):
        obs = np.zeros((3, self.size, self.size), dtype=np.uint8)
        obs = (self.grid == 1).astype(np.uint8)        # obstacles
        obs[self.goal[0], self.goal[1]] = 3             # goal
        obs[self.agent_position[0], self.agent_position[1]] = 2  # agent
        obs = obs[np.newaxis, :, :]

        observation = obs

        symbols = {0: '.', 1: '#', 2: 'A', 3: 'G'}
        print("\n".join(" ".join(symbols.get(v, '?') for v in row) for row in observation))
        print()