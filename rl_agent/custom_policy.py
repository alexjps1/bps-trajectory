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


class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        image_observation_space = observation_space['image']
        coords_observation_space = observation_space['coords']

        # image input
        n_input_channels = image_observation_space.shape[0]
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )


        # self.cnn = nn.Sequential(
        #         nn.Conv2d(n_input_channels, 32, 3, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(32, 48, 3, stride=2, padding=1),               
        #         nn.ReLU(),
        #         nn.Conv2d(48, 64, 3, stride=2, padding=1),                
        #         nn.ReLU(),
        #         nn.Flatten()
        #     )

        self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 1), 
                nn.ReLU(),
                nn.MaxPool2d(3),  

                nn.Conv2d(32, 48, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(48, 48, 1),
                nn.ReLU(),
                nn.MaxPool2d(3),  

                nn.Conv2d(48, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 1), 
                nn.ReLU(),
                nn.MaxPool2d(3),  

                # nn.Conv2d(64, 80, 3, stride=1, padding=1),     
                # nn.ReLU(),
                # nn.Conv2d(80, 80, 1),
                # nn.ReLU(),
                # nn.MaxPool2d(3),

                nn.Flatten(),
            )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(image_observation_space.sample()[None]).float()
            ).shape[1]

        # coordinate input

        # self.l1 = nn.Sequential(
        #     nn.Linear(4, 4), nn.ReLU()
        # )

        self.l1 = nn.Sequential(
                nn.Linear(4, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
        print(n_flatten)

        self.linear = nn.Sequential(nn.Linear(n_flatten + 128, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, features_dim),
                                    nn.ReLU()
                                    )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        image_observations = observations['image']
        coords_observations = observations['coords']

        cnn_output = self.cnn(image_observations)
        f1_output = self.l1(coords_observations)
        
        return self.linear(torch.cat([cnn_output, f1_output], dim = 1))
    


class CustomCNNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomCNN,  #custom extractor
            features_extractor_kwargs=dict(features_dim=256), 
            **kwargs
        )



class CustomCNN_BPS(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(CustomCNN_BPS, self).__init__(observation_space, features_dim)
    

        image_observation_space = observation_space['image']
        coords_observation_space = observation_space['coords']

        # image input
        n_input_channels = image_observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),       

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(image_observation_space.sample()[None]).float()
            ).shape[1]

        self.l1 = nn.Sequential(
                nn.Linear(4, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
        print(n_flatten)

        self.linear = nn.Sequential(nn.Linear(n_flatten + 128, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, features_dim),
                                    nn.ReLU()
                                    )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        image_observations = observations['image']
        coords_observations = observations['coords']

        cnn_output = self.cnn(image_observations)
        f1_output = self.l1(coords_observations)
        
        return self.linear(torch.cat([cnn_output, f1_output], dim = 1))
    


class CustomCNNPolicyBPS(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomCNN_BPS,  #custom extractor
            features_extractor_kwargs=dict(features_dim=256), 
            **kwargs
        )
