"""PyTorch CNN definitions used only for training."""

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.layers(x))


class BoardExtractor(BaseFeaturesExtractor):
    """Residual CNN for a 16-channel 3×3 observation."""

    def __init__(self, observation_space: gym.spaces.Box, *, filters: int = 64,
                 residual_blocks: int = 4, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        channels, height, width = observation_space.shape
        self.network = nn.Sequential(
            nn.Conv2d(channels, filters, 3, padding=1), nn.ReLU(),
            *[_ResidualBlock(filters) for _ in range(residual_blocks)],
            nn.Flatten(), nn.Linear(filters * height * width, features_dim), nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations.float())
