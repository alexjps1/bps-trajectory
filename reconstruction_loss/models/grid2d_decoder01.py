# pytorch imports
from typing import Tuple, Union

# other imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

# constants
PRECISION = np.float32


class BPSOccupancyGrid2dDecoder01(nn.Module):
    num_basis_points: int
    scene_ndim: int
    input_dim: int
    num_conv_layers: int

    def __init__(self, num_basis_points: int, feature_dim: int, scene_dims: Tuple[int, int], num_conv_layers: int = 4):
        """
        Initialize the model (BPS Code to 2D Occupancy Grid Decoder).

        Parameters
        ----------
        num_basis_points: int
        feature_dim: int
        scene_dims: Tuple[int, int]
        num_conv_layers: int
            Number of transposed convolution layers (upsampling steps).
        """
        super(BPSOccupancyGrid2dDecoder01, self).__init__()

        if feature_dim not in range(1, 4):
            raise ValueError(
                "feature_dim should be 1 for scalar-based BPS encoding, 2 or 3 for diff vector-based"
            )

        # set attributes
        self.num_basis_points = num_basis_points
        self.scene_ndim = feature_dim
        self.input_dim = num_basis_points * feature_dim
        self.output_dims = scene_dims
        self.num_conv_layers = num_conv_layers

        # determine  initial spatial size for the 2D tensor reshape
        # assume output dimensions are powers of 2
        target_size = scene_dims[0]
        initial_dim = target_size // (2**num_conv_layers)

        # feature bottleneck
        self.init_channels = 512
        self.initial_spatial_dim = initial_dim
        self.fc_start = nn.Linear(
            self.input_dim,
            self.init_channels * self.initial_spatial_dim * self.initial_spatial_dim,
        )
        self.bn_start = nn.BatchNorm1d(
            self.init_channels * self.initial_spatial_dim * self.initial_spatial_dim
        )

        layers = []
        in_channels = self.init_channels
        out_channels = self.init_channels // 2

        for i in range(num_conv_layers):
            # transposed convolution: increases resolution (stride=2) and halves channels
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,  # Commonly used for upsampling
                    stride=2,
                    padding=1,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            in_channels = out_channels
            # halve channels until the last layer, which must output 1 channel
            out_channels = in_channels // 2 if i < num_conv_layers - 2 else 1

        # final convolution to ensure precise output size and smooth the result
        layers.append(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        )

        # sigmoid activation: output is a probability map in range [0, 1]
        layers.append(nn.Sigmoid())

        self.deconv_stack = nn.Sequential(*layers)

    def forward(self, bps_encoded_scene: torch.Tensor) -> torch.Tensor:
        # flatten input
        x = bps_encoded_scene.view(bps_encoded_scene.size(0), -1)

        x = torch.relu(self.bn_start(self.fc_start(x)))
        x = x.view(
            -1, self.init_channels, self.initial_spatial_dim, self.initial_spatial_dim
        )
        predicted_occupancy_grid = self.deconv_stack(x)

        # remove redundant dimension (b, 1, h, w) -> (b, h, w)
        out = predicted_occupancy_grid.squeeze(1)
        return out
