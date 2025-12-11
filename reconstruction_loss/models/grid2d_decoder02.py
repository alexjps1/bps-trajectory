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


class BPSOccupancyGrid2dDecoder02(nn.Module):
    """
    Reduced complexity compared to BPSOccupancyGrid2dDecoder01.
    Differences:
        - reduced initial channels to 128
        - parameter reduction
        - fc dropout (configurable, default 30%)
        - conv dropout (half of fc droput)
    """

    num_basis_points: int
    scene_ndim: int
    input_dim: int
    num_conv_layers: int

    def __init__(
        self,
        num_basis_points: int,
        feature_dim: int,
        scene_dims: Tuple[int, int],
        num_conv_layers: int = 4,
        init_channels: int = 128,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize the regularized model (BPS Code to 2D Occupancy Grid Decoder).

        Parameters
        ----------
        num_basis_points: int
            Number of basis points in the BPS encoding
        feature_dim: int
            Dimension of each BPS feature (1 for scalar, 2+ for difference vectors)
        scene_dims: Tuple[int, int]
            Output dimensions of the occupancy grid
        num_conv_layers: int
            Number of transposed convolution layers (upsampling steps)
        init_channels: int
            Number of initial channels after FC layer (reduced from 512 to 128)
        dropout_rate: float
            Dropout probability for regularization
        """
        super(BPSOccupancyGrid2dDecoder02, self).__init__()

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
        self.init_channels = init_channels
        self.dropout_rate = dropout_rate

        # determine initial spatial size for the 2D tensor reshape
        # assume output dimensions are powers of 2
        target_size = scene_dims[0]
        initial_dim = target_size // (2**num_conv_layers)
        self.initial_spatial_dim = initial_dim

        # reduced complexity feature bottleneck with dropout
        self.fc_start = nn.Linear(
            self.input_dim,
            self.init_channels * self.initial_spatial_dim * self.initial_spatial_dim,
        )
        self.bn_start = nn.BatchNorm1d(
            self.init_channels * self.initial_spatial_dim * self.initial_spatial_dim
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        # build transposed convolution layers
        layers = []
        in_channels = self.init_channels
        out_channels = self.init_channels // 2

        for i in range(num_conv_layers):
            # transposed convolution: increases resolution (stride=2) and reduces channels
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            # add dropout to intermediate layers for additional regularization
            if i < num_conv_layers - 1:  # not on the last layer
                layers.append(
                    nn.Dropout2d(self.dropout_rate * 0.5)
                )  # lighter dropout for conv layers

            in_channels = out_channels
            # halve channels until the last layer, which must output appropriate channels for final conv
            out_channels = max(in_channels // 2, 8) if i < num_conv_layers - 2 else 8

        # final convolution to ensure precise output size and smooth the result
        layers.append(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        )

        # sigmoid activation: output is a probability map in range [0, 1]
        layers.append(nn.Sigmoid())

        self.deconv_stack = nn.Sequential(*layers)

    def forward(self, bps_encoded_scene: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Parameters
        ----------
        bps_encoded_scene: torch.Tensor
            BPS encoded scene representation

        Returns
        -------
        torch.Tensor
            Reconstructed occupancy grid of shape (batch_size, height, width)
        """
        # flatten input
        x = bps_encoded_scene.view(bps_encoded_scene.size(0), -1)

        # fully connected layer with regularization
        x = self.fc_start(x)
        x = self.bn_start(x)
        x = torch.relu(x)
        x = self.dropout(x)  # dropout for regularization

        # reshape for convolutional layers
        x = x.view(
            -1, self.init_channels, self.initial_spatial_dim, self.initial_spatial_dim
        )

        # transposed convolution stack
        predicted_occupancy_grid = self.deconv_stack(x)

        # remove redundant dimension (batch_size, 1, height, width) -> (batch_size, height, width)
        out = predicted_occupancy_grid.squeeze(1)
        return out

    def get_parameter_count(self) -> int:
        """
        Return the total number of trainable parameters in the model.
        Useful for monitoring model complexity.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
