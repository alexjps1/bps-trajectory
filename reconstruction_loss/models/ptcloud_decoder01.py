# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# other imports
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Union, Tuple

# constants
PRECISION = np.float32

class BPSPointCloudDecoder01(nn.Module):
    num_basis_points: int
    scene_ndim: int
    input_dim: int
    max_num_scene_points: int

    def __init__(self, num_basis_points: int, max_num_scene_points: int, feature_dim: int):
        """
        Initialize the model.

        Parameters
        ----------
        num_basis_points: int
            Corresponds to the length of input to model
        num_scene_points: int
            Corresponds to length of output of model
        feature_dim: int
            1 for scalar-based BPS encoding, 2 or 3 for difference vector-based BPS encoding
        """
        super(BPSPointCloudDecoder01, self).__init__()

        if feature_dim not in range(1, 4):
            raise ValueError("feature_dim should be 1 for scalar-based BPS encoding, 2 or 3 for diff vector-based")

        # input dimensions
        self.num_basis_points = num_basis_points
        self.max_num_scene_points = max_num_scene_points
        self.scene_ndim = feature_dim
        self.input_dim = num_basis_points * feature_dim  # K * D

        global_feature_size = 128
        expansion_factor = 4

        # 1 feature bottleneck
        self.fc1 = nn.Linear(self.input_dim, global_feature_size)
        self.bn1 = nn.BatchNorm1d(global_feature_size)

        # 2 expansion layer
        self.fc2 = nn.Linear(global_feature_size, global_feature_size * expansion_factor)
        self.bn2 = nn.BatchNorm1d(global_feature_size * expansion_factor)

        # 3 final output layer
        # always assume 3d coordinates (if scene is 2d, we can cut off the z coord)
        self.output_layer = nn.Linear(global_feature_size * expansion_factor, self.max_num_scene_points * 3)

    def forward(self, bps_encoded_scene: torch.Tensor):

        # flatten input
        x = bps_encoded_scene.view(bps_encoded_scene.size(0), -1)

        # run through model layers
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))

        # Maps to a flat vector of all coordinates
        x = self.output_layer(x)

        # Reshape to (Batch Size, max num scene points, 3 Coordinates)
        point_cloud_out = x.view(-1, self.max_num_scene_points, 3)

        return point_cloud_out
