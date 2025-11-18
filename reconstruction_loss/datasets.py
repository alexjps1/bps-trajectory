"""
Dataset for Scene Reconstruction Loss
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2025-11-18
"""

# imports
import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Union, Tuple, List

# import bps from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)
import bps

# constants
PRECISION = np.float32

class Scenes2dDataset(Dataset):
    data: np.ndarray
    bps: np.ndarray
    bps_encoding_type: str
    norm_bound_shape: str
    target_encoding: str
    max_num_scene_points: int

    def __init__(self,
        data_array: np.ndarray,
        basis_point_cloud: np.ndarray,
        bps_encoding_type: str,
        norm_bound_shape: str,
        target_encoding: str,
        max_num_scene_points: int
    ) -> None:
        """
        Initializes the dataset.

        Parameters
        ----------
        data_array: np.ndarray
            Array of occupancy grids. Should have 3 dims, because it's an array of 2d scenes.
        basis_point_cloud: np.ndarray
            Array of 2d points in scene
        bps_encoding_type: str "scalar" or "diff"
            use scalar for 1d features (scalar BPS encoding)
            use diff for 2d features (difference vector BPS encoding)
        norm_bound_shape: str "ncube" or "nsphere"
            which shape to use for scene normalization
            use ncube with grid-based BPS or nsphere with random sampling-based BPS
        target_encoding: str "cloud" or "grid"
            representation of target (original scene) to use for loss fn
            if cloud, __getitem__ outputs [bps_encoded, scene_point_cloud, num_points_in_scene]
            if grid, __getitem__ outputs [bps_encoded, scene_occupancy_grid]
        max_num_scene_points: int
            used to pad target point clouds to prevent tensor-stacking issues in train/test loop
        """
        if not isinstance(data_array, np.ndarray):
            raise TypeError("data_array must be np.ndarray (not a filepath, for example)")
        if bps_encoding_type not in ["scalar", "diff"]:
            raise ValueError("bps_encoding_type must be 'scalar' or 'diff'")
        if target_encoding not in ["cloud", "grid"]:
            raise ValueError("original_scene_encoding must be 'cloud' or 'grid'")
        self.bps_encoding_type = bps_encoding_type
        self.data = data_array
        self.bps = basis_point_cloud
        self.target_encoding = target_encoding
        self.norm_bound_shape = norm_bound_shape
        self.max_num_scene_points = max_num_scene_points

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return self.data.shape[0]

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> \
        Union[Tuple[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns the BPS-encoded and target scene representations for a given sample.
        The target scene representation can be in point cloud or grid form depending on self.target_encoding

        Parameters
        ----------
        idx: int or torch.Tensor
            Index for scene to be loaded

        Returns
        -------
        [bps_encoded_tensor, scene_occupancy_grid_tensor]
            if original_scene_encoding == "grid"
        [bps_encoded_tensor, scene_point_cloud_tensor, num_points_in_scene]
            if original_scene_encoding == "cloud"
        """
        index: Union[int, torch.Tensor, List]
        if isinstance(idx, torch.Tensor):
            index = idx.tolist()
        else:
            index = idx

        scene_occupancy_grid_raw: np.ndarray = self.data[index]

        # turn all nonzero elements into 1 (otherwise binary cross entropy breaks)
        scene_occupancy_grid = (scene_occupancy_grid_raw != 0).astype(int)

        # create bps encoding
        scene_point_cloud: np.ndarray = bps.create_point_cloud(scene_occupancy_grid)
        encoded_result = bps.encode_scene(scene_point_cloud, self.bps, self.bps_encoding_type, self.norm_bound_shape)
        assert isinstance(encoded_result, np.ndarray)  # encode_scene should return an ndarray based on our encoding_type
        bps_encoded: np.ndarray = encoded_result
        bps_encoded_tensor: torch.Tensor = torch.from_numpy(bps_encoded.astype(PRECISION))

        if self.target_encoding == "grid":
            scene_occupancy_grid_tensor: torch.Tensor = torch.from_numpy(scene_occupancy_grid.astype(PRECISION))
            return (bps_encoded_tensor, scene_occupancy_grid_tensor)
        elif self.target_encoding == "cloud":
            # pad the scene point cloud so tensors stack nicely
            padded_scene_point_cloud: np.ndarray = np.pad(
                scene_point_cloud,
                pad_width=((0, self.max_num_scene_points - scene_point_cloud.shape[0]), (0, 0)),
                mode='constant',
                constant_values=0
            )
            scene_point_cloud_tensor: torch.Tensor = torch.from_numpy(padded_scene_point_cloud)
            num_points_in_scene: int = scene_point_cloud.shape[0]
            return (bps_encoded_tensor, scene_point_cloud_tensor, num_points_in_scene)
        else:
            raise ValueError("original_scene_encoding should be one of 'grid' or 'cloud'")
