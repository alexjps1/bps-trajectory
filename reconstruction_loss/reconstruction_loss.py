"""
Training and Testing Script for Scene Reconstruction Loss
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2025-11-11
"""

# imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from typing import Tuple

# enable importing from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)

# our imports
from models.ptcloud_decoder01 import BPSPointCloudDecoder01
from models.grid2d_decoder01 import BPSOccupancyGrid2dDecoder01
from datasets import Scenes2dDataset
import loops
import bps


# constants
PRECISION = np.float32


def main():
    # bps settings
    bps_type: str = "grid"  # grid or sampling
    sampling_shape: str = "ncube"  # ncube or nsphere
    sampling_num_basis_points: int = 200
    grid_size: int = 16  # the number of basis points will be grid_size squared

    # model type
    model_type = "grid"  # grid or cloud

    # other hyperparameter settings
    encoding_type: str = "scalar"  # scalar or diff
    norm_bound_shape: str = "ncube"  # ncube or nsphere
    batch_size: int = 64
    num_epochs: int = 50
    scene_dims: Tuple[int, int] = (64, 64)

    # basis point set generation
    basis_point_cloud: np.ndarray
    num_basis_points: int
    if bps_type == "sampling":
        basis_point_cloud = bps.generate_bps_sampling(sampling_num_basis_points, 2, sampling_shape, 1.0, random_seed=13)
        num_basis_points = sampling_num_basis_points
    elif bps_type == "grid":
        basis_point_cloud = bps.generate_bps_ngrid(grid_size, 2)
        num_basis_points = basis_point_cloud.shape[0]  # grid_size squared
    else:
        raise ValueError("bps_type must be 'sampling' or 'grid'")

    # model initialization
    decoder: nn.Module
    feature_dim: int = 1 if encoding_type == "scalar" else 2  # if working in 2d
    max_num_scene_points: int = scene_dims[0] * scene_dims[1]
    if model_type == "cloud":
        decoder = BPSPointCloudDecoder01(
            num_basis_points,
            max_num_scene_points,
            feature_dim
        )
    elif model_type == "grid":
        decoder = BPSOccupancyGrid2dDecoder01(
            num_basis_points,
            feature_dim,
            scene_dims,
            num_conv_layers=4
        )
    else:
        raise ValueError("model_type must be 'grid' or 'cloud'")


    # dataset and loader
    scenes_data_array: np.ndarray = np.load('./environments.npy')
    scenes2d: Scenes2dDataset = Scenes2dDataset(scenes_data_array, basis_point_cloud, encoding_type, norm_bound_shape, model_type, max_num_scene_points)

    # data split
    total_size: int = len(scenes2d)
    train_size: int = int(0.8 * total_size)
    val_size: int = int(0.1 * total_size)
    test_size: int = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        scenes2d,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42))

    # dataloaders
    train_dataloader: DataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader: DataLoader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader: DataLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # train the model
    if model_type == "cloud":
        print("Starting training loop for point-cloud based decoder...")
        loops.train_bps_cloud_decoder_2d(
            decoder,
            train_dataloader,
            val_dataloader,
            num_basis_points,
            encoding_type,
            norm_bound_shape,
            batch_size,
            num_epochs
        )
    elif model_type == "grid":
        print("Starting training loop for occupancy grid based decoder")
        loops.train_bps_grid_decoder_2d(
            decoder,
            train_dataloader,
            val_dataloader,
            num_basis_points,
            encoding_type,
            norm_bound_shape,
            batch_size,
            num_epochs
        )

    exit()

if __name__ == "__main__":
    main()
