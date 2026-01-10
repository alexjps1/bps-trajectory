"""
Script for ML-based Scene Reconstruction
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2025-11-11
"""

# standard library imports
import argparse
import json
import os
import sys
from enum import EnumType
from typing import Union

# third party imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# enable importing from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.insert(0, parent_dir)

# first party imports
import bps
import loops
from datasets import Scenes2dDataset
from models.grid2d_decoder02 import BPSOccupancyGrid2dDecoder02
from models.ptcloud_decoder01 import BPSPointCloudDecoder01


def load_config(config_path: str):
    """Load a JSON config file as dict"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def main(config_path: Union[str, None] = None):
    # Load configuration
    if config_path:
        config = load_config(config_path)
        print(f"Loaded configuration from {config_path}")
    else:
        # Default configuration (fallback)
        config = {
            "bps_settings": {
                "bps_type": "grid",  # grid or sampling
                "sampling_shape": "ncube",  # ncube or sphere (ignored if bps_type is grid)
                "sampling_num_basis_points": 200,  # (ignored if bps_type is grid)
                "grid_size": 32,  # num points on each axis of grid (ignored if bps_type is sampling)
            },
            "model": {
                "model_type": "grid",  # grid or cloud
                "num_conv_layers": 4,  # relevant only for grid model (CNN-based)
            },
            "training": {
                "encoding_type": "diff",  # diff or scalar
                "encoding_signed": True,
                "norm_bound_shape": "ncube",  # ncube or sphere
                "batch_size": 64,
                "num_epochs": 5,
                "scene_dims": [64, 64],
            },
            "data": {
                "data_directory": "/home/alexjps/TrajectoryPlanning/scenes2d",
                "train_split": 0.8,
                "val_split": 0.1,
                "random_seed": 42,  # used for data splitting and reproducibility
            },
        }
        print("Using default configuration")

    # Extract settings from config
    bps_type = config["bps_settings"]["bps_type"]
    sampling_shape = config["bps_settings"]["sampling_shape"]
    sampling_num_basis_points = config["bps_settings"]["sampling_num_basis_points"]
    grid_size = config["bps_settings"]["grid_size"]

    model_type = config["model"]["model_type"]
    num_conv_layers = config["model"]["num_conv_layers"]

    encoding_type = config["training"]["encoding_type"]
    encoding_signed = config["training"]["encoding_signed"]
    norm_bound_shape = config["training"]["norm_bound_shape"]
    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    scene_dims = tuple(config["training"]["scene_dims"])

    data_directory = config["data"]["data_directory"]
    train_split = config["data"]["train_split"]
    val_split = config["data"]["val_split"]
    random_seed = config["data"]["random_seed"]

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
    model_name: str
    feature_dim: int = 1 if encoding_type == "scalar" else 2  # if working in 2d
    max_num_scene_points: int = scene_dims[0] * scene_dims[1]
    if model_type == "cloud":
        model_name = "cloud01"
        decoder = BPSPointCloudDecoder01(num_basis_points, max_num_scene_points, feature_dim)
    elif model_type == "grid":
        model_name = "grid02"
        decoder = BPSOccupancyGrid2dDecoder02(
            num_basis_points, feature_dim, scene_dims, num_conv_layers=num_conv_layers
        )
    else:
        raise ValueError("model_type must be 'grid' or 'cloud'")

    # dataset and loader
    scenes2d: Scenes2dDataset = Scenes2dDataset(
        data_directory,
        basis_point_cloud,
        encoding_type,
        norm_bound_shape,
        model_type,
        max_num_scene_points,
        bps_encoding_signed=encoding_signed,
    )

    # data split
    total_size: int = len(scenes2d)
    train_size: int = int(train_split * total_size)
    val_size: int = int(val_split * total_size)
    test_size: int = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        scenes2d,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    # dataloaders
    train_dataloader: DataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader: DataLoader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader: DataLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # train the model
    print(f"""Training model with the following hyperparameters:
# bps settings
bps_type: {bps_type}
sampling_shape: {sampling_shape}
sampling_num_basis_points: {sampling_num_basis_points}
grid_size: {grid_size}

# model type
model_type: {model_type}

# other hyperparameter settings
encoding_type: {encoding_type}
encoding_signed: {encoding_signed}
norm_bound_shape: {norm_bound_shape}
batch_size: {batch_size}
num_epochs: {num_epochs}
scene_dims: {scene_dims})

""")
    if model_type == "cloud":
        print("Starting training loop for point-cloud based decoder...")
        loops.train_bps_cloud_decoder_2d(
            decoder,
            train_dataloader,
            val_dataloader,
            num_basis_points,
            encoding_type,
            norm_bound_shape,
            bps_type,
            model_name,
            batch_size,
            num_epochs,
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
            bps_type,
            model_name,
            batch_size,
            num_epochs,
        )

    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scene Reconstruction Loss Training")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    args = parser.parse_args()

    main(args.config)
