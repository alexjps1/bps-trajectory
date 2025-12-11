"""
Dataset for Scene Reconstruction Loss
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2025-11-18
"""

# imports
import glob
import os
import sys
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# import bps from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.insert(0, parent_dir)
import bps

# constants
PRECISION = np.float32
ArrayLike = np.ndarray | torch.Tensor


class Scenes2dDataset(Dataset):
    """
    Memory-mapped dataset for providing 2d scenes loaded from several NPY files.
    Scenes are provided alongside the ground truth / target encoding of choice (either cloud or grid)
    """

    bps_encoding_type: str
    basis_point_cloud: np.ndarray
    bps_encoding_type: str
    norm_bound_shape: str
    target_encoding: str
    max_num_scene_points: int
    as_numpy: bool
    memmapped_arrays: List
    file_boundaries: List
    total_samples: int


    def __init__(
        self,
        data_directory: str,
        basis_point_cloud: np.ndarray,
        bps_encoding_type: str,
        norm_bound_shape: str,
        target_encoding: str,
        max_num_scene_points: int,
        file_pattern: str = "*.npy",
        as_numpy: bool = False
    ) -> None:
        """
        Initializes the memory-mapped dataset.

        Parameters
        ----------
        data_directory: str
            Path to directory containing NPY files
        basis_point_cloud: np.ndarray
            Array of 2d points in scene
        bps_encoding_type: str "scalar" or "diff"
            use scalar for 1d features (scalar BPS encoding)
            use diff for 2d features (difference vector BPS encoding)
        norm_bound_shape: str "ncube" or "nsphere" or "none"
            which shape to use for scene normalization, or "none" for no normalization
        target_encoding: str "cloud" or "grid"
            representation of target (original scene) to use for loss fn
        max_num_scene_points: int
            used to pad target point clouds to prevent tensor-stacking issues
        file_pattern: str (optional)
            Pattern to match NPY files (default: "*.npy")
        as_numpy: bool (optional)
            If True, scenes are provided as numpy.ndarray instead of torch.Tensor (default false)
        """
        if bps_encoding_type not in ["scalar", "diff"]:
            raise ValueError("bps_encoding_type must be 'scalar' or 'diff'")
        if target_encoding not in ["cloud", "grid"]:
            raise ValueError("target_encoding must be 'cloud' or 'grid'")

        self.bps_encoding_type = bps_encoding_type
        self.bps = basis_point_cloud
        self.target_encoding = target_encoding
        self.norm_bound_shape = norm_bound_shape
        self.max_num_scene_points = max_num_scene_points
        self.as_numpy = as_numpy

        # Get all NPY files from the directory
        npy_file_paths = sorted(glob.glob(os.path.join(data_directory, file_pattern)))
        if not npy_file_paths:
            raise ValueError(
                f"No NPY files found in {data_directory} with pattern {file_pattern}"
            )
        else:
            print(
                f"Using the following {len(npy_file_paths)} NPY files for the dataset:"
            )
            for filename in npy_file_paths:
                print(filename)

        # Initialize memory-mapped arrays and track file boundaries
        self.memmapped_arrays = []
        self.file_boundaries = []  # cumulative indices for each file
        cumulative_size = 0

        for file_path in npy_file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"NPY file not found: {file_path}")

            # Load as memory-mapped array (mode='r' for read-only)
            mmap_array = np.load(file_path, mmap_mode="r")
            self.memmapped_arrays.append(mmap_array)

            # Track where each file's samples start/end in the global index
            cumulative_size += mmap_array.shape[0]
            self.file_boundaries.append(cumulative_size)

        self.total_samples = cumulative_size
        print(f"Total number samples in dataset: {self.total_samples}")
        print()

    @classmethod
    def from_directory(
        cls,
        data_directory: str,
        basis_point_cloud: np.ndarray,
        bps_encoding_type: str,
        norm_bound_shape: str,
        target_encoding: str,
        max_num_scene_points: int,
        file_pattern: str = "*.npy",
    ):
        """
        Create dataset from all NPY files in a directory.

        Parameters
        ----------
        data_directory: str
            Path to directory containing NPY files
        file_pattern: str
            Pattern to match NPY files (default: "*.npy")
        """
        return cls(
            data_directory,
            basis_point_cloud,
            bps_encoding_type,
            norm_bound_shape,
            target_encoding,
            max_num_scene_points,
            file_pattern,
        )

    def _find_file_and_local_index(self, global_idx: int) -> Tuple[int, int]:
        """
        Given a global index, find which file it belongs to and the local index within that file.

        Returns
        -------
        Tuple[int, int]: (file_index, local_index)
        """
        if global_idx >= self.total_samples:
            raise IndexError(
                f"Index {global_idx} out of range for dataset of size {self.total_samples}"
            )

        # Find which file contains this index
        for file_idx, boundary in enumerate(self.file_boundaries):
            if global_idx < boundary:
                # Calculate local index within this file
                start_idx = 0 if file_idx == 0 else self.file_boundaries[file_idx - 1]
                local_idx = global_idx - start_idx
                return file_idx, local_idx

        # Should never reach here
        raise IndexError(f"Could not locate index {global_idx}")

    def __len__(self):
        """Returns the total number of samples across all files."""
        return self.total_samples

    def __getitem__(
        self, idx: Union[int, torch.Tensor]
    ) -> Union[
        Tuple[ArrayLike, ArrayLike, int],
        Tuple[ArrayLike, ArrayLike],
    ]:
        """
        Returns the BPS-encoded and target scene representations for a given sample.

        Parameters
        ----------
        idx: int or torch.Tensor
            Global index for scene to be loaded

        Returns
        -------
        Same as Scenes2dDataset.__getitem__
        """
        # Convert tensor index to int if needed
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        # Find which file and local index
        file_idx, local_idx = self._find_file_and_local_index(idx)

        # Load the specific sample (only this sample is loaded into memory)
        scene_occupancy_grid_raw = self.memmapped_arrays[file_idx][local_idx]

        # Rest of the processing is identical to the original dataset
        scene_occupancy_grid = (scene_occupancy_grid_raw != 0).astype(int)

        # Create BPS encoding
        scene_point_cloud: np.ndarray = bps.create_point_cloud(scene_occupancy_grid)
        encoded_result = bps.encode_scene(
            scene_point_cloud, self.bps, self.bps_encoding_type, self.norm_bound_shape
        )
        assert isinstance(encoded_result, np.ndarray)

        bps_encoded_arr: ArrayLike
        if self.as_numpy:
            bps_encoded_arr = encoded_result.astype(PRECISION)
        else:
            bps_encoded_arr = torch.from_numpy(encoded_result.astype(PRECISION))

        if self.target_encoding == "grid":
            # generate grid encoding and return
            scene_occupancy_grid_arr: ArrayLike
            if self.as_numpy:
                scene_occupancy_grid_arr = scene_occupancy_grid.astype(PRECISION)
            else:
                scene_occupancy_grid_arr = torch.from_numpy(scene_occupancy_grid.astype(PRECISION))
            return (bps_encoded_arr, scene_occupancy_grid_arr)
        elif self.target_encoding == "cloud":
            # pad the scene point cloud so tensors stack nicely
            padded_scene_point_cloud: np.ndarray = np.pad(
                scene_point_cloud,
                pad_width=(
                    (0, self.max_num_scene_points - scene_point_cloud.shape[0]),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0,
            )

            scene_point_cloud_arr: ArrayLike
            if self.as_numpy:
                scene_point_cloud_arr = padded_scene_point_cloud.astype(PRECISION)
            else:
                scene_point_cloud_arr = torch.from_numpy(padded_scene_point_cloud.astype(PRECISION))
            num_points_in_scene: int = scene_point_cloud.shape[0]
            return (bps_encoded_arr, scene_point_cloud_arr, num_points_in_scene)
        else:
            raise ValueError("target_encoding should be one of 'grid' or 'cloud'")
