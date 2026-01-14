"""
Dataset for Frame Prediction in Dynamic Scenes
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2025-01-06
"""

# standard library imports
import glob
import os
from typing import List, Tuple

# third party imports
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class DynamicScenes2dDataset(Dataset):
    """
    Memory-mapped dataset for providing dynamic 2d scenes loaded from several NPY files.
    A dynamic scene is a tensor of dimensions [frames, rows, cols] (with an additional first dim for batches)

    This dataset does not provide BPS encodings, only occupancy grids (scene ground truths).
    All NPY files should contain the same number of frames per scene to avoid problems.

    Note that tensors are provided with dtype torch.float32 (or numpy arrays with dtype np.float32 when as_numpy=True)
    """

    as_numpy: bool
    memmapped_arrays: List
    file_boundaries: List
    total_samples: int
    num_input_frames: int
    num_target_frames: int

    def __init__(
        self,
        data_directory: str,
        num_input_frames: int,
        num_target_frames: int,
        as_numpy: bool = False,
        file_pattern: str = "*.npy",
    ) -> None:
        """
        Initializes the memory-mapped dataset.

        Parameters
        ----------
        data_directory: str
            Path to directory containing NPY files
        num_input_frames: int
            Number of frames used as input; frame at index num_input_frames is the first target frame
        num_target_frames: int (optinal)
            Number of frames to serve as prediction targets for the model.
        as_numpy: bool (optional)
            If True, scenes are provided as numpy.ndarray instead of torch.Tensor (default false)
        file_pattern: str (optional)
            Pattern to match NPY files (default: "*.npy")
        """
        self.as_numpy = as_numpy
        self.num_input_frames = num_input_frames
        self.num_target_frames = num_target_frames

        # Get all NPY files from the directory
        npy_file_paths = sorted(glob.glob(os.path.join(data_directory, file_pattern)))
        if not npy_file_paths:
            raise ValueError(f"No NPY files found in {data_directory} with pattern {file_pattern}")
        else:
            print(f"Using the following {len(npy_file_paths)} NPY files for the dataset:")
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

    def _find_file_and_local_index(self, global_idx: int) -> tuple[int, int]:
        """
        Given a global index, find which file it belongs to and the local index within that file.

        Returns
        -------
        tuple[int, int]: (file_index, local_index)
        """
        if global_idx >= self.total_samples:
            raise IndexError(f"Index {global_idx} out of range for dataset of size {self.total_samples}")

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[NDArray, NDArray]:
        """
        Returns input frames and target frames for the requested sample.

        Parameters
        ----------
        idx: int
            Global index for dynamic scene to be loaded

        Returns
        -------
        Tuple of (input_frames, target_frames):
            input_frames: shape [num_input_frames, rows, cols]
            target_frames: shape [num_target_frames, rows, cols]
        """
        # Find which file and local index
        file_idx, local_idx = self._find_file_and_local_index(idx)

        # Load the specific sample (copy from mmap to avoid holding references)
        scene_raw = np.array(self.memmapped_arrays[file_idx][local_idx])

        if scene_raw.shape[0] < self.num_input_frames + self.num_target_frames:
            raise ValueError(
                "The specified number of input frames + target frames exceeds the number of frames available in this particular scene."
            )

        # Split into input and target
        input_frames = scene_raw[: self.num_input_frames]
        target_frames = scene_raw[self.num_input_frames : self.num_input_frames + self.num_target_frames]

        # Return in requested type with 32-bit floats
        if self.as_numpy:
            return (
                input_frames.astype(np.float32),
                target_frames.astype(np.float32),
            )
        else:
            return (
                torch.from_numpy(input_frames.astype(np.float32)),
                torch.from_numpy(target_frames.astype(np.float32)),
            )
