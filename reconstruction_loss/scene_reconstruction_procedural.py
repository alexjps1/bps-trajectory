"""
Procedural Method for Scene Reconstruction
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2025-12-02


Limitations of this implementation:
- BPS type is a square grid (no sampling, no nsphere)
- Scene is 2d
- Scene is a square
- BPS grid length <= scene grid length
- BPS encoding type is "scalar" (no diff vectors)
- No normalization of scenes before encoding
    (This unfortunately poses some validity issues when comparing to ML-based reconstruction loss)
"""

# standard library imports
import argparse
import os
import sys

# third party imports
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm

# enable importing from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.insert(0, parent_dir)

# first party imports
import bps
from datasets import Scenes2dDataset
from reconstruction_vis import visualize_grid_difference

# constants
DEFAULT_DATASET_DIR = "/home/alexjps/Dokumente/Uni/ADLR/TrajectoryPlanning/scenes2d"


def interpolate_distance_map(bps_grid: NDArray[np.float64], bps_grid_length: int, output_grid_length: int) -> NDArray[np.float64]:
    """
    Creates a distance map of a 2d square grid-based scene by upsampling basis point distances using bilinear interpolation.
    This is the first step in reconstructing a scene procedurally (next, apply a threshold to distance map vals to decide what counts as "occupied")

    NOTE: These operations are theoretically parallelizeable assuming identical scene and bps grid size.
          Unfortuantely, this implementation does not take full advantage of numpy and is not parallelized.

    Args
    ----
    bps_grid: np.ndarray
        two-dimensional ndarray of shape (bps_grid_length, bps_grid_length)
    bps_grid_length: int
        side length of the square bps grid
    output_grid_length: int
        side length of the square scene to be reconstructed by this bilinear interpolation

    Returns
    -------
    np.ndarray:
        two-dimensional ndarray of shape (output_grid_length, output_grid_length)
        representing distance field upsampled to original scene dimensions
    """
    # check inputs
    if bps_grid.shape != (bps_grid_length, bps_grid_length):
        raise ValueError(f"input basis_grid must be square and match given bps_grid_length {bps_grid_length}")
    if bps_grid_length > output_grid_length:
        raise ValueError(f"bps_grid_length {bps_grid_length} should be smaller or equal to output_grid_length {output_grid_length}")

    grid_length_scale_factor = output_grid_length / bps_grid_length

    interpolated_map = np.zeros((output_grid_length, output_grid_length), dtype=np.float64)

    for x in range(output_grid_length):
        for y in range(output_grid_length):
            # map the output coordinate (x, y) back to the input space (xs, ys)
            xs = x / grid_length_scale_factor
            ys = y / grid_length_scale_factor

            # find the four surrounding basis grid indices (i1, j1, i2, j2)
            i1 = int(xs)
            j1 = int(ys)

            i2 = min(i1 + 1, bps_grid_length - 1)
            j2 = min(j1 + 1, bps_grid_length - 1)

            # calculate interpolation weights (the fractional parts)
            wx = xs - i1
            wy = ys - j1

            # get the values of the four corner points
            Q11 = bps_grid[i1, j1]
            Q21 = bps_grid[i2, j1]
            Q12 = bps_grid[i1, j2]
            Q22 = bps_grid[i2, j2]

            # interpolate along the x-axis (r1, r2)
            R1 = Q11 * (1 - wx) + Q21 * wx
            R2 = Q12 * (1 - wx) + Q22 * wx

            # interpolate along the y-axis
            interpolated_map[x, y] = R1 * (1 - wy) + R2 * wy

    return interpolated_map


def apply_threshold(distance_map: NDArray[np.float64], threshold: float = 0.5) -> NDArray[np.int64]:
    """
    Given a distance map of an upsampled 2d square grid scene, apply the threshold to decide which grid cells are occupied.
    """
    reconstructed_scene: NDArray[np.int64] = (distance_map < threshold).astype(np.int64)
    return reconstructed_scene


def main(args: argparse.Namespace):
    # unpack args
    bps_grid_length: int = args.bpsgridlen
    scene_length: int = args.scenelen if args.scenelen is not None else 64
    show_visualization: bool = args.visual

    # bps grid generation based on scene length and with NO normalization
    step_size: int = scene_length // bps_grid_length
    max_pixel_coord: int = (bps_grid_length - 1) * step_size
    basis_points_grid: NDArray[np.float64] = bps.generate_bps_ngrid(bps_grid_length, 2, 0, max_pixel_coord)

    # dataset
    dataset_directory: str = DEFAULT_DATASET_DIR
    scenes2d: Scenes2dDataset = Scenes2dDataset(
        dataset_directory,
        basis_points_grid,
        bps_encoding_type="scalar",  # this method doesn't work with difference vectors
        norm_bound_shape="none",  # because
        target_encoding="grid",
        max_num_scene_points=(scene_length**2),
        as_numpy=True,
        grid_shape_for_grid_basis=(bps_grid_length, bps_grid_length),
    )

    # dataloader
    dataloader: DataLoader = DataLoader(dataset=scenes2d, batch_size=1, shuffle=True)

    # loss function
    criterion = F.binary_cross_entropy

    total_loss: float = 0.0

    # this assumes batch size of 1
    bps_encoding_np: NDArray[np.float64]
    loop: tqdm = tqdm(dataloader, desc="Reconstructing... ")
    for i, (bps_encoding, target_grid) in enumerate(loop):
        bps_encoding_np = bps_encoding[0].numpy()
        distance_map: NDArray[np.float64] = interpolate_distance_map(bps_encoding_np, bps_grid_length, scene_length)
        predicted_grid: torch.Tensor = torch.from_numpy(apply_threshold(distance_map))

        # have to convert to float bc tensors otherwise get interpreted as bool tensors
        loss: float = float(criterion(predicted_grid.float(), target_grid[0].float(), reduction="mean"))
        total_loss += loss

        visualize_grid_difference(
            predicted_grid,
            target_grid[0],
            show_window=show_visualization,
            save_image=False,
        )

        print(f"loss of {loss}")

    average_loss: float = total_loss / len(scenes2d)

    print(f"The average loss was {average_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procedural Scene Reconstruction")
    parser.add_argument(
        "-b",
        "--bpsgridlen",
        type=int,
        help="Side length of the square BPS grid to use in reconstruction",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--scenelen",
        nargs="?",
        type=int,
        help="Side length of the 2d square scene grids present in the dataset being used",
    )
    parser.add_argument(
        "-v",
        "--visual",
        action="store_true",
        help="Set this flag to show a visualization of every generated reconstruction",
    )
    args = parser.parse_args()
    main(args)
