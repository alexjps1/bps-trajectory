"""
Scene Prediction Visualization Script
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2026-01-14

Initially just a copy of the scene reconstruction visualization script
"""

from typing import Any, Dict

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import KDTree

# constants
ArrayLike = np.ndarray | torch.Tensor


def visualize_grid_difference(
    grid_pred: ArrayLike,
    grid_target: ArrayLike,
    show_window: bool = False,
    save_image: bool = True,
    output_path: str = "grid_comparison.png",
) -> Dict[str, Any]:
    """
    Visualize grid occupancy prediction vs target with color coding and save as PNG.

    Args
    ----
    grid_predicted: torch.Tensor or np.ndarray
        Predicted occupancy grid (values between 0 and 1)
    grid_target: torch.Tensor or np.ndarray
        Target occupancy grid (binary values 0 or 1)
    show_window: bool (optional)
        Whether to show the visualization in a window (suspends program until closed, default false)
    save_image: bool (optional)
        Whether to save image at the given output_path (default true)
    output_path: str (optional)
        path to which to save visualization image if save_image is true

    Returns
    -------
    dict:
        Dictionary containing accuracy metrics
    """
    # Convert tensors to numpy arrays
    if isinstance(grid_pred, torch.Tensor):
        grid_pred = grid_pred.detach().cpu().numpy()
    if isinstance(grid_target, torch.Tensor):
        grid_target = grid_target.detach().cpu().numpy()

    # Create RGB visualization
    height, width = grid_pred.shape
    rgb_image = np.zeros((height, width, 3))

    # Green channel: correct predictions (high where pred matches target)
    correct_occupied = (grid_pred > 0.5) & (grid_target > 0.5)  # True positives
    correct_free = (grid_pred <= 0.5) & (grid_target <= 0.5)  # True negatives
    rgb_image[:, :, 1] = (correct_occupied | correct_free).astype(float)

    # Red channel: incorrect predictions
    false_positive = (grid_pred > 0.5) & (grid_target <= 0.5)  # Predicted occupied but actually free
    false_negative = (grid_pred <= 0.5) & (grid_target > 0.5)  # Predicted free but actually occupied
    rgb_image[:, :, 0] = (false_positive | false_negative).astype(float)

    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot predicted grid
    ax1.imshow(grid_pred, cmap="gray_r", vmin=0, vmax=1)
    ax1.set_title("Predicted Occupancy")
    ax1.axis("off")

    # Plot target grid
    ax2.imshow(grid_target, cmap="gray_r", vmin=0, vmax=1)
    ax2.set_title("Target Occupancy")
    ax2.axis("off")

    # Plot comparison (red = incorrect, green = correct)
    ax3.imshow(rgb_image)
    ax3.set_title("Comparison (Green=Correct, Red=Incorrect)")
    ax3.axis("off")

    plt.tight_layout()

    # Save the plot as PNG
    if show_window:
        plt.show()
    if save_image:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    # Calculate accuracy metrics
    total_pixels = height * width
    correct_pixels = np.sum(correct_occupied | correct_free)
    accuracy = correct_pixels / total_pixels

    # Return metrics
    return {
        "accuracy": accuracy,
        "correct_pixels": int(correct_pixels),
        "total_pixels": int(total_pixels),
        "output_path": output_path,
    }


def visualize_cloud_difference(
    cloud_pred: torch.Tensor,
    cloud_target: torch.Tensor,
    output_filepath: str = "cloud_comparison.png",
    colormap_max_err: float = 0.1,
) -> None:
    """
    Visualize two 2D point clouds side by side using matplotlib and save to the given output_filepath.
    This function works exclusively with 2D point clouds and will raise an error if 3D clouds are provided.
    """
    # Convert tensors to numpy arrays
    pred_np = cloud_pred.detach().cpu().numpy()
    target_np = cloud_target.detach().cpu().numpy()

    # Check that we're working with 2D point clouds
    if pred_np.shape[1] != 2:
        raise ValueError(
            f"visualize_cloud_difference expects 2D point clouds, but predicted cloud has {pred_np.shape[1]} dimensions. Use a 3D visualization function for 3D clouds."
        )

    if target_np.shape[1] != 2:
        raise ValueError(
            f"visualize_cloud_difference expects 2D point clouds, but target cloud has {target_np.shape[1]} dimensions. Use a 3D visualization function for 3D clouds."
        )

    # 1. Build KDTree on the Target cloud
    tree_T = KDTree(target_np)

    # 2. Query P for its nearest neighbors in T
    # distances: distance from P[i] to T[nn_indices[i]]
    # nn_indices: index of nearest neighbor in T
    distances, nn_indices = tree_T.query(pred_np, k=1)

    # 3. Define Color Mapping
    # Normalize and clamp distances to [0, 1]
    norm_distances = np.clip(distances / colormap_max_err, 0, 1)

    # Use a custom colormap: [Black -> Red]
    custom_colors = [(0, 0, 0), (1, 0, 0)]  # Black to Red
    cmap_custom = colors.LinearSegmentedColormap.from_list("BlackRed", custom_colors)

    # Get the RGB colors for each predicted point
    P_colors = cmap_custom(norm_distances)

    # 4. Plot the 2D Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"2D Predicted Cloud Error (Nearest Neighbor - Max Error: {colormap_max_err})")

    # Plot Target Cloud (Context)
    ax.scatter(
        target_np[:, 0],
        target_np[:, 1],
        c="lightgray",
        s=20,
        alpha=0.5,
        label="Target (T)",
    )

    # Plot Predicted Cloud (Colored by Error)
    ax.scatter(
        pred_np[:, 0],
        pred_np[:, 1],
        c=P_colors,
        s=40,
        alpha=1.0,
        label="Predicted (P)",
    )

    # Draw Error Lines
    for i in range(len(pred_np)):
        p_coords = pred_np[i]
        t_coords = target_np[nn_indices[i]]
        line_color = P_colors[i]

        ax.plot(
            [p_coords[0], t_coords[0]],
            [p_coords[1], t_coords[1]],
            color=line_color,
            linewidth=0.5,
        )

    # Add a color bar legend for the distance
    sm = cm.ScalarMappable(cmap=cmap_custom, norm=colors.Normalize(vmin=0, vmax=colormap_max_err))
    sm.set_array([])  # Required for older matplotlib versions
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Distance Error (d)", rotation=270, labelpad=15)

    # Set equal aspect ratio (important for 2D point clouds)
    ax.set_aspect("equal")

    # Set axis limits based on the combined data range
    all_x = np.concatenate([pred_np[:, 0], target_np[:, 0]])
    all_y = np.concatenate([pred_np[:, 1], target_np[:, 1]])

    x_margin = (all_x.max() - all_x.min()) * 0.05
    y_margin = (all_y.max() - all_y.min()) * 0.05

    ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    # Save the plot as PNG
    plt.savefig(output_filepath, dpi=150, bbox_inches="tight")
    plt.close()  # Close the figure to free memory


# Example usage (P and T would be your actual data)
# P = np.random.rand(1000, 3) + np.array([0.1, 0, 0])
# T = np.random.rand(1000, 3)
# visualize_chamfer_error(P, T)`
