"""
Scene Prediction Visualization Script
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2026-01-14

Initially just a copy of the scene reconstruction visualization script
"""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

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
