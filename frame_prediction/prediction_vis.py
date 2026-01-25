"""
Scene Prediction Visualization Script
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2026-01-14

Initially just a copy of the scene reconstruction visualization script
"""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

# constants
ArrayLike = np.ndarray | torch.Tensor


def visualize_grid_difference(
    grid_pred: ArrayLike,
    grid_target: ArrayLike,
    show_window: bool,
    save_image: bool,
    output_path: Path,
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


def visualize_time_series_grid_difference(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    num_steps: int = 5,
    n_rows: int = 3,
    n_cols: int = 3,
    threshold: float = 0.5,
    show_window: bool = False,
    save_image: bool = True
    

):
    """
    Visualize grid occupancy predictions vs targets with color coding and save as PNG.

    Args
    ----
    predictions: torch.Tensor or np.ndarray
        Predicted occupancy grids (N, 64, 64)
    targets: torch.Tensor or np.ndarray
        Target occupancy grids (N, 64, 64)
    output_path: str (optional)
        path to which to save visualization image if save_image is true
    num_steps: int
        Number of future steps that are predicted
    n_rows: int
        Number of rows of the plot grid
    n_cols: int
        Number of columns of the plot grid
    threshold: float
        Thresholds when a predicted value gets rounded to 1
    show_window: bool (optional)
        Whether to show the visualization in a window (suspends program until closed, default false)
    save_image: bool (optional)
        Whether to save image at the given output_path (default true)
    

    Returns
    -------
    dict:
        Dictionary containing accuracy metrics
    """
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10,10))
    index = 0
    for i in range(n_rows):
        for j in range(n_cols):
            
            original = (targets[num_steps + index] > threshold).astype(int)
            reconstruction = (predictions[index] > threshold).astype(int)

            match = np.array([(original == 1) & (reconstruction == 1)], dtype=int)

            

            img = original + 2* reconstruction + match
            img = img.squeeze(0)

            rgb = np.ones((img.shape[0], img.shape[1], 3))

            rgb[img == 0] = [1, 1, 1]
            rgb[img == 1] = [0.25, 0.35, 0.85]
            rgb[img == 2] = [0.85, 0.25, 0.25]
            rgb[img == 4] = [0.25, 0.75, 0.35]
            axs[i][j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            axs[i][j].imshow(rgb)
            axs[i][j].set_title(f"t + {index + 1}")

            tp = np.sum(match)
            precision = tp / np.sum(reconstruction == 1)

            fn = np.sum(np.array([(original == 1) & (reconstruction == 0)], dtype=int))
            recall = tp/(tp+fn)
            f1 = 2*(precision * recall)/(precision+recall)
            axs[i][j].text(0.5, 0.08, f"F1: {f1:.3f}", ha='center', va='top', transform=axs[i][j].transAxes)

            index += 1



    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='True Positive'),
        Patch(facecolor='red', edgecolor='black', label='False Positive'),
        Patch(facecolor='blue', edgecolor='black', label='False Negative')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Autoregressive Prediction with Timesteps t-4 to t as Input")
    if show_window:
        plt.show()
    if save_image:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    return {
        "F1": f1,
        "correct_pixels": int(tp),
        "total_pixels": int(original.shape[0]**2),
        "output_path": output_path,
    }
