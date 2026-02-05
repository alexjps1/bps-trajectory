"""
Scene Prediction Visualization Script
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2026-01-14

Initially just a copy of the scene reconstruction visualization script
"""

import math
from pathlib import Path
from typing import Any, Dict, List, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

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
    false_positive = (grid_pred > 0.5) & (
        grid_target <= 0.5
    )  # Predicted occupied but actually free
    false_negative = (grid_pred <= 0.5) & (
        grid_target > 0.5
    )  # Predicted free but actually occupied
    rgb_image[:, :, 0] = (false_positive | false_negative).astype(float)

    # Create the plot
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

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
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    output_path: str | Path,
    loss_metrics: List[Dict[str, float]] | None = None,
    input_frames: np.ndarray | torch.Tensor | None = None,
    n_rows: int | None = None,
    n_cols: int | None = None,
    threshold: float = 0.5,
    show_window: bool = False,
    save_image: bool = True,
    target_frame_offset: int = 0,
) -> None:
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
    loss_metrics: List[Dict[str, float]] (optional)
        a list containing the
    input_frames: torch.Tensor or np.ndarray (optional)
        input frames given to the model
        If provided, they will be shown. If not, then not.
    target_frame_offset: int (optional)
        If using an offset between the last input frame and the first predicted frame, pass it here.
        This just updates the axes' titles so they don´t start with t+1.
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
    # use numpy arrays for everything
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    if input_frames is not None and isinstance(input_frames, torch.Tensor):
        input_frames = input_frames.numpy()

    # Capture max prediction value before discretization for debugging
    max_pred_value = float(np.max(predictions))

    # resolve absolute path
    if isinstance(output_path, str):
        output_path = Path(output_path).resolve()

    # input validation
    if len(predictions.shape) != 3:
        raise ValueError(
            f"visualize_time_series_grid_difference received predictions tensor with shape {predictions.shape}, but shape must have length 3"
        )
    if len(targets.shape) != 3:
        raise ValueError(
            f"visualize_time_series_grid_difference received targets tensor with shape {targets.shape}, but shape must have length 3"
        )
    if input_frames is not None and len(input_frames.shape) != 3:
        raise ValueError(
            f"visualize_time_series_grid_difference received input_frames tensor with shape {input_frames.shape}, but shape must have length 3"
        )
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError(
            f"in visualize_time_series_grid_difference, predictions tensor has {predictions.shape[0]} time steps but targets tensor has {targets.shape[0]}. Predictions shape: {predictions.shape}, Targets shape: {targets.shape}"
        )

    # figure out how many input truth frames were given
    num_input_frames: int
    if input_frames is None:
        num_input_frames = 0
    else:
        num_input_frames = input_frames.shape[0]

    # figure out number of predicted/target frames (same number)
    num_predicted_frames: int = predictions.shape[0]

    # determine grid dimensions and num plots needed
    # If we have input frames, we need to round up to ensure predictions start on a new row
    if n_cols is None:
        if n_rows is None:
            n_cols = 5
        else:
            # If rows are fixed, calculate cols needed
            total_plots = num_input_frames + num_predicted_frames
            n_cols = math.ceil(total_plots / n_rows)

    # calculate rows needed if not provided
    if n_rows is None:
        # Calculate rows for input frames (rounded up to full rows)
        input_rows = math.ceil(num_input_frames / n_cols) if num_input_frames > 0 else 0
        # Calculate rows for predicted frames
        pred_rows = math.ceil(num_predicted_frames / n_cols)
        n_rows = input_rows + pred_rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 4.5))

    # flatten axs for easy iteration if it's a grid
    if n_rows * n_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]  # Handle single plot case

    cmap_prediction = {
        0: (1, 1, 1),  # white for true negative (prediction=0, target=0)
        1: (0.25, 0.35, 0.85),  # blue for false negative (prediction=0, target=1)
        2: (0.85, 0.25, 0.25),  # red for false positive (prediction=1, target=0)
        4: (0.25, 0.75, 0.35),  # green for true positive (prediction=1, target=0)
    }

    plot_idx = 0

    # plot input frames if provided
    for i in range(num_input_frames):
        ax = axs[plot_idx]

        frame = cast(np.ndarray, input_frames)[i]

        ax.imshow(frame, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"Input t-{num_input_frames - i}", fontsize=14, fontweight="bold")
        ax.axis("off")  # Hide ticks

        plot_idx += 1

    # After plotting input frames, ensure predicted frames start on a new row
    if num_input_frames > 0 and plot_idx % n_cols != 0:
        # Hide the blank axes before moving to next row
        old_plot_idx = plot_idx
        # Move to the start of the next row
        plot_idx = ((plot_idx // n_cols) + 1) * n_cols
        # Hide all the blank spaces between input frames and predicted frames
        for j in range(old_plot_idx, plot_idx):
            axs[j].axis("off")

    # plot predictions vs targets difference maps
    for i in range(num_predicted_frames):
        if plot_idx >= len(axs):
            break  # Avoid index error if grid is too small

        ax = axs[plot_idx]

        # Get data for this step
        target_grid = targets[i]
        pred_grid = predictions[i]

        # Binarize
        target_bin = (target_grid > threshold).astype(int)
        pred_bin = (pred_grid > threshold).astype(int)

        match = ((target_bin == 1) & (pred_bin == 1)).astype(int)
        img_code = target_bin + 2 * pred_bin + match

        # Create RGB Image
        rgb = np.ones((target_grid.shape[0], target_grid.shape[1], 3))
        for val, color in cmap_prediction.items():
            rgb[img_code == val] = color

        ax.imshow(rgb)
        # Calculate the actual frame index accounting for offset
        actual_frame_idx = i + 1 + target_frame_offset
        ax.set_title(f"Pred t+{actual_frame_idx}", fontsize=14, fontweight="bold")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Render Metrics text below the plot
        if loss_metrics is not None and i < len(loss_metrics):
            m = loss_metrics[i]

            # Format text as two columns, left-aligned within each
            col1_text = f"BCE: {m.get('bce', 0):.3f}\nMSE: {m.get('mse', 0):.3f}\nF1:  {m.get('f1', 0):.3f}"
            col2_text = f"BCE bin: {m.get('bce_bin', 0):.3f}\nMSE bin: {m.get('mse_bin', 0):.3f}\nAcc:     {m.get('accuracy', 0):.3f}"

            # Place left column
            ax.text(
                0.0,
                -0.02,
                col1_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=13,
                family="monospace",
            )
            # Place right column
            ax.text(
                1.0,
                -0.02,
                col2_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=13,
                family="monospace",
            )

        plot_idx += 1

    # hide any unused subplots
    for j in range(plot_idx, len(axs)):
        axs[j].axis("off")

    # legend
    legend_elements = [
        Patch(
            facecolor=cmap_prediction[4], edgecolor="black", label="True Positive"
        ),  # Green
        Patch(
            facecolor=cmap_prediction[1], edgecolor="black", label="False Negative"
        ),  # Blue
        Patch(
            facecolor=cmap_prediction[2], edgecolor="black", label="False Positive"
        ),  # Red
    ]

    if input_frames is not None:
        legend_elements.append(
            Patch(
                facecolor=(0.5, 0.5, 0.5),  # gray
                edgecolor="black",
                label="Input Frame",
            )
        )

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )

    plt.tight_layout()

    # Add debug info: max prediction value (before discretization)
    fig.text(
        0.98,
        0.02,
        f"max pred: {max_pred_value:.4f}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="lightgrey",
        alpha=0.7,
    )

    if show_window:
        plt.show()

    if save_image:
        # create dir for image if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()
