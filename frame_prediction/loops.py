"""
Training and Evaluation Loops for Scene-to-Scene Frame Prediction
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2026-01-06
"""

# constant
GPU_VRAM_LOGGING = True

# standard library imports
import os
from typing import Any

# third party imports
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# first party imports
from prediction_vis import visualize_grid_difference
from torch.utils.data import DataLoader
from tqdm import tqdm


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    loss: float,
    filepath: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        filepath,
    )


def train_scene_to_scene(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_target_frames: int,
    device: torch.device,
    num_epochs: int,
    epochs_between_evals: int,
    learning_rate: float,
    checkpoint_dir: str,
    training_run_name_prefix: str,
    training_run_name: str,
    trial: Any | None = None,
) -> float:
    """
    Train a scene-to-scene frame prediction model.
    Logs MSE and BCE loss but optimizes on BCE.
    Supports multi-step prediction by autoregressively predicting multiple target frames.

    Parameters
    ----------
    model: nn.Module
        The scene-to-scene prediction model
    train_dataloader: DataLoader
        DataLoader for training data
    val_dataloader: DataLoader
        DataLoader for validation data
    num_input_frames: int
        Number of input frames used for prediction
    num_target_frames: int
        Number of target frames to predict
    frame_dims: tuple[int, int]
        Dimensions of each frame (height, width)
    device: torch.device
        Device to run training on (default: CPU)
    model_name: str
        Name prefix for checkpoint files
    batch_size: int
        Batch size (for checkpoint naming)
    num_epochs: int
        Number of training epochs
    epochs_between_evals: int
        Evaluate and checkpoint every N epochs (minimum 1)
    learning_rate: float
        Learning rate for optimizer
    checkpoint_dir: str
        Directory to save checkpoints
    trial: Any | None
        Optional Optuna trial for reporting and pruning
    """
    model.train()
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    bce_criterion = F.binary_cross_entropy
    mse_criterion = F.mse_loss

    # checkpoint setup
    best_val_loss: float = float("inf")
    checkpoint_path: str = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_path, exist_ok=True)
    improvement_threshold: float = 0.01

    epochs_between_evals = max(1, epochs_between_evals)

    for epoch in range(num_epochs):
        model.train()
        loop: tqdm = tqdm(train_dataloader, desc=f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}]", ncols=80)

        epoch_bce_loss: float = 0.0
        epoch_bce_loss_bin: float = 0.0
        epoch_mse_loss: float = 0.0
        epoch_mse_loss_bin: float = 0.0

        predicted_frames: torch.Tensor
        for i, (input_frames, target_frames) in enumerate(loop):
            optimizer.zero_grad()

            # move data to device
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)

            # Memory logging
            if GPU_VRAM_LOGGING:
                if i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
                    reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
                    loop.write(
                        f"\n[Batch {i}] GPU Mem after data load  || "
                        f"Allocated: {allocated_gb:.2f} GB | Reserved: {reserved_gb:.2f} GB"
                    )

            # forward pass
            if num_target_frames == 1:
                # Use the more efficient forward method for single-frame prediction
                predicted_frames = model.forward(input_frames)
            else:
                # Use multi-step for predicting multiple frames autoregressively
                predicted_frames = model.forward_multi_step(input_frames, num_target_frames)
            predicted_frames_bin: torch.Tensor = (predicted_frames >= 0.5).float()

            # loss calculation over all predicted frames
            batch_bce: torch.Tensor = bce_criterion(predicted_frames, target_frames, reduction="mean")
            batch_bce_bin: torch.Tensor = bce_criterion(predicted_frames_bin, target_frames, reduction="mean")
            batch_mse: torch.Tensor = mse_criterion(predicted_frames, target_frames, reduction="mean")
            batch_mse_bin: torch.Tensor = mse_criterion(predicted_frames_bin, target_frames, reduction="mean")

            # backward pass and optimization (optimizing BCE)
            batch_bce.backward()

            # Memory logging
            if GPU_VRAM_LOGGING:
                if i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
                    reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
                    loop.write(
                        f"[Batch {i}] GPU Mem after backward() || "
                        f"Allocated: {allocated_gb:.2f} GB | Reserved: {reserved_gb:.2f} GB"
                    )

            optimizer.step()

            epoch_bce_loss += batch_bce.item()
            epoch_bce_loss_bin += batch_bce_bin.item()
            epoch_mse_loss += batch_mse.item()
            epoch_mse_loss_bin += batch_mse_bin.item()

            loop.set_postfix(bce=batch_bce.item())

        avg_train_bce_loss = epoch_bce_loss / len(train_dataloader)
        avg_train_bce_loss_bin = epoch_bce_loss_bin / len(train_dataloader)
        avg_train_mse_loss = epoch_mse_loss / len(train_dataloader)
        avg_train_mse_loss_bin = epoch_mse_loss_bin / len(train_dataloader)

        if epoch % epochs_between_evals == 0 or epoch == num_epochs - 1:
            avg_val_bce_loss, avg_val_bce_loss_bin, avg_val_mse_loss, avg_val_mse_loss_bin = evaluate_scene_to_scene(
                model,
                val_dataloader,
                num_target_frames,
                device,
                training_run_name_prefix=training_run_name_prefix,
                training_run_name=training_run_name,
                max_visualizations=3,
                epoch=epoch,
            )
            print(
                f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}] | Train | BCE: {avg_train_bce_loss:.6f} | BCE bin: {avg_train_bce_loss_bin:.6f} | MSE: {avg_train_mse_loss:.6f} | MSE bin: {avg_train_mse_loss_bin:.6f}\n"
                f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}] | Val   | BCE: {avg_val_bce_loss:.6f} | BCE bin: {avg_val_bce_loss_bin:.6f} | MSE: {avg_val_mse_loss:.6f} | MSE bin: {avg_val_mse_loss_bin:.6f}\n"
            )

            # checkpoint on evaluation epochs if improved
            if avg_val_bce_loss < best_val_loss - improvement_threshold:
                print(
                    f"              Validation BCE Loss improved from {best_val_loss:.6f} to {avg_val_bce_loss:.6f}. Saving..."
                )
                best_val_loss = avg_val_bce_loss
                save_filename: str = f"{training_run_name_prefix}_e{epoch}.pt"
                save_filepath: str = os.path.join(checkpoint_path, save_filename)
                save_checkpoint(model, epoch, optimizer, best_val_loss, save_filepath)
            if trial is not None:
                trial.report(avg_val_bce_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        else:
            print(
                f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}] | Train | BCE: {avg_train_bce_loss:.6f} | BCE bin: {avg_train_bce_loss_bin:.6f} | MSE: {avg_train_mse_loss:.6f} | MSE bin: {avg_train_mse_loss_bin:.6f}"
            )

    return best_val_loss


def evaluate_scene_to_scene(
    model: nn.Module,
    dataloader: DataLoader,
    num_target_frames: int,
    device: torch.device,
    training_run_name_prefix: str,
    training_run_name: str,
    max_visualizations: int = 0,
    epoch: int | None = None,
) -> tuple[float, float, float, float]:
    """
    Evaluate a scene-to-scene frame prediction model.

    Parameters
    ----------
    model: nn.Module
        The scene-to-scene prediction model
    dataloader: DataLoader
        DataLoader for evaluation data (validation or test)
    num_target_frames: int
        Number of target frames to predict
    device: torch.device
        Device to run evaluation on (default: CPU)

    Returns
    -------
    tuple[float, float, float, float]
        Average BCE loss, BCE loss (binarized), MSE loss, MSE loss (binarized)
    """
    model.eval()

    total_bce_loss: float = 0.0
    total_bce_loss_bin: float = 0.0
    total_mse_loss: float = 0.0
    total_mse_loss_bin: float = 0.0

    bce_criterion = F.binary_cross_entropy
    mse_criterion = F.mse_loss

    images_dir = os.path.join(os.path.dirname(__file__), "runs", training_run_name_prefix, "images")
    should_visualize = max_visualizations > 0 and epoch is not None and epoch % 5 == 0
    if should_visualize:
        os.makedirs(images_dir, exist_ok=True)

    saved_visualizations = 0

    predicted_frames: torch.Tensor
    with torch.no_grad():
        for input_frames, target_frames in dataloader:
            # move data to device
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)

            if num_target_frames == 1:
                predicted_frames = model.forward(input_frames)
            else:
                predicted_frames = model.forward_multi_step(input_frames, num_target_frames)
            predicted_frames_bin: torch.Tensor = (predicted_frames >= 0.5).float()
            if should_visualize and saved_visualizations < max_visualizations:
                batch_size = predicted_frames.size(0)
                for batch_idx in range(batch_size):
                    if saved_visualizations >= max_visualizations:
                        break
                    os.makedirs(images_dir, exist_ok=True)
                    if epoch is not None:
                        image_filename = f"{training_run_name}_e{epoch:03d}_img{saved_visualizations:02d}.png"
                    else:
                        image_filename = f"{training_run_name}_img{saved_visualizations:02d}.png"
                    image_path = os.path.join(images_dir, image_filename)
                    visualize_grid_difference(
                        predicted_frames[batch_idx, 0],
                        target_frames[batch_idx, 0],
                        show_window=False,
                        output_path=image_path,
                    )
                    saved_visualizations += 1

            batch_bce_loss: torch.Tensor = bce_criterion(predicted_frames, target_frames, reduction="mean")
            batch_bce_loss_bin: torch.Tensor = bce_criterion(predicted_frames_bin, target_frames, reduction="mean")
            batch_mse_loss: torch.Tensor = mse_criterion(predicted_frames, target_frames, reduction="mean")
            batch_mse_loss_bin: torch.Tensor = mse_criterion(predicted_frames_bin, target_frames, reduction="mean")

            total_bce_loss += batch_bce_loss.item()
            total_bce_loss_bin += batch_bce_loss_bin.item()
            total_mse_loss += batch_mse_loss.item()
            total_mse_loss_bin += batch_mse_loss_bin.item()

    model.train()

    avg_bce = total_bce_loss / len(dataloader)
    avg_bce_bin = total_bce_loss_bin / len(dataloader)
    avg_mse = total_mse_loss / len(dataloader)
    avg_mse_bin = total_mse_loss_bin / len(dataloader)

    return avg_bce, avg_bce_bin, avg_mse, avg_mse_bin
