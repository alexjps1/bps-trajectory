"""
Training and Evaluation Loops for Scene-to-Scene Frame Prediction
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2026-01-06
"""

# standard library imports
import os
from typing import Any

# third party imports
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    num_input_frames: int,
    num_target_frames: int,
    frame_dims: tuple[int, int],
    device: torch.device = torch.device("cpu"),
    model_name: str = "lstm_s2s",
    batch_size: int = 32,
    num_epochs: int = 50,
    epochs_between_evals: int = 1,
    learning_rate: float = 0.001,
    checkpoint_dir: str = "checkpoints",
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
    checkpoint_filename_prefix: str = f"{model_name}_frames{num_input_frames}_target{num_target_frames}_dims{frame_dims[0]}x{frame_dims[1]}_bs{batch_size}"
    improvement_threshold: float = 0.01

    epochs_between_evals = max(1, epochs_between_evals)

    for epoch in range(num_epochs):
        model.train()
        loop: tqdm = tqdm(train_dataloader, desc=f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}]")

        epoch_bce_loss: float = 0.0
        epoch_bce_loss_bin: float = 0.0
        epoch_mse_loss: float = 0.0
        epoch_mse_loss_bin: float = 0.0

        for i, (input_frames, target_frames) in enumerate(loop):
            optimizer.zero_grad()

            # move data to device
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)

            # forward pass: predict multiple frames autoregressively
            predicted_frames: torch.Tensor = model.forward_multi_step(input_frames, num_target_frames)
            predicted_frames_bin: torch.Tensor = (predicted_frames >= 0.5).float()

            # loss calculation over all predicted frames
            batch_bce: torch.Tensor = bce_criterion(predicted_frames, target_frames, reduction="mean")
            batch_bce_bin: torch.Tensor = bce_criterion(predicted_frames_bin, target_frames, reduction="mean")
            batch_mse: torch.Tensor = mse_criterion(predicted_frames, target_frames, reduction="mean")
            batch_mse_bin: torch.Tensor = mse_criterion(predicted_frames_bin, target_frames, reduction="mean")

            # backward pass and optimization (optimizing BCE)
            batch_bce.backward()
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
                save_filename: str = f"{checkpoint_filename_prefix}_e{epoch}.pt"
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
    device: torch.device = torch.device("cpu"),
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

    with torch.no_grad():
        for input_frames, target_frames in dataloader:
            # move data to device
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)

            predicted_frames: torch.Tensor = model.forward_multi_step(input_frames, num_target_frames)
            predicted_frames_bin: torch.Tensor = (predicted_frames >= 0.5).float()

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
