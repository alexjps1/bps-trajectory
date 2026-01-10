"""
Training and Evaluation Loops for Scene-to-Scene Frame Prediction
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2026-01-06
"""

# standard library imports
import os

# third party imports
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
    frame_dims: tuple[int, int],
    model_name: str = "lstm_s2s",
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Train a scene-to-scene frame prediction model.
    Logs MSE and BCE loss but optimizes on BCE.

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
    frame_dims: tuple[int, int]
        Dimensions of each frame (height, width)
    model_name: str
        Name prefix for checkpoint files
    batch_size: int
        Batch size (for checkpoint naming)
    num_epochs: int
        Number of training epochs
    learning_rate: float
        Learning rate for optimizer
    checkpoint_dir: str
        Directory to save checkpoints
    """
    model.train()
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    bce_criterion = F.binary_cross_entropy
    mse_criterion = F.mse_loss

    # checkpoint setup
    best_val_loss: float = float("inf")
    checkpoint_path: str = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_filename_prefix: str = (
        f"{model_name}_frames{num_input_frames}_dims{frame_dims[0]}x{frame_dims[1]}_bs{batch_size}"
    )
    improvement_threshold: float = 0.01

    for epoch in range(num_epochs):
        model.train()
        loop: tqdm = tqdm(train_dataloader, desc=f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}]")

        epoch_bce_loss: float = 0.0
        epoch_bce_loss_bin: float = 0.0
        epoch_mse_loss: float = 0.0
        epoch_mse_loss_bin: float = 0.0

        for i, (input_frames, target_frame) in enumerate(loop):
            optimizer.zero_grad()

            # forward pass
            predicted_frame: torch.Tensor = model(input_frames)
            predicted_frame_bin: torch.Tensor = (predicted_frame >= 0.5).float()

            # ensure target is on same device and dtype
            target_frame = target_frame.to(device=predicted_frame.device, dtype=predicted_frame.dtype)

            # loss calculation
            batch_bce: torch.Tensor = bce_criterion(predicted_frame, target_frame, reduction="mean")
            batch_bce_bin: torch.Tensor = bce_criterion(predicted_frame_bin, target_frame, reduction="mean")
            batch_mse: torch.Tensor = mse_criterion(predicted_frame, target_frame, reduction="mean")
            batch_mse_bin: torch.Tensor = mse_criterion(predicted_frame_bin, target_frame, reduction="mean")

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

        # validation
        avg_val_bce_loss, avg_val_bce_loss_bin, avg_val_mse_loss, avg_val_mse_loss_bin = evaluate_scene_to_scene(
            model,
            val_dataloader,
        )
        print(
            f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}] | Train | BCE: {avg_train_bce_loss:.6f} | BCE bin: {avg_train_bce_loss_bin:.6f} | MSE: {avg_train_mse_loss:.6f} | MSE bin: {avg_train_mse_loss_bin:.6f}\n"
            f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}] | Val   | BCE: {avg_val_bce_loss:.6f} | BCE bin: {avg_val_bce_loss_bin:.6f} | MSE: {avg_val_mse_loss:.6f} | MSE bin: {avg_val_mse_loss_bin:.6f}\n"
        )

        # checkpoint on odd epochs if improved
        if epoch % 2 == 1:
            if avg_val_bce_loss < best_val_loss - improvement_threshold:
                print(
                    f"              Validation BCE Loss improved from {best_val_loss:.6f} to {avg_val_bce_loss:.6f}. Saving..."
                )
                best_val_loss = avg_val_bce_loss
                save_filename: str = f"{checkpoint_filename_prefix}_e{epoch}.pt"
                save_filepath: str = os.path.join(checkpoint_path, save_filename)
                save_checkpoint(model, epoch, optimizer, best_val_loss, save_filepath)


def evaluate_scene_to_scene(
    model: nn.Module,
    dataloader: DataLoader,
) -> tuple[float, float, float, float]:
    """
    Evaluate a scene-to-scene frame prediction model.

    Parameters
    ----------
    model: nn.Module
        The scene-to-scene prediction model
    dataloader: DataLoader
        DataLoader for evaluation data (validation or test)

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
        for input_frames, target_frame in dataloader:
            predicted_frame: torch.Tensor = model(input_frames)
            predicted_frame_bin: torch.Tensor = (predicted_frame >= 0.5).float()
            target_frame = target_frame.to(device=predicted_frame.device, dtype=predicted_frame.dtype)

            batch_bce_loss: torch.Tensor = bce_criterion(predicted_frame, target_frame, reduction="mean")
            batch_bce_loss_bin: torch.Tensor = bce_criterion(predicted_frame_bin, target_frame, reduction="mean")
            batch_mse_loss: torch.Tensor = mse_criterion(predicted_frame, target_frame, reduction="mean")
            batch_mse_loss_bin: torch.Tensor = mse_criterion(predicted_frame_bin, target_frame, reduction="mean")

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
