"""
Training and Testing Loops for Scene Reconstruction Loss
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2025-11-18
"""

# standard library imports
import os

# third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch3d.loss import chamfer_distance
from torch.utils.data import DataLoader
from tqdm import tqdm

# first party imports
from reconstruction_vis import visualize_cloud_difference, visualize_grid_difference


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


"""
Train and Evaluation Loop for Occupancy Grid Decoder
"""


def train_bps_grid_decoder_2d(
    decoder: nn.Module,
    train_dataloader: DataLoader,  # make sure the associated dataset has target_encoding="grid"
    val_dataloader: DataLoader,  # for validation and deciding which state dicts to save
    num_basis_points: int,
    encoding_type: str,
    norm_bound_shape: str,
    bps_type: str,
    model_name: str = "grid",
    batch_size: int = 64,
    num_epochs: int = 8,
) -> None:
    """
    Train a 2d occupancy-grid decoder.
    Logs MSE and BCE loss but optimizes on BCE
    """
    decoder.train()
    optimizer: optim.Optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    bce_criterion = F.binary_cross_entropy
    mse_criterion = F.mse_loss

    # checkpoint setup
    best_val_loss: float = 0.4
    checkpoint_path: str = os.path.abspath("/home/alexjps/TrajectoryPlanning/reconstruction_loss/checkpoints")
    checkpoint_filename_prefix: str = f"{model_name}_bps{bps_type.title()}_pts{num_basis_points}_enc{encoding_type.title()}_norm{norm_bound_shape.title()}_bs{batch_size}"
    improvement_threshold: float = 0.05

    for epoch in range(num_epochs):
        loop: tqdm = tqdm(train_dataloader, desc=f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}]")

        epoch_bce_loss: float = 0.0
        epoch_bce_loss_bin: float = 0.0
        epoch_mse_loss: float = 0.0
        epoch_mse_loss_bin: float = 0.0

        for i, (bps_encoding, target_grid) in enumerate(loop):
            optimizer.zero_grad()

            # forward pass
            predicted_grid: torch.Tensor = decoder(bps_encoding)
            predicted_grid_bin: torch.Tensor = (predicted_grid >= 0.5).float()

            # loss and batch loss calculation
            target_grid = target_grid.to(device=predicted_grid.device, dtype=predicted_grid.dtype)

            batch_bce: torch.Tensor = bce_criterion(predicted_grid, target_grid, reduction="mean")
            batch_bce_bin: torch.Tensor = bce_criterion(predicted_grid_bin, target_grid, reduction="mean")
            batch_mse: torch.Tensor = mse_criterion(predicted_grid, target_grid, reduction="mean")
            batch_mse_bin: torch.Tensor = mse_criterion(predicted_grid_bin, target_grid, reduction="mean")

            # backward pass and optimization (optimizing BCE, not MSE)
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
        avg_val_bce_loss, avg_val_bce_loss_bin, avg_val_mse_loss, avg_val_mse_loss_bin = evaluate_bps_grid_decoder(
            decoder,
            val_dataloader,
            f"/home/alexjps/TrajectoryPlanning/reconstruction_loss/images/{model_name}_bps{bps_type.title()}_pts{num_basis_points}_enc{encoding_type.title()}_norm{norm_bound_shape.title()}_bs{batch_size}.png",
        )
        print(f"""[Epoch {epoch:02d}/{(num_epochs - 1):02d}] | Train | BCE: {avg_train_bce_loss:.6f} | BCE bin: {avg_train_bce_loss_bin:.6f} | MSE: {avg_train_mse_loss:.6f} | MSE bin: {avg_train_mse_loss_bin:.6f}
[Epoch {epoch:02d}/{(num_epochs - 1):02d}] | Val   | BCE: {avg_val_bce_loss:.6f} | BCE bin: {avg_val_bce_loss_bin:.6f} | MSE: {avg_val_mse_loss:.6f} | MSE bin: {avg_val_mse_loss_bin:.6f}
""")

        # check whether to checkpoint on every even epoch
        if epoch % 2 == 1:
            if avg_val_bce_loss < best_val_loss - improvement_threshold:
                print(
                    f"              Validation BCE Loss improved from {best_val_loss} to {avg_val_bce_loss}. Saving..."
                )
                best_val_loss = avg_val_bce_loss
                save_filename: str = f"{checkpoint_filename_prefix}_e{epoch}.pt"
                save_filepath: str = os.path.join(checkpoint_path, save_filename)
                save_checkpoint(decoder, epoch, optimizer, best_val_loss, save_filepath)


def evaluate_bps_grid_decoder(
    decoder: nn.Module,
    dataloader: DataLoader,  # either test or validation is OK
    visualization_filepath: str,
) -> tuple[float, float, float, float]:
    decoder.eval()

    total_bce_loss: float = 0.0
    total_bce_loss_bin: float = 0.0
    total_mse_loss: float = 0.0
    total_mse_loss_bin: float = 0.0

    bce_criterion = F.binary_cross_entropy
    mse_criterion = F.mse_loss

    visualization_generated: bool = False

    with torch.no_grad():
        for bps_encoding, target_grid in dataloader:
            predicted_grid: torch.Tensor = decoder(bps_encoding)
            predicted_grid_bin: torch.Tensor = (predicted_grid >= 0.5).float()
            target_grid = target_grid.to(device=predicted_grid.device, dtype=predicted_grid.dtype)

            if not visualization_generated:
                visualize_grid_difference(
                    predicted_grid[0],
                    target_grid[0],
                    show_window=False,
                    output_path=visualization_filepath,
                )
                visualization_generated = True

            batch_bce_loss: torch.Tensor = bce_criterion(predicted_grid, target_grid, reduction="mean")
            batch_bce_loss_bin: torch.Tensor = bce_criterion(predicted_grid_bin, target_grid, reduction="mean")
            batch_mse_loss: torch.Tensor = mse_criterion(predicted_grid, target_grid, reduction="mean")
            batch_mse_loss_bin: torch.Tensor = mse_criterion(predicted_grid_bin, target_grid, reduction="mean")

            total_bce_loss += batch_bce_loss.item()
            total_bce_loss_bin += batch_bce_loss_bin.item()
            total_mse_loss += batch_mse_loss.item()
            total_mse_loss_bin += batch_mse_loss_bin.item()

    decoder.train()

    avg_bce = total_bce_loss / len(dataloader)
    avg_bce_bin = total_bce_loss_bin / len(dataloader)
    avg_mse = total_mse_loss / len(dataloader)
    avg_mse_bin = total_mse_loss_bin / len(dataloader)

    return avg_bce, avg_bce_bin, avg_mse, avg_mse_bin


"""
Training and Evaluation Loops for Point Cloud Decoder
"""


def train_bps_cloud_decoder_2d(
    decoder: nn.Module,
    train_dataloader: DataLoader,  # make sure the associated dataset has target_encoding="cloud"
    val_dataloader: DataLoader,  # for validation and deciding when to save state dicts
    num_basis_points: int,
    encoding_type: str,
    norm_bound_shape: str,
    bps_type: str,
    model_name: str = "cloud",
    batch_size: int = 64,
    num_epochs: int = 50,
) -> None:
    decoder.train()
    optimizer: optim.Optimizer = optim.Adam(decoder.parameters(), lr=0.001)

    # checkpoint setup
    best_val_loss: float = 500
    checkpoint_path: str = os.path.abspath("/home/alexjps/TrajectoryPlanning/reconstruction_loss/checkpoints")
    checkpoint_filename_prefix: str = f"{model_name}_bps{bps_type.title()}_pts{num_basis_points}_enc{encoding_type.title()}_norm{norm_bound_shape.title()}_bs{batch_size}"
    improvement_threshold: float = 50

    # training loop
    for epoch in range(num_epochs):
        loop: tqdm = tqdm(train_dataloader, desc=f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}]")

        epoch_loss: float = 0.0

        for i, (bps_encoding, target_cloud, num_points) in enumerate(loop):
            optimizer.zero_grad()

            predicted_cloud: torch.Tensor = decoder(bps_encoding)

            batch_loss: torch.Tensor = torch.tensor(0.0, device=bps_encoding.device, requires_grad=True)
            actual_batch_size: int = bps_encoding.size(0)

            for j in range(actual_batch_size):
                # truncate model output to actual number of points in ea scene (can't be done in batches)
                num_points_j: int = num_points[j].item()
                predicted_cloud_j: torch.Tensor = predicted_cloud[j, :num_points_j, :]
                target_cloud_j: torch.Tensor = target_cloud[j]

                # remove z coordinate from each predicted point (we're training in 2d only)
                predicted_cloud_j_2d: torch.Tensor = predicted_cloud_j[:, :2]

                assert target_cloud_j.size(1) == 2

                loss_tuple, loss_normals = chamfer_distance(
                    x=predicted_cloud_j_2d.unsqueeze(0),
                    y=target_cloud_j.unsqueeze(0),
                    batch_reduction=None,
                    point_reduction=None,
                    single_directional=False,
                    x_normals=None,
                    y_normals=None,
                )

                loss_forward, loss_backward = loss_tuple

                loss_j: torch.Tensor = loss_forward.mean() + loss_backward.mean()

                batch_loss = batch_loss + loss_j

            batch_loss = batch_loss / actual_batch_size

            # backward pass and optimization
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            loop.set_postfix(loss=batch_loss.item())

        avg_train_loss = epoch_loss / len(train_dataloader)

        # validation
        avg_val_loss = evaluate_bps_cloud_decoder(
            decoder,
            val_dataloader,
            f"/home/alexjps/TrajectoryPlanning/reconstruction_loss/images/{model_name}_bps{bps_type.title()}_pts{num_basis_points}_enc{encoding_type.title()}_norm{norm_bound_shape.title()}_bs{batch_size}.png",
        )
        print(
            f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}] Avg Train Loss: {avg_train_loss:.4f} | Avg Validation Loss: {avg_val_loss:.4f}"
        )

        # check whether to checkpoint on every even epoch
        if epoch % 2 == 1:
            if avg_val_loss < best_val_loss - improvement_threshold:
                print(f"              Validation Loss improved from {best_val_loss} to {avg_val_loss}. Saving...")
                best_val_loss = avg_val_loss
                save_filename: str = f"{checkpoint_filename_prefix}_e{epoch}.pt"
                save_filepath: str = os.path.join(checkpoint_path, save_filename)
                save_checkpoint(decoder, epoch, optimizer, best_val_loss, save_filepath)


def evaluate_bps_cloud_decoder(
    decoder: nn.Module,
    dataloader: DataLoader,  # either test or validation is OK
    visualization_filepath: str,
) -> float:
    decoder.eval()
    total_loss: float = 0.0

    visualization_generated: bool = False

    with torch.no_grad():
        for bps_encoding, target_cloud, num_points in dataloader:
            predicted_cloud: torch.Tensor = decoder(bps_encoding)
            actual_batch_size: int = bps_encoding.size(0)
            batch_loss: torch.Tensor = torch.tensor(0.0, device=bps_encoding.device)

            for j in range(actual_batch_size):
                # truncate and process points for Chamfer distance
                num_points_j: int = num_points[j].item()
                predicted_cloud_j: torch.Tensor = predicted_cloud[j, :num_points_j, :]
                predicted_cloud_j_2d: torch.Tensor = predicted_cloud_j[:, :2]

                # try visualization thing
                if not visualization_generated:
                    visualize_cloud_difference(predicted_cloud_j_2d, target_cloud[j], visualization_filepath)
                    visualization_generated = True

                loss_tuple, _ = chamfer_distance(
                    x=predicted_cloud_j_2d.unsqueeze(0),
                    y=target_cloud[j].unsqueeze(0),
                    batch_reduction=None,
                    point_reduction=None,
                )
                loss_forward, loss_backward = loss_tuple
                loss_j: torch.Tensor = loss_forward.mean() + loss_backward.mean()
                batch_loss = batch_loss + loss_j

            # average loss per scene
            total_loss += (batch_loss.item() / actual_batch_size) * actual_batch_size

    decoder.train()
    return total_loss / len(dataloader)
