"""
Training and Evaluation Loops for Scene-to-Scene Frame Prediction
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2026-01-06
"""

# standard library imports
import os
from pathlib import Path
from typing import Dict, cast

# third party imports
import matplotlib.pyplot as plt
import optuna

# first party imports
import prediction_vis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

# constants
GPU_VRAM_LOGGING = False
THIS_DIR = Path(__file__).resolve().parent


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    loss: float,
    filepath: Path,
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
    run_name: str,
    run_path: Path,
    teacher_forcing: str = "off",
    scheduled_sampling_decay: float = 0.99,
    trial=None,
    writer: SummaryWriter | None = None,
) -> Dict[str, float]:
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
    num_target_frames: int
        Number of target frames to predict
    device: torch.device
        Device to run training on (default: CPU)
    num_epochs: int
        Number of training epochs
    epochs_between_evals: int
        Evaluate and checkpoint every N epochs (minimum 1)
    learning_rate: float
        Learning rate for optimizer
    run_name: str
        Run identifier used in checkpoint and image filenames
    run_path: Path
        Base directory for run artifacts (checkpoints/images)
    teacher_forcing: str
        Teacher forcing mode: "always", "off", or "scheduled_sampling"
    scheduled_sampling_decay: float
        Controls the linear decay speed of teacher forcing probability.
        Only used when teacher_forcing is "scheduled_sampling".
        p(epoch) = max(1.0 - epoch * decay / num_epochs, 0.0)
        A value of 1.0 means p reaches 0 at the final epoch.
        A value of 2.0 means p reaches 0 at the halfway point.
    trial: optuna.trial.Trial | None
        Optional Optuna trial for reporting and pruning

    Returns
    -------
    float
        Best validation BCE loss observed during training
    """

    model.train()
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    bce_criterion = F.binary_cross_entropy
    mse_criterion = F.mse_loss

    # path for this run
    os.makedirs(run_path, exist_ok=True)

    # checkpoint setup
    improvement_threshold: float = 0.01
    checkpoint_path: Path = run_path / "checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)

    epochs_between_evals = max(1, epochs_between_evals)

    best_val_bce_loss: float = float("inf")
    avg_train_bce_loss: float = float("inf")
    avg_train_bce_loss_bin: float = float("inf")
    avg_train_mse_loss: float = float("inf")
    avg_train_mse_loss_bin: float = float("inf")

    # training loop
    if num_epochs <= 0:
        raise ValueError(f"Training loop called with num_epochs={num_epochs}")
    for epoch in range(num_epochs):
        model.train()
        loop: tqdm = tqdm(
            train_dataloader,
            desc=f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}]",
            ncols=80,
        )

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
                # Determine teacher forcing probability for this epoch
                if teacher_forcing == "always":
                    tf_prob = 1.0
                elif teacher_forcing == "scheduled_sampling":
                    tf_prob = max(1.0 - epoch * scheduled_sampling_decay / num_epochs, 0.0)
                else:  # "off"
                    tf_prob = 0.0

                # Use multi-step for predicting multiple frames autoregressively
                predicted_frames = model.forward_multi_step(input_frames, num_target_frames, tf_prob, target_frames)

            # loss calculation over all predicted frames
            # this is the loss metric with which we actually optimize
            batch_bce: torch.Tensor = bce_criterion(predicted_frames, target_frames, reduction="mean")

            # Use detached copies for other loss calculations
            # This saves memory by excluding them from computational graph, which is fine because we're not using them to optimize
            predicted_frames_detached: torch.Tensor = predicted_frames.detach()
            predicted_frames_bin: torch.Tensor = (predicted_frames_detached >= 0.5).float()

            batch_bce_bin: torch.Tensor = bce_criterion(predicted_frames_bin, target_frames, reduction="mean")
            batch_mse: torch.Tensor = mse_criterion(predicted_frames_detached, target_frames, reduction="mean")
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

        if writer is not None:
            writer.add_scalar("Train/BCE", avg_train_bce_loss, epoch)
            writer.add_scalar("Train/BCE_bin", avg_train_bce_loss_bin, epoch)
            writer.add_scalar("Train/MSE", avg_train_mse_loss, epoch)
            writer.add_scalar("Train/MSE_bin", avg_train_mse_loss_bin, epoch)

        if epoch % epochs_between_evals != 0 and epoch != num_epochs - 1:
            # no evaluation for this epoch
            print(
                f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}] | Train | BCE: {avg_train_bce_loss:.6f} | BCE bin: {avg_train_bce_loss_bin:.6f} | MSE: {avg_train_mse_loss:.6f} | MSE bin: {avg_train_mse_loss_bin:.6f}"
            )
        else:
            # perform an evaluation
            eval_summary_dict: Dict[str, float] = evaluate_scene_to_scene(
                model,
                val_dataloader,
                num_target_frames,
                device,
                run_name=run_name,
                run_path=run_path,
                max_visualizations=3,
                epoch=epoch,
                writer=writer,
            )
            # unpack values from the evaluation
            avg_val_bce_loss = eval_summary_dict["avg_val_bce"]
            avg_val_bce_loss_bin = eval_summary_dict["avg_val_bce_bin"]
            avg_val_mse_loss = eval_summary_dict["avg_val_mse"]
            avg_val_mse_loss_bin = eval_summary_dict["avg_val_mse_bin"]
            avg_val_f1 = eval_summary_dict["avg_val_f1"]
            avg_val_accuracy = eval_summary_dict["avg_val_accuracy"]

            print(
                f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}] | Train | BCE: {avg_train_bce_loss:.6f} | BCE bin: {avg_train_bce_loss_bin:.6f} | MSE: {avg_train_mse_loss:.6f} | MSE bin: {avg_train_mse_loss_bin:.6f}\n"
                f"[Epoch {epoch:02d}/{(num_epochs - 1):02d}] | Val   | BCE: {avg_val_bce_loss:.6f} | BCE bin: {avg_val_bce_loss_bin:.6f} | MSE: {avg_val_mse_loss:.6f} | MSE bin: {avg_val_mse_loss_bin:.6f} | F1: {avg_val_f1:.6f} | Acc: {avg_val_accuracy:.6f}\n"
            )

            # write metrics to tensorboard
            if writer is not None:
                writer.add_scalar("Val/BCE", avg_val_bce_loss, epoch)
                writer.add_scalar("Val/BCE_bin", avg_val_bce_loss_bin, epoch)
                writer.add_scalar("Val/MSE", avg_val_mse_loss, epoch)
                writer.add_scalar("Val/MSE_bin", avg_val_mse_loss_bin, epoch)
                writer.add_scalar("Val/F1", avg_val_f1, epoch)
                writer.add_scalar("Val/Accuracy", avg_val_accuracy, epoch)

            # checkpoint on evaluation epochs if improved
            if avg_val_bce_loss < best_val_bce_loss - improvement_threshold:
                print(
                    f"              Validation BCE Loss improved from {best_val_bce_loss:.6f} to {avg_val_bce_loss:.6f}. Saving..."
                )
                save_filename: str = f"{run_name}_e{epoch:02d}.pt"
                save_filepath: Path = checkpoint_path / save_filename
                save_checkpoint(model, epoch, optimizer, avg_val_bce_loss, save_filepath)

            if trial is not None:
                trial.report(avg_val_bce_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # update the best_val_loss to help optuna optimize
            best_val_bce_loss = min(best_val_bce_loss, avg_val_bce_loss)

    if float("inf") in [
        best_val_bce_loss,
        avg_train_bce_loss,
        avg_train_bce_loss_bin,
        avg_train_mse_loss,
        avg_train_mse_loss_bin,
    ]:
        raise ValueError("Infinite loss found in one of the loss measures. There was a problem with the training loop.")

    return {
        "best_val_bce_loss": best_val_bce_loss,  # bce loss is used by the parameter optimizer (Adam) and by hyperparameter optimizer (Optuna)
        "final_train_bce_loss": avg_train_bce_loss,
        "final_train_bce_loss_bin": avg_train_bce_loss_bin,
        "final_train_mse_loss": avg_train_mse_loss,
        "final_train_mse_loss_bin": avg_train_mse_loss_bin,
    }


# You may need to import this at the top of loops.py
from prediction_vis import (
    visualize_grid_difference,
    visualize_time_series_grid_difference,
)


def evaluate_scene_to_scene(
    model: nn.Module,
    dataloader: DataLoader,
    num_target_frames: int,
    device: torch.device,
    run_name: str,
    run_path: Path,
    max_visualizations: int = 0,
    epoch: int | None = None,
    writer: SummaryWriter | None = None,
) -> Dict[str, float]:
    """
    Evaluate a scene-to-scene frame prediction model.
    """
    model.eval()

    total_bce_loss: float = 0.0
    total_bce_loss_bin: float = 0.0
    total_mse_loss: float = 0.0
    total_mse_loss_bin: float = 0.0
    total_f1_score: float = 0.0
    total_accuracy: float = 0.0

    bce_criterion = F.binary_cross_entropy
    mse_criterion = F.mse_loss

    # prepare visualizations
    saved_visualizations = 0
    should_visualize = max_visualizations > 0 and epoch is not None and epoch % 5 == 0
    images_path: Path | None = None
    if should_visualize:
        images_path = run_path / "images"
        os.makedirs(images_path, exist_ok=True)

    predicted_frames: torch.Tensor
    with torch.no_grad():
        for input_frames, target_frames in dataloader:
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)

            if num_target_frames == 1:
                predicted_frames = model.forward(input_frames)
                # Ensure dimension consistency: add temporal dim if missing
                # Model01 returns (B, H, W) -> (B, 1, H, W)
                # Model02 returns (B, 1, H, W) which already matches target shape
                if predicted_frames.ndim == 3:
                    predicted_frames = predicted_frames.unsqueeze(1)
            else:
                predicted_frames = model.forward_multi_step(input_frames, num_target_frames)

            # Detach for metrics calculation to save VRAM
            predictions_detached = predicted_frames.detach()
            predicted_frames_bin = (predictions_detached >= 0.5).float()

            # --- Metrics Calculation (Global) ---
            target_frames_bin = (target_frames >= 0.5).float()

            # Accuracy
            batch_accuracy = (predicted_frames_bin == target_frames_bin).float().mean()

            # F1 Score
            tp = (predicted_frames_bin * target_frames_bin).sum()
            fp = (predicted_frames_bin * (1 - target_frames_bin)).sum()
            fn = ((1 - predicted_frames_bin) * target_frames_bin).sum()
            batch_f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)

            total_accuracy += batch_accuracy.item()
            total_f1_score += batch_f1.item()

            # --- Visualizations ---
            if should_visualize and saved_visualizations < max_visualizations:
                batch_size = predictions_detached.size(0)

                for sample_idx in range(batch_size):
                    if saved_visualizations >= max_visualizations:
                        break

                    # 1. Calculate Per-Frame Metrics (BCE, MSE, F1, Acc)
                    sample_metrics = []
                    actual_T = predictions_detached.shape[1]

                    for t in range(actual_T):
                        # Extract frames (C, H, W)
                        pred_t = predictions_detached[sample_idx, t]
                        target_t = target_frames[sample_idx, t]

                        # Discretize (Binarize)
                        pred_bin = (pred_t >= 0.5).float()
                        target_bin = (target_t >= 0.5).float()

                        # --- Continuous Metrics ---
                        loss_bce = bce_criterion(pred_t, target_t).item()
                        loss_mse = mse_criterion(pred_t, target_t).item()

                        # --- Discrete Metrics ---
                        # BCE on binarized output (Note: can be unstable if prediction is confidently wrong)
                        loss_bce_bin = bce_criterion(pred_bin, target_bin).item()
                        loss_mse_bin = mse_criterion(pred_bin, target_bin).item()

                        # Accuracy: Percentage of pixels matching exactly
                        acc = (pred_bin == target_bin).float().mean().item()

                        # F1 Score
                        tp = (pred_bin * target_bin).sum()
                        fp = (pred_bin * (1 - target_bin)).sum()
                        fn = ((1 - pred_bin) * target_bin).sum()

                        f1 = (2 * tp / (2 * tp + fp + fn + 1e-7)).item()

                        sample_metrics.append(
                            {
                                "bce": loss_bce,
                                "bce_bin": loss_bce_bin,
                                "mse": loss_mse,
                                "mse_bin": loss_mse_bin,
                                "f1": f1,
                                "accuracy": acc,
                            }
                        )

                    # 2. Define Base Filename
                    if epoch is not None:
                        base_fname = f"{run_name}_e{epoch:03d}_img{saved_visualizations:02d}"
                    else:
                        base_fname = f"{run_name}_img{saved_visualizations:02d}"

                    img_dir = cast(Path, images_path)

                    # 3. Call Visualization 1: Single Frame Difference (Legacy)
                    # Only for non-autoregressive (single frame) models
                    if num_target_frames == 1:
                        prediction_vis.visualize_grid_difference(
                            predictions_detached[sample_idx, 0].cpu(),
                            target_frames[sample_idx, 0].cpu(),
                            save_image=True,
                            show_window=False,
                            output_path=img_dir / f"{base_fname}_singlediff.png",
                        )

                    # 4. Call Visualization 2: Time Series Grid Difference (New)
                    # Works for both single-frame and multi-frame
                    prediction_vis.visualize_time_series_grid_difference(
                        predictions=predictions_detached[sample_idx].squeeze(1).cpu(),  # (T, H, W)
                        targets=target_frames[sample_idx].squeeze(1).cpu(),  # (T, H, W)
                        input_frames=input_frames[sample_idx].squeeze(1).cpu(),  # (T_in, H, W)
                        loss_metrics=sample_metrics,
                        output_path=img_dir / f"{base_fname}_tseries.png",
                        save_image=True,
                        show_window=False,
                    )

                    # Log to TensorBoard (using the tseries image if available)
                    if writer is not None and epoch is not None:
                        try:
                            # Prefer tseries image for tensorboard
                            tb_img_path = img_dir / f"{base_fname}_tseries.png"
                            img_array = plt.imread(str(tb_img_path))
                            writer.add_image(
                                f"Prediction/Image_{saved_visualizations:02d}",
                                img_array,
                                global_step=epoch,
                                dataformats="HWC",
                            )
                        except Exception as e:
                            print(f"Failed to log image to TensorBoard: {e}")

                    saved_visualizations += 1

            # --- Global Loss Accumulation ---
            # (Must use attached 'predicted_frames' for BCE to ensure graph correctness if this were training,
            #  but technically in eval() it doesn't matter. Keeping consistent with training loop logic.)
            batch_bce_loss: torch.Tensor = bce_criterion(predicted_frames, target_frames, reduction="mean")
            batch_bce_loss_bin: torch.Tensor = bce_criterion(predicted_frames_bin, target_frames, reduction="mean")
            batch_mse_loss: torch.Tensor = mse_criterion(predictions_detached, target_frames, reduction="mean")
            batch_mse_loss_bin: torch.Tensor = mse_criterion(predicted_frames_bin, target_frames, reduction="mean")

            total_bce_loss += batch_bce_loss.item()
            total_bce_loss_bin += batch_bce_loss_bin.item()
            total_mse_loss += batch_mse_loss.item()
            total_mse_loss_bin += batch_mse_loss_bin.item()

    model.train()

    avg_val_bce = total_bce_loss / len(dataloader)
    avg_val_bce_bin = total_bce_loss_bin / len(dataloader)
    avg_val_mse = total_mse_loss / len(dataloader)
    avg_val_mse_bin = total_mse_loss_bin / len(dataloader)
    avg_val_f1 = total_f1_score / len(dataloader)
    avg_val_accuracy = total_accuracy / len(dataloader)

    return {
        "avg_val_bce": avg_val_bce,
        "avg_val_bce_bin": avg_val_bce_bin,
        "avg_val_mse": avg_val_mse,
        "avg_val_mse_bin": avg_val_mse_bin,
        "avg_val_f1": avg_val_f1,
        "avg_val_accuracy": avg_val_accuracy,
    }
