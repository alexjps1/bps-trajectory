"""
Training Entry Point for Scene-to-Scene Frame Prediction
(with option for Optuna hyperparameter tuning)
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2026-01-10
"""

# standard library imports
import argparse
import atexit
import copy
import sys
from collections.abc import Sized
from datetime import datetime
from pathlib import Path
from typing import Dict, cast

# third party imports
import json5
import optuna
import torch
from optuna.trial import Trial
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# first party imports
import loops
from datasets import DynamicScenes2dDataset
from models.lstm_scene_to_scene01 import LSTMSceneToScene01
from models.lstm_scene_to_scene02 import LSTMSceneToScene02

# constants
THIS_DIR = Path(__file__).resolve().parent


class Tee:
    """A helper class to tee output to a file and a stream."""

    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)
        self.flush()

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()


def setup_logging(log_dir: Path, log_name: str) -> None:
    """
    Redirect stdout and stderr to both the console and log files.

    Parameters
    ----------
    log_dir: Path
        Directory to store the log files
    log_name: str
        Base name for the .out and .err log files

    Returns
    -------
    None
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    out_log_path = log_dir / f"{log_name}.out"
    err_log_path = log_dir / f"{log_name}.err"

    # Open log files
    out_file = open(out_log_path, "a")
    err_file = open(err_log_path, "a")

    # write a separator to tell when a new terminal session started
    separator = f"--- New Terminal Output from {datetime.now().isoformat()} ---"
    out_file.write(separator)
    err_file.write(separator)

    # Ensure files are closed on exit
    atexit.register(out_file.close)
    atexit.register(err_file.close)

    # Keep original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Tee stdout and stderr
    sys.stdout = Tee(original_stdout, out_file)
    sys.stderr = Tee(original_stderr, err_file)

    print(f"Logging stdout to: {out_log_path}")
    print(f"Logging stderr to: {err_log_path}")


def objective(
    trial: Trial,
    study_name: str,
    search_space_config: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Optuna objective function for hyperparameter optimization.

    Parameters
    ----------
    trial: Trial
        Optuna trial object for suggesting hyperparameters
    study_name: str
        Name of the Optuna study (used for run naming)
    search_space_config: dict
        Base run configuration with hyperparameter types and acceptable value lists/ranges
        Has similar formatting to the single run config, contained in the "run" attribute of an optuna study config JSON.
    train_dataloader: DataLoader
        Training data loader
    val_dataloader: DataLoader
        Validation data loader
    device: torch.device
        Device to run training on

    Returns
    -------
    float
        Best validation BCE loss of this trial (lower is better)
    """
    # Suggest hyperparameters
    hyperparams = suggest_hyperparams(trial, search_space_config)
    hidden_dim = hyperparams["hidden_dim"]
    num_lstm_layers = hyperparams["num_lstm_layers"]
    dropout_rate = hyperparams["dropout_rate"]
    learning_rate = hyperparams["learning_rate"]
    batch_size = hyperparams["batch_size"]

    # Extract fixed parameters from config
    data_config = search_space_config["data"]
    training_config = search_space_config["training"]
    num_target_frames = data_config["num_target_frames"]
    frame_dims = tuple(training_config["frame_dims"])
    num_epochs = training_config["num_epochs"]

    # Recreate dataloaders with suggested batch size
    train_dataloader_trial = DataLoader(
        dataset=train_dataloader.dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader_trial = DataLoader(
        dataset=val_dataloader.dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Model initialization
    model_config = search_space_config["model"]
    if model_config["type"] == "linear":
        model = LSTMSceneToScene01(
            frame_dims=frame_dims,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
            dropout_rate=dropout_rate,
        )
    elif model_config["type"] == "conv":
        model = LSTMSceneToScene02(
            frame_dims=frame_dims,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
        )
    else:
        raise ValueError("Provided model type in config is not defined.")
    model = model.to(device)

    run_name: str = f"{study_name}_t{trial.number:02d}"
    run_path: Path = THIS_DIR / "runs" / study_name / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    # print info about the current run to the console
    trial_run_config = copy.deepcopy(search_space_config)
    trial_run_config["model"]["hidden_dim"] = hidden_dim
    trial_run_config["model"]["num_lstm_layers"] = num_lstm_layers
    trial_run_config["model"]["dropout_rate"] = dropout_rate
    trial_run_config["training"]["batch_size"] = batch_size
    trial_run_config["training"]["learning_rate"] = learning_rate
    trial_run_config["training"]["num_epochs"] = num_epochs
    print_run_config(trial_run_config, trial.number)
    print(f"[Trial {trial.number}] Model parameters: {model.get_parameter_count():,}")

    # save that info to a json file in the run directory
    with open(run_path / "config.json5", "w") as f:
        json5.dump(trial_run_config, f)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=run_path / "tensorboard")

    # Train with pruning (the loop handles logging and pruning when trial object passed into it)
    try:
        train_summary_dict: Dict[str, float] = loops.train_scene_to_scene(
            model=model,
            train_dataloader=train_dataloader_trial,
            val_dataloader=val_dataloader_trial,
            num_target_frames=num_target_frames,
            device=device,
            num_epochs=num_epochs,
            epochs_between_evals=1,
            learning_rate=learning_rate,
            run_name=run_name,
            run_path=run_path,
            trial=trial,
            writer=writer,
        )
    finally:
        writer.close()

    return train_summary_dict["best_val_bce_loss"]


def suggest_hyperparams(trial: Trial, run_config: dict) -> dict:
    """
    Suggest hyperparameters for all tunable entries in the run configuration.

    Parameters
    ----------
    trial: Trial
        Optuna trial used to suggest values
    run_config: dict
        Run configuration containing model/training hyperparameter spaces

    Returns
    -------
    dict
        Mapping of hyperparameter names to suggested values
    """
    model_space = run_config["model"]
    training_space = run_config["training"]
    hyperparams: dict[str, object] = {}

    for name, space in model_space.items():
        hyperparams[name] = suggest_from_space(trial, name, space)

    for name, space in training_space.items():
        if name in {
            "num_epochs",
            "epochs_between_evals",
            "frame_dims",
            "checkpoint_dir",
        }:
            continue
        hyperparams[name] = suggest_from_space(trial, name, space)

    return hyperparams


def suggest_from_space(trial: Trial, name: str, space: object) -> object:
    """
    Resolve a single hyperparameter space entry.

    Parameters
    ----------
    trial: Trial
        Optuna trial used to suggest values
    name: str
        Hyperparameter name
    space: object
        Either a constant value or a dict defining a search space

    Returns
    -------
    object
        Suggested or constant value for the hyperparameter
    """
    if not isinstance(space, dict) or "type" not in space:
        return space

    space_type = space["type"]
    values = space.get("vals")
    use_log = bool(space.get("log", False))

    if space_type == "categorical":
        assert isinstance(values, list)
        return trial.suggest_categorical(name, values)
    if space_type == "int":
        if isinstance(values, list) and len(values) == 2:
            return trial.suggest_int(name, values[0], values[1], log=use_log)
        return values
    if space_type == "float":
        if isinstance(values, list) and len(values) == 2:
            return trial.suggest_float(name, values[0], values[1], log=use_log)
        return values
    if space_type == "constant":
        if isinstance(values, list):
            return values[0] if values else values
        return values

    raise ValueError(f"Unsupported hyperparameter type '{space_type}' for '{name}'")


def print_run_config(run_config: dict, trial_num: int | None = None) -> None:
    if trial_num is not None:
        print(f"[Trial {trial_num}] hyperparameters")
    else:
        print("[Training run hyperparameters (no run number)]")
    print(json5.dumps(run_config, indent=2, sort_keys=True))


def resolve_config_value(value: object) -> object:
    """
    Resolve a config entry that may be a search space into a concrete value.

    Parameters
    ----------
    value: object
        Either a constant value or a dict with a "vals" entry

    Returns
    -------
    object
        Concrete value to use from the config
    """
    if not isinstance(value, dict) or "type" not in value:
        return value

    values = value.get("vals")
    if isinstance(values, list):
        return values[0] if values else values
    return values


def get_dataset(run_config: dict) -> DynamicScenes2dDataset:
    """
    Create a dataset instance based on the run configuration.

    Parameters
    ----------
    run_config: dict
        Run configuration containing data settings

    Returns
    -------
    DynamicScenes2dDataset
        Dataset configured from the run settings
    """
    data_config = run_config["data"]
    dataset = DynamicScenes2dDataset(
        data_directory=(THIS_DIR / data_config["data_directory"]),
        as_numpy=False,
        num_input_frames=data_config["num_input_frames"],
        num_target_frames=data_config["num_target_frames"],
        file_pattern=data_config["file_pattern"],
        target_frame_offset=data_config.get("target_frame_offset", 0),
    )
    return dataset


def get_split_datasets(
    run_config: dict, dataset: DynamicScenes2dDataset
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Returns train, val, test (in that order) datasets for the given dataset.

    run_config is used to defined the train/test/val split and random seed.
    Returns all three datasets even if given split makes a dataset empty.
    """
    data_config = run_config["data"]
    if data_config["train_split"] + data_config["val_split"] > 1:
        raise ValueError("Values for data split exceed 1")

    # Data split
    total_size = len(dataset)
    train_size = int(data_config["train_split"] * total_size)
    val_size = int(data_config["val_split"] * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(data_config["random_seed"]),
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train, val, test (in that order) dataloaders for the given datasets and batch size.
    """
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def load_config(train_config_path: str | None, study_config_path: str | None) -> tuple[bool, dict, dict | None]:
    """
    Returns is_optuna_study, run_config, study_config
    """
    if train_config_path:
        # no optuna study
        run_config: dict
        with open(train_config_path) as f:
            run_config = json5.load(f)
        assert run_config.get("name") is not None
        return False, run_config, None
    elif study_config_path:
        # is optuna study
        with open(study_config_path) as f:
            study_config = json5.load(f)
            run_config = study_config["run"]
        assert run_config.get("name") is None
        return True, run_config, study_config
    else:
        raise ValueError("Neither a single-run nor optuna study config found")


def run_single_training(run_config: dict, dataset: DynamicScenes2dDataset, device: torch.device) -> None:
    """
    Note: Does not use resolve_config_value to read values from run_config.
    It is assumed that a single run config file will not contain tuning info

    Parameters
    ----------
    run_config: dict
        Single-run configuration with fixed hyperparameters
    dataset: DynamicScenes2dDataset
        Full dataset to split into train/val/test
    device: torch.device
        Device used for training and evaluation

    Returns
    -------
    None
    """
    # config shortcut vars
    training_config = run_config["training"]
    data_config = run_config["data"]
    model_config = run_config["model"]

    # make sure not using the optuna study format for training config
    assert isinstance(training_config["batch_size"], int)
    assert isinstance(training_config["num_epochs"], int)
    assert isinstance(training_config["epochs_between_evals"], int)
    assert isinstance(training_config["learning_rate"], (int, float))
    assert isinstance(model_config["hidden_dim"], int)
    assert isinstance(model_config["num_lstm_layers"], int)
    assert isinstance(model_config["dropout_rate"], (int, float))

    # split dataset and get dataloaders
    train_dataset, val_dataset, test_dataset = get_split_datasets(run_config, dataset)
    train_dl, val_dl, test_dl = get_dataloaders(train_dataset, val_dataset, test_dataset, training_config["batch_size"])

    num_target_frames = data_config["num_target_frames"]
    frame_dims = tuple(training_config["frame_dims"])

    if model_config["type"] == "linear":
        model = LSTMSceneToScene01(
            frame_dims=frame_dims,
            hidden_dim=model_config["hidden_dim"],
            num_lstm_layers=model_config["num_lstm_layers"],
            dropout_rate=model_config["dropout_rate"],
        )
    elif model_config["type"] == "conv":
        model = LSTMSceneToScene02(
            frame_dims=frame_dims,
            hidden_dim=model_config["hidden_dim"],
            num_lstm_layers=model_config["num_lstm_layers"],
        )
    else:
        raise ValueError("Provided model type in config is not defined.")
    model = model.to(device)

    # create dir for this run
    run_name = run_config["name"]
    run_path: Path = THIS_DIR / "runs" / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    # print and save the run config
    print_run_config(run_config)
    with open(run_path / "config.json5", "w") as f:
        json5.dump(run_config, f)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=run_path / "tensorboard")

    try:
        train_summary_dict: Dict[str, float] = loops.train_scene_to_scene(
            model=model,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            num_target_frames=num_target_frames,
            device=device,
            num_epochs=training_config["num_epochs"],
            epochs_between_evals=training_config["epochs_between_evals"],
            learning_rate=training_config["learning_rate"],
            run_name=run_name,
            run_path=run_path,
            writer=writer,
        )
        print(f"Completed training with best val bce loss {train_summary_dict['best_val_bce_loss']}")
    finally:
        writer.close()

    print("\nEvaluating on test set...")
    eval_summary_dict: Dict[str, float] = loops.evaluate_scene_to_scene(
        model=model,
        dataloader=test_dl,
        num_target_frames=num_target_frames,
        device=device,
        run_name=run_name,
        run_path=run_path,
    )

    # unpack values from the evaluation
    test_bce = eval_summary_dict["avg_val_bce"]
    test_bce_bin = eval_summary_dict["avg_val_bce_bin"]
    test_mse = eval_summary_dict["avg_val_mse"]
    test_mse_bin = eval_summary_dict["avg_val_mse_bin"]
    test_f1 = eval_summary_dict["avg_val_f1"]
    test_accuracy = eval_summary_dict["avg_val_accuracy"]

    print(
        f"Test Results | BCE: {test_bce:.6f} | BCE bin: {test_bce_bin:.6f} | MSE: {test_mse:.6f} | MSE bin: {test_mse_bin:.6f} | F1: {test_f1:.6f} | Acc: {test_accuracy:.6f}\n"
    )


def run_optuna_study(
    run_config: dict,
    study_config: dict,
    dataset: DynamicScenes2dDataset,
    device: torch.device,
) -> None:
    study_name = study_config["study"]["name"]

    storage = f"sqlite:///{THIS_DIR}/runs/optuna_storage/{study_config['study']['storage']}"
    n_trials = study_config["optimization"]["n_trials"]
    timeout = study_config["optimization"]["timeout_seconds"]
    pruner_type = study_config["pruner"]["type"]

    data_config = run_config["data"]
    training_config = run_config["training"]
    batch_size = cast(int, resolve_config_value(training_config.get("batch_size", 1)))

    train_dataset, val_dataset, test_dataset = get_split_datasets(run_config, dataset)
    train_dl, val_dl, _ = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)
    train_size = len(cast(Sized, train_dataset))
    val_size = len(cast(Sized, val_dataset))
    test_size = len(cast(Sized, test_dataset))
    data_directory = data_config["data_directory"]

    print(
        f"""
Optuna Hyperparameter Tuning for LSTM Frame Prediction
=======================================================

# Data Settings
data_directory: {data_directory}
train_size: {train_size}
val_size: {val_size}
test_size: {test_size}

# Optuna Settings
n_trials: {n_trials}
study_name: {study_name}
storage: {storage}
pruner: {pruner_type}
    """
    )

    # Create pruner
    if pruner_type == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    elif pruner_type == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(study_config["study"].get("load_if_exists", True)),
        direction="minimize",
        pruner=pruner,
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, study_name, run_config, train_dl, val_dl, device),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Optimization Complete")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation BCE loss: {study.best_trial.value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Train final model with best hyperparameters and evaluate on test set
    print("\n" + "-" * 60)
    print("Training final model with best hyperparameters...")
    print("-" * 60)

    best_params = study.best_trial.params
    frame_dims = tuple(training_config["frame_dims"])
    num_target_frames = data_config["num_target_frames"]

    if run_config["type"] == "linear":
        final_model = LSTMSceneToScene01(
            frame_dims=frame_dims,
            hidden_dim=best_params["hidden_dim"],
            num_lstm_layers=best_params["num_lstm_layers"],
            dropout_rate=best_params["dropout_rate"],
        )
    elif run_config["type"] == "conv":
        final_model = LSTMSceneToScene02(
            frame_dims=frame_dims,
            hidden_dim=best_params["hidden_dim"],
            num_lstm_layers=best_params["num_lstm_layers"],
        )
    else:
        raise ValueError("Provided model type in config is not defined.")

    final_model = final_model.to(device)

    # Recreate dataloaders with best batch size
    train_dataloader_final, val_dataloader_final, test_dataloader_final = get_dataloaders(
        train_dataset, val_dataset, test_dataset, best_params["batch_size"]
    )

    final_run_name: str = f"{study_name}_finalrun"
    final_run_path: Path = THIS_DIR / "runs" / study_name / final_run_name

    # Full training with best hyperparameters
    train_summary_dict: Dict[str, float] = loops.train_scene_to_scene(
        model=final_model,
        train_dataloader=train_dataloader_final,
        val_dataloader=val_dataloader_final,
        num_target_frames=num_target_frames,
        device=device,
        num_epochs=cast(int, resolve_config_value(training_config["num_epochs"])),
        epochs_between_evals=cast(int, resolve_config_value(training_config["epochs_between_evals"])),
        learning_rate=best_params["learning_rate"],
        run_name=final_run_name,
        run_path=final_run_path,
    )
    print(f"Completed final training with best val bce loss {train_summary_dict['best_val_bce_loss']}")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    eval_summary_dict: Dict[str, float] = loops.evaluate_scene_to_scene(
        model=final_model,
        dataloader=test_dataloader_final,
        num_target_frames=num_target_frames,
        device=device,
        run_name=final_run_name,
        run_path=final_run_path,
    )
    # unpack values from the test
    test_bce = eval_summary_dict["avg_val_bce"]
    test_bce_bin = eval_summary_dict["avg_val_bce_bin"]
    test_mse = eval_summary_dict["avg_val_mse"]
    test_mse_bin = eval_summary_dict["avg_val_mse_bin"]
    test_f1 = eval_summary_dict["avg_val_f1"]
    test_accuracy = eval_summary_dict["avg_val_accuracy"]

    print(
        f"Final Test Results | BCE: {test_bce:.6f} | BCE bin: {test_bce_bin:.6f} | MSE: {test_mse:.6f} | MSE bin: {test_mse_bin:.6f} | F1: {test_f1:.6f} | Acc: {test_accuracy:.6f}\n"
    )


def evaluate_checkpoint(
    run_config: dict,
    checkpoint_path: Path,
    dataset: DynamicScenes2dDataset,
    device: torch.device,
) -> None:
    """
    Load a model from a checkpoint and evaluate it on the test set.
    Also checks the maximum predicted value to detect dead models.
    """
    print(f"\n--- Evaluating Checkpoint: {checkpoint_path} ---\n")

    # 1. Initialize Model
    training_config = run_config["training"]
    data_config = run_config["data"]
    model_config = run_config["model"]
    frame_dims = tuple(training_config["frame_dims"])

    if model_config["type"] == "linear":
        model = LSTMSceneToScene01(
            frame_dims=frame_dims,
            hidden_dim=model_config["hidden_dim"],
            num_lstm_layers=model_config["num_lstm_layers"],
            dropout_rate=model_config["dropout_rate"],
        )
    elif model_config["type"] == "conv":
        model = LSTMSceneToScene02(
            frame_dims=frame_dims,
            hidden_dim=model_config["hidden_dim"],
            num_lstm_layers=model_config["num_lstm_layers"],
        )
    else:
        raise ValueError("Provided model type in config is not defined.")

    model = model.to(device)

    # 2. Load Checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print("Loading model weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 3. Prepare Data
    print("Preparing test dataset...")
    train_dataset, val_dataset, test_dataset = get_split_datasets(run_config, dataset)
    batch_size = training_config.get("batch_size", 1)

    # We only need the test dataloader
    _, _, test_dl = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)

    # 4. Max Value Check
    print("Checking model output range (max value)...")
    max_pred = -float("inf")
    num_target_frames = data_config["num_target_frames"]

    # Use tqdm if possible, or just simple loop
    with torch.no_grad():
        for inputs, _ in test_dl:
            inputs = inputs.to(device)
            if num_target_frames == 1:
                preds = model(inputs)
            else:
                preds = model.forward_multi_step(inputs, num_target_frames)

            # Check max value in this batch
            current_max = preds.max().item()
            if current_max > max_pred:
                max_pred = current_max

    print(f"Maximum predicted scalar value in test set: {max_pred:.6f}")
    if max_pred < 0.01:
        print("WARNING: Model appears to be outputting near-zero values (Dead ReLU/Sigmoid saturation?).")

    # 5. Standard Evaluation
    print("Running standard evaluation metrics...")
    # Use parent directory of checkpoint for output (likely 'checkpoints'),
    # so we go up one level to the run directory
    run_path = checkpoint_path.parent.parent
    eval_run_name = f"eval_{checkpoint_path.stem}"

    eval_summary = loops.evaluate_scene_to_scene(
        model=model,
        dataloader=test_dl,
        num_target_frames=num_target_frames,
        device=device,
        run_name=eval_run_name,
        run_path=run_path,
        max_visualizations=5,  # Generate a few visualizations during explicit eval
        epoch=999,  # Dummy epoch number for file naming
    )

    print("\nEvaluation Summary:")
    print(json5.dumps(eval_summary, indent=2))


def main(
    train_config_path: str | None,
    study_config_path: str | None,
    evaluate_checkpoint_path: Path | None = None,
) -> None:
    # Load configuration to get the study name for logging
    is_optuna_study: bool
    run_config: dict | None
    study_config: dict | None
    is_optuna_study, run_config, study_config = load_config(train_config_path, study_config_path)

    # Setup logging to file
    log_dir_name: str
    if is_optuna_study:
        log_dir_name = cast(dict, study_config)["study"]["name"]
        setup_logging(log_dir=(THIS_DIR / "runs" / log_dir_name / "logs"), log_name=log_dir_name)
        print("Running an Optuna study with multiple training runs.")
    else:
        log_dir_name = run_config["name"]
        setup_logging(log_dir=(THIS_DIR / "runs" / log_dir_name / "logs"), log_name=log_dir_name)
        print("Running a single training run.")

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # get dataset
    dataset = get_dataset(run_config)

    if evaluate_checkpoint_path:
        if is_optuna_study:
            print("Warning: Evaluating using a study config. Using the 'run' section as base config.")

        evaluate_checkpoint(run_config, evaluate_checkpoint_path, dataset, device)
        return

    if is_optuna_study:
        run_optuna_study(run_config, cast(dict, study_config), dataset, device)
    else:
        run_single_training(run_config, dataset, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train",
        "-t",
        type=Path,
        metavar="FILE",
        help="Path to json file with configuration for one training run.",
        required=False,
    )
    group.add_argument(
        "--study",
        "-s",
        type=Path,
        metavar="FILE",
        help="Path to json file containing optuna study settings, inluding which hyperparameters to tune and how many runs to perform.",
        required=False,
    )

    parser.add_argument(
        "--evaluate",
        "-e",
        type=Path,
        metavar="FILE",
        help="Path to a checkpoint .pt file to evaluate. Requires --train to provide the model configuration.",
        required=False,
    )

    args = parser.parse_args()

    if not (args.train or args.study):
        raise ValueError("Must provide a config file, either as --train or as --study")

    if args.train and args.study:
        raise ValueError("Cannot provide two config files.")

    main(args.train, args.study, args.evaluate)
