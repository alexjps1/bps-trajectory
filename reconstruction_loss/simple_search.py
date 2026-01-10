"""
Simple Grid Search for ML-based Reconstruction
for ADLR Trajectory Planning Project
Mortiz Schüler and Alexander João Peterson Santos
2025-11-11
"""

# standard library imports
import itertools
import json
import os
import subprocess
from datetime import datetime


def create_config(base_config, param_values, run_id):
    """Create a config with the given parameter values"""
    config = json.loads(json.dumps(base_config))  # deep copy

    # Apply parameter overrides
    for param_path, value in param_values.items():
        keys = param_path.split(".")
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value

    # Save config
    config_filename = f"config_run_{run_id:03d}.json"
    config_path = os.path.join("reconstruction_loss", "configs", config_filename)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def run_training(config_path, run_id, param_values):
    """Run a single training with timeout and error handling"""

    # Create descriptive log name
    param_str = "_".join([f"{k.split('.')[-1]}-{v}" for k, v in param_values.items()])
    log_filename = f"run_{run_id:03d}_{param_str}.log"
    log_path = os.path.join("reconstruction_loss", "logs", log_filename)

    print(f"Run {run_id}: {param_str}")
    print(f"Config: {config_path}")
    print(f"Log: {log_path}")

    # Run command
    cmd = [
        "python3",
        "-u",
        "reconstruction_loss/scene_reconstruction_ml.py",
        "--config",
        config_path,
    ]

    try:
        with open(log_path, "w") as log_file:
            # Write header
            log_file.write(f"Run {run_id} - {datetime.now()}\n")
            log_file.write(f"Parameters: {param_values}\n")
            log_file.write("=" * 50 + "\n\n")
            log_file.flush()

            # Run with timeout (30 minutes max per run)
            # Set TQDM_DISABLE to prevent progress bar spam in logs
            env = os.environ.copy()
            env["TQDM_DISABLE"] = "1"

            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=2700,  # 45 minutes
                text=True,
                env=env,
            )

            # Write result
            log_file.write(f"\n{'=' * 50}\n")
            log_file.write(f"Return code: {result.returncode}\n")

        if result.returncode == 0:
            print(f"✓ Success")
            return True
        else:
            print(f"✗ Failed (code {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout (30 min)")
        with open(log_path, "a") as log_file:
            log_file.write(f"\n{'=' * 50}\n")
            log_file.write("TIMEOUT - Training exceeded 30 minutes\n")
        return False

    except Exception as e:
        print(f"✗ Error: {e}")
        with open(log_path, "a") as log_file:
            log_file.write(f"\n{'=' * 50}\n")
            log_file.write(f"ERROR: {e}\n")
        return False


def main():
    # Load base config
    base_config_path = "reconstruction_loss/example_config.json"
    with open(base_config_path, "r") as f:
        base_config = json.load(f)

    # ==============================================
    # DEFINE SEARCH PARAMETERS HERE
    # ==============================================
    search_params = {
        "bps_settings.grid_size": [8, 16, 32, 64],
        "norm_bound_shape": ["ncube", "none"],
    }
    # ==============================================

    # Generate all combinations
    param_names = list(search_params.keys())
    param_values = list(search_params.values())
    combinations = list(itertools.product(*param_values))

    print("Hyperparameter Search")
    print(f"Base config: {base_config_path}")
    print(f"Parameters: {search_params}")
    print(f"Total combinations: {len(combinations)}")
    print()

    # Create directories
    os.makedirs("reconstruction_loss/logs", exist_ok=True)
    os.makedirs("reconstruction_loss/configs", exist_ok=True)
    os.makedirs("reconstruction_loss/images", exist_ok=True)
    os.makedirs("reconstruction_loss/checkpoints", exist_ok=True)

    # Confirm
    response = input("Continue? (y/n): ")
    if response.lower() != "y":
        print("Cancelled")
        return

    # Run search
    successful = 0
    failed = 0

    print(f"\nStarting search at {datetime.now()}")
    print("=" * 60)

    for i, combination in enumerate(combinations, 1):
        param_dict = dict(zip(param_names, combination))

        print(f"\n[{i}/{len(combinations)}] ", end="")

        try:
            # Create config
            config_path = create_config(base_config, param_dict, i)

            # Run training
            if run_training(config_path, i, param_dict):
                successful += 1
            else:
                failed += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user!")
            break
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Search completed at {datetime.now()}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")

    # Show log files
    print("\nLog files in: reconstruction_loss/logs/")
    print("Config files in: reconstruction_loss/configs/")


if __name__ == "__main__":
    main()
