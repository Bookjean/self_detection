"""Interactive Training Script for Self Detection Models.

Allows selection of:
- Model architecture (stage1_mlp, simple_mlp, deep_mlp, tcn, tcn_light, etc.)
- DataLoader type (standard/shuffled, sequence, no_shuffle)
- Training hyperparameters

Output files are saved with format: {model}_{loader}_{YYYYMMDD_HHMMSS}.pth

Usage:
    python train_example.py                    # Interactive mode
    python train_example.py --auto             # Use defaults
    python train_example.py --model tcn --loader sequence
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from self_detection_mlp.data_loader import DataLoaderFactory
from self_detection_mlp.model import ModelRegistry
from self_detection_mlp.trainer import Trainer
from self_detection_mlp.utils import set_seed, get_device


def print_banner():
    """Print welcome banner."""
    print()
    print("=" * 65)
    print("   Self Detection Training Framework")
    print("   - Select Model & DataLoader interactively")
    print("   - Supports MLP, TCN, and more")
    print("=" * 65)


def find_data_files(project_root: Path) -> list:
    """Find available data files."""
    data_files = []
    for pattern in ["*.txt", "*.csv", "data/*.txt", "data/*.csv"]:
        data_files.extend(project_root.glob(pattern))
    return sorted(set(data_files))


def select_from_list(options: list, prompt: str, default: int = 0) -> int:
    """Interactive selection from list."""
    print(f"\n{prompt}")
    print("-" * 50)
    for i, opt in enumerate(options):
        marker = " (default)" if i == default else ""
        print(f"  [{i}] {opt}{marker}")

    while True:
        try:
            choice = input(f"\nSelect [0-{len(options)-1}] (Enter for default): ").strip()
            if choice == "":
                return default
            idx = int(choice)
            if 0 <= idx < len(options):
                return idx
            print(f"Invalid choice. Please enter 0-{len(options)-1}")
        except ValueError:
            print("Please enter a number.")


def get_user_config(project_root: Path) -> dict:
    """Get configuration through interactive prompts."""
    config = {}

    # 1. Select data file
    data_files = find_data_files(project_root)
    if not data_files:
        print("Error: No data files found in project directory.")
        print("Please place a .txt or .csv file in the project root.")
        sys.exit(1)

    data_options = [str(f.relative_to(project_root)) for f in data_files]
    idx = select_from_list(data_options, "Select Data File:", default=0)
    config["data_file"] = str(data_files[idx])

    # 2. Select model
    ModelRegistry.list_models()
    model_names = ModelRegistry.get_available_models()
    idx = select_from_list(model_names, "Select Model:", default=0)
    config["model_name"] = model_names[idx]

    # Check if model requires sequence data
    requires_seq = ModelRegistry.requires_sequence(config["model_name"])

    # 3. Select dataloader type
    DataLoaderFactory.list_loaders()
    loader_types = list(DataLoaderFactory.AVAILABLE_LOADERS.keys())

    if requires_seq:
        print(f"  Note: {config['model_name']} requires sequence data.")
        print(f"  Auto-selecting 'sequence' loader.")
        config["loader_type"] = "sequence"

        # Ask for sequence length
        seq_len = input("  Sequence length [32]: ").strip()
        config["seq_len"] = int(seq_len) if seq_len else 32
    else:
        idx = select_from_list(loader_types, "Select DataLoader Type:", default=0)
        config["loader_type"] = loader_types[idx]
        config["seq_len"] = 1  # Not used for non-sequence models

    # 4. Training parameters
    print("\nTraining Parameters (press Enter for defaults):")
    print("-" * 50)

    epochs = input("  Epochs [200]: ").strip()
    config["epochs"] = int(epochs) if epochs else 200

    batch_size = input("  Batch size [64]: ").strip()
    config["batch_size"] = int(batch_size) if batch_size else 64

    lr = input("  Learning rate [0.001]: ").strip()
    config["learning_rate"] = float(lr) if lr else 0.001

    train_ratio = input("  Train ratio [0.7]: ").strip()
    config["train_ratio"] = float(train_ratio) if train_ratio else 0.7

    seed = input("  Random seed [42]: ").strip()
    config["seed"] = int(seed) if seed else 42

    return config


def generate_filename(model_name: str, loader_type: str) -> str:
    """Generate filename with model, loader, and timestamp.

    Format: {model}_{loader}_{YYYYMMDD_HHMMSS}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{loader_type}_{timestamp}"


def train_model(config: dict, project_root: Path):
    """Train model with given configuration."""
    set_seed(config["seed"])
    device = get_device()

    # Check if model requires sequence
    requires_seq = ModelRegistry.requires_sequence(config["model_name"])

    # Auto-adjust loader for sequence models
    if requires_seq and config["loader_type"] != "sequence":
        print(f"\nNote: {config['model_name']} requires sequence data.")
        print(f"      Changing loader from '{config['loader_type']}' to 'sequence'")
        config["loader_type"] = "sequence"

    print("\n" + "=" * 65)
    print("Training Configuration")
    print("=" * 65)
    for k, v in config.items():
        print(f"  {k:20s}: {v}")
    print(f"  {'device':20s}: {device}")
    print(f"  {'requires_sequence':20s}: {requires_seq}")
    print("=" * 65)

    # ========== 1. Create DataLoaders ==========
    print("\n[1/4] Creating DataLoaders...")
    factory = DataLoaderFactory(config["data_file"])

    data_info = factory.get_info()
    print(f"  Loaded {data_info['num_samples']} samples")
    print(f"  Input shape: {data_info['input_shape']}")
    print(f"  Output shape: {data_info['output_shape']}")

    # Get sequence length for sequence models
    seq_len = config.get("seq_len", 32) if requires_seq else 1

    train_loader, val_loader, loader_info = factory.create_dataloaders(
        loader_type=config["loader_type"],
        train_ratio=config["train_ratio"],
        batch_size=config["batch_size"],
        shuffle_train=True,
        normalize=True,
        window_size=seq_len,
        seed=config["seed"],
    )

    print(f"  Train samples: {loader_info['train_samples']}")
    print(f"  Val samples: {loader_info['val_samples']}")
    print(f"  Shuffle: {loader_info['shuffle_train']}")
    if requires_seq:
        print(f"  Sequence length: {seq_len}")

    # ========== 2. Create Model ==========
    print("\n[2/4] Creating model...")
    model = ModelRegistry.create_model(config["model_name"])
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {config['model_name']}")
    print(f"  Parameters: {total_params:,}")

    if hasattr(model, 'get_config'):
        model_config = model.get_config()
        for k, v in model_config.items():
            if k != "model":
                print(f"    {k}: {v}")

    # ========== 3. Train ==========
    print("\n[3/4] Training...")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    history = {"train_loss": [], "val_loss": []}
    best_state = None

    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # Progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{config['epochs']} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n  Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  Best val loss: {best_val_loss:.6f}")

    # ========== 4. Evaluate & Save ==========
    print("\n[4/4] Evaluating and saving...")

    # Collect all validation predictions
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            all_preds.append(y_pred)
            all_targets.append(y_batch.numpy())

    y_val_pred_norm = np.concatenate(all_preds, axis=0)
    y_val_true_norm = np.concatenate(all_targets, axis=0)

    # Denormalize
    y_val_pred = factory.denormalize_y(y_val_pred_norm)
    y_val_true = factory.denormalize_y(y_val_true_norm)

    # Calculate residuals
    residuals = y_val_true - y_val_pred

    print("\n  Compensation Results (Validation Set):")
    print("  " + "-" * 50)
    print("  Sensor | Std Before | Std After | Reduction (%)")
    print("  " + "-" * 50)

    for i in range(min(4, y_val_true.shape[1])):
        std_before = np.std(y_val_true[:, i])
        std_after = np.std(residuals[:, i])
        reduction = (1 - std_after / std_before) * 100 if std_before > 0 else 0
        print(f"  raw{i+1}   | {std_before:10.2f} | {std_after:9.2f} | {reduction:6.1f}%")

    # Generate filename with model, loader, timestamp
    base_filename = generate_filename(config["model_name"], config["loader_type"])

    # Save model
    output_dir = project_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_filename = f"{base_filename}.pth"
    model_path = output_dir / model_filename

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "model_config": model.get_config() if hasattr(model, 'get_config') else {},
        "norm_params": factory.get_norm_params(),
        "history": history,
        "best_val_loss": best_val_loss,
        "seq_len": seq_len if requires_seq else None,
    }
    torch.save(checkpoint, model_path)
    print(f"\n  Model saved to: {model_path}")

    # Save normalization params separately for realtime use
    norm_filename = f"norm_{base_filename}.pt"
    norm_path = output_dir / norm_filename
    norm_params = factory.get_norm_params()
    torch.save({
        "stage1_means": torch.tensor(norm_params["y_mean"]),
        "stage1_stds": torch.tensor(norm_params["y_std"]),
        "input_means": torch.tensor(norm_params["X_mean"]),
        "input_stds": torch.tensor(norm_params["X_std"]),
    }, norm_path)
    print(f"  Normalization params saved to: {norm_path}")

    # Also save as default normalization_params.pt for realtime use
    default_norm_path = output_dir / "normalization_params.pt"
    torch.save({
        "stage1_means": torch.tensor(norm_params["y_mean"]),
        "stage1_stds": torch.tensor(norm_params["y_std"]),
        "input_means": torch.tensor(norm_params["X_mean"]),
        "input_stds": torch.tensor(norm_params["X_std"]),
    }, default_norm_path)

    # ========== Summary ==========
    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)
    print(f"Model: {config['model_name']}")
    print(f"DataLoader: {config['loader_type']}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Output files:")
    print(f"  - {model_filename}")
    print(f"  - {norm_filename}")
    print(f"Output directory: {output_dir}/")
    print("=" * 65)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Self Detection model")
    parser.add_argument("--auto", action="store_true",
                        help="Use default settings without prompts")
    parser.add_argument("--model", type=str, default="stage1_mlp",
                        choices=ModelRegistry.get_available_models(),
                        help="Model to use")
    parser.add_argument("--loader", type=str, default="standard",
                        choices=list(DataLoaderFactory.AVAILABLE_LOADERS.keys()),
                        help="DataLoader type")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data file")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=32,
                        help="Sequence length for TCN and other sequence models")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    project_root = Path(__file__).parent.parent

    print_banner()

    if args.auto:
        # Auto mode with command line args
        data_file = args.data
        if data_file is None:
            data_files = find_data_files(project_root)
            if data_files:
                data_file = str(data_files[0])
            else:
                print("Error: No data file found. Use --data to specify.")
                sys.exit(1)

        config = {
            "data_file": data_file,
            "model_name": args.model,
            "loader_type": args.loader,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "seq_len": args.seq_len,
        }
    else:
        # Interactive mode
        config = get_user_config(project_root)

    train_model(config, project_root)


if __name__ == "__main__":
    main()
