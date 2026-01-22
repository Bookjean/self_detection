"""Two-Stage Training Script for Self Detection.

Workflow:
    1. Train Stage 1: joint angles (6) -> raw sensors (4)
    2. Compute residuals: residual = raw - stage1_prediction
    3. Train Stage 2: residual sequence (K x 4) -> residual correction (4)

Output files:
    - stage1_{model}_{timestamp}.pth
    - stage2_{model}_{timestamp}.pth

Usage:
    python train_two_stage.py                     # Interactive mode
    python train_two_stage.py --auto              # Use defaults
    python train_two_stage.py --stage1 stage1_mlp --stage2 stage2_tcn
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from self_detection_mlp.data_loader import DataLoaderFactory
from self_detection_mlp.model import ModelRegistry
from self_detection_mlp.utils import set_seed, get_device


def print_banner():
    print()
    print("=" * 65)
    print("   Two-Stage Self Detection Training")
    print("   Stage 1: Joint angles -> Raw sensors (baseline)")
    print("   Stage 2: Residual sequence -> Residual correction")
    print("=" * 65)


def find_data_files(project_root: Path) -> list:
    data_files = []
    for pattern in ["*.txt", "*.csv", "data/*.txt", "data/*.csv"]:
        data_files.extend(project_root.glob(pattern))
    return sorted(set(data_files))


def select_from_list(options: list, prompt: str, default: int = 0) -> int:
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


def get_stage1_models() -> list:
    """Get models suitable for Stage 1 (joint input)."""
    return ["stage1_mlp", "simple_mlp", "deep_mlp"]


def get_stage2_models() -> list:
    """Get models suitable for Stage 2 (residual input)."""
    return ["stage2_tcn", "stage2_memory"]


def get_user_config(project_root: Path) -> dict:
    config = {}

    # 1. Select data file
    data_files = find_data_files(project_root)
    if not data_files:
        print("Error: No data files found.")
        sys.exit(1)

    data_options = [str(f.relative_to(project_root)) for f in data_files]
    idx = select_from_list(data_options, "Select Data File:", default=0)
    config["data_file"] = str(data_files[idx])

    # 2. Select Stage 1 model
    stage1_models = get_stage1_models()
    idx = select_from_list(stage1_models, "Select Stage 1 Model:", default=0)
    config["stage1_model"] = stage1_models[idx]

    # 3. Select Stage 2 model
    stage2_models = get_stage2_models()
    idx = select_from_list(stage2_models, "Select Stage 2 Model:", default=0)
    config["stage2_model"] = stage2_models[idx]

    # 4. Stage 2 sequence length (memory window K)
    print("\nStage 2 Parameters:")
    print("-" * 50)
    seq_len = input("  Memory window K (sequence length) [10]: ").strip()
    config["seq_len"] = int(seq_len) if seq_len else 10

    # 5. Training parameters
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


def create_residual_sequences(residuals: np.ndarray, seq_len: int) -> tuple:
    """Create sequences from residual data.

    Args:
        residuals: Shape (N, 4)
        seq_len: Memory window K

    Returns:
        X: Residual sequences, shape (N-seq_len, seq_len, 4)
        y: Target residuals, shape (N-seq_len, 4)
    """
    N = len(residuals)
    X_list = []
    y_list = []

    for i in range(seq_len, N):
        # Input: past K residuals r(t-K), ..., r(t-1)
        X_list.append(residuals[i-seq_len:i])
        # Target: current residual r(t)
        y_list.append(residuals[i])

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


def train_stage1(config: dict, factory: DataLoaderFactory, device: torch.device) -> tuple:
    """Train Stage 1 model.

    Returns:
        model: Trained Stage 1 model
        norm_params: Normalization parameters
    """
    print("\n" + "=" * 65)
    print("STAGE 1: Training baseline model")
    print("=" * 65)

    # Create dataloaders (standard, not sequence)
    train_loader, val_loader, loader_info = factory.create_dataloaders(
        loader_type="standard",
        train_ratio=config["train_ratio"],
        batch_size=config["batch_size"],
        shuffle_train=True,
        normalize=True,
        seed=config["seed"],
    )

    print(f"  Train samples: {loader_info['train_samples']}")
    print(f"  Val samples: {loader_info['val_samples']}")

    # Create model
    model = ModelRegistry.create_model(config["stage1_model"])
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {config['stage1_model']}")
    print(f"  Parameters: {total_params:,}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    best_state = None

    for epoch in range(config["epochs"]):
        # Train
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

        # Validate
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

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{config['epochs']} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n  Best validation loss: {best_val_loss:.6f}")

    return model, factory.get_norm_params()


def compute_residuals(model: nn.Module, X: np.ndarray, y: np.ndarray,
                      norm_params: dict, device: torch.device) -> np.ndarray:
    """Compute residuals in ORIGINAL SCALE.

    Formula: residual = raw_original - stage1_prediction_original

    Args:
        model: Trained Stage 1 model
        X: Joint angles (N, 6) - ORIGINAL SCALE
        y: Raw sensors (N, 4) - ORIGINAL SCALE
        norm_params: Normalization parameters
        device: Torch device

    Returns:
        residuals: Shape (N, 4) - ORIGINAL SCALE
    """
    model.eval()

    # Normalize input for model (model expects normalized input)
    X_norm = (X - norm_params['X_mean']) / (norm_params['X_std'] + 1e-8)

    # Predict (model outputs normalized values)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_norm = model(X_tensor).cpu().numpy()

    # Denormalize prediction to original scale
    y_pred_original = y_pred_norm * norm_params['y_std'] + norm_params['y_mean']

    # Compute residuals in original scale
    # residual = raw_original - predicted_original
    residuals = y - y_pred_original

    return residuals


def train_stage2(config: dict, residuals: np.ndarray, device: torch.device) -> tuple:
    """Train Stage 2 model on residual sequences.

    Args:
        config: Training configuration
        residuals: Residual data, shape (N, 4)
        device: Torch device

    Returns:
        model: Trained Stage 2 model
        residual_norm_params: Normalization parameters for residuals
    """
    print("\n" + "=" * 65)
    print("STAGE 2: Training residual correction model")
    print("=" * 65)

    seq_len = config["seq_len"]

    # Create sequences
    X_seq, y_seq = create_residual_sequences(residuals, seq_len)
    print(f"  Created {len(X_seq)} sequences (seq_len={seq_len})")

    # Normalize residuals
    residual_mean = np.mean(residuals, axis=0)
    residual_std = np.std(residuals, axis=0) + 1e-8

    X_seq_norm = (X_seq - residual_mean) / residual_std
    y_seq_norm = (y_seq - residual_mean) / residual_std

    residual_norm_params = {
        'mean': residual_mean,
        'std': residual_std,
    }

    # Split train/val (temporal order preserved, NO shuffle)
    n_samples = len(X_seq)
    n_train = int(n_samples * config["train_ratio"])

    # Temporal split: train = first portion, val = later portion
    X_train = torch.tensor(X_seq_norm[:n_train], dtype=torch.float32)
    y_train = torch.tensor(y_seq_norm[:n_train], dtype=torch.float32)
    X_val = torch.tensor(X_seq_norm[n_train:], dtype=torch.float32)
    y_val = torch.tensor(y_seq_norm[n_train:], dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    print(f"  Train samples: {n_train} (temporal: 0 ~ {n_train-1})")
    print(f"  Val samples: {n_samples - n_train} (temporal: {n_train} ~ {n_samples-1})")

    # Create Stage 2 model
    model = ModelRegistry.create_model(
        config["stage2_model"],
        seq_len=seq_len,
        input_dim=4,
        output_dim=4,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {config['stage2_model']}")
    print(f"  Parameters: {total_params:,}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30
    best_state = None

    for epoch in range(config["epochs"]):
        # Train
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

        # Validate
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

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{config['epochs']} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n  Best validation loss: {best_val_loss:.6f}")

    return model, residual_norm_params, best_val_loss


def train_two_stage(config: dict, project_root: Path):
    """Run two-stage training."""
    set_seed(config["seed"])
    device = get_device()

    print("\n" + "=" * 65)
    print("Configuration")
    print("=" * 65)
    for k, v in config.items():
        print(f"  {k:20s}: {v}")
    print(f"  {'device':20s}: {device}")
    print("=" * 65)

    # Load data
    print("\n[1/5] Loading data...")
    factory = DataLoaderFactory(config["data_file"])
    data_info = factory.get_info()
    print(f"  Loaded {data_info['num_samples']} samples")

    # Get raw data for residual computation later
    X_all, y_all = factory.get_raw_data()

    # ========== Stage 1 ==========
    print("\n[2/5] Training Stage 1...")
    stage1_model, norm_params = train_stage1(config, factory, device)

    # ========== Compute Residuals ==========
    print("\n[3/5] Computing residuals (in original scale)...")

    # Compute residuals: residual = raw_original - stage1_pred_original
    # Both X_all and y_all are in ORIGINAL SCALE
    residuals = compute_residuals(stage1_model, X_all, y_all, norm_params, device)

    print(f"  Residual shape: {residuals.shape}")
    print(f"  Residual stats:")
    for i in range(4):
        print(f"    ch{i+1}: mean={residuals[:, i].mean():.2f}, std={residuals[:, i].std():.2f}")

    # ========== Stage 2 ==========
    print("\n[4/5] Training Stage 2...")
    stage2_model, residual_norm_params, stage2_val_loss = train_stage2(
        config, residuals, device
    )

    # ========== Save Models ==========
    print("\n[5/5] Saving models...")

    output_dir = project_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Stage 1
    stage1_filename = f"stage1_{config['stage1_model']}_{timestamp}.pth"
    stage1_path = output_dir / stage1_filename
    torch.save({
        "model_state_dict": stage1_model.state_dict(),
        "config": config,
        "model_config": stage1_model.get_config() if hasattr(stage1_model, 'get_config') else {},
        "norm_params": norm_params,
        "stage": 1,
    }, stage1_path)
    print(f"  Stage 1 saved: {stage1_filename}")

    # Save Stage 2
    stage2_filename = f"stage2_{config['stage2_model']}_{timestamp}.pth"
    stage2_path = output_dir / stage2_filename
    torch.save({
        "model_state_dict": stage2_model.state_dict(),
        "config": config,
        "model_config": stage2_model.get_config() if hasattr(stage2_model, 'get_config') else {},
        "norm_params": norm_params,  # Stage 1 norm params for reference
        "residual_norm_params": residual_norm_params,  # Stage 2 specific
        "seq_len": config["seq_len"],
        "stage": 2,
        "best_val_loss": stage2_val_loss,
    }, stage2_path)
    print(f"  Stage 2 saved: {stage2_filename}")

    # Save combined config for realtime use
    combined_filename = f"two_stage_config_{timestamp}.pt"
    combined_path = output_dir / combined_filename
    torch.save({
        "stage1_model_file": stage1_filename,
        "stage2_model_file": stage2_filename,
        "stage1_model_name": config["stage1_model"],
        "stage2_model_name": config["stage2_model"],
        "norm_params": norm_params,
        "residual_norm_params": residual_norm_params,
        "seq_len": config["seq_len"],
    }, combined_path)
    print(f"  Combined config saved: {combined_filename}")

    # ========== Summary ==========
    print("\n" + "=" * 65)
    print("TWO-STAGE TRAINING COMPLETE")
    print("=" * 65)
    print(f"Stage 1: {config['stage1_model']}")
    print(f"Stage 2: {config['stage2_model']} (seq_len={config['seq_len']})")
    print(f"\nOutput files:")
    print(f"  - {stage1_filename}")
    print(f"  - {stage2_filename}")
    print(f"  - {combined_filename}")
    print(f"\nOutput directory: {output_dir}/")
    print("=" * 65)
    print("\nTo use in realtime:")
    print(f"  1. Load Stage 1: {stage1_filename}")
    print(f"  2. Load Stage 2: {stage2_filename}")
    print("  3. Compensation: y_corrected = raw - stage1(joints) - stage2(residual_seq)")
    print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser(description="Two-Stage Self Detection Training")
    parser.add_argument("--auto", action="store_true", help="Use defaults")
    parser.add_argument("--stage1", type=str, default="stage1_mlp",
                        choices=get_stage1_models(), help="Stage 1 model")
    parser.add_argument("--stage2", type=str, default="stage2_tcn",
                        choices=get_stage2_models(), help="Stage 2 model")
    parser.add_argument("--data", type=str, default=None, help="Data file path")
    parser.add_argument("--seq-len", type=int, default=10, help="Stage 2 sequence length (K)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).parent.parent

    print_banner()

    if args.auto:
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
            "stage1_model": args.stage1,
            "stage2_model": args.stage2,
            "seq_len": args.seq_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
        }
    else:
        config = get_user_config(project_root)

    train_two_stage(config, project_root)


if __name__ == "__main__":
    main()
