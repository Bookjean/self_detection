#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 Self-Detection Field MLP Training Script

[CONCEPT - SINGLE LINEAGE]
- Input  : 6 joint angles (q)        ※ no normalization
- Output : N proximity sensors       ※ normalized by (raw - BASE) / SCALE
- Model  : Stage1SelfDetectionFieldMLP (256-256-128)

The model learns:
    ŝ_norm(q) ≈ (s_raw(q) - BASE) / SCALE

Compensation (runtime):
    y_comp = y_raw - (ŝ_raw - BASE)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# project import
sys.path.insert(0, str(Path(__file__).parent.parent))

from self_detection_mlp.model import (
    Stage1SelfDetectionFieldMLP,
    OutputNormSpec,
)
from self_detection_mlp.data_loader import DataLoaderFactory


# ============================================================
# Constants
# ============================================================
N_JOINTS = 6
BASE_VALUE = 4e7
SCALE_VALUE = 1e6


# ============================================================
# Utilities
# ============================================================
def print_banner():
    print("\n" + "=" * 70)
    print(" Stage 1 Self-Detection Field MLP Training")
    print(" Input : 6 joint angles (NO normalization)")
    print(" Output: N sensors, normalized by (raw - BASE) / SCALE")
    print(" Model : 256-256-128 MLP")
    print("=" * 70)


# ============================================================
# Training
# ============================================================
def train_stage1(config: dict, project_root: Path):

    # -------------------------
    # Reproducibility
    # -------------------------
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[CONFIGURATION]")
    for k, v in config.items():
        print(f"  {k:18s}: {v}")
    print(f"  {'device':18s}: {device}")

    # -------------------------
    # Load data
    # -------------------------
    print("\n[1/4] Loading data...")
    factory = DataLoaderFactory(
        config["data_file"],
        num_sensors=config["num_sensors"],
    )

    data_info = factory.get_info()
    output_dim = data_info["output_shape"][1]

    print(f"  Samples      : {data_info['num_samples']}")
    print(f"  Input shape  : {data_info['input_shape']}")
    print(f"  Output shape : {data_info['output_shape']}")

    # output normalization ONLY (baseline-based)
    norm = OutputNormSpec(
        baseline=BASE_VALUE,
        scale=SCALE_VALUE,
    )

    train_loader, val_loader, loader_info = factory.create_dataloaders(
        loader_type="standard",
        train_ratio=config["train_ratio"],
        batch_size=config["batch_size"],
        shuffle_train=True,
        normalize=False,   # 🔴 DataLoader의 정규화 사용 안 함
        seed=config["seed"],
    )

    print(f"  Train samples: {loader_info['train_samples']}")
    print(f"  Val samples  : {loader_info['val_samples']}")

    # -------------------------
    # Model
    # -------------------------
    print("\n[2/4] Creating model...")

    model = Stage1SelfDetectionFieldMLP(
        input_dim=N_JOINTS,
        output_dim=output_dim,
        hidden_dims=(256, 256, 128),
        activation="relu",
        dropout=0.0,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture : 6 -> 256 -> 256 -> 128 -> {output_dim}")
    print(f"  Parameters  : {num_params:,}")

    # -------------------------
    # Optimization (train_self_detection.py 스타일)
    # -------------------------
    print("\n[3/4] Training...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience = config.get("patience", 30)
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse_raw": [],
        "learning_rate": [],
    }

    for epoch in range(config["epochs"]):
        # -------- Train --------
        model.train()
        train_losses = []

        for X, y_raw in train_loader:
            X = X[:, :N_JOINTS].to(device)     # joint angles only
            y_raw = y_raw.to(device)

            # baseline-based normalization
            y_norm = norm.normalize(y_raw)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y_norm)
            loss.backward()
            
            # Gradient clipping (train_self_detection.py 스타일)
            grad_clip = config.get("grad_clip", 1.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            optimizer.step()

            train_losses.append(loss.item())

        # -------- Validate --------
        model.eval()
        val_losses = []
        rmse_list = []
        with torch.no_grad():
            for X, y_raw in val_loader:
                X = X[:, :N_JOINTS].to(device)
                y_raw = y_raw.to(device)
                y_norm = norm.normalize(y_raw)

                y_pred = model(X)
                loss = criterion(y_pred, y_norm)
                val_losses.append(loss.item())
                
                # RMSE on raw scale (more interpretable)
                y_pred_raw = norm.denormalize(y_pred)
                rmse = torch.sqrt(torch.mean((y_pred_raw - y_raw) ** 2)).item()
                rmse_list.append(rmse)

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_rmse_raw = float(np.mean(rmse_list)) if rmse_list else float("inf")

        # Scheduler step (validation loss 기준)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse_raw"].append(val_rmse_raw)
        history["learning_rate"].append(float(new_lr))

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        # Print progress (train_self_detection.py 스타일)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{config['epochs']} | "
                f"Train {train_loss:.6e} | Val {val_loss:.6e} | "
                f"ValRMSE(raw) {val_rmse_raw:.1f} | LR {new_lr:.2e}"
            )

        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest validation loss: {best_val:.6e}")

    # -------------------------
    # Save model
    # -------------------------
    print("\n[4/4] Saving model...")

    out_dir = project_root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stage1_self_detection_{timestamp}.pth"
    path = out_dir / filename

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "output_norm": {
                "baseline": BASE_VALUE,
                "scale": SCALE_VALUE,
            },
            "training_config": config,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "history": history,
        },
        path,
    )

    print(f"  Saved model: {path}")
    print(f"  Best epoch: {best_epoch+1}, best val_loss: {best_val:.6e}")
    print("\nTRAINING COMPLETE")

    return model, best_val


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage1 Self-Detection Field MLP Training"
    )
    parser.add_argument("--data", type=str, default=None, help="Data file path. If omitted, uses latest robot_data_*.txt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 regularization")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-sensors", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    return parser.parse_args()

def _find_latest_robot_data(project_root: Path) -> str | None:
    candidates = sorted(project_root.glob("robot_data_*.txt"), reverse=True)
    if not candidates:
        return None
    return str(candidates[0])


def main():
    args = parse_args()
    project_root = Path(__file__).parent.parent

    print_banner()

    data_path = args.data
    if not data_path:
        data_path = _find_latest_robot_data(project_root)
        if not data_path:
            print("[ERROR] --data not provided and no robot_data_*.txt found in project root.")
            sys.exit(1)
        print(f"[INFO] Using latest data file: {Path(data_path).name}")

    config = {
        "data_file": data_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "num_sensors": args.num_sensors,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
    }

    train_stage1(config, project_root)


if __name__ == "__main__":
    main()
