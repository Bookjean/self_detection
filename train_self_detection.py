#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Detection Compensation Training Script

논문 기반: "Real-Time Obstacle Avoidance Using Dual-Type Proximity Sensor 
           for Safe Human-Robot Interaction"

Section IV. SELF-SENSING COMPENSATION:
  "To perform machine learning, it is important to build training data.
   After attaching a sensor to an actual robot, we collected sensor data 
   for each joint while the robot moved randomly."
  
  "The output is the data of the seven sensors currently attached. 
   A total of 100,000 training data were employed, three layers and 
   10 nodes for each layer were used."
   
  "Before compensation, the change in sensor data is up to 50,000.
   After applying the compensation, the change is up to 5,000."

핵심 포인트 (논문):
  1. ToF는 학습에 사용하지 않음 (논문: "tof를 사용안해")
  2. 장애물 없는 환경에서 수집된 모든 데이터 사용
  3. 관절 상태 → 센서 값 예측 (모든 변화 = self-detection)
  4. 실시간: measured - predicted = 실제 장애물 신호

센서 구성 (사용자 시스템 - 16채널):
  - Forearm (A1): prox_11 ~ prox_81 (8채널) - 링크 절반씩 4개 원형 배치
  - Lowerarm (A2): prox_12 ~ prox_82 (8채널) - 링크 절반씩 4개 원형 배치

입력: 관절 위치 (6) + 관절 속도 (6) = 12차원
출력: 16채널 proximity 센서 값 예측

사용법:
  python train_self_detection.py --data_dir /path/to/csv --epochs 100

저자: GitHub Copilot
날짜: 2026-01-26
"""

import os
import glob
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# =============================================================================
# Robot-data Stage1 (6 joints -> 8 sensors) pipeline (keeps your data format/I-O)
# =============================================================================
from pathlib import Path

try:
    # your project data loader + model definitions
    from self_detection_mlp.data_loader import DataLoaderFactory
    from self_detection_mlp.model import Stage1SelfDetectionFieldMLP, OutputNormSpec
    _ROBOT_PIPE_AVAILABLE = True
except Exception:
    _ROBOT_PIPE_AVAILABLE = False


# ============================================================
# 설정 및 상수
# ============================================================
BASE_VALUE = 4e7  # Proximity 센서 baseline (장애물 없을 때 값)

# 컬럼 이름 정의
JOINT_POS_COLS = [f"joint_pos_{i}" for i in range(6)]
JOINT_VEL_COLS = [f"joint_vel_{i}" for i in range(6)]
INPUT_COLS = JOINT_POS_COLS + JOINT_VEL_COLS  # 12차원 입력

# Proximity 센서 컬럼 (16채널)
# 센서 명명: prox_XY where X=위치(1-8), Y=방향(1 or 2)
#
# Lowerarm (8개): 51, 52, 61, 62, 71, 72, 81, 82
#   - 링크 절반씩 4개 위치, 각 위치에 2개 센서 (원형 배치)
PROX_LOWERARM_COLS = ['prox_51', 'prox_52', 'prox_61', 'prox_62',
                       'prox_71', 'prox_72', 'prox_81', 'prox_82']
#
# Forearm (8개): 11, 12, 21, 22, 31, 32, 41, 42  
#   - 링크 절반씩 4개 위치, 각 위치에 2개 센서 (원형 배치)
PROX_FOREARM_COLS = ['prox_11', 'prox_12', 'prox_21', 'prox_22',
                      'prox_31', 'prox_32', 'prox_41', 'prox_42']
#
PROX_COLS = PROX_LOWERARM_COLS + PROX_FOREARM_COLS  # 16채널 (Lowerarm 먼저, Forearm 다음)


# =============================================================================
# New: Robot-data (robot_data_*.txt) training utilities
# =============================================================================

def _find_latest_robot_data(project_root: Path) -> str | None:
    candidates = sorted(project_root.glob("robot_data_*.txt"), reverse=True)
    if not candidates:
        return None
    return str(candidates[0])


def train_robot_stage1(args) -> None:
    """Train Stage1 MLP on robot_data_*.txt (Input: 6 joints, Output: N raw sensors).

    Keeps:
    - data loading: robot_data_*.txt via DataLoaderFactory
    - input/output dims: 6 -> num_sensors(8)
    - checkpoint location: ./models
    """
    if not _ROBOT_PIPE_AVAILABLE:
        raise RuntimeError("Robot-data pipeline imports failed. Check self_detection_mlp package.")

    project_root = Path(__file__).parent
    data_file = args.data_file or _find_latest_robot_data(project_root)
    if not data_file:
        raise FileNotFoundError("No robot_data_*.txt found and --data_file not provided.")

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "=" * 60)
    print("Robot Stage1 Training (6 joints -> sensors)")
    print("=" * 60)
    print(f"Data file: {data_file}")
    print(f"Num sensors: {args.num_sensors}")
    print(f"Epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}, wd: {args.weight_decay}")
    print(f"Train ratio: {args.train_ratio}")
    print("=" * 60)

    # Data
    factory = DataLoaderFactory(data_file, num_sensors=args.num_sensors)
    info = factory.get_info()
    output_dim = int(info["output_shape"][1])

    # We keep your "data loading mechanism" and do output normalization here (baseline/scale)
    train_loader, val_loader, loader_info = factory.create_dataloaders(
        loader_type="standard",
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        shuffle_train=True,
        normalize=False,  # do not use DataLoaderFactory normalization here
        seed=args.seed,
    )

    # Output normalization spec (paper-style)
    norm = OutputNormSpec(baseline=float(args.baseline), scale=float(args.scale))

    # Model
    model = Stage1SelfDetectionFieldMLP(
        input_dim=6,
        output_dim=output_dim,
        hidden_dims=(256, 256, 128),
        activation="relu",
        dropout=float(args.dropout),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_rmse_raw": [], "learning_rate": []}
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0

    for epoch in range(int(args.epochs)):
        model.train()
        train_losses = []

        for X, y_raw in train_loader:
            X = X[:, :6].to(device)  # joint angles only
            y_raw = y_raw.to(device)
            y_norm = norm.normalize(y_raw)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")

        # Validation
        model.eval()
        val_losses = []
        rmse_list = []
        with torch.no_grad():
            for X, y_raw in val_loader:
                X = X[:, :6].to(device)
                y_raw = y_raw.to(device)
                y_norm = norm.normalize(y_raw)
                y_pred = model(X)
                vloss = criterion(y_pred, y_norm).item()
                val_losses.append(vloss)

                # RMSE on raw scale (more interpretable)
                y_pred_raw = norm.denormalize(y_pred)
                rmse = torch.sqrt(torch.mean((y_pred_raw - y_raw) ** 2)).item()
                rmse_list.append(rmse)

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        val_rmse_raw = float(np.mean(rmse_list)) if rmse_list else float("inf")

        # Scheduler uses validation loss (normalized space)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse_raw"].append(val_rmse_raw)
        history["learning_rate"].append(float(new_lr))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Train {train_loss:.6e} | Val {val_loss:.6e} | "
                f"ValRMSE(raw) {val_rmse_raw:.1f} | LR {new_lr:.2e}"
            )

        # Early stopping on val_loss
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(args.patience):
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = project_root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stage1_mlp_{ts}.pth"
    path = out_dir / filename

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "output_norm": {"baseline": float(args.baseline), "scale": float(args.scale)},
            "training_config": {
                "data_file": str(data_file),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "train_ratio": float(args.train_ratio),
                "seed": int(args.seed),
                "num_sensors": int(args.num_sensors),
                "dropout": float(args.dropout),
                "grad_clip": float(args.grad_clip),
                "patience": int(args.patience),
            },
            "best_val_loss": float(best_val),
            "best_epoch": int(best_epoch),
            "history": history,
        },
        path,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE (robot stage1)")
    print("=" * 60)
    print(f"Saved: {path}")
    print(f"Best epoch: {best_epoch+1}, best val_loss: {best_val:.6e}")
    print("=" * 60)


# ============================================================
# 데이터 전처리 함수
# ============================================================
def load_and_preprocess_csv(file_path: str) -> pd.DataFrame:
    """
    CSV 파일 로드 및 전처리
    
    논문 방식: ToF 없이, 장애물 없는 환경에서 수집된 모든 데이터 사용
    """
    df = pd.read_csv(file_path)
    
    # 필요한 컬럼 확인
    missing_cols = []
    for col in INPUT_COLS + PROX_COLS:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"Warning: Missing columns in {file_path}: {missing_cols[:5]}...")
        return None
    
    return df


def load_all_data(data_dir: str) -> pd.DataFrame:
    """
    모든 Self_Detection CSV 파일 로드
    
    논문: "we collected sensor data for each joint while the robot moved randomly"
    """
    pattern = os.path.join(data_dir, "*Self_Detection*.csv")
    files = sorted(glob.glob(pattern))
    
    # 하위 디렉토리도 검색
    pattern_sub = os.path.join(data_dir, "**/*Self_Detection*.csv")
    files_sub = sorted(glob.glob(pattern_sub, recursive=True))
    files = list(set(files + files_sub))
    files = sorted(files)
    
    if len(files) == 0:
        raise RuntimeError(f"No CSV files found matching: {pattern}")
    
    print(f"Found {len(files)} CSV files")
    
    dfs = []
    for f in files:
        df = load_and_preprocess_csv(f)
        if df is not None:
            df['source_file'] = os.path.basename(f)
            dfs.append(df)
            print(f"  Loaded: {os.path.basename(f)} ({len(df)} samples)")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal samples: {len(combined)}")
    
    return combined


# ============================================================
# PyTorch Dataset
# ============================================================
class SelfDetectionDataset(Dataset):
    """
    Self-Detection 학습용 Dataset
    
    논문 방식:
      - 입력: 관절 상태 (위치 + 속도)
      - 출력: proximity 센서 값 (self-detection 포함)
      - 장애물 없는 환경에서 수집된 모든 데이터 사용 (ToF gating 없음)
    """
    
    def __init__(self, df: pd.DataFrame, normalize_output: bool = True):
        # 입력 데이터: 관절 위치 + 속도 (12차원)
        self.X = df[INPUT_COLS].to_numpy(dtype=np.float32)
        
        # 출력 데이터: proximity 센서 값 (16채널)
        self.Y_raw = df[PROX_COLS].to_numpy(dtype=np.float32)
        
        # 입력 정규화
        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0) + 1e-8
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        
        # 출력 정규화 (센서 값이 4e7 근처이므로 정규화 필요)
        self.normalize_output = normalize_output
        if normalize_output:
            # baseline 기준으로 정규화: (값 - baseline) / scale
            # self-detection으로 인한 변화량을 학습
            self.Y_baseline = BASE_VALUE
            self.Y_scale = 1e6  # 스케일 조정 (변화량이 수천~수만 단위)
            self.Y = (self.Y_raw - self.Y_baseline) / self.Y_scale
        else:
            self.Y = self.Y_raw
            self.Y_baseline = 0
            self.Y_scale = 1
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_normalized[idx]),
            torch.from_numpy(self.Y[idx].astype(np.float32))
        )
    
    def get_normalization_params(self):
        """모델 저장 시 정규화 파라미터도 저장"""
        return {
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'Y_baseline': self.Y_baseline,
            'Y_scale': self.Y_scale
        }


# ============================================================
# MLP 모델 (논문 구조 기반)
# ============================================================
class SelfDetectionMLP(nn.Module):
    """
    Self-Detection 보정용 MLP
    
    논문: "three layers and 10 nodes for each layer were used"
    
    하지만 우리는 16채널 출력이 필요하므로 더 큰 네트워크 사용
    """
    
    def __init__(self, in_dim=12, hidden_dims=(256, 256, 128), out_dim=16, dropout=0.1):
        super().__init__()
        
        # 히든 레이어 구성
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(d, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            d = h
        
        # 출력 레이어
        layers.append(nn.Linear(d, out_dim))
        
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim
    
    def forward(self, x):
        """
        Args:
            x: (batch, in_dim) 관절 상태
        Returns:
            y: (batch, out_dim) 예측 센서 값
        """
        return self.net(x)


class SelfDetectionMLPTrunkHead(nn.Module):
    """
    Trunk + Head 구조의 MLP (더 정밀한 채널별 예측)
    
    - Trunk: 공유 특성 추출기
    - Heads: 채널별 독립 예측기 (16개)
    
    이 구조는 채널 간 공통 패턴은 공유하면서
    채널별 특성은 개별 head에서 학습
    """
    
    def __init__(self, in_dim=12, trunk_hidden=(256, 256), trunk_dim=128,
                 head_hidden=64, out_dim=16, dropout=0.1):
        super().__init__()
        
        # Trunk: 공유 representation 학습
        layers = []
        d = in_dim
        for h in trunk_hidden:
            layers.extend([
                nn.Linear(d, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            d = h
        layers.extend([
            nn.Linear(d, trunk_dim),
            nn.ReLU(inplace=True)
        ])
        self.trunk = nn.Sequential(*layers)
        
        # Heads: 채널별 독립 예측기
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(head_hidden, 1)
            ) for _ in range(out_dim)
        ])
        
        self.out_dim = out_dim
    
    def forward(self, x):
        z = self.trunk(x)  # (batch, trunk_dim)
        outputs = [head(z) for head in self.heads]  # list of (batch, 1)
        y = torch.cat(outputs, dim=1)  # (batch, out_dim)
        return y


# ============================================================
# 손실 함수
# ============================================================
def compute_loss(pred, target):
    """
    MSE 손실
    
    논문에서는 단순히 센서 값을 예측하는 regression 문제로 접근
    ToF gating 없이 모든 샘플 사용
    """
    return nn.functional.mse_loss(pred, target)


# ============================================================
# 평가 메트릭
# ============================================================
@torch.no_grad()
def evaluate_metrics(model, dataloader, device, dataset):
    """
    검증 데이터에 대한 평가 메트릭 계산
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        all_preds.append(pred.cpu())
        all_targets.append(Y.cpu())
    
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # 원래 스케일로 복원
    preds_original = preds * dataset.Y_scale + dataset.Y_baseline
    targets_original = targets * dataset.Y_scale + dataset.Y_baseline
    
    # 오차 계산 (원래 스케일)
    errors = np.abs(preds_original - targets_original)
    
    # 메트릭
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)
    
    # Within X 비율 (논문: 보정 후 5000 이내로 감소)
    within_1000 = np.mean(errors <= 1000)
    within_5000 = np.mean(errors <= 5000)
    within_10000 = np.mean(errors <= 10000)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'within_1000': within_1000,
        'within_5000': within_5000,
        'within_10000': within_10000
    }


@torch.no_grad()
def evaluate_per_channel(model, dataloader, device, dataset):
    """
    채널별 평가 메트릭
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        all_preds.append(pred.cpu())
        all_targets.append(Y.cpu())
    
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # 원래 스케일로 복원
    preds_original = preds * dataset.Y_scale + dataset.Y_baseline
    targets_original = targets * dataset.Y_scale + dataset.Y_baseline
    
    results = []
    for ch, col in enumerate(PROX_COLS):
        ch_pred = preds_original[:, ch]
        ch_target = targets_original[:, ch]
        ch_error = np.abs(ch_pred - ch_target)
        
        mae = np.mean(ch_error)
        rmse = np.sqrt(np.mean(ch_error ** 2))
        max_err = np.max(ch_error)
        within_5000 = np.mean(ch_error <= 5000)
        
        results.append({
            'channel': col,
            'mae': mae,
            'rmse': rmse,
            'max_error': max_err,
            'within_5000': within_5000
        })
    
    return results


# ============================================================
# 학습 함수
# ============================================================
def train_model(model, train_loader, val_loader, train_dataset, val_dataset, device, args):
    """
    모델 학습
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_rmse = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_rmse': [], 'val_mae': [], 'val_within_5000': [], 'learning_rate': []}
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer.zero_grad()
            pred = model(X)
            loss = compute_loss(pred, Y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        
        # Validation
        val_metrics = evaluate_metrics(model, val_loader, device, val_dataset)
        
        # 학습률 변화 추적
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['rmse'])
        new_lr = optimizer.param_groups[0]['lr']
        
        # 학습률이 변경되었을 때 알림
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # 현재 학습률 기록
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_within_5000'].append(val_metrics['within_5000'])
        history['learning_rate'].append(current_lr)
        
        # Best model 저장
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_metrics['rmse'],
                'normalization': train_dataset.get_normalization_params(),
            }, os.path.join(args.out_dir, 'best_model.pt'))
        
        # 로그 출력
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val RMSE: {val_metrics['rmse']:.1f} | "
                  f"Val MAE: {val_metrics['mae']:.1f} | "
                  f"W5000: {val_metrics['within_5000']:.3f} | "
                  f"LR: {current_lr:.2e}")
    
    print(f"\nBest model at epoch {best_epoch+1} with Val RMSE: {best_val_rmse:.1f}")
    
    return history


# ============================================================
# ONNX 변환
# ============================================================
def export_to_onnx(model, norm_params, out_path, device):
    """
    PyTorch 모델을 ONNX로 변환
    """
    model.eval()
    
    # 더미 입력
    dummy_input = torch.randn(1, 12).to(device)
    
    # ONNX 변환
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=['joint_state'],
        output_names=['predicted_prox'],
        dynamic_axes={
            'joint_state': {0: 'batch_size'},
            'predicted_prox': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    print(f"ONNX model saved: {out_path}")


# ============================================================
# 시각화
# ============================================================
def plot_training_history(history, out_dir):
    """학습 히스토리 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0,0].plot(history['train_loss'], label='Train Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Validation RMSE
    axes[0,1].plot(history['val_rmse'], label='Val RMSE', color='orange')
    axes[0,1].axhline(y=5000, color='r', linestyle='--', label='Target (5000)')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('RMSE')
    axes[0,1].set_title('Validation RMSE (논문 목표: 5000 이내)')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Within 5000 ratio
    axes[1,0].plot(history['val_within_5000'], label='Within 5000', color='green')
    axes[1,0].axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Ratio')
    axes[1,0].set_title('Within 5000 Ratio')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Learning Rate
    axes[1,1].plot(history['learning_rate'], label='Learning Rate', color='purple')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Learning Rate')
    axes[1,1].set_title('Learning Rate Schedule')
    axes[1,1].set_yscale('log')  # 로그 스케일로 표시
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Training history saved: {os.path.join(out_dir, 'training_history.png')}")


def plot_channel_metrics(channel_results, out_dir):
    """채널별 메트릭 시각화"""
    channels = [r['channel'] for r in channel_results]
    rmse_values = [r['rmse'] for r in channel_results]
    within_5000_values = [r['within_5000'] for r in channel_results]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    x = np.arange(len(channels))
    
    # RMSE per channel
    colors = ['green' if r < 5000 else 'orange' if r < 10000 else 'red' for r in rmse_values]
    axes[0].bar(x, rmse_values, color=colors)
    axes[0].axhline(y=5000, color='r', linestyle='--', label='Target (5000)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channels, rotation=45)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE per Channel (논문: 보정 후 5000 이내)')
    axes[0].legend()
    axes[0].grid(True, axis='y')
    
    # Within 5000 ratio per channel
    axes[1].bar(x, within_5000_values, color='steelblue')
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels, rotation=45)
    axes[1].set_ylabel('Within 5000 Ratio')
    axes[1].set_title('Within 5000 Ratio per Channel')
    axes[1].legend()
    axes[1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'channel_metrics.png'), dpi=150)
    plt.close()
    print(f"Channel metrics saved: {os.path.join(out_dir, 'channel_metrics.png')}")


def plot_prediction_comparison(model, val_loader, val_dataset, device, out_dir, num_samples=1000):
    """예측값 vs 실제값 비교 시각화"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, Y in val_loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            all_preds.append(pred.cpu())
            all_targets.append(Y.cpu())
    
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # 원래 스케일로 복원
    preds_original = preds * val_dataset.Y_scale + val_dataset.Y_baseline
    targets_original = targets * val_dataset.Y_scale + val_dataset.Y_baseline
    
    # 일부 채널에 대해 scatter plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    sample_channels = [0, 2, 4, 7, 8, 10, 12, 15]  # 각 arm에서 일부 채널
    
    for idx, ch in enumerate(sample_channels):
        ax = axes[idx // 4, idx % 4]
        
        # 샘플 선택
        n = min(num_samples, len(preds_original))
        indices = np.random.choice(len(preds_original), n, replace=False)
        
        pred_ch = preds_original[indices, ch]
        target_ch = targets_original[indices, ch]
        
        ax.scatter(target_ch, pred_ch, alpha=0.3, s=5)
        
        # 대각선 (완벽한 예측)
        min_val = min(target_ch.min(), pred_ch.min())
        max_val = max(target_ch.max(), pred_ch.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{PROX_COLS[ch]}')
        ax.grid(True)
    
    plt.suptitle('Prediction vs Actual (Sample Channels)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'prediction_comparison.png'), dpi=150)
    plt.close()
    print(f"Prediction comparison saved: {os.path.join(out_dir, 'prediction_comparison.png')}")


# ============================================================
# 메인 함수
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Self-Detection Compensation Training')

    # NEW: mode switch
    parser.add_argument(
        '--mode',
        type=str,
        default='robot_stage1',
        choices=['robot_stage1', 'legacy16'],
        help='robot_stage1: robot_data_*.txt (6->8). legacy16: old CSV 16ch pipeline.',
    )

    # NEW: robot-stage1 args (keeps your data saving under ./models)
    parser.add_argument('--data_file', type=str, default=None, help='robot_data_*.txt path (default: latest in project root)')
    parser.add_argument('--num_sensors', type=int, default=8, help='Number of sensors (4 or 8)')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--baseline', type=float, default=4e7)
    parser.add_argument('--scale', type=float, default=1e6)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=30)
    
    # 데이터 경로
    parser.add_argument('--data_dir', type=str, 
                        default='/home/sj/self_detection_mlp/data/',
                        help='CSV 파일이 있는 디렉토리')
    parser.add_argument('--out_dir', type=str, default='./self_detection_model',
                        help='모델 저장 디렉토리')
    
    # 학습 파라미터
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    # 모델 구조
    parser.add_argument('--model_type', type=str, default='trunk_head',
                        choices=['simple', 'trunk_head'],
                        help='모델 구조 선택')
    parser.add_argument('--hidden_dims', type=str, default='256,256,128',
                        help='Simple 모델 hidden layer sizes (comma separated)')
    parser.add_argument('--trunk_hidden', type=str, default='256,256',
                        help='Trunk hidden layer sizes (comma separated)')
    parser.add_argument('--trunk_dim', type=int, default=128)
    parser.add_argument('--head_hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 기타
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()

    # New default path: robot stage1
    if args.mode == 'robot_stage1':
        train_robot_stage1(args)
        return
    
    # 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 출력 디렉토리 생성
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==========================================================
    # 데이터 로드
    # ==========================================================
    print("\n" + "="*60)
    print("Loading data (논문 방식: ToF 없이 모든 데이터 사용)...")
    print("="*60)
    
    df = load_all_data(args.data_dir)
    
    # 데이터 통계
    print(f"\nData statistics:")
    print(f"  Joint positions range:")
    for col in JOINT_POS_COLS:
        print(f"    {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
    
    print(f"\n  Proximity sensor range (sample channels):")
    for col in PROX_COLS[:4]:  # 일부만 출력
        print(f"    {col}: [{df[col].min():.0f}, {df[col].max():.0f}]")
    
    # self-detection 변화량 분석
    print(f"\n  Self-detection variation (from baseline {BASE_VALUE:.0f}):")
    for col in PROX_COLS[:4]:
        deviation = df[col] - BASE_VALUE
        print(f"    {col}: min={deviation.min():.0f}, max={deviation.max():.0f}")
    
    # ==========================================================
    # Train/Val 분할
    # ==========================================================
    N = len(df)
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    n_val = int(N * args.val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    df_train = df.iloc[train_indices].reset_index(drop=True)
    df_val = df.iloc[val_indices].reset_index(drop=True)
    
    print(f"\nTrain samples: {len(df_train)}")
    print(f"Val samples: {len(df_val)}")
    
    # ==========================================================
    # Dataset & DataLoader
    # ==========================================================
    train_dataset = SelfDetectionDataset(df_train)
    val_dataset = SelfDetectionDataset(df_val)
    
    # Validation에 train의 정규화 파라미터 사용
    val_dataset.X_mean = train_dataset.X_mean
    val_dataset.X_std = train_dataset.X_std
    val_dataset.Y_baseline = train_dataset.Y_baseline
    val_dataset.Y_scale = train_dataset.Y_scale
    
    # Validation 데이터 재정규화
    val_dataset.X_normalized = (val_dataset.X - val_dataset.X_mean) / val_dataset.X_std
    val_dataset.Y = (val_dataset.Y_raw - val_dataset.Y_baseline) / val_dataset.Y_scale
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # ==========================================================
    # 모델 생성
    # ==========================================================
    if args.model_type == 'simple':
        hidden_dims = tuple(int(x) for x in args.hidden_dims.split(','))
        model = SelfDetectionMLP(
            in_dim=12,
            hidden_dims=hidden_dims,
            out_dim=16,
            dropout=args.dropout
        ).to(device)
    else:  # trunk_head
        trunk_hidden = tuple(int(x) for x in args.trunk_hidden.split(','))
        model = SelfDetectionMLPTrunkHead(
            in_dim=12,
            trunk_hidden=trunk_hidden,
            trunk_dim=args.trunk_dim,
            head_hidden=args.head_hidden,
            out_dim=16,
            dropout=args.dropout
        ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel type: {args.model_type}")
    print(f"Model parameters: {num_params:,}")
    
    # ==========================================================
    # 학습
    # ==========================================================
    print("\n" + "="*60)
    print("Training (논문: 관절 상태 → 센서 값 예측)...")
    print("="*60)
    
    history = train_model(model, train_loader, val_loader, 
                          train_dataset, val_dataset, device, args)
    
    # ==========================================================
    # Best 모델 로드 및 최종 평가
    # ==========================================================
    print("\n" + "="*60)
    print("Final Evaluation...")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(args.out_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 전체 메트릭
    val_metrics = evaluate_metrics(model, val_loader, device, val_dataset)
    print(f"\n[Overall Metrics]")
    print(f"  RMSE: {val_metrics['rmse']:.1f}")
    print(f"  MAE: {val_metrics['mae']:.1f}")
    print(f"  Max Error: {val_metrics['max_error']:.1f}")
    print(f"  Within 1000: {val_metrics['within_1000']:.3f}")
    print(f"  Within 5000: {val_metrics['within_5000']:.3f} (논문 목표)")
    print(f"  Within 10000: {val_metrics['within_10000']:.3f}")
    
    # 채널별 메트릭
    channel_results = evaluate_per_channel(model, val_loader, device, val_dataset)
    
    print(f"\n[Per-Channel Metrics]")
    print(f"{'Channel':<10} {'RMSE':>10} {'MAE':>10} {'W5000':>10}")
    print("-" * 45)
    for r in channel_results:
        print(f"{r['channel']:<10} {r['rmse']:>10.1f} {r['mae']:>10.1f} {r['within_5000']:>10.3f}")
    
    # Worst 채널 출력
    worst_channels = sorted(channel_results, key=lambda x: -x['rmse'])[:5]
    print(f"\n[Worst 5 Channels by RMSE]")
    for r in worst_channels:
        print(f"  {r['channel']}: RMSE={r['rmse']:.1f}, W5000={r['within_5000']:.3f}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nBest model saved: {os.path.join(args.out_dir, 'best_model.pt')}")
    print(f"Best epoch: {checkpoint.get('epoch', 'N/A') + 1}")
    print(f"Best Val RMSE: {checkpoint.get('val_rmse', 'N/A'):.1f}")
    print(f"\nTo export model and generate visualizations, run:")
    print(f"  python export_model.py --model_dir {args.out_dir}")


if __name__ == '__main__':
    main()
