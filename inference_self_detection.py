#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Detection Compensation Inference & Validation Script

학습된 모델을 사용하여 추론하고 결과를 시각화하는 스크립트
학습 데이터셋에서 랜덤 파일을 가져와 학습이 잘 되었는지 확인

사용법:
  python inference_self_detection.py --model_dir ./self_detection_model
  python inference_self_detection.py --model_dir ./self_detection_model
  python inference_self_detection.py --model_dir ./self_detection_model --csv_file /path/to/specific.csv

저자: GitHub Copilot
날짜: 2026-01-26
"""

import os
import glob
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# =============================================================================
# Robot-data Stage1 inference (6 joints -> 8 sensors) (keeps your data format)
# =============================================================================
from pathlib import Path

try:
    from self_detection_mlp.data_loader import DataLoaderFactory
    from self_detection_mlp.model import Stage1SelfDetectionFieldMLP, OutputNormSpec
    _ROBOT_PIPE_AVAILABLE = True
except Exception:
    _ROBOT_PIPE_AVAILABLE = False


# ============================================================
# 설정 및 상수
# ============================================================
BASE_VALUE = 4e7  # Proximity 센서 baseline

# 컬럼 이름 정의
JOINT_POS_COLS = [f"joint_pos_{i}" for i in range(6)]
JOINT_VEL_COLS = [f"joint_vel_{i}" for i in range(6)]
INPUT_COLS = JOINT_POS_COLS + JOINT_VEL_COLS  # 12차원 입력

# Proximity 센서 컬럼 (16채널)
# Lowerarm (8개): 51, 52, 61, 62, 71, 72, 81, 82
PROX_LOWERARM_COLS = ['prox_51', 'prox_52', 'prox_61', 'prox_62',
                       'prox_71', 'prox_72', 'prox_81', 'prox_82']
# Forearm (8개): 11, 12, 21, 22, 31, 32, 41, 42  
PROX_FOREARM_COLS = ['prox_11', 'prox_12', 'prox_21', 'prox_22',
                      'prox_31', 'prox_32', 'prox_41', 'prox_42']
PROX_COLS = PROX_LOWERARM_COLS + PROX_FOREARM_COLS  # 16채널


def _find_latest_robot_data(project_root: Path) -> str | None:
    candidates = sorted(project_root.glob("robot_data_*.txt"), reverse=True)
    if not candidates:
        return None
    return str(candidates[0])


def _find_latest_stage1_ckpt(models_dir: Path) -> str | None:
    cands = sorted(models_dir.glob("stage1_*.pth"), reverse=True)
    if not cands:
        return None
    return str(cands[0])


@torch.no_grad()
def run_robot_stage1_inference(model_file: str, data_file: str, out_dir: str, num_sensors: int = 8):
    """Run inference on robot_data_*.txt and plot raw/pred/comp for 8 sensors."""
    if not _ROBOT_PIPE_AVAILABLE:
        raise RuntimeError("Robot-data pipeline imports failed. Check self_detection_mlp package.")

    project_root = Path(__file__).parent
    models_dir = project_root / "models"

    # Load checkpoint
    ckpt_path = models_dir / model_file if not os.path.isabs(model_file) else Path(model_file)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {}) or {}
    out_dim = int(cfg.get("output_dim", num_sensors))
    hidden_dims = tuple(cfg.get("hidden_dims", (256, 256, 128)))

    model = Stage1SelfDetectionFieldMLP(
        input_dim=int(cfg.get("input_dim", 6)),
        output_dim=out_dim,
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Norm spec (new checkpoints)
    on = ckpt.get("output_norm", {"baseline": BASE_VALUE, "scale": 1e6})
    norm = OutputNormSpec(baseline=float(on.get("baseline", BASE_VALUE)), scale=float(on.get("scale", 1e6)))

    # Data
    factory = DataLoaderFactory(data_file, num_sensors=num_sensors)
    # Keep your loader, no normalization here
    train_loader, val_loader, _ = factory.create_dataloaders(
        loader_type="standard",
        train_ratio=0.7,
        batch_size=512,
        shuffle_train=False,
        normalize=False,
        seed=42,
    )

    # Build full arrays from both splits for visualization
    def _collect(loader):
        Xs, Ys = [], []
        for X, y in loader:
            Xs.append(X[:, :6])
            Ys.append(y)
        X = torch.cat(Xs, dim=0)
        Y = torch.cat(Ys, dim=0)
        return X, Y

    X_tr, Y_tr = _collect(train_loader)
    X_va, Y_va = _collect(val_loader)
    X = torch.cat([X_tr, X_va], dim=0)
    Y_raw = torch.cat([Y_tr, Y_va], dim=0)

    # Predict
    Y_pred_norm = model(X).cpu()
    Y_pred_raw = norm.denormalize(Y_pred_norm)

    # Compensation (paper form): comp = measured - (pred_raw - BASE)
    comp = Y_raw[:, :out_dim] - (Y_pred_raw - float(norm.baseline))

    # Metrics (raw scale)
    rmse = torch.sqrt(torch.mean((Y_pred_raw - Y_raw[:, :out_dim]) ** 2)).item()
    mae = torch.mean(torch.abs(Y_pred_raw - Y_raw[:, :out_dim])).item()
    print(f"[Robot Stage1] RMSE(raw): {rmse:.1f}, MAE(raw): {mae:.1f}  (N={out_dim})")

    os.makedirs(out_dir, exist_ok=True)

    # Plot time-series for first 8 sensors
    import matplotlib.pyplot as plt
    n_sensors_plot = min(num_sensors, 8)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    t = np.arange(len(Y_raw))
    for i in range(n_sensors_plot):
        ax = axes[i]
        ax.plot(t, Y_raw[:, i].numpy(), label="raw", color="blue", alpha=0.7, linewidth=0.8)
        ax.plot(t, Y_pred_raw[:, i].numpy(), label="pred", color="orange", alpha=0.7, linewidth=0.8)
        ax.plot(t, comp[:, i].numpy(), label="comp", color="red", alpha=0.5, linewidth=0.8)
        ax.set_title(f"Sensor {i+1}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "robot_stage1_timeseries.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")


# ============================================================
# MLP 모델 (학습 코드와 동일한 구조)
# ============================================================
class SelfDetectionMLPTrunkHead(nn.Module):
    """Trunk + Head 구조의 MLP"""
    
    def __init__(self, in_dim=12, trunk_hidden=(256, 256), trunk_dim=128,
                 head_hidden=64, out_dim=16, dropout=0.1):
        super().__init__()
        
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
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(head_hidden, 1)
            ) for _ in range(out_dim)
        ])
        
        self.out_dim = out_dim
    
    def forward(self, x):
        z = self.trunk(x)
        outputs = [head(z) for head in self.heads]
        y = torch.cat(outputs, dim=1)
        return y


class SelfDetectionMLP(nn.Module):
    """Simple MLP"""
    
    def __init__(self, in_dim=12, hidden_dims=(256, 256, 128), out_dim=16, dropout=0.1):
        super().__init__()
        
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(d, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            d = h
        layers.append(nn.Linear(d, out_dim))
        
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# 추론 함수
# ============================================================
def load_model(model_dir, device):
    """저장된 모델과 정규화 파라미터 로드"""
    
    # 정규화 파라미터 로드
    norm_path = os.path.join(model_dir, 'normalization_params.npz')
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalization params not found: {norm_path}")
    
    norm_data = np.load(norm_path, allow_pickle=True)
    norm_params = {
        'X_mean': norm_data['X_mean'],
        'X_std': norm_data['X_std'],
        'Y_baseline': float(norm_data['Y_baseline']),
        'Y_scale': float(norm_data['Y_scale'])
    }
    
    # 모델 로드
    model_path = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 모델 구조 추론 (TrunkHead가 기본)
    try:
        model = SelfDetectionMLPTrunkHead(
            in_dim=12, trunk_hidden=(256, 256), trunk_dim=128,
            head_hidden=64, out_dim=16, dropout=0.0  # 추론 시 dropout 비활성화
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        model = SelfDetectionMLP(
            in_dim=12, hidden_dims=(256, 256, 128), out_dim=16, dropout=0.0
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Best epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val RMSE: {checkpoint.get('val_rmse', 'N/A'):.1f}")
    
    return model, norm_params


def get_random_csv_file(data_dir):
    """데이터 디렉토리에서 랜덤 CSV 파일 선택"""
    pattern = os.path.join(data_dir, "*Self_Detection*.csv")
    files = sorted(glob.glob(pattern))
    
    # 하위 디렉토리도 검색
    pattern_sub = os.path.join(data_dir, "**/*Self_Detection*.csv")
    files_sub = sorted(glob.glob(pattern_sub, recursive=True))
    files = list(set(files + files_sub))
    
    if len(files) == 0:
        raise RuntimeError(f"No CSV files found matching: {pattern}")
    
    selected = random.choice(files)
    print(f"Selected random file: {os.path.basename(selected)}")
    return selected


def run_inference(model, norm_params, csv_file, device):
    """단일 CSV 파일에 대해 추론 수행"""
    
    # 데이터 로드
    df = pd.read_csv(csv_file)
    
    # 입력 데이터 준비
    X = df[INPUT_COLS].to_numpy(dtype=np.float32)
    Y_actual = df[PROX_COLS].to_numpy(dtype=np.float32)
    timestamps = df['timestamp'].to_numpy() if 'timestamp' in df.columns else np.arange(len(df))
    
    # 정규화
    X_normalized = (X - norm_params['X_mean']) / norm_params['X_std']
    
    # 추론
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_normalized).to(device)
        Y_pred_normalized = model(X_tensor).cpu().numpy()
    
    # 역정규화
    Y_pred = Y_pred_normalized * norm_params['Y_scale'] + norm_params['Y_baseline']
    
    # 보정된 값 계산 (측정값 - 예측된 self-detection 성분)
    # compensated = measured - (predicted - baseline)
    Y_compensated = Y_actual - (Y_pred - BASE_VALUE)
    
    return {
        'timestamps': timestamps,
        'actual': Y_actual,
        'predicted': Y_pred,
        'compensated': Y_compensated,
        'joint_pos': df[JOINT_POS_COLS].to_numpy(),
        'joint_vel': df[JOINT_VEL_COLS].to_numpy()
    }


# ============================================================
# 시각화 함수
# ============================================================
def plot_time_series(results, out_dir, filename):
    """시간에 따른 센서 값 변화 시각화"""
    
    timestamps = results['timestamps']
    actual = results['actual']
    predicted = results['predicted']
    compensated = results['compensated']
    
    # Lowerarm과 Forearm 분리해서 플롯
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    
    for ch_idx, col in enumerate(PROX_COLS):
        row = ch_idx // 4
        col_idx = ch_idx % 4
        ax = axes[row, col_idx]
        
        # 실제값, 예측값, 보정값 플롯
        ax.plot(timestamps, actual[:, ch_idx], 'b-', alpha=0.7, label='Actual', linewidth=0.5)
        ax.plot(timestamps, predicted[:, ch_idx], 'r-', alpha=0.7, label='Predicted', linewidth=0.5)
        ax.plot(timestamps, compensated[:, ch_idx], 'g-', alpha=0.7, label='Compensated', linewidth=0.5)
        
        # Baseline 표시
        ax.axhline(y=BASE_VALUE, color='k', linestyle='--', alpha=0.3, label='Baseline')
        
        ax.set_title(f'{col}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Sensor Value')
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True, alpha=0.3)
        
        # Y축 범위 조정 (baseline 중심)
        y_range = max(abs(actual[:, ch_idx] - BASE_VALUE).max(), 
                      abs(predicted[:, ch_idx] - BASE_VALUE).max()) * 1.2
        ax.set_ylim(BASE_VALUE - y_range, BASE_VALUE + y_range)
    
    plt.suptitle(f'Self-Detection Compensation Results\n{os.path.basename(filename)}', fontsize=14)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'inference_time_series.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Time series plot saved: {out_path}")


def plot_error_distribution(results, out_dir):
    """오차 분포 시각화"""
    
    actual = results['actual']
    predicted = results['predicted']
    errors = np.abs(predicted - actual)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 전체 오차 히스토그램
    ax = axes[0, 0]
    ax.hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=5000, color='r', linestyle='--', label='Target (5000)')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution (All Channels)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 채널별 RMSE
    ax = axes[0, 1]
    rmse_per_channel = np.sqrt(np.mean(errors**2, axis=0))
    colors = ['green' if r < 5000 else 'orange' if r < 10000 else 'red' for r in rmse_per_channel]
    x = np.arange(len(PROX_COLS))
    ax.bar(x, rmse_per_channel, color=colors)
    ax.axhline(y=5000, color='r', linestyle='--', label='Target (5000)')
    ax.set_xticks(x)
    ax.set_xticklabels(PROX_COLS, rotation=45, ha='right')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE per Channel')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 3. Lowerarm 오차 박스플롯
    ax = axes[1, 0]
    lowerarm_errors = errors[:, :8]  # 처음 8개가 Lowerarm
    ax.boxplot(lowerarm_errors, labels=PROX_LOWERARM_COLS)
    ax.axhline(y=5000, color='r', linestyle='--', label='Target (5000)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Lowerarm Error Distribution')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 4. Forearm 오차 박스플롯
    ax = axes[1, 1]
    forearm_errors = errors[:, 8:]  # 나머지 8개가 Forearm
    ax.boxplot(forearm_errors, labels=PROX_FOREARM_COLS)
    ax.axhline(y=5000, color='r', linestyle='--', label='Target (5000)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Forearm Error Distribution')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'inference_error_distribution.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Error distribution plot saved: {out_path}")


def plot_compensation_effect(results, out_dir):
    """Self-Detection 보정 효과 시각화"""
    
    actual = results['actual']
    predicted = results['predicted']
    compensated = results['compensated']
    
    # Self-detection 변화량 계산
    self_detection_before = actual - BASE_VALUE  # 보정 전 변화량
    self_detection_after = compensated - BASE_VALUE  # 보정 후 변화량
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 보정 전 변화량 분포
    ax = axes[0, 0]
    ax.hist(self_detection_before.flatten(), bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(x=0, color='k', linestyle='-', linewidth=2)
    ax.axvline(x=-5000, color='r', linestyle='--', label='±5000')
    ax.axvline(x=5000, color='r', linestyle='--')
    ax.set_xlabel('Deviation from Baseline')
    ax.set_ylabel('Count')
    ax.set_title(f'Before Compensation\n(Std: {self_detection_before.std():.0f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 보정 후 변화량 분포
    ax = axes[0, 1]
    ax.hist(self_detection_after.flatten(), bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(x=0, color='k', linestyle='-', linewidth=2)
    ax.axvline(x=-5000, color='r', linestyle='--', label='±5000')
    ax.axvline(x=5000, color='r', linestyle='--')
    ax.set_xlabel('Deviation from Baseline')
    ax.set_ylabel('Count')
    ax.set_title(f'After Compensation\n(Std: {self_detection_after.std():.0f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 채널별 보정 전후 표준편차 비교
    ax = axes[1, 0]
    std_before = np.std(self_detection_before, axis=0)
    std_after = np.std(self_detection_after, axis=0)
    
    x = np.arange(len(PROX_COLS))
    width = 0.35
    ax.bar(x - width/2, std_before, width, label='Before', color='blue', alpha=0.7)
    ax.bar(x + width/2, std_after, width, label='After', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(PROX_COLS, rotation=45, ha='right')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Self-Detection Variation per Channel')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 4. 보정 효과 (감소율)
    ax = axes[1, 1]
    reduction = (1 - std_after / (std_before + 1e-8)) * 100
    colors = ['green' if r > 50 else 'orange' if r > 0 else 'red' for r in reduction]
    ax.bar(x, reduction, color=colors)
    ax.axhline(y=50, color='r', linestyle='--', label='50% reduction')
    ax.set_xticks(x)
    ax.set_xticklabels(PROX_COLS, rotation=45, ha='right')
    ax.set_ylabel('Reduction (%)')
    ax.set_title('Self-Detection Reduction Rate')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'inference_compensation_effect.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Compensation effect plot saved: {out_path}")
    
    # 통계 출력
    print("\n" + "="*60)
    print("Compensation Effect Summary")
    print("="*60)
    print(f"  Before: Std = {self_detection_before.std():.0f}, "
          f"Max = {np.abs(self_detection_before).max():.0f}")
    print(f"  After:  Std = {self_detection_after.std():.0f}, "
          f"Max = {np.abs(self_detection_after).max():.0f}")
    print(f"  Reduction: {(1 - self_detection_after.std() / self_detection_before.std()) * 100:.1f}%")


def plot_scatter_comparison(results, out_dir, num_samples=2000):
    """실제값 vs 예측값 scatter plot"""
    
    actual = results['actual']
    predicted = results['predicted']
    
    # 샘플 선택
    n = min(num_samples, len(actual))
    indices = np.random.choice(len(actual), n, replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Lowerarm 4개, Forearm 4개 선택해서 표시
    sample_channels = [0, 2, 4, 6, 8, 10, 12, 14]  # 각 링크에서 일부 채널
    
    for idx, ch in enumerate(sample_channels):
        ax = axes[idx // 4, idx % 4]
        
        a = actual[indices, ch]
        p = predicted[indices, ch]
        
        ax.scatter(a, p, alpha=0.3, s=5)
        
        # 대각선 (완벽한 예측)
        min_val = min(a.min(), p.min())
        max_val = max(a.max(), p.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # ±5000 범위 표시
        ax.plot([min_val, max_val], [min_val + 5000, max_val + 5000], 'g--', alpha=0.5)
        ax.plot([min_val, max_val], [min_val - 5000, max_val - 5000], 'g--', alpha=0.5, label='±5000')
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{PROX_COLS[ch]}')
        ax.legend(loc='upper left', fontsize=6)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Actual vs Predicted (Sample Channels)')
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'inference_scatter.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Scatter plot saved: {out_path}")


def print_metrics(results):
    """추론 결과 메트릭 출력"""
    
    actual = results['actual']
    predicted = results['predicted']
    errors = np.abs(predicted - actual)
    
    print("\n" + "="*60)
    print("Inference Metrics")
    print("="*60)
    
    # 전체 메트릭
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)
    within_1000 = np.mean(errors <= 1000)
    within_5000 = np.mean(errors <= 5000)
    within_10000 = np.mean(errors <= 10000)
    
    print(f"\n[Overall Metrics]")
    print(f"  MAE: {mae:.1f}")
    print(f"  RMSE: {rmse:.1f}")
    print(f"  Max Error: {max_error:.1f}")
    print(f"  Within 1000: {within_1000*100:.1f}%")
    print(f"  Within 5000: {within_5000*100:.1f}% (논문 목표)")
    print(f"  Within 10000: {within_10000*100:.1f}%")
    
    # 채널별 메트릭
    print(f"\n[Per-Channel RMSE]")
    print(f"{'Channel':<10} {'RMSE':>10} {'MAE':>10} {'W5000':>10}")
    print("-" * 45)
    
    for ch_idx, col in enumerate(PROX_COLS):
        ch_error = errors[:, ch_idx]
        ch_rmse = np.sqrt(np.mean(ch_error ** 2))
        ch_mae = np.mean(ch_error)
        ch_w5000 = np.mean(ch_error <= 5000)
        print(f"{col:<10} {ch_rmse:>10.1f} {ch_mae:>10.1f} {ch_w5000*100:>9.1f}%")
    
    # Lowerarm vs Forearm 비교
    print(f"\n[Link Comparison]")
    lowerarm_rmse = np.sqrt(np.mean(errors[:, :8] ** 2))
    forearm_rmse = np.sqrt(np.mean(errors[:, 8:] ** 2))
    print(f"  Lowerarm RMSE: {lowerarm_rmse:.1f}")
    print(f"  Forearm RMSE: {forearm_rmse:.1f}")


# ============================================================
# 메인 함수
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Self-Detection Inference & Validation')

    parser.add_argument(
        '--mode',
        type=str,
        default='robot_stage1',
        choices=['robot_stage1', 'legacy16'],
        help='robot_stage1: robot_data_*.txt (6->8). legacy16: old CSV 16ch pipeline.',
    )
    parser.add_argument('--model_file', type=str, default=None, help='stage1_*.pth in ./models (default: latest)')
    parser.add_argument('--data_file', type=str, default=None, help='robot_data_*.txt (default: latest in project root)')
    parser.add_argument('--num_sensors', type=int, default=8)
    
    parser.add_argument('--model_dir', type=str, default='./self_detection_model',
                        help='모델이 저장된 디렉토리')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/sj/self_detection_mlp/data',
                        help='CSV 파일 디렉토리 (랜덤 선택 시 사용)')
    parser.add_argument('--csv_file', type=str, default=None,
                        help='특정 CSV 파일 (지정하지 않으면 랜덤 선택)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='결과 저장 디렉토리 (기본: model_dir)')
    parser.add_argument('--seed', type=int, default=None,
                        help='랜덤 시드')
    
    args = parser.parse_args()

    if args.mode == 'robot_stage1':
        project_root = Path(__file__).parent
        models_dir = project_root / "models"
        model_file = args.model_file or _find_latest_stage1_ckpt(models_dir)
        if not model_file:
            raise FileNotFoundError("No stage1_*.pth found in ./models and --model_file not provided.")
        if not args.data_file:
            data_file = _find_latest_robot_data(project_root)
            if not data_file:
                raise FileNotFoundError("No robot_data_*.txt found and --data_file not provided.")
        else:
            data_file = args.data_file

        # If we got absolute ckpt, keep it; else pass basename and let loader use ./models
        model_arg = model_file if os.path.isabs(model_file) else os.path.basename(model_file)
        out_dir = args.out_dir if args.out_dir else './inference_out'
        run_robot_stage1_inference(model_arg, data_file, out_dir, num_sensors=args.num_sensors)
        return
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    out_dir = args.out_dir if args.out_dir else args.model_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 로드
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    model, norm_params = load_model(args.model_dir, device)
    
    # CSV 파일 선택
    print("\n" + "="*60)
    print("Selecting test file...")
    print("="*60)
    if args.csv_file:
        csv_file = args.csv_file
        print(f"Using specified file: {os.path.basename(csv_file)}")
    else:
        csv_file = get_random_csv_file(args.data_dir)
    
    # 추론 실행
    print("\n" + "="*60)
    print("Running inference...")
    print("="*60)
    results = run_inference(model, norm_params, csv_file, device)
    print(f"Processed {len(results['timestamps'])} samples")
    
    # 메트릭 출력
    print_metrics(results)
    
    # 시각화
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    plot_time_series(results, out_dir, csv_file)
    plot_error_distribution(results, out_dir)
    plot_compensation_effect(results, out_dir)
    plot_scatter_comparison(results, out_dir)
    
    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)
    print(f"\nOutput files saved to: {out_dir}")
    print(f"  - inference_time_series.png")
    print(f"  - inference_error_distribution.png")
    print(f"  - inference_compensation_effect.png")
    print(f"  - inference_scatter.png")


if __name__ == '__main__':
    main()
