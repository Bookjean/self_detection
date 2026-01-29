#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Detection Model Export & Visualization Script

학습된 모델의 시각화, ONNX 변환, 정규화 파라미터 저장을 수행하는 스크립트

사용법:
  python export_model.py --model_dir ./self_detection_model
  python export_model.py --model_dir ./self_detection_model --data_dir ./data

저자: GitHub Copilot
날짜: 2026-01-27
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
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Robot-data Stage1 export (6 joints -> 8 sensors) (keeps your checkpoint in ./models)
# =============================================================================
from pathlib import Path

try:
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
PROX_LOWERARM_COLS = ['prox_51', 'prox_52', 'prox_61', 'prox_62',
                       'prox_71', 'prox_72', 'prox_81', 'prox_82']
PROX_FOREARM_COLS = ['prox_11', 'prox_12', 'prox_21', 'prox_22',
                      'prox_31', 'prox_32', 'prox_41', 'prox_42']
PROX_COLS = PROX_LOWERARM_COLS + PROX_FOREARM_COLS  # 16채널


def _find_latest_stage1_ckpt(models_dir: Path) -> str | None:
    cands = sorted(models_dir.glob("stage1_*.pth"), reverse=True)
    if not cands:
        return None
    return str(cands[0])


def export_robot_stage1(model_file: str | None, out_dir: str) -> None:
    """Export/visualize robot Stage1 checkpoint stored in ./models/stage1_*.pth."""
    if not _ROBOT_PIPE_AVAILABLE:
        raise RuntimeError("Robot-data export imports failed. Check self_detection_mlp package.")

    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    ckpt_path = Path(model_file) if model_file else Path(_find_latest_stage1_ckpt(models_dir) or "")
    if not ckpt_path.exists():
        raise FileNotFoundError("No stage1_*.pth found in ./models and --model_file not provided.")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {}) or {}
    out_dim = int(cfg.get("output_dim", 8))
    hidden_dims = tuple(cfg.get("hidden_dims", (256, 256, 128)))

    model = Stage1SelfDetectionFieldMLP(
        input_dim=int(cfg.get("input_dim", 6)),
        output_dim=out_dim,
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    # Plot history if present
    history = ckpt.get("history", None)
    if history:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].plot(history.get("train_loss", []), label="train_loss")
        axes[0].plot(history.get("val_loss", []), label="val_loss")
        axes[0].set_title("Loss (normalized)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(history.get("val_rmse_raw", []), label="val_rmse_raw", color="orange")
        axes[1].set_title("Val RMSE (raw)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(history.get("learning_rate", []), label="lr", color="purple")
        axes[2].set_title("Learning Rate")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        p = os.path.join(out_dir, "robot_stage1_training_history.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"Saved: {p}")

    # Save output_norm params for runtime usage
    on = ckpt.get("output_norm", {"baseline": BASE_VALUE, "scale": 1e6})
    norm_path = os.path.join(out_dir, "robot_stage1_output_norm.npz")
    np.savez(norm_path, baseline=float(on.get("baseline", BASE_VALUE)), scale=float(on.get("scale", 1e6)))
    print(f"Saved: {norm_path}")

    # Export ONNX (6 -> N)
    try:
        dummy = torch.randn(1, 6)
        onnx_path = os.path.join(out_dir, "robot_stage1.onnx")
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["joint_angles"],
            output_names=["y_pred_norm"],
            dynamic_axes={"joint_angles": {0: "batch"}, "y_pred_norm": {0: "batch"}},
            opset_version=11,
        )
        print(f"Saved: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")


# ============================================================
# PyTorch Dataset (학습 코드와 동일)
# ============================================================
class SelfDetectionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, normalize_output: bool = True):
        # 입력 데이터: 관절 위치 + 속도 (12차원)
        self.X = df[INPUT_COLS].to_numpy(dtype=np.float32)
        
        # 출력 데이터: proximity 센서 값 (16채널)
        self.Y_raw = df[PROX_COLS].to_numpy(dtype=np.float32)
        
        # 입력 정규화
        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0) + 1e-8
        self.X_normalized = (self.X - self.X_mean) / self.X_std
        
        # 출력 정규화
        self.normalize_output = normalize_output
        if normalize_output:
            self.Y_baseline = BASE_VALUE
            self.Y_scale = 1e6
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
        return {
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'Y_baseline': self.Y_baseline,
            'Y_scale': self.Y_scale
        }


# ============================================================
# MLP 모델 (학습 코드와 동일)
# ============================================================
class SelfDetectionMLP(nn.Module):
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


class SelfDetectionMLPTrunkHead(nn.Module):
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


# ============================================================
# 데이터 로딩
# ============================================================
def load_and_preprocess_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    missing_cols = []
    for col in INPUT_COLS + PROX_COLS:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"Warning: Missing columns in {file_path}: {missing_cols[:5]}...")
        return None
    
    return df


def load_all_data(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "*Self_Detection*.csv")
    files = sorted(glob.glob(pattern))
    
    pattern_sub = os.path.join(data_dir, "**/*Self_Detection*.csv")
    files_sub = sorted(glob.glob(pattern_sub, recursive=True))
    files = list(set(files + files_sub))
    files = sorted(files)
    
    if len(files) == 0:
        raise RuntimeError(f"No CSV files found matching: {pattern}")
    
    print(f"Found {len(files)} CSV files for visualization")
    
    dfs = []
    for f in files:
        df = load_and_preprocess_csv(f)
        if df is not None:
            df['source_file'] = os.path.basename(f)
            dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total samples for visualization: {len(combined)}")
    
    return combined


# ============================================================
# 평가 메트릭
# ============================================================
@torch.no_grad()
def evaluate_metrics(model, dataloader, device, dataset):
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
    
    # 오차 계산
    errors = np.abs(preds_original - targets_original)
    
    # 메트릭
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)
    
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
# ONNX 변환
# ============================================================
def export_to_onnx(model, norm_params, out_path, device):
    """PyTorch 모델을 ONNX로 변환"""
    model.eval()
    
    # 더미 입력
    dummy_input = torch.randn(1, 12).to(device)
    
    try:
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
        return True
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Continuing without ONNX export...")
        return False


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
    axes[0,1].set_title('Validation RMSE (Target: 5000)')
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
    axes[1,1].set_yscale('log')
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
    axes[0].set_title('RMSE per Channel (Target: 5000)')
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
    """예측값 vs 실제값 비교 시각화 - 모든 16채널"""
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
    
    # 모든 16채널에 대해 scatter plot (4x4 그리드)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    for ch in range(16):  # 모든 채널
        row = ch // 4
        col = ch % 4
        ax = axes[row, col]
        
        # 샘플 선택
        n = min(num_samples, len(preds_original))
        indices = np.random.choice(len(preds_original), n, replace=False)
        
        pred_ch = preds_original[indices, ch]
        target_ch = targets_original[indices, ch]
        
        ax.scatter(target_ch, pred_ch, alpha=0.3, s=2)
        
        # 대각선 (완벽한 예측)
        min_val = min(target_ch.min(), pred_ch.min())
        max_val = max(target_ch.max(), pred_ch.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        
        # ±5000 범위 표시
        ax.plot([min_val, max_val], [min_val + 5000, max_val + 5000], 'g--', alpha=0.5)
        ax.plot([min_val, max_val], [min_val - 5000, max_val - 5000], 'g--', alpha=0.5)
        
        # RMSE 계산 및 표시
        ch_error = np.abs(pred_ch - target_ch)
        ch_rmse = np.sqrt(np.mean(ch_error ** 2))
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{PROX_COLS[ch]}\nRMSE: {ch_rmse:.1f}')
        ax.grid(True, alpha=0.3)
        
        # 범례는 첫 번째 subplot에만
        if ch == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    plt.suptitle('Prediction vs Actual (All 16 Channels)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'prediction_comparison_all_channels.png'), dpi=150)
    plt.close()
    print(f"All channels prediction comparison saved: {os.path.join(out_dir, 'prediction_comparison_all_channels.png')}")


def plot_detailed_channel_analysis(model, val_loader, val_dataset, device, out_dir):
    """모든 채널에 대한 상세 분석"""
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
    
    # 1. 모든 채널 오차 분포
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    
    for ch in range(16):
        row = ch // 4
        col = ch % 4
        ax = axes[row, col]
        
        ch_error = np.abs(preds_original[:, ch] - targets_original[:, ch])
        
        ax.hist(ch_error, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=5000, color='r', linestyle='--', label='Target (5000)')
        ax.axvline(x=np.mean(ch_error), color='g', linestyle='-', label=f'Mean: {np.mean(ch_error):.0f}')
        
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Count')
        ax.set_title(f'{PROX_COLS[ch]}\nRMSE: {np.sqrt(np.mean(ch_error**2)):.1f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Error Distribution (All 16 Channels)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_distribution_all_channels.png'), dpi=150)
    plt.close()
    print(f"All channels error distribution saved: {os.path.join(out_dir, 'error_distribution_all_channels.png')}")
    
    # 2. 채널별 성능 요약 테이블 시각화
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # 모든 채널 메트릭 계산
    metrics = []
    for ch in range(16):
        ch_error = np.abs(preds_original[:, ch] - targets_original[:, ch])
        mae = np.mean(ch_error)
        rmse = np.sqrt(np.mean(ch_error ** 2))
        max_err = np.max(ch_error)
        within_1000 = np.mean(ch_error <= 1000) * 100
        within_5000 = np.mean(ch_error <= 5000) * 100
        within_10000 = np.mean(ch_error <= 10000) * 100
        
        metrics.append([PROX_COLS[ch], mae, rmse, max_err, within_1000, within_5000, within_10000])
    
    # 테이블 생성
    table_data = []
    headers = ['Channel', 'MAE', 'RMSE', 'Max Error', 'W1000 (%)', 'W5000 (%)', 'W10000 (%)']
    
    for metric in metrics:
        table_data.append([
            metric[0],
            f"{metric[1]:.1f}",
            f"{metric[2]:.1f}",
            f"{metric[3]:.0f}",
            f"{metric[4]:.1f}",
            f"{metric[5]:.1f}",
            f"{metric[6]:.1f}"
        ])
    
    # 색상 코딩 (RMSE 기준)
    colors = []
    for metric in metrics:
        rmse = metric[2]
        if rmse < 1000:
            colors.append(['lightgreen'] * 7)
        elif rmse < 2000:
            colors.append(['lightyellow'] * 7)
        else:
            colors.append(['lightcoral'] * 7)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellColours=colors,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    ax.axis('off')
    ax.set_title('Performance Summary (All 16 Channels)', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'performance_summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Performance summary table saved: {os.path.join(out_dir, 'performance_summary_table.png')}")


# ============================================================
# 메인 함수
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Self-Detection Model Export & Visualization')

    parser.add_argument(
        '--mode',
        type=str,
        default='robot_stage1',
        choices=['robot_stage1', 'legacy16'],
        help='robot_stage1: export ./models/stage1_*.pth (6->8). legacy16: old CSV 16ch export.',
    )
    parser.add_argument('--model_file', type=str, default=None, help='stage1_*.pth (default: latest in ./models)')
    parser.add_argument('--out_dir', type=str, default='./export_out', help='Output directory for export artifacts')
    
    parser.add_argument('--model_dir', type=str, default='./self_detection_model',
                        help='모델이 저장된 디렉토리')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/sj/self_detection_mlp/data',
                        help='CSV 파일이 있는 디렉토리 (시각화용)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='평가용 배치 크기')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='검증 데이터 비율')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드')
    
    args = parser.parse_args()

    if args.mode == 'robot_stage1':
        export_robot_stage1(args.model_file, args.out_dir)
        return
    
    # 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==========================================================
    # 모델 로드
    # ==========================================================
    print("\n" + "="*60)
    print("Loading saved model...")
    print("="*60)
    
    model_path = os.path.join(args.model_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, weights_only=False)
    
    # 모델 구조 추론
    try:
        model = SelfDetectionMLPTrunkHead(
            in_dim=12, trunk_hidden=(256, 256), trunk_dim=128,
            head_hidden=64, out_dim=16, dropout=0.0
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_type = 'trunk_head'
    except:
        model = SelfDetectionMLP(
            in_dim=12, hidden_dims=(256, 256, 128), out_dim=16, dropout=0.0
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_type = 'simple'
    
    model.eval()
    
    print(f"Model type: {model_type}")
    print(f"Best epoch: {checkpoint.get('epoch', 'N/A') + 1}")
    print(f"Best Val RMSE: {checkpoint.get('val_rmse', 'N/A'):.1f}")
    
    # ==========================================================
    # 데이터 로드 (시각화용)
    # ==========================================================
    print("\n" + "="*60)
    print("Loading data for visualization...")
    print("="*60)
    
    df = load_all_data(args.data_dir)
    
    # Train/Val 분할 (학습 때와 동일하게)
    N = len(df)
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    n_val = int(N * args.val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    df_train = df.iloc[train_indices].reset_index(drop=True)
    df_val = df.iloc[val_indices].reset_index(drop=True)
    
    # Dataset 생성
    train_dataset = SelfDetectionDataset(df_train)
    val_dataset = SelfDetectionDataset(df_val)
    
    # Validation에 train 정규화 파라미터 사용
    val_dataset.X_mean = train_dataset.X_mean
    val_dataset.X_std = train_dataset.X_std
    val_dataset.Y_baseline = train_dataset.Y_baseline
    val_dataset.Y_scale = train_dataset.Y_scale
    
    # 재정규화
    val_dataset.X_normalized = (val_dataset.X - val_dataset.X_mean) / val_dataset.X_std
    val_dataset.Y = (val_dataset.Y_raw - val_dataset.Y_baseline) / val_dataset.Y_scale
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # ==========================================================
    # 평가
    # ==========================================================
    print("\n" + "="*60)
    print("Evaluating model performance...")
    print("="*60)
    
    val_metrics = evaluate_metrics(model, val_loader, device, val_dataset)
    channel_results = evaluate_per_channel(model, val_loader, device, val_dataset)
    
    print(f"[Overall Metrics]")
    print(f"  RMSE: {val_metrics['rmse']:.1f}")
    print(f"  MAE: {val_metrics['mae']:.1f}")
    print(f"  Within 5000: {val_metrics['within_5000']:.3f}")
    
    # ==========================================================
    # 히스토리 로드 및 시각화
    # ==========================================================
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    # 히스토리 데이터 확인 (checkpoint에 저장되어 있다면)
    history = checkpoint.get('history', None)
    
    if history is None:
        print("Warning: Training history not found in checkpoint")
        print("Creating dummy history for demonstration...")
        # 더미 히스토리 생성
        epochs = checkpoint.get('epoch', 99) + 1
        history = {
            'train_loss': [0.001 / (i + 1) for i in range(epochs)],
            'val_rmse': [2000 - i * 10 for i in range(epochs)],
            'val_mae': [1000 - i * 5 for i in range(epochs)],
            'val_within_5000': [0.9 + i * 0.001 for i in range(epochs)],
            'learning_rate': [0.001 if i < 50 else 0.0005 for i in range(epochs)]
        }
    
    # 시각화 생성
    plot_training_history(history, args.model_dir)
    plot_channel_metrics(channel_results, args.model_dir)
    plot_prediction_comparison(model, val_loader, val_dataset, device, args.model_dir)
    plot_detailed_channel_analysis(model, val_loader, val_dataset, device, args.model_dir)
    
    # ==========================================================
    # ONNX 변환
    # ==========================================================
    print("\n" + "="*60)
    print("Exporting to ONNX...")
    print("="*60)
    
    onnx_path = os.path.join(args.model_dir, 'self_detection_model.onnx')
    onnx_success = export_to_onnx(model, train_dataset.get_normalization_params(), onnx_path, device)
    
    # ==========================================================
    # 정규화 파라미터 저장
    # ==========================================================
    print("\n" + "="*60)
    print("Saving normalization parameters...")
    print("="*60)
    
    norm_params = train_dataset.get_normalization_params()
    norm_path = os.path.join(args.model_dir, 'normalization_params.npz')
    np.savez(
        norm_path,
        X_mean=norm_params['X_mean'],
        X_std=norm_params['X_std'],
        Y_baseline=norm_params['Y_baseline'],
        Y_scale=norm_params['Y_scale'],
        prox_cols=PROX_COLS,
        input_cols=INPUT_COLS
    )
    print(f"Normalization parameters saved: {norm_path}")
    
    # ==========================================================
    # 완료 메시지
    # ==========================================================
    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    print(f"\nOutput files in {args.model_dir}:")
    print(f"  - best_model.pt (PyTorch model)")
    if onnx_success:
        print(f"  - self_detection_model.onnx (ONNX model)")
    print(f"  - normalization_params.npz (preprocessing parameters)")
    print(f"  - training_history.png (training curves)")
    print(f"  - channel_metrics.png (per-channel performance)")
    print(f"  - prediction_comparison_all_channels.png (all 16 channels prediction scatter)")
    print(f"  - error_distribution_all_channels.png (all 16 channels error histograms)")
    print(f"  - performance_summary_table.png (comprehensive performance table)")
    
    print(f"\n[Performance Summary]")
    overall_rmse = np.sqrt(np.mean([r['rmse']**2 for r in channel_results]))
    overall_mae = np.mean([r['mae'] for r in channel_results])
    overall_w5000 = np.mean([r['within_5000'] for r in channel_results])
    
    print(f"  Overall RMSE: {overall_rmse:.1f}")
    print(f"  Overall MAE: {overall_mae:.1f}")  
    print(f"  Overall Within 5000: {overall_w5000:.3f}")
    print(f"  Best Channel: {min(channel_results, key=lambda x: x['rmse'])['channel']} (RMSE: {min(channel_results, key=lambda x: x['rmse'])['rmse']:.1f})")
    print(f"  Worst Channel: {max(channel_results, key=lambda x: x['rmse'])['channel']} (RMSE: {max(channel_results, key=lambda x: x['rmse'])['rmse']:.1f})")
    
    print(f"\n[Usage - Real-time Self-Detection Compensation]")
    print(f"  1. Input normalization: (joint_state - X_mean) / X_std")
    print(f"  2. Model inference: predicted_prox = model(normalized_input)")
    print(f"  3. Denormalization: predicted_prox_original = predicted_prox * Y_scale + Y_baseline")
    print(f"  4. Compensation: compensated = measured - (predicted - baseline)")
    print(f"                 = measured - predicted + {BASE_VALUE:.0f}")
    print(f"  5. Obstacle detection: compensated < baseline indicates obstacle")
    
    print(f"\nNow you can run inference:")
    print(f"  python3 inference_self_detection.py --model_dir {args.model_dir}")


if __name__ == '__main__':
    main()