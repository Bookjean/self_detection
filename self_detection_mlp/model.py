#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Self-Detection Models (Single Lineage)

✅ 목표
- "논문 계보(1번)"의 의미로 Stage 계보(2번)를 정렬/흡수
- 입력: 6 joint angles (q) ONLY
- 출력: N sensors (raw)  ※ N=8(또는 16) 그대로 사용
- 입력 정규화: 사용 안 함 (너의 요구)
- 출력 정규화: baseline 기준 정규화(논문 방식)만 지원
    y_norm = (y_raw - BASE_VALUE) / Y_SCALE

❗ 핵심 개념 변화(중요)
- 기존 Stage1StaticOffsetMLP(=baseline b(q)) 개념 폐기
- Stage1은 "self-detection field" 전체를 예측:
    ŝ(q) ≈ y_raw (obstacle-free)
  학습은 보통 baseline 기준 정규화된 y_norm을 타깃으로 한다.

보상(실시간)에서의 표준 수식:
    y_pred_raw = y_pred_norm * Y_SCALE + BASE_VALUE
    y_comp     = y_meas - (y_pred_raw - BASE_VALUE)
              = y_meas - y_pred_raw + BASE_VALUE
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


# =============================================================================
# Normalization Spec (Output-only)
# =============================================================================

@dataclass(frozen=True)
class OutputNormSpec:
    """Output normalization spec: baseline + scale."""
    baseline: float = 4e7
    scale: float = 1e6

    def normalize(self, y_raw: torch.Tensor) -> torch.Tensor:
        return (y_raw - self.baseline) / self.scale

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.scale + self.baseline


# =============================================================================
# Stage 1 (Unified): Self-Detection Field MLP
# =============================================================================

class Stage1SelfDetectionFieldMLP(nn.Module):
    """
    Stage 1: Self-Detection Field Model

    Input : q (6 joint angles), shape (B, 6)   [NO input normalization]
    Output: y_pred_norm (N sensors), shape (B, N)
            where y_pred_norm ≈ (y_raw - BASE_VALUE) / Y_SCALE
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 8,
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
        self.dropout = float(dropout)

        act = {
            "relu": nn.ReLU(inplace=True),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        }.get(activation, nn.ReLU(inplace=True))

        layers: List[nn.Module] = []
        d = input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(act)
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            d = h
        layers.append(nn.Linear(d, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: (B, 6) joint angles (raw, not normalized)
        Returns:
            y_pred_norm: (B, output_dim)
        """
        return self.net(q)

    def get_config(self) -> Dict:
        return {
            "model": "Stage1SelfDetectionFieldMLP",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
            "dropout": self.dropout,
        }


class Stage1SelfDetectionTrunkHeadMLP(nn.Module):
    """
    Stage 1 (optional): Trunk + Head 구조 (채널별 특성 분리)

    - Trunk: 공유 representation
    - Head : 채널별 1D 회귀기

    Input : (B, 6)
    Output: (B, N) normalized
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 8,
        trunk_hidden: Tuple[int, ...] = (256, 256),
        trunk_dim: int = 128,
        head_hidden: int = 64,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.trunk_hidden = tuple(trunk_hidden)
        self.trunk_dim = int(trunk_dim)
        self.head_hidden = int(head_hidden)
        self.activation = activation
        self.dropout = float(dropout)

        act = {
            "relu": nn.ReLU(inplace=True),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        }.get(activation, nn.ReLU(inplace=True))

        # trunk
        layers: List[nn.Module] = []
        d = input_dim
        for h in self.trunk_hidden:
            layers.append(nn.Linear(d, h))
            layers.append(act)
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            d = h
        layers.append(nn.Linear(d, self.trunk_dim))
        layers.append(act)
        self.trunk = nn.Sequential(*layers)

        # heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.trunk_dim, self.head_hidden),
                act,
                nn.Linear(self.head_hidden, 1),
            )
            for _ in range(output_dim)
        ])

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        z = self.trunk(q)  # (B, trunk_dim)
        ys = [head(z) for head in self.heads]  # list[(B,1)]
        return torch.cat(ys, dim=1)  # (B, N)

    def get_config(self) -> Dict:
        return {
            "model": "Stage1SelfDetectionTrunkHeadMLP",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "trunk_hidden": list(self.trunk_hidden),
            "trunk_dim": self.trunk_dim,
            "head_hidden": self.head_hidden,
            "activation": self.activation,
            "dropout": self.dropout,
        }


# =============================================================================
# Stage 2 (Residual-only): Memory MLP / Residual TCN
# =============================================================================

class Stage2ResidualMemoryMLP(nn.Module):
    """
    Stage 2: Residual-only Memory MLP

    Input : residual history only (NO joint angles, NO raw)
            r_hist shape (B, K, N) or (B, K*N)
    Output: residual correction r_hat(t) (B, N)

    NOTE:
    - 이 Stage2는 '손 신호를 지우지 않기' 위해
      구조적으로 raw-following 되지 않게(작게) 유지해야 한다.
    """

    def __init__(
        self,
        memory_window: int = 5,
        output_dim: int = 8,
        hidden_dim: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.memory_window = int(memory_window)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        in_dim = self.memory_window * self.output_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, residual_history: torch.Tensor) -> torch.Tensor:
        if residual_history.dim() == 3:
            b = residual_history.shape[0]
            residual_history = residual_history.reshape(b, -1)
        return self.net(residual_history)

    def get_config(self) -> Dict:
        return {
            "model": "Stage2ResidualMemoryMLP",
            "memory_window": self.memory_window,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }


class TCNBlock(nn.Module):
    """Dilated causal conv block with residual connection."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation)
        self.act1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, padding=pad, dilation=dilation)
        self.act2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        y = self.conv1(x)
        y = y[..., :x.size(-1)]  # causal crop
        y = self.drop1(self.act1(y))

        y = self.conv2(y)
        y = y[..., :x.size(-1)]
        y = self.drop2(self.act2(y))

        return y + self.downsample(x)


class Stage2ResidualTCN(nn.Module):
    """
    Stage 2: Residual-only TCN

    Input : residual_seq shape (B, K, N)  (r(t-K) ... r(t-1))
    Output: r_hat(t) shape (B, N)
    """

    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 8,
        seq_len: int = 5,
        channels: Tuple[int, ...] = (32, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.seq_len = int(seq_len)
        self.channels = tuple(channels)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

        layers: List[nn.Module] = []
        c_in = self.input_dim
        dilation = 1
        for c_out in self.channels:
            layers.append(TCNBlock(c_in, c_out, kernel_size=self.kernel_size, dilation=dilation, dropout=self.dropout))
            c_in = c_out
            dilation *= 2

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(c_in, self.output_dim, kernel_size=1)

    def forward(self, residual_seq: torch.Tensor) -> torch.Tensor:
        # (B, K, N) -> (B, N, K)
        x = residual_seq.transpose(1, 2)
        z = self.tcn(x)         # (B, C, K)
        y = self.head(z)        # (B, N, K)
        return y[..., -1]       # (B, N)

    def get_config(self) -> Dict:
        return {
            "model": "Stage2ResidualTCN",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "seq_len": self.seq_len,
            "channels": list(self.channels),
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
        }


# =============================================================================
# Optional Wrapper (Two-Stage)
# =============================================================================

class TwoStageCompensator(nn.Module):
    """
    Two-stage compensator wrapper.

    stage1: predicts normalized self-detection field ŝ_norm(q)
    stage2: predicts residual correction r̂(t) (same normalized scale), optional

    This module returns normalized predictions.
    Denormalize outside using OutputNormSpec.
    """

    def __init__(
        self,
        stage1: nn.Module,
        stage2: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2

    def forward(
        self,
        joint_angles: torch.Tensor,
        residual_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y1 = self.stage1(joint_angles)
        y2 = None
        if self.stage2 is not None and residual_history is not None:
            y2 = self.stage2(residual_history)
        return y1, y2


# =============================================================================
# Model Registry (Cleaned)
# =============================================================================

class ModelRegistry:
    """
    Clean registry aligned with "논문 계보" meaning.

    - stage1_field_mlp        : recommended default (256-256-128)
    - stage1_trunk_head_mlp   : optional per-channel head model
    - stage2_memory           : residual-only memory MLP
    - stage2_tcn              : residual-only TCN
    """

    MODELS = {
        "stage1_field_mlp": {
            "class": Stage1SelfDetectionFieldMLP,
            "description": "Stage1: Self-detection field MLP (q -> sensors, output normalized by baseline/scale)",
            "default_config": {
                "input_dim": 6,
                "output_dim": 8,
                "hidden_dims": (256, 256, 128),
                "activation": "relu",
                "dropout": 0.0,
            },
            "requires_sequence": False,
        },
        "stage1_trunk_head_mlp": {
            "class": Stage1SelfDetectionTrunkHeadMLP,
            "description": "Stage1: Trunk+Head self-detection MLP (q -> sensors, normalized)",
            "default_config": {
                "input_dim": 6,
                "output_dim": 8,
                "trunk_hidden": (256, 256),
                "trunk_dim": 128,
                "head_hidden": 64,
                "activation": "relu",
                "dropout": 0.0,
            },
            "requires_sequence": False,
        },
        "stage2_memory": {
            "class": Stage2ResidualMemoryMLP,
            "description": "Stage2: Residual-only memory MLP (residual history -> residual correction)",
            "default_config": {
                "memory_window": 5,
                "output_dim": 8,
                "hidden_dim": 12,
                "dropout": 0.0,
            },
            "requires_sequence": True,
        },
        "stage2_tcn": {
            "class": Stage2ResidualTCN,
            "description": "Stage2: Residual-only TCN (residual seq -> residual correction)",
            "default_config": {
                "input_dim": 8,
                "output_dim": 8,
                "seq_len": 5,
                "channels": (32, 32),
                "kernel_size": 3,
                "dropout": 0.1,
            },
            "requires_sequence": True,
        },
    }

    @classmethod
    def list_models(cls) -> None:
        print("\nAvailable Models:")
        print("-" * 80)
        for name, info in cls.MODELS.items():
            print(f"  {name:22s} : {info['description']}")
        print("-" * 80)

    @classmethod
    def get_model_info(cls, name: str) -> Dict:
        if name not in cls.MODELS:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls.MODELS.keys())}")
        return cls.MODELS[name]

    @classmethod
    def create_model(cls, name: str, **kwargs) -> nn.Module:
        info = cls.get_model_info(name)
        cfg = dict(info["default_config"])
        cfg.update(kwargs)
        return info["class"](**cfg)

    @classmethod
    def get_available_models(cls) -> List[str]:
        return list(cls.MODELS.keys())

    @classmethod
    def requires_sequence(cls, name: str) -> bool:
        return cls.get_model_info(name).get("requires_sequence", False)


# =============================================================================
# Backward compatibility shims (older code/checkpoints)
# =============================================================================

class SimpleMLP(Stage1SelfDetectionFieldMLP):
    """Backward compatible alias for older checkpoints/configs.

    Older code used `SimpleMLP` with hidden_dims=[256,256,128] and output in either:
    - z-score normalized space (with `norm_params` in checkpoint), or
    - raw space (rare)

    The architecture matches Stage1SelfDetectionFieldMLP.
    """

    def get_config(self) -> Dict:
        cfg = super().get_config()
        cfg["model"] = "SimpleMLP"
        return cfg


class Stage1StaticOffsetMLP(nn.Module):
    """Legacy Stage1 baseline MLP (kept for compatibility with older scripts).

    Architecture: (B,6) -> ... -> (B,output_dim)
    using a repeated hidden_dim for num_layers.
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 8,
        hidden_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        layers: List[nn.Module] = []
        d = self.input_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(d, self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            d = self.hidden_dim
        layers.append(nn.Linear(d, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.net(q)

    def get_config(self) -> Dict:
        return {
            "model": "Stage1StaticOffsetMLP",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
