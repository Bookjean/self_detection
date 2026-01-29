"""Self Detection MLP training package for capacitive proximity sensors.

Based on train_self_detection.py, inference_self_detection.py, export_model.py structure:
- Input: 6 joint angles (j1-j6) - NO input normalization
- Output: N raw sensor values (raw1-rawN, N=8) - baseline/scale normalization
- Single-stage MLP architecture (Stage1SelfDetectionFieldMLP)
- Output normalization using OutputNormSpec (baseline/scale)
"""

__version__ = "0.3.0"
__author__ = "Robot Sensing Lab"

from .data_loader import DataLoader, DataLoaderFactory
from .model import (
    Stage1SelfDetectionFieldMLP,
    Stage1SelfDetectionTrunkHeadMLP,
    # Backward compatible names
    SimpleMLP,
    Stage1StaticOffsetMLP,
    Stage2ResidualMemoryMLP,
    Stage2ResidualTCN,
    OutputNormSpec,
)
from .trainer import Trainer
from .evaluator import Evaluator

__all__ = [
    "DataLoader",
    "DataLoaderFactory",
    "Stage1SelfDetectionFieldMLP",
    "Stage1SelfDetectionTrunkHeadMLP",
    "SimpleMLP",
    "Stage1StaticOffsetMLP",
    "Stage2ResidualMemoryMLP",
    "Stage2ResidualTCN",
    "OutputNormSpec",
    "Trainer",
    "Evaluator",
]
