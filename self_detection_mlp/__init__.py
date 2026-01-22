"""Self Detection MLP training package for capacitive proximity sensors.

Based on README specifications:
- Input: 6 joint angles (j1-j6)
- Output: 4 raw sensor values (raw1-raw4)
- Two-stage compensation architecture
"""

__version__ = "0.2.0"
__author__ = "Robot Sensing Lab"

from .data_loader import DataLoader
from .preprocessing import OutputNormalizer, prepare_training_data, save_normalization_params
from .model import Stage1StaticOffsetMLP, Stage2ResidualMemoryMLP, SelfDetectionMLP
from .trainer import Trainer
from .evaluator import Evaluator

__all__ = [
    "DataLoader",
    "OutputNormalizer",
    "prepare_training_data",
    "save_normalization_params",
    "Stage1StaticOffsetMLP",
    "Stage2ResidualMemoryMLP",
    "SelfDetectionMLP",
    "Trainer",
    "Evaluator",
]
