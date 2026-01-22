"""Utility functions for Self Detection package."""

import numpy as np
import torch
from typing import Tuple


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device() -> str:
    """Get available device.
    
    Returns:
        "cuda" if GPU available, else "cpu"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def sensor_compensation(y_raw: np.ndarray,
                       y_baseline: np.ndarray) -> np.ndarray:
    """Apply baseline compensation to raw sensor readings.
    
    Formula: y_corrected = y_raw - y_baseline
    
    Args:
        y_raw: Raw sensor readings, shape (N, 4)
        y_baseline: Predicted baseline, shape (N, 4)
        
    Returns:
        Compensated readings, shape (N, 4)
    """
    return y_raw - y_baseline


def compute_improvement_ratio(metric_before: np.ndarray,
                              metric_after: np.ndarray) -> np.ndarray:
    """Compute improvement ratio (percentage).
    
    Args:
        metric_before: Metric before compensation, shape (4,)
        metric_after: Metric after compensation, shape (4,)
        
    Returns:
        Improvement ratio in percentage, shape (4,)
    """
    improvement = (metric_before - metric_after) / (metric_before + 1e-8) * 100
    return np.clip(improvement, 0, 100)  # Clip to [0, 100]%


class DataNormalizer:
    """General data normalizer (min-max or z-score)."""
    
    def __init__(self, method: str = "zscore"):
        """Initialize normalizer.
        
        Args:
            method: "zscore" or "minmax"
        """
        self.method = method
        self.params = {}
    
    def fit(self, data: np.ndarray) -> None:
        """Fit normalizer on data.
        
        Args:
            data: Shape (N, D) or (N,)
        """
        if self.method == "zscore":
            self.params["mean"] = np.mean(data, axis=0)
            self.params["std"] = np.std(data, axis=0)
            self.params["std"][self.params["std"] == 0] = 1.0
        elif self.method == "minmax":
            self.params["min"] = np.min(data, axis=0)
            self.params["max"] = np.max(data, axis=0)
            self.params["max"] = np.where(
                self.params["max"] == self.params["min"],
                self.params["min"] + 1,
                self.params["max"]
            )
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data.
        
        Args:
            data: Shape (N, D) or (N,)
            
        Returns:
            Normalized data
        """
        if self.method == "zscore":
            return (data - self.params["mean"]) / self.params["std"]
        elif self.method == "minmax":
            return (data - self.params["min"]) / (self.params["max"] - self.params["min"])
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Transform back to original scale.
        
        Args:
            data: Normalized data
            
        Returns:
            Original scale data
        """
        if self.method == "zscore":
            return data * self.params["std"] + self.params["mean"]
        elif self.method == "minmax":
            return data * (self.params["max"] - self.params["min"]) + self.params["min"]
