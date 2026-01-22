"""Preprocessing utilities for Self Detection training.

Based on README specifications:
- Input: 6 joint angles (j1-j6) - NO velocity
- Output: 4 raw sensor values (raw1-raw4)
- Normalization: Output only (Z-score standardization)
"""

import numpy as np
import torch
from typing import Tuple, Dict


class OutputNormalizer:
    """Standardize and denormalize output channels (raw1-raw4).

    Normalization: y_norm = (y - mean) / std
    Denormalization: y = y_norm * std + mean
    """

    def __init__(self):
        """Initialize normalizer."""
        self.means = None
        self.stds = None
        self.is_fitted = False

    def fit(self, outputs: np.ndarray) -> None:
        """Compute statistics from training data.

        Args:
            outputs: Shape (N, 4) - raw1, raw2, raw3, raw4
        """
        self.means = np.mean(outputs, axis=0)  # Shape: (4,)
        self.stds = np.std(outputs, axis=0)    # Shape: (4,)

        # Avoid division by zero
        self.stds[self.stds == 0] = 1.0

        self.is_fitted = True

    def normalize(self, outputs: np.ndarray) -> np.ndarray:
        """Normalize outputs.

        Args:
            outputs: Shape (N, 4) or (4,)

        Returns:
            Normalized outputs with same shape
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        return (outputs - self.means) / self.stds

    def denormalize(self, outputs_norm: np.ndarray) -> np.ndarray:
        """Denormalize outputs.

        Args:
            outputs_norm: Shape (N, 4) or (4,)

        Returns:
            Denormalized outputs with same shape
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        return outputs_norm * self.stds + self.means

    def get_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get stored mean and std.

        Returns:
            (means, stds)
        """
        return self.means.copy(), self.stds.copy()

    def save_torch(self, path: str, prefix: str = "stage1") -> None:
        """Save normalization parameters as PyTorch tensors.

        Args:
            path: Path to save .pt file
            prefix: Prefix for parameter names (e.g., "stage1", "stage2_target")
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        torch.save({
            f"{prefix}_means": torch.tensor(self.means, dtype=torch.float64),
            f"{prefix}_stds": torch.tensor(self.stds, dtype=torch.float64),
        }, path)

    def load_torch(self, path: str, prefix: str = "stage1") -> None:
        """Load normalization parameters from PyTorch file.

        Args:
            path: Path to .pt file
            prefix: Prefix for parameter names
        """
        params = torch.load(path, map_location='cpu')
        self.means = params[f"{prefix}_means"].numpy()
        self.stds = params[f"{prefix}_stds"].numpy()
        self.is_fitted = True


def prepare_training_data(
    joint_angles: np.ndarray,
    raw_sensors: np.ndarray,
    train_ratio: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, OutputNormalizer]:
    """Prepare training and validation data for Stage 1 model.

    Based on README:
    - Input: 6 joint angles (j1-j6)
    - Output: 4 raw sensor values (normalized)
    - NO velocity computation

    Args:
        joint_angles: Shape (N, 6) - j1, j2, j3, j4, j5, j6
        raw_sensors: Shape (N, 4) - raw1, raw2, raw3, raw4
        train_ratio: Train/total data ratio (time-based split)

    Returns:
        (X_train, y_train_norm, X_val, y_val_norm, normalizer)
        - X shape: (M, 6) - joint angles only
        - y shape: (M, 4) - normalized raw sensor values
    """
    assert joint_angles.shape[0] == raw_sensors.shape[0], \
        "joint_angles and raw_sensors must have same number of samples"
    assert joint_angles.shape[1] == 6, \
        f"Expected 6 joint angles, got {joint_angles.shape[1]}"
    assert raw_sensors.shape[1] == 4, \
        f"Expected 4 raw sensors, got {raw_sensors.shape[1]}"

    # Input: joint angles only (no velocity)
    X = joint_angles  # (N, 6)
    y = raw_sensors   # (N, 4)

    # Time-based split (NOT random!)
    split_idx = int(len(X) * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Normalize outputs only
    normalizer = OutputNormalizer()
    normalizer.fit(y_train)
    y_train_norm = normalizer.normalize(y_train)
    y_val_norm = normalizer.normalize(y_val)

    return X_train, y_train_norm, X_val, y_val_norm, normalizer


def prepare_stage2_data(
    residuals: np.ndarray,
    memory_window: int = 5,
    train_ratio: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, OutputNormalizer]:
    """Prepare training data for Stage 2 residual memory model.

    Args:
        residuals: Shape (N, 4) - Stage 1 residuals (raw - predicted_baseline)
        memory_window: Number of past samples to use (K)
        train_ratio: Train/total data ratio

    Returns:
        (X_train, y_train_norm, X_val, y_val_norm, normalizer)
        - X shape: (M, memory_window, 4) - past residual history
        - y shape: (M, 4) - current residual (normalized)
    """
    N = len(residuals)

    # Create sliding window data
    X_list = []
    y_list = []

    for i in range(memory_window, N):
        # Input: past K residuals [r(t-K), ..., r(t-1)]
        X_list.append(residuals[i - memory_window:i])
        # Target: current residual r(t)
        y_list.append(residuals[i])

    X = np.array(X_list)  # (N-K, K, 4)
    y = np.array(y_list)  # (N-K, 4)

    # Time-based split
    split_idx = int(len(X) * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Normalize targets
    normalizer = OutputNormalizer()
    normalizer.fit(y_train)
    y_train_norm = normalizer.normalize(y_train)
    y_val_norm = normalizer.normalize(y_val)

    return X_train, y_train_norm, X_val, y_val_norm, normalizer


def save_normalization_params(
    stage1_normalizer: OutputNormalizer,
    path: str,
    stage2_normalizer: OutputNormalizer = None
) -> None:
    """Save all normalization parameters to a single file.

    Args:
        stage1_normalizer: Stage 1 output normalizer
        path: Path to save .pt file
        stage2_normalizer: Optional Stage 2 target normalizer
    """
    params = {
        "stage1_means": torch.tensor(stage1_normalizer.means, dtype=torch.float64),
        "stage1_stds": torch.tensor(stage1_normalizer.stds, dtype=torch.float64),
    }

    if stage2_normalizer is not None:
        params["stage2_target_means"] = torch.tensor(stage2_normalizer.means, dtype=torch.float64)
        params["stage2_target_stds"] = torch.tensor(stage2_normalizer.stds, dtype=torch.float64)

    torch.save(params, path)
    print(f"Saved normalization params to {path}")


def load_normalization_params(path: str) -> Dict[str, np.ndarray]:
    """Load normalization parameters from file.

    Args:
        path: Path to .pt file

    Returns:
        Dictionary with numpy arrays
    """
    params = torch.load(path, map_location='cpu')
    return {k: v.numpy() for k, v in params.items()}
