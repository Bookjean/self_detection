"""Data loading utilities for Self Detection training."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import json
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader


class DataLoader:
    """Load CSV log data for Self Detection training.

    Expected data format (CSV or TXT with # comments):
        timestamp,
        j1, j2, j3, j4, j5, j6,
        prox1, prox2, prox3, prox4, prox5, prox6, prox7, prox8,
        raw1, raw2, raw3, raw4, raw5, raw6, raw7, raw8,
        tof1, tof2, tof3, tof4, tof5, tof6, tof7, tof8
    """

    COLUMN_NAMES = [
        "timestamp",
        "j1", "j2", "j3", "j4", "j5", "j6",
        "prox1", "prox2", "prox3", "prox4", "prox5", "prox6", "prox7", "prox8",
        "raw1", "raw2", "raw3", "raw4", "raw5", "raw6", "raw7", "raw8",
        "tof1", "tof2", "tof3", "tof4", "tof5", "tof6", "tof7", "tof8"
    ]
    JOINT_COLUMNS = ["j1", "j2", "j3", "j4", "j5", "j6"]
    RAW_SENSOR_COLUMNS = ["raw1", "raw2", "raw3", "raw4", "raw5", "raw6", "raw7", "raw8"]
    PROX_SENSOR_COLUMNS = ["prox1", "prox2", "prox3", "prox4", "prox5", "prox6", "prox7", "prox8"]
    TOF_COLUMNS = ["tof1", "tof2", "tof3", "tof4", "tof5", "tof6", "tof7", "tof8"]

    def __init__(self, file_path: str):
        """Initialize DataLoader.

        Args:
            file_path: Path to CSV/TXT log file
        """
        self.file_path = Path(file_path)
        self.data = None
        self.norm_params: Dict[str, Dict[str, np.ndarray]] = {}

    def load(self) -> pd.DataFrame:
        """Load CSV/TXT file.

        Automatically handles:
        - Comment lines starting with #
        - Files with or without header
        - Whitespace around values

        Returns:
            Loaded DataFrame
        """
        # Try to detect if file has header by reading first non-comment line
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    first_data_line = line
                    break

        # Check if first data line looks like a header (contains letters)
        has_header = any(c.isalpha() for c in first_data_line.split(',')[0])

        if has_header:
            self.data = pd.read_csv(
                self.file_path,
                comment='#',
                skipinitialspace=True
            )
        else:
            # No header, use predefined column names
            self.data = pd.read_csv(
                self.file_path,
                comment='#',
                header=None,
                names=self.COLUMN_NAMES,
                skipinitialspace=True
            )

        # Strip whitespace from column names if present
        self.data.columns = self.data.columns.str.strip()

        return self.data
    
    def get_joint_angles(self, use_all_joints: bool = True) -> np.ndarray:
        """Extract joint angles.

        Args:
            use_all_joints: If True, return all 6 joints (j1-j6).
                           If False, return j2-j6 only (legacy behavior).

        Returns:
            Shape: (N, 6) if use_all_joints else (N, 5)
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        if use_all_joints:
            return self.data[self.JOINT_COLUMNS].values  # j1-j6, shape (N, 6)
        else:
            return self.data[["j2", "j3", "j4", "j5", "j6"]].values  # j2-j6, shape (N, 5)
    
    def get_raw_sensors(self, num_sensors: int = 4) -> np.ndarray:
        """Extract AKF-processed sensor outputs (raw1-raw8 or raw1-raw4).
        
        Args:
            num_sensors: Number of sensors to extract (4 or 8). Default: 4 for backward compatibility.
        
        Returns:
            Shape: (N, num_sensors) - raw1~raw4 if num_sensors=4, raw1~raw8 if num_sensors=8
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        if num_sensors == 4:
            # Backward compatibility: return first 4 sensors
            columns = ["raw1", "raw2", "raw3", "raw4"]
        elif num_sensors == 8:
            columns = self.RAW_SENSOR_COLUMNS
        else:
            raise ValueError(f"num_sensors must be 4 or 8, got {num_sensors}")
        
        return self.data[columns].values
    
    def get_timestamps(self) -> np.ndarray:
        """Extract timestamps.

        Handles format: "YYYY-MM-DD HHMMSS.mmm" (e.g., "2026-01-16 163828.867")

        Returns:
            Shape: (N,) - numpy datetime64 array
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        timestamps_raw = self.data["timestamp"].astype(str).str.strip()

        def parse_timestamp(ts: str) -> pd.Timestamp:
            """Parse custom timestamp format."""
            try:
                # Try standard format first
                return pd.to_datetime(ts)
            except Exception:
                pass

            # Handle "YYYY-MM-DD HHMMSS.mmm" format
            try:
                parts = ts.split()
                if len(parts) == 2:
                    date_part = parts[0]
                    time_part = parts[1]

                    # Parse time: HHMMSS.mmm
                    if '.' in time_part:
                        time_main, ms = time_part.split('.')
                    else:
                        time_main = time_part
                        ms = "0"

                    # Pad time_main to 6 digits (HHMMSS)
                    time_main = time_main.zfill(6)
                    hh = time_main[0:2]
                    mm = time_main[2:4]
                    ss = time_main[4:6]

                    formatted = f"{date_part} {hh}:{mm}:{ss}.{ms}"
                    return pd.to_datetime(formatted)
            except Exception:
                pass

            raise ValueError(f"Cannot parse timestamp: {ts}")

        timestamps = timestamps_raw.apply(parse_timestamp)
        return timestamps.values

    def get_timestamp_deltas(self) -> np.ndarray:
        """Compute time deltas in seconds.

        Returns:
            Shape: (N-1,). Time difference between consecutive samples in seconds.
        """
        timestamps = self.get_timestamps()
        # Convert to seconds since epoch
        timestamps_seconds = timestamps.astype('datetime64[ns]').astype(np.int64) / 1e9
        deltas = np.diff(timestamps_seconds)
        return deltas

    def normalize(self, data: np.ndarray, name: str, fit: bool = True) -> np.ndarray:
        """Z-score 정규화 (Standardization).

        Args:
            data: 정규화할 데이터. Shape: (N, D)
            name: 정규화 파라미터 저장 키 (예: "joints", "sensors")
            fit: True면 mean/std 계산 후 저장, False면 기존 파라미터 사용

        Returns:
            정규화된 데이터. Shape: (N, D)
        """
        if fit:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std = np.where(std == 0, 1.0, std)  # 0으로 나누기 방지
            self.norm_params[name] = {"mean": mean, "std": std}
        else:
            if name not in self.norm_params:
                raise RuntimeError(f"정규화 파라미터 '{name}'이 없습니다. fit=True로 먼저 호출하세요.")
            mean = self.norm_params[name]["mean"]
            std = self.norm_params[name]["std"]

        return (data - mean) / std

    def denormalize(self, data: np.ndarray, name: str) -> np.ndarray:
        """Z-score 역정규화.

        Args:
            data: 역정규화할 데이터. Shape: (N, D) 또는 (D,)
            name: 정규화 파라미터 키

        Returns:
            원래 스케일로 복원된 데이터
        """
        if name not in self.norm_params:
            raise RuntimeError(f"정규화 파라미터 '{name}'이 없습니다.")

        mean = self.norm_params[name]["mean"]
        std = self.norm_params[name]["std"]

        return data * std + mean

    def save_norm_params(self, path: str) -> None:
        """정규화 파라미터를 JSON 파일로 저장.

        Args:
            path: 저장할 파일 경로
        """
        params_to_save = {}
        for name, params in self.norm_params.items():
            params_to_save[name] = {
                "mean": params["mean"].tolist(),
                "std": params["std"].tolist()
            }

        with open(path, 'w') as f:
            json.dump(params_to_save, f, indent=2)

    def load_norm_params(self, path: str) -> None:
        """JSON 파일에서 정규화 파라미터 로드.

        Args:
            path: 로드할 파일 경로
        """
        with open(path, 'r') as f:
            params_loaded = json.load(f)

        self.norm_params = {}
        for name, params in params_loaded.items():
            self.norm_params[name] = {
                "mean": np.array(params["mean"]),
                "std": np.array(params["std"])
            }


# ============================================================================
# PyTorch Dataset Classes (Row-wise Shuffling Support)
# ============================================================================

class SelfDetectionDataset(Dataset):
    """PyTorch Dataset for Self Detection training.

    Each sample is a complete row (time step) containing:
    - Joint angles (j1-j6)
    - Raw sensor values (raw1-raw4)

    Row-wise shuffling preserves the correspondence between
    joint angles and sensor values at each time step.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Initialize dataset.

        Args:
            X: Input features (e.g., joint angles), shape (N, input_dim)
            y: Target values (e.g., raw sensors), shape (N, output_dim)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    """Dataset for sequence/window-based models (e.g., Stage 2).

    Creates sliding window sequences while preserving temporal order
    within each window.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int = 5):
        """Initialize sequence dataset.

        Args:
            X: Input features, shape (N, input_dim)
            y: Target values, shape (N, output_dim)
            window_size: Number of past samples to include
        """
        self.window_size = window_size
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

        # Valid indices (need enough history)
        self.valid_indices = list(range(window_size, len(X)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        actual_idx = self.valid_indices[idx]
        # Get window of past X values
        X_window = self.X[actual_idx - self.window_size:actual_idx]  # (window_size, input_dim)
        y_target = self.y[actual_idx]  # (output_dim,)
        return X_window, y_target


class DataLoaderFactory:
    """Factory for creating data loaders with various configurations.

    Supports:
    - Row-wise shuffling (preserves time step integrity)
    - Train/val split
    - Batch loading
    - Multiple dataset types
    """

    AVAILABLE_LOADERS = {
        "standard": "Standard row-wise shuffled DataLoader",
        "sequence": "Sequence/window-based DataLoader for temporal models",
        "no_shuffle": "No shuffling (preserves time order)",
    }

    def __init__(self, file_path: str, num_sensors: int = 4):
        """Initialize factory with data file.

        Args:
            file_path: Path to CSV/TXT data file
            num_sensors: Number of sensors to use (4 or 8). Default: 4 for backward compatibility.
        """
        self.file_path = file_path
        self.num_sensors = num_sensors
        self.data_loader = DataLoader(file_path)
        self.data_loader.load()

        # Extract data
        self.joint_angles = self.data_loader.get_joint_angles(use_all_joints=True)
        self.raw_sensors = self.data_loader.get_raw_sensors(num_sensors=num_sensors)

        # Normalization parameters
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def get_info(self) -> Dict:
        """Get dataset information."""
        return {
            "file": str(self.file_path),
            "num_samples": len(self.joint_angles),
            "input_shape": self.joint_angles.shape,
            "output_shape": self.raw_sensors.shape,
        }

    def normalize(self, X: np.ndarray, y: np.ndarray,
                  fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize data (Z-score standardization).

        Args:
            X: Input features
            y: Target values
            fit: If True, compute and save statistics. If False, use saved stats.

        Returns:
            Normalized (X, y)
        """
        if fit:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.X_std = np.where(self.X_std == 0, 1.0, self.X_std)

            self.y_mean = np.mean(y, axis=0)
            self.y_std = np.std(y, axis=0)
            self.y_std = np.where(self.y_std == 0, 1.0, self.y_std)

        X_norm = (X - self.X_mean) / self.X_std
        y_norm = (y - self.y_mean) / self.y_std

        return X_norm, y_norm

    def denormalize_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Denormalize target values."""
        return y_norm * self.y_std + self.y_mean

    def get_norm_params(self) -> Dict:
        """Get normalization parameters (output only)."""
        return {
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }

    def get_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get raw (unnormalized) data.

        Returns:
            (X, y): Joint angles and raw sensors, both unnormalized
        """
        return self.joint_angles.copy(), self.raw_sensors.copy()

    def create_dataloaders(
        self,
        loader_type: str = "standard",
        train_ratio: float = 0.7,
        batch_size: int = 64,
        shuffle_train: bool = True,
        normalize: bool = True,
        window_size: int = 5,
        seed: int = 42,
    ) -> Tuple[TorchDataLoader, TorchDataLoader, Dict]:
        """Create train and validation DataLoaders.

        Args:
            loader_type: Type of loader ("standard", "sequence", "no_shuffle")
            train_ratio: Ratio of data for training
            batch_size: Batch size
            shuffle_train: Whether to shuffle training data
            normalize: Whether to normalize data
            window_size: Window size for sequence loader
            seed: Random seed for reproducibility

        Returns:
            (train_loader, val_loader, info_dict)
        """
        np.random.seed(seed)

        X = self.joint_angles.copy()
        y = self.raw_sensors.copy()

        # Split indices
        n_samples = len(X)
        n_train = int(n_samples * train_ratio)

        if loader_type == "no_shuffle":
            # Time-ordered split (no randomization)
            train_idx = np.arange(n_train)
            val_idx = np.arange(n_train, n_samples)
        else:
            # Random split (shuffle indices for train/val split)
            indices = np.random.permutation(n_samples)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Normalize only output (sensor values), not input (joint angles)
        if normalize:
            # Only normalize y (sensor values), keep X (joint angles) as is
            self.y_mean = np.mean(y_train, axis=0)
            self.y_std = np.std(y_train, axis=0)
            self.y_std = np.where(self.y_std == 0, 1.0, self.y_std)
            
            y_train = (y_train - self.y_mean) / self.y_std
            y_val = (y_val - self.y_mean) / self.y_std

        # Create datasets
        if loader_type == "sequence":
            train_dataset = SequenceDataset(X_train, y_train, window_size)
            val_dataset = SequenceDataset(X_val, y_val, window_size)
            shuffle_train = shuffle_train  # Can shuffle sequence indices
        else:
            train_dataset = SelfDetectionDataset(X_train, y_train)
            val_dataset = SelfDetectionDataset(X_val, y_val)

        # Create DataLoaders
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            drop_last=False,
        )

        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        info = {
            "loader_type": loader_type,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "batch_size": batch_size,
            "shuffle_train": shuffle_train,
            "normalized": normalize,
            "train_val_split_randomized": loader_type != "no_shuffle",  # 랜덤화 여부 표시
        }

        return train_loader, val_loader, info

    @classmethod
    def list_loaders(cls) -> None:
        """Print available loader types."""
        print("\nAvailable DataLoader Types:")
        print("-" * 50)
        for name, desc in cls.AVAILABLE_LOADERS.items():
            print(f"  {name:15s} : {desc}")
        print()
