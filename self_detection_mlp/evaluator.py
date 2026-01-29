"""Evaluation utilities for Self Detection model (based on inference_self_detection.py structure)."""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader

from .model import OutputNormSpec


class Evaluator:
    """Evaluate Self Detection model performance (inference_self_detection.py style).
    
    Metrics:
    - RMSE (Root Mean Squared Error) on raw scale
    - MAE (Mean Absolute Error) on raw scale
    - Per-channel metrics
    - Compensation metrics
    """
    
    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute RMSE.
        
        Args:
            y_true: True values, shape (N, C)
            y_pred: Predicted values, shape (N, C)
            
        Returns:
            RMSE value
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute MAE.
        
        Args:
            y_true: True values, shape (N, C)
            y_pred: Predicted values, shape (N, C)
            
        Returns:
            MAE value
        """
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def compute_compensation_metrics(
        y_raw: np.ndarray,
        y_pred_raw: np.ndarray,
        norm_spec: OutputNormSpec,
    ) -> Dict:
        """Compute compensation metrics (inference_self_detection.py style).
        
        Args:
            y_raw: Raw sensor measurements, shape (N, C)
            y_pred_raw: Predicted raw values, shape (N, C)
            norm_spec: Output normalization spec
            
        Returns:
            Dictionary with compensation metrics
        """
        # Compensation: comp = measured - (pred_raw - BASE)
        y_compensated = y_raw - (y_pred_raw - float(norm_spec.baseline))
        
        # Prediction error (raw scale)
        pred_rmse = Evaluator.compute_rmse(y_raw, y_pred_raw)
        pred_mae = Evaluator.compute_mae(y_raw, y_pred_raw)
        
        # Compensated signal statistics
        comp_std = float(np.std(y_compensated, axis=0).mean())
        raw_std = float(np.std(y_raw, axis=0).mean())
        std_reduction = raw_std - comp_std
        
        comp_ptp = float(np.ptp(y_compensated, axis=0).mean())
        raw_ptp = float(np.ptp(y_raw, axis=0).mean())
        ptp_reduction = raw_ptp - comp_ptp
        
        return {
            "pred_rmse": pred_rmse,
            "pred_mae": pred_mae,
            "compensated_std": comp_std,
            "raw_std": raw_std,
            "std_reduction": std_reduction,
            "compensated_ptp": comp_ptp,
            "raw_ptp": raw_ptp,
            "ptp_reduction": ptp_reduction,
        }
    
    @staticmethod
    @torch.no_grad()
    def evaluate_model(
        model: torch.nn.Module,
        dataloader: DataLoader,
        norm_spec: OutputNormSpec,
        device: torch.device,
    ) -> Dict:
        """Evaluate model on dataset (inference_self_detection.py style).
        
        Args:
            model: Trained model
            dataloader: Data loader
            norm_spec: Output normalization spec
            device: Device to use
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        
        all_y_raw = []
        all_y_pred_norm = []
        
        for X, y_raw in dataloader:
            X = X[:, :6].to(device)  # joint angles only
            y_raw = y_raw.to(device)
            
            # Predict (normalized)
            y_pred_norm = model(X)
            
            all_y_raw.append(y_raw.cpu())
            all_y_pred_norm.append(y_pred_norm.cpu())
        
        # Concatenate all batches
        y_raw = torch.cat(all_y_raw, dim=0).numpy()
        y_pred_norm = torch.cat(all_y_pred_norm, dim=0).numpy()
        
        # Denormalize predictions
        y_pred_raw = norm_spec.denormalize(torch.from_numpy(y_pred_norm)).numpy()
        
        # Compute metrics
        rmse = Evaluator.compute_rmse(y_raw, y_pred_raw)
        mae = Evaluator.compute_mae(y_raw, y_pred_raw)
        
        # Compensation metrics
        comp_metrics = Evaluator.compute_compensation_metrics(
            y_raw, y_pred_raw, norm_spec
        )
        
        return {
            "rmse": rmse,
            "mae": mae,
            **comp_metrics,
        }
    
    @staticmethod
    @torch.no_grad()
    def evaluate_per_channel(
        model: torch.nn.Module,
        dataloader: DataLoader,
        norm_spec: OutputNormSpec,
        device: torch.device,
        num_sensors: int = 8,
    ) -> list:
        """Evaluate model per channel (inference_self_detection.py style).
        
        Args:
            model: Trained model
            dataloader: Data loader
            norm_spec: Output normalization spec
            device: Device to use
            num_sensors: Number of sensors
            
        Returns:
            List of per-channel metrics
        """
        model.eval()
        
        all_y_raw = []
        all_y_pred_norm = []
        
        for X, y_raw in dataloader:
            X = X[:, :6].to(device)
            y_raw = y_raw.to(device)
            y_pred_norm = model(X)
            all_y_raw.append(y_raw.cpu())
            all_y_pred_norm.append(y_pred_norm.cpu())
        
        y_raw = torch.cat(all_y_raw, dim=0).numpy()
        y_pred_norm = torch.cat(all_y_pred_norm, dim=0).numpy()
        y_pred_raw = norm_spec.denormalize(torch.from_numpy(y_pred_norm)).numpy()
        
        results = []
        for ch in range(min(num_sensors, y_raw.shape[1])):
            ch_raw = y_raw[:, ch]
            ch_pred = y_pred_raw[:, ch]
            ch_error = np.abs(ch_raw - ch_pred)
            
            mae = float(np.mean(ch_error))
            rmse = float(np.sqrt(np.mean(ch_error ** 2)))
            max_err = float(np.max(ch_error))
            
            # Compensation metrics
            ch_comp = ch_raw - (ch_pred - float(norm_spec.baseline))
            comp_std = float(np.std(ch_comp))
            raw_std = float(np.std(ch_raw))
            std_reduction = raw_std - comp_std
            
            results.append({
                "channel": ch,
                "mae": mae,
                "rmse": rmse,
                "max_error": max_err,
                "raw_std": raw_std,
                "compensated_std": comp_std,
                "std_reduction": std_reduction,
            })
        
        return results
    
    @staticmethod
    def print_report(metrics: Dict, channel_metrics: Optional[list] = None) -> None:
        """Print evaluation report.
        
        Args:
            metrics: Overall metrics dictionary
            channel_metrics: Optional per-channel metrics list
        """
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        
        print("\nOverall Metrics:")
        print(f"  RMSE (raw): {metrics.get('rmse', 0):.1f}")
        print(f"  MAE (raw): {metrics.get('mae', 0):.1f}")
        print(f"  Prediction RMSE: {metrics.get('pred_rmse', 0):.1f}")
        print(f"  Prediction MAE: {metrics.get('pred_mae', 0):.1f}")
        
        print("\nCompensation Metrics:")
        print(f"  Raw std: {metrics.get('raw_std', 0):.1f}")
        print(f"  Compensated std: {metrics.get('compensated_std', 0):.1f}")
        print(f"  Std reduction: {metrics.get('std_reduction', 0):.1f}")
        print(f"  Raw ptp: {metrics.get('raw_ptp', 0):.1f}")
        print(f"  Compensated ptp: {metrics.get('compensated_ptp', 0):.1f}")
        print(f"  Ptp reduction: {metrics.get('ptp_reduction', 0):.1f}")
        
        if channel_metrics:
            print("\nPer-Channel Metrics:")
            for ch_metrics in channel_metrics:
                ch = ch_metrics.get("channel", 0)
                print(f"  Channel {ch+1}:")
                print(f"    MAE: {ch_metrics.get('mae', 0):.1f}")
                print(f"    RMSE: {ch_metrics.get('rmse', 0):.1f}")
                print(f"    Max error: {ch_metrics.get('max_error', 0):.1f}")
                print(f"    Std reduction: {ch_metrics.get('std_reduction', 0):.1f}")
        
        print("=" * 60 + "\n")
