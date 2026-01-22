"""Evaluation utilities for Self Detection model."""

import numpy as np
import torch
from typing import Dict, Tuple


class Evaluator:
    """Evaluate Self Detection model performance.
    
    Metrics:
    - Standard deviation reduction per channel
    - Peak-to-peak reduction per channel
    """
    
    @staticmethod
    def std_reduction(y_original: np.ndarray, 
                      y_compensated: np.ndarray) -> np.ndarray:
        """Compute standard deviation reduction per channel.
        
        Args:
            y_original: Original sensor values, shape (N, 4)
            y_compensated: Compensated sensor values (y_original - baseline), shape (N, 4)
            
        Returns:
            Std reduction per channel, shape (4,). Values > 0 indicate improvement.
        """
        std_original = np.std(y_original, axis=0)
        std_compensated = np.std(y_compensated, axis=0)
        
        return std_original - std_compensated
    
    @staticmethod
    def peak_to_peak_reduction(y_original: np.ndarray,
                               y_compensated: np.ndarray) -> np.ndarray:
        """Compute peak-to-peak reduction per channel.
        
        Args:
            y_original: Original sensor values, shape (N, 4)
            y_compensated: Compensated sensor values, shape (N, 4)
            
        Returns:
            Peak-to-peak reduction per channel, shape (4,).
        """
        ptp_original = np.ptp(y_original, axis=0)
        ptp_compensated = np.ptp(y_compensated, axis=0)
        
        return ptp_original - ptp_compensated
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error.
        
        Args:
            y_true: True values, shape (N, 4)
            y_pred: Predicted values, shape (N, 4)
            
        Returns:
            MSE value
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean absolute error.
        
        Args:
            y_true: True values, shape (N, 4)
            y_pred: Predicted values, shape (N, 4)
            
        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def evaluate(y_original: np.ndarray,
                y_baseline_pred: np.ndarray,
                y_baseline_true: np.ndarray = None) -> Dict:
        """Comprehensive evaluation.
        
        Args:
            y_original: Original sensor readings, shape (N, 4)
            y_baseline_pred: Predicted baseline, shape (N, 4)
            y_baseline_true: Ground truth baseline (optional), shape (N, 4)
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_compensated = y_original - y_baseline_pred
        
        results = {
            "std_reduction": Evaluator.std_reduction(y_original, y_compensated),
            "ptp_reduction": Evaluator.peak_to_peak_reduction(y_original, y_compensated),
            "original_std": np.std(y_original, axis=0),
            "compensated_std": np.std(y_compensated, axis=0),
            "original_ptp": np.ptp(y_original, axis=0),
            "compensated_ptp": np.ptp(y_compensated, axis=0),
        }
        
        if y_baseline_true is not None:
            results["baseline_mse"] = Evaluator.mse(y_baseline_true, y_baseline_pred)
            results["baseline_mae"] = Evaluator.mae(y_baseline_true, y_baseline_pred)
        
        return results
    
    @staticmethod
    def print_report(eval_dict: Dict, channel_names: list = None) -> None:
        """Print evaluation report.
        
        Args:
            eval_dict: Evaluation results from evaluate()
            channel_names: Optional channel names (default: raw1, raw2, raw3, raw4)
        """
        if channel_names is None:
            channel_names = ["raw1", "raw2", "raw3", "raw4"]
        
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print("\nStandard Deviation Reduction (larger is better):")
        for i, name in enumerate(channel_names):
            std_red = eval_dict["std_reduction"][i]
            print(f"  {name}: {std_red:.4f}")
        
        print("\nPeak-to-Peak Reduction (larger is better):")
        for i, name in enumerate(channel_names):
            ptp_red = eval_dict["ptp_reduction"][i]
            print(f"  {name}: {ptp_red:.4f}")
        
        print("\nOriginal vs Compensated Std Dev:")
        for i, name in enumerate(channel_names):
            orig = eval_dict["original_std"][i]
            comp = eval_dict["compensated_std"][i]
            print(f"  {name}: {orig:.4f} → {comp:.4f}")
        
        if "baseline_mse" in eval_dict:
            print(f"\nBaseline Prediction MSE: {eval_dict['baseline_mse']:.6f}")
            print(f"Baseline Prediction MAE: {eval_dict['baseline_mae']:.6f}")
        
        print("="*60 + "\n")
