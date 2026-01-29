"""Training utilities for Self Detection MLP (based on train_self_detection.py structure)."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .model import OutputNormSpec


class Trainer:
    """Trainer for Self Detection MLP (train_self_detection.py style).
    
    Features:
    - AdamW optimizer with ReduceLROnPlateau scheduler
    - Gradient clipping
    - Early stopping
    - History tracking (train_loss, val_loss, val_rmse_raw, learning_rate)
    - Output normalization using OutputNormSpec
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        norm_spec: Optional[OutputNormSpec] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use ("cpu" or "cuda")
            optimizer: Optimizer (if None, will create AdamW)
            scheduler: Learning rate scheduler (if None, will create ReduceLROnPlateau)
            norm_spec: Output normalization spec (if None, will use default)
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.norm_spec = norm_spec or OutputNormSpec(baseline=4e7, scale=1e6)
        
        # Setup optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-3,
                weight_decay=1e-5,
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler if not provided
        if scheduler is None:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
            )
        else:
            self.scheduler = scheduler
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse_raw": [],
            "learning_rate": [],
        }
        self.criterion = nn.MSELoss()
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        grad_clip: float = 1.0,
    ) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            grad_clip: Gradient clipping max norm (0 to disable)
            
        Returns:
            Average training loss
        """
        self.model.train()
        train_losses = []
        
        for X, y_raw in train_loader:
            # Extract joint angles (first 6 columns)
            X = X[:, :6].to(self.device)
            y_raw = y_raw.to(self.device)
            
            # Normalize output using OutputNormSpec
            y_norm = self.norm_spec.normalize(y_raw)
            
            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y_norm)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=grad_clip
                )
            
            self.optimizer.step()
            train_losses.append(loss.item())
        
        return float(np.mean(train_losses)) if train_losses else float("inf")
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
    ) -> Tuple[float, float]:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            (val_loss_normalized, val_rmse_raw)
        """
        self.model.eval()
        val_losses = []
        rmse_list = []
        
        with torch.no_grad():
            for X, y_raw in val_loader:
                X = X[:, :6].to(self.device)
                y_raw = y_raw.to(self.device)
                
                # Normalize output
                y_norm = self.norm_spec.normalize(y_raw)
                
                # Forward pass
                y_pred = self.model(X)
                loss = self.criterion(y_pred, y_norm)
                val_losses.append(loss.item())
                
                # RMSE on raw scale (more interpretable)
                y_pred_raw = self.norm_spec.denormalize(y_pred)
                rmse = torch.sqrt(torch.mean((y_pred_raw - y_raw) ** 2)).item()
                rmse_list.append(rmse)
        
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        val_rmse_raw = float(np.mean(rmse_list)) if rmse_list else float("inf")
        
        return val_loss, val_rmse_raw
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 200,
        patience: int = 30,
        grad_clip: float = 1.0,
        verbose: bool = True,
        print_every: int = 5,
    ) -> Dict:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            patience: Early stopping patience
            grad_clip: Gradient clipping max norm
            verbose: Print training progress
            print_every: Print progress every N epochs
            
        Returns:
            Training history dictionary
        """
        best_val = float("inf")
        best_epoch = 0
        best_state = None
        wait = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, grad_clip=grad_clip)
            
            # Validate
            val_loss, val_rmse_raw = self.validate(val_loader)
            
            # Scheduler step (uses validation loss)
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr and verbose:
                print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_rmse_raw"].append(val_rmse_raw)
            self.history["learning_rate"].append(float(new_lr))
            
            # Print progress
            if verbose and ((epoch + 1) % print_every == 0 or epoch == 0):
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train {train_loss:.6e} | Val {val_loss:.6e} | "
                    f"ValRMSE(raw) {val_rmse_raw:.1f} | LR {new_lr:.2e}"
                )
            
            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return {
            "history": self.history,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
        }
    
    def get_history(self) -> Dict:
        """Get training history.
        
        Returns:
            History dictionary
        """
        return self.history.copy()
    
    def save_checkpoint(
        self,
        path: str,
        model_config: Dict,
        training_config: Dict,
        best_val_loss: float,
        best_epoch: int,
        history: Dict,
    ) -> None:
        """Save model checkpoint (train_self_detection.py format).
        
        Args:
            path: Path to save checkpoint
            model_config: Model configuration dict
            training_config: Training configuration dict
            best_val_loss: Best validation loss
            best_epoch: Best epoch index
            history: Training history
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": model_config,
                "output_norm": {
                    "baseline": float(self.norm_spec.baseline),
                    "scale": float(self.norm_spec.scale),
                },
                "training_config": training_config,
                "best_val_loss": float(best_val_loss),
                "best_epoch": int(best_epoch),
                "history": history,
            },
            path,
        )
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Update norm_spec if present
        if "output_norm" in checkpoint:
            on = checkpoint["output_norm"]
            self.norm_spec = OutputNormSpec(
                baseline=float(on.get("baseline", 4e7)),
                scale=float(on.get("scale", 1e6)),
            )
        
        return checkpoint
