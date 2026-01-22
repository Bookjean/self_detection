"""Training loop for Self Detection MLP."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path


class Trainer:
    """Trainer for Self Detection MLP.
    
    Handles training loop, validation, and model checkpointing.
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use ("cpu" or "cuda")
        """
        self.model = model.to(device)
        self.device = device
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "epoch": []
        }
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-5,
              patience: int = 20,
              verbose: bool = True) -> Dict:
        """Train the model.
        
        Args:
            X_train: Training input, shape (N, 10)
            y_train: Training target (normalized), shape (N, 4)
            X_val: Validation input, shape (M, 10)
            y_val: Validation target (normalized), shape (M, 4)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).float().to(self.device)
        X_val_t = torch.from_numpy(X_val).float().to(self.device)
        y_val_t = torch.from_numpy(y_val).float().to(self.device)
        
        # Setup
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=learning_rate, 
                                     weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            indices = np.arange(len(X_train_t))
            np.random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X_train_t[batch_idx]
                y_batch = y_train_t[batch_idx]
                
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            train_loss /= num_batches
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                y_val_pred = self.model(X_val_t)
                val_loss = criterion(y_val_pred, y_val_t).item()
            
            # Logging
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.history
    
    def get_history(self) -> Dict:
        """Get training history.
        
        Returns:
            History dictionary
        """
        return self.history.copy()
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.model.get_config()
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
