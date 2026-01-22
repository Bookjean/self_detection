"""MLP models for Self Detection - Two-Stage Compensation.

Based on README specifications:
- Input: 6 joint angles (j1-j6)
- Output: 4 sensor raw values (raw1-raw4)
- Stage 1: Static posture-dependent baseline model
- Stage 2: Short-term memory residual model
"""

import torch
import torch.nn as nn
from typing import Optional


class Stage1StaticOffsetMLP(nn.Module):
    """Stage 1: Static posture-dependent baseline model.

    Learns: b(q) = static offset as function of joint angles

    Architecture (from README):
        Input(6) → Dense(32) + ReLU → Dense(32) + ReLU → Dense(32) + ReLU → Dense(4)

    Purpose:
        Remove the main posture-dependent baseline fluctuations (36-73% variance reduction).
    """

    def __init__(self, input_dim: int = 6, output_dim: int = 4,
                 hidden_dim: int = 32, num_layers: int = 3, dropout: float = 0.0):
        """Initialize Stage 1 MLP.

        Args:
            input_dim: Input dimension (6 joint angles: j1-j6)
            output_dim: Output dimension (4 sensors: raw1-raw4)
            hidden_dim: Hidden layer dimension (default 32)
            num_layers: Number of hidden layers (default 3)
            dropout: Dropout rate (default 0.0, not used in README spec)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build network
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict static offset b(q).

        Args:
            x: Joint angles, shape (batch_size, 6)

        Returns:
            Predicted static offset (normalized), shape (batch_size, 4)
        """
        return self.net(x)

    def get_config(self) -> dict:
        return {
            "model": "Stage1StaticOffsetMLP",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }


class Stage2ResidualMemoryMLP(nn.Module):
    """Stage 2: Short-term memory residual model.

    Learns: r̂(t) = g(r(t-1), r(t-2), ..., r(t-K))

    Purpose:
        Remove hysteresis and dielectric relaxation effects from residual signal.

    Input: Residual history only (NOT raw signal, NOT joint angles)

    Design constraints (from README):
        - Small network (8-16 hidden units)
        - Short memory window (K=3-10 samples ≈ 30-100 ms)
        - Prevents raw signal following behavior
    """

    def __init__(self, memory_window: int = 5, output_dim: int = 4,
                 hidden_dim: int = 12, dropout: float = 0.0):
        """Initialize Stage 2 Memory MLP.

        Args:
            memory_window: Number of past residual samples to use (K=3-10)
            output_dim: Output dimension (4 sensors)
            hidden_dim: Hidden dimension (8-16 recommended)
            dropout: Dropout rate
        """
        super().__init__()
        self.memory_window = memory_window
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Input: memory_window * 4 (4 sensor residuals × K past samples)
        input_dim = memory_window * output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, residual_history: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict residual correction r̂(t).

        Args:
            residual_history: Past residuals, shape (batch_size, memory_window, 4)
                              or flattened (batch_size, memory_window * 4)

        Returns:
            Predicted residual correction, shape (batch_size, 4)
        """
        # Flatten if needed: (batch_size, memory_window, 4) → (batch_size, memory_window*4)
        if residual_history.dim() == 3:
            batch_size = residual_history.shape[0]
            residual_history = residual_history.reshape(batch_size, -1)

        return self.net(residual_history)

    def get_config(self) -> dict:
        return {
            "model": "Stage2ResidualMemoryMLP",
            "memory_window": self.memory_window,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
        }


class TwoStageCompensator(nn.Module):
    """Combined Two-Stage Self Detection Compensator.

    Compensation formula:
        y_corrected(t) = raw(t) - b̂(q(t)) - r̂(t)

    Where:
        - b̂(q) = Stage 1 output (static baseline)
        - r̂(t) = Stage 2 output (residual correction)
    """

    def __init__(self, stage1: Stage1StaticOffsetMLP, stage2: Stage2ResidualMemoryMLP = None):
        """Initialize Two-Stage Compensator.

        Args:
            stage1: Trained Stage 1 model
            stage2: Trained Stage 2 model (optional)
        """
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2

    def forward(self, joint_angles: torch.Tensor,
                residual_history: torch.Tensor = None) -> tuple:
        """Forward pass through both stages.

        Args:
            joint_angles: Current joint angles, shape (batch_size, 6)
            residual_history: Past residuals for Stage 2, shape (batch_size, K, 4)

        Returns:
            tuple: (stage1_output, stage2_output or None)
        """
        stage1_out = self.stage1(joint_angles)

        stage2_out = None
        if self.stage2 is not None and residual_history is not None:
            stage2_out = self.stage2(residual_history)

        return stage1_out, stage2_out


# Legacy alias for backward compatibility
class SelfDetectionMLP(Stage1StaticOffsetMLP):
    """Legacy alias for Stage1StaticOffsetMLP."""
    pass


# ============================================================================
# Additional Model Architectures
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP with configurable architecture.

    A basic feedforward network for quick experiments.
    """

    def __init__(self, input_dim: int = 6, output_dim: int = 4,
                 hidden_dims: list = [64, 32], activation: str = "relu"):
        """Initialize Simple MLP.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ("relu", "tanh", "leaky_relu")
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Select activation
        act_fn = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }.get(activation, nn.ReLU())

        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_config(self) -> dict:
        return {
            "model": "SimpleMLP",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
        }


class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block.

    Dilated causal convolution with residual connection.
    Based on "An Empirical Evaluation of Generic Convolutional and
    Recurrent Networks for Sequence Modeling" (Bai et al., 2018).
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.2):
        """Initialize TCN Block.

        Args:
            c_in: Input channels
            c_out: Output channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor for causal convolution
            dropout: Dropout rate
        """
        super().__init__()
        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=kernel_size,
                               padding=pad, dilation=dilation)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=kernel_size,
                               padding=pad, dilation=dilation)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # Residual connection (downsample if channels differ)
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, channels, seq_len)

        Returns:
            Output tensor, shape (batch, c_out, seq_len)
        """
        # First conv + causal crop
        y = self.conv1(x)
        y = y[..., :x.size(-1)]  # Causal: crop to input length
        y = self.drop1(self.act1(y))

        # Second conv + causal crop
        y = self.conv2(y)
        y = y[..., :x.size(-1)]
        y = self.drop2(self.act2(y))

        # Residual connection
        return y + self.downsample(x)


class Stage2ResidualTCN(nn.Module):
    """
    Stage 2: Residual-only Temporal Convolutional Network.

    Learns:
        r̂(t) = g(r(t-1), r(t-2), ..., r(t-K))

    ✔ Input  : residual history only (NO joint angles, NO raw)
    ✔ Output : residual correction at current time step
    ✔ Role   : Replace Stage2ResidualMemoryMLP with temporal model
    """

    def __init__(
        self,
        input_dim: int = 4,          # residual dimension (fixed)
        output_dim: int = 4,
        seq_len: int = 5,            # memory window K
        channels: tuple = (32, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.channels = channels
        self.kernel_size = kernel_size

        # -----------------------------
        # Build TCN blocks
        # -----------------------------
        layers = []
        c_in = input_dim
        dilation = 1

        for c_out in channels:
            layers.append(
                TCNBlock(
                    c_in=c_in,
                    c_out=c_out,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            c_in = c_out
            dilation *= 2  # exponential dilation

        self.tcn = nn.Sequential(*layers)

        # Output projection (1x1 conv)
        self.head = nn.Conv1d(c_in, output_dim, kernel_size=1)

    def forward(self, residual_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual_seq:
                shape (batch, seq_len, 4)
                r(t-K) ... r(t-1)

        Returns:
            residual correction r̂(t)
                shape (batch, 4)
        """
        # (B, T, C) -> (B, C, T)
        x = residual_seq.transpose(1, 2)

        # TCN
        z = self.tcn(x)              # (B, C', T)

        # Output head
        y = self.head(z)             # (B, 4, T)

        # Return last time step only
        return y[..., -1]

    def get_config(self) -> dict:
        return {
            "model": "Stage2ResidualTCN",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "seq_len": self.seq_len,
            "channels": self.channels,
            "kernel_size": self.kernel_size,
        }


class TCN(nn.Module):
    """Temporal Convolutional Network for sequence-to-one regression.

    Takes a sequence of inputs and predicts output at the last time step.
    Suitable for self-detection with temporal dependencies.

    Architecture:
        Input(seq_len, in_dim) -> TCNBlocks with increasing dilation -> Output(out_dim)
    """

    def __init__(self, input_dim: int = 6, output_dim: int = 4,
                 channels: tuple = (64, 64, 64, 64), kernel_size: int = 3,
                 dropout: float = 0.2):
        """Initialize TCN.

        Args:
            input_dim: Input feature dimension (e.g., 6 joint angles)
            output_dim: Output dimension (e.g., 4 raw sensors)
            channels: Tuple of channel sizes for each TCN block
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channels = channels
        self.kernel_size = kernel_size

        # Build TCN layers with exponentially increasing dilation
        layers = []
        c_in = input_dim
        dilation = 1
        for c_out in channels:
            layers.append(TCNBlock(c_in, c_out, kernel_size, dilation, dropout))
            c_in = c_out
            dilation *= 2  # Exponential dilation: 1, 2, 4, 8, ...

        self.net = nn.Sequential(*layers)

        # Output head: 1x1 conv to map to output dimension
        self.head = nn.Conv1d(c_in, output_dim, kernel_size=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_seq: Input sequence, shape (batch, seq_len, input_dim)

        Returns:
            Output at last time step, shape (batch, output_dim)
        """
        # Transpose: (B, T, C) -> (B, C, T) for Conv1d
        x = x_seq.transpose(1, 2)

        # TCN layers
        z = self.net(x)  # (B, channels[-1], T)

        # Output head
        y = self.head(z)  # (B, output_dim, T)

        # Return last time step only
        return y[..., -1]  # (B, output_dim)

    def get_config(self) -> dict:
        return {
            "model": "TCN",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "channels": self.channels,
            "kernel_size": self.kernel_size,
        }


class TCNLight(nn.Module):
    """Lightweight TCN for faster inference.

    Smaller architecture suitable for real-time applications.
    """

    def __init__(self, input_dim: int = 6, output_dim: int = 4,
                 channels: tuple = (32, 32), kernel_size: int = 3,
                 dropout: float = 0.1):
        """Initialize lightweight TCN."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channels = channels

        layers = []
        c_in = input_dim
        dilation = 1
        for c_out in channels:
            layers.append(TCNBlock(c_in, c_out, kernel_size, dilation, dropout))
            c_in = c_out
            dilation *= 2

        self.net = nn.Sequential(*layers)
        self.head = nn.Conv1d(c_in, output_dim, kernel_size=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x = x_seq.transpose(1, 2)
        z = self.net(x)
        y = self.head(z)
        return y[..., -1]

    def get_config(self) -> dict:
        return {
            "model": "TCNLight",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "channels": self.channels,
        }


class DeepMLP(nn.Module):
    """Deeper MLP with residual connections and batch normalization.

    For more complex patterns that need deeper networks.
    """

    def __init__(self, input_dim: int = 6, output_dim: int = 4,
                 hidden_dim: int = 64, num_layers: int = 5,
                 use_batch_norm: bool = True, dropout: float = 0.1):
        """Initialize Deep MLP.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Hidden layers with optional residual connections
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            self.layers.append(layer)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
        return self.output_proj(x)

    def get_config(self) -> dict:
        return {
            "model": "DeepMLP",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """Registry for available models.

    Allows easy selection and instantiation of different model architectures.
    """

    MODELS = {
        "stage1_mlp": {
            "class": Stage1StaticOffsetMLP,
            "description": "Stage 1: Static baseline MLP (6 joints -> 4 sensors)",
            "default_config": {
                "input_dim": 6,
                "output_dim": 4,
                "hidden_dim": 32,
                "num_layers": 3,
            },
            "requires_sequence": False,
        },
        "stage2_memory": {
            "class": Stage2ResidualMemoryMLP,
            "description": "Stage 2: Residual memory MLP for hysteresis correction",
            "default_config": {
                "memory_window": 5,
                "output_dim": 4,
                "hidden_dim": 12,
            },
            "requires_sequence": True,
        },
        "stage2_tcn": {
            "class": Stage2ResidualTCN,
            "description": "Stage 2: Residual TCN for temporal hysteresis correction",
            "default_config": {
                "input_dim": 4,
                "output_dim": 4,
                "seq_len": 5,
                "channels": (32, 32),
                "kernel_size": 3,
                "dropout": 0.1,
            },
            "requires_sequence": True,
        },
        "simple_mlp": {
            "class": SimpleMLP,
            "description": "Simple MLP with configurable hidden layers",
            "default_config": {
                "input_dim": 6,
                "output_dim": 4,
                "hidden_dims": [64, 32],
            },
            "requires_sequence": False,
        },
        "deep_mlp": {
            "class": DeepMLP,
            "description": "Deep MLP with residual connections and batch norm",
            "default_config": {
                "input_dim": 6,
                "output_dim": 4,
                "hidden_dim": 64,
                "num_layers": 5,
            },
            "requires_sequence": False,
        },
        "tcn": {
            "class": TCN,
            "description": "TCN: Temporal Convolutional Network (sequence model)",
            "default_config": {
                "input_dim": 6,
                "output_dim": 4,
                "channels": (64, 64, 64, 64),
                "kernel_size": 3,
                "dropout": 0.2,
            },
            "requires_sequence": True,
        },
        "tcn_light": {
            "class": TCNLight,
            "description": "TCN Light: Lightweight TCN for real-time inference",
            "default_config": {
                "input_dim": 6,
                "output_dim": 4,
                "channels": (32, 32),
                "kernel_size": 3,
                "dropout": 0.1,
            },
            "requires_sequence": True,
        },
    }

    @classmethod
    def list_models(cls) -> None:
        """Print available models."""
        print("\nAvailable Models:")
        print("-" * 60)
        for name, info in cls.MODELS.items():
            print(f"  {name:15s} : {info['description']}")
        print()

    @classmethod
    def get_model_info(cls, name: str) -> dict:
        """Get model information."""
        if name not in cls.MODELS:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls.MODELS.keys())}")
        return cls.MODELS[name]

    @classmethod
    def create_model(cls, name: str, **kwargs) -> nn.Module:
        """Create a model instance.

        Args:
            name: Model name from registry
            **kwargs: Override default config parameters

        Returns:
            Instantiated model
        """
        if name not in cls.MODELS:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls.MODELS.keys())}")

        model_info = cls.MODELS[name]
        config = model_info["default_config"].copy()
        config.update(kwargs)

        return model_info["class"](**config)

    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model names."""
        return list(cls.MODELS.keys())

    @classmethod
    def requires_sequence(cls, name: str) -> bool:
        """Check if model requires sequence input.

        Args:
            name: Model name

        Returns:
            True if model needs sequence data (e.g., TCN), False for single-step models
        """
        if name not in cls.MODELS:
            raise ValueError(f"Unknown model: {name}")
        return cls.MODELS[name].get("requires_sequence", False)
