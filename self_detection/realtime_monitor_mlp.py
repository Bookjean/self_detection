#!/usr/bin/env python3
"""
Real-time Self Detection Monitoring Node - MLP Only (Stage 1)

Compensation:
    y_corrected = raw - b̂(q)

Where:
    b̂(q) = MLP prediction (posture-dependent baseline)

Logging:
    ~/rb10_Proximity/logs/compensated_mlp_*.txt
"""

import os
import glob
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState, Range


try:
    import torch
    from self_detection_mlp.model import Stage1StaticOffsetMLP
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"[WARN] PyTorch not available: {e}")


class RealtimeMonitorMLP(Node):
    """MLP-only self detection compensation node."""

    def __init__(self):
        super().__init__('realtime_monitor_mlp')

        # Parameters
        self.declare_parameter('model_file', '')
        self.declare_parameter('log_rate', 100.0)  # Hz

        self.cb_group = ReentrantCallbackGroup()

        # Data
        self.raw_data = [0.0] * 4
        self.joint_states = None
        self.joint_names = None
        self.raw_received = False

        # Model
        self.model = None
        self.norm_params = None

        # Load model
        if TORCH_AVAILABLE:
            self._load_model()

        # Subscribers
        for i in range(1, 5):
            self.create_subscription(
                Range,
                f'/raw_distance{i}',
                lambda msg, idx=i-1: self.raw_callback(msg, idx),
                10,
                callback_group=self.cb_group
            )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10,
            callback_group=self.cb_group
        )

        # Logging
        log_rate = self.get_parameter('log_rate').value
        log_dir = os.path.expanduser('~/rb10_Proximity/logs')
        os.makedirs(log_dir, exist_ok=True)

        ts = self.get_clock().now().nanoseconds
        self.log_path = os.path.join(log_dir, f'compensated_mlp_{ts}.txt')
        self.log_file = open(self.log_path, 'w')
        self.log_file.write('# raw1 raw2 raw3 raw4 (MLP compensated)\n')

        self.log_timer = self.create_timer(
            1.0 / log_rate,
            self.log_callback,
            callback_group=self.cb_group
        )

        self.get_logger().info('=' * 60)
        self.get_logger().info('MLP-Only Self Detection Realtime Monitor')
        self.get_logger().info(f'Model loaded: {self.model is not None}')
        self.get_logger().info(f'Logging @{log_rate}Hz -> {self.log_path}')
        self.get_logger().info('=' * 60)

    def _load_model(self):
        """Load MLP model from file."""
        model_dir = os.path.expanduser('~/rb10_Proximity/src/self_detection/models')

        # Check parameter first
        model_file = self.get_parameter('model_file').value
        if model_file:
            path = os.path.join(model_dir, model_file)
            if os.path.exists(path):
                self._load_checkpoint(path)
                return

        # Auto-find latest stage1 model
        patterns = [
            os.path.join(model_dir, 'stage1_*.pth'),
            os.path.join(model_dir, 'stage1_*.pt'),
        ]

        model_files = []
        for pattern in patterns:
            model_files.extend(glob.glob(pattern))

        if model_files:
            latest = sorted(model_files)[-1]
            self._load_checkpoint(latest)
        else:
            self.get_logger().warn(f'No model found in {model_dir}')

    def _load_checkpoint(self, path: str):
        """Load model checkpoint."""
        self.get_logger().info(f'Loading model: {path}')

        ckpt = torch.load(path, map_location='cpu', weights_only=False)

        # Get model config
        model_cfg = ckpt.get('model_config', {})
        self.norm_params = ckpt.get('norm_params', None)

        # Create model
        self.model = Stage1StaticOffsetMLP(
            input_dim=model_cfg.get('input_dim', 6),
            output_dim=model_cfg.get('output_dim', 4),
            hidden_dim=model_cfg.get('hidden_dim', 32),
            num_layers=model_cfg.get('num_layers', 3),
        )

        # Load weights
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.model.eval()

        self.get_logger().info(f'Model loaded successfully')
        if self.norm_params:
            self.get_logger().info(f'Normalization params loaded')

    def raw_callback(self, msg, idx):
        """Raw sensor callback."""
        self.raw_data[idx] = msg.range
        self.raw_received = True

    def joint_callback(self, msg):
        """Joint state callback."""
        self.joint_states = np.array(msg.position)
        self.joint_names = msg.name

    def _run_inference(self):
        """Run MLP inference and compute compensation."""
        if not self.raw_received or self.joint_states is None:
            return None

        if self.model is None:
            return None

        # Get joint angles (first 6)
        joints = self.joint_states[:6]

        # Normalize input if norm_params available
        if self.norm_params is not None:
            joints_norm = (joints - self.norm_params['X_mean']) / (self.norm_params['X_std'] + 1e-8)
        else:
            joints_norm = joints

        # Inference
        with torch.no_grad():
            x = torch.tensor(joints_norm, dtype=torch.float32).unsqueeze(0)
            y_pred_norm = self.model(x).squeeze().numpy()

        # Denormalize output
        if self.norm_params is not None:
            y_pred = y_pred_norm * self.norm_params['y_std'] + self.norm_params['y_mean']
        else:
            y_pred = y_pred_norm

        # Compensation: y_corrected = raw - baseline_prediction
        raw = np.array(self.raw_data)
        compensated = raw - y_pred

        return compensated

    def log_callback(self):
        """Timer callback for logging compensated values."""
        result = self._run_inference()
        if result is None:
            return

        self.log_file.write(
            f"{result[0]:.6f} {result[1]:.6f} {result[2]:.6f} {result[3]:.6f}\n"
        )

    def destroy_node(self):
        """Clean up on shutdown."""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            self.get_logger().info(f'Log saved: {self.log_path}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealtimeMonitorMLP()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
