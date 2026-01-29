#!/usr/bin/env python3
"""
Real-time Self Detection Monitoring Node
(Stage 1: Self-Detection Field MLP)

Based on inference_self_detection.py structure.

Model:
    ŝ_norm(q) ≈ (s_raw(q) - BASE) / SCALE

Compensation (inference_self_detection.py style):
    y_pred_raw = denormalize(y_pred_norm)
    y_comp = y_meas - (y_pred_raw - BASE)
"""

import os
import glob
import numpy as np
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState, Range

import torch

from self_detection_mlp.model import (
    Stage1SelfDetectionFieldMLP,
    OutputNormSpec,
)

# =========================
# Constants
# =========================
N_JOINTS = 6
N_SENSORS = 8
BASE_VALUE = 4e7
SCALE_VALUE = 1e6


class RealtimeMonitorMLP(Node):
    """Real-time self-detection monitor node (inference_self_detection.py style)."""

    def __init__(self):
        super().__init__('realtime_monitor_mlp')

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('model_file', '')
        self.declare_parameter('log_rate', 100.0)

        log_rate = float(self.get_parameter('log_rate').value)
        model_file = self.get_parameter('model_file').value

        self.cb_group = ReentrantCallbackGroup()

        # -------------------------
        # Data buffers
        # -------------------------
        self.raw_data = np.zeros(N_SENSORS, dtype=np.float32)
        self.joint_states = None
        self.raw_received = False

        # -------------------------
        # Normalization spec (will be loaded from checkpoint)
        # -------------------------
        self.norm_spec = OutputNormSpec(
            baseline=BASE_VALUE,
            scale=SCALE_VALUE,
        )
        # Legacy (z-score) normalization params (old checkpoints)
        self.legacy_norm_params = None
        self._ckpt_mode = "baseline_scale"  # or "zscore"

        # -------------------------
        # Load model (inference_self_detection.py style)
        # -------------------------
        self.model = None
        self.model_output_dim = N_SENSORS
        self.model_path = None
        self._load_model(model_file)

        if self.model is None:
            self.get_logger().error("=" * 60)
            self.get_logger().error("ERROR: Model load failed. Node aborted.")
            self.get_logger().error("=" * 60)
            self._model_load_failed = True
            return

        # -------------------------
        # Subscribers
        # -------------------------
        for i in range(N_SENSORS):
            self.create_subscription(
                Range,
                f'/raw_distance{i+1}',
                lambda msg, idx=i: self.raw_callback(msg, idx),
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

        # -------------------------
        # Publishers
        # -------------------------
        self.comp_pubs = []
        for i in range(N_SENSORS):
            self.comp_pubs.append(
                self.create_publisher(
                    Range,
                    f'/compensated_mlp_distance{i+1}',
                    10
                )
            )

        # -------------------------
        # Logging
        # -------------------------
        log_dir = os.path.expanduser('~/rb10_Proximity/logs')
        os.makedirs(log_dir, exist_ok=True)

        # Extract model name for log filename
        model_name = os.path.splitext(os.path.basename(self.model_path))[0] if self.model_path else "unknown"
        now = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f'compensated_mlp_{model_name}_{now}.txt'
        self.log_path = os.path.join(log_dir, filename)
        self.log_file = open(self.log_path, 'w')

        header = (
            "# timestamp "
            + " ".join([f"j{i+1}" for i in range(N_JOINTS)]) + " "
            + " ".join([f"raw{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"comp{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"pred{i+1}" for i in range(N_SENSORS)]) + "\n"
        )
        self.log_file.write(header)
        self.log_file.flush()

        # -------------------------
        # Timer
        # -------------------------
        self.timer = self.create_timer(
            1.0 / log_rate,
            self.timer_callback,
            callback_group=self.cb_group
        )

        self.get_logger().info("=" * 60)
        self.get_logger().info("Realtime Self Detection Monitor (Stage1)")
        self.get_logger().info(f"Model: {os.path.basename(self.model_path)}")
        self.get_logger().info(f"Model output dim: {self.model_output_dim}")
        self.get_logger().info(f"Normalization: {self._ckpt_mode}")
        self.get_logger().info(f"Log: {self.log_path}")
        self.get_logger().info("=" * 60)

    # ======================================================
    # Model loading (inference_self_detection.py style)
    # ======================================================
    def _load_model(self, model_file: str):
        """Load model checkpoint (inference_self_detection.py style)."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'models')

        if not model_file:
            self.get_logger().error("model_file parameter is empty.")
            return

        path = os.path.join(model_dir, model_file)
        if not os.path.exists(path):
            self.get_logger().error(f"Model not found: {path}")
            return

        # Load checkpoint (inference_self_detection.py style)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('model_config', {}) or {}
        self.model_output_dim = int(cfg.get('output_dim', N_SENSORS))
        input_dim = int(cfg.get('input_dim', N_JOINTS))
        hidden_dims = tuple(cfg.get('hidden_dims', (256, 256, 128)))

        # Determine normalization type from checkpoint (inference_self_detection.py style)
        if 'output_norm' in ckpt:
            # New checkpoints: baseline/scale normalization
            on = ckpt.get('output_norm') or {}
            self.norm_spec = OutputNormSpec(
                baseline=float(on.get('baseline', BASE_VALUE)),
                scale=float(on.get('scale', SCALE_VALUE)),
            )
            self.legacy_norm_params = None
            self._ckpt_mode = "baseline_scale"
            self.get_logger().info(
                f"Using baseline/scale normalization: "
                f"baseline={self.norm_spec.baseline}, scale={self.norm_spec.scale}"
            )
        elif 'norm_params' in ckpt:
            # Old checkpoints: z-score normalization
            np_ = ckpt.get('norm_params') or {}
            y_mean = np_.get('y_mean', None)
            y_std = np_.get('y_std', None)
            if y_mean is not None and y_std is not None:
                self.legacy_norm_params = {
                    'y_mean': np.array(y_mean, dtype=np.float32),
                    'y_std': np.array(y_std, dtype=np.float32),
                }
                self._ckpt_mode = "zscore"
                self.get_logger().info("Using z-score normalization (legacy)")
            else:
                self.legacy_norm_params = None
                self._ckpt_mode = "baseline_scale"
        else:
            # No normalization params found
            self.get_logger().warn("No normalization parameters found. Using defaults.")
            self.legacy_norm_params = None
            self._ckpt_mode = "baseline_scale"

        # Build model (inference_self_detection.py style)
        self.model = Stage1SelfDetectionFieldMLP(
            input_dim=input_dim,
            output_dim=self.model_output_dim,
            hidden_dims=hidden_dims,
        )
        self.model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.model.eval()

        self.model_path = path

    # ======================================================
    # Callbacks
    # ======================================================
    def raw_callback(self, msg, idx):
        """Raw sensor callback."""
        self.raw_data[idx] = msg.range
        self.raw_received = True

    def joint_callback(self, msg):
        """Joint state callback."""
        self.joint_states = np.array(msg.position[:N_JOINTS], dtype=np.float32)

    # ======================================================
    # Inference + compensation (inference_self_detection.py style)
    # ======================================================
    def timer_callback(self):
        """Timer callback for inference and compensation (inference_self_detection.py style)."""
        if not self.raw_received or self.joint_states is None:
            return

        # Inference (inference_self_detection.py style)
        with torch.no_grad():
            q = torch.from_numpy(self.joint_states).unsqueeze(0)
            y_pred_norm = self.model(q).squeeze(0).numpy()

        # Prepare data
        raw_full = self.raw_data.copy()
        raw = raw_full[:self.model_output_dim]

        # Denormalize and compute compensation (inference_self_detection.py style)
        y_pred_full = np.zeros(N_SENSORS, dtype=np.float32)
        comp_full = raw_full.copy()

        if self._ckpt_mode == "zscore" and self.legacy_norm_params is not None:
            # Old checkpoint: z-score normalization
            y_mean = self.legacy_norm_params['y_mean'][:self.model_output_dim]
            y_std = self.legacy_norm_params['y_std'][:self.model_output_dim]
            y_pred_raw = y_pred_norm[:self.model_output_dim] * y_std + y_mean
            y_pred_full[:self.model_output_dim] = y_pred_raw
            # Compensation: comp = raw - pred_raw
            comp_full[:self.model_output_dim] = raw - y_pred_raw
        else:
            # New checkpoint: baseline/scale normalization (inference_self_detection.py style)
            y_norm = y_pred_norm[:self.model_output_dim]
            y_pred_raw = self.norm_spec.denormalize(torch.from_numpy(y_norm)).numpy().astype(np.float32)
            y_pred_full[:self.model_output_dim] = y_pred_raw
            # Compensation (inference_self_detection.py style): comp = measured - (pred_raw - BASE)
            comp_full[:self.model_output_dim] = raw - (y_pred_raw - float(self.norm_spec.baseline))

        # Logging
        now = self.get_clock().now()
        t = now.nanoseconds / 1e9

        line = (
            f"{t:.9f} "
            + " ".join(f"{x:.6f}" for x in self.joint_states) + " "
            + " ".join(f"{x:.6f}" for x in raw_full) + " "
            + " ".join(f"{x:.6f}" for x in comp_full) + " "
            + " ".join(f"{x:.6f}" for x in y_pred_full)
            + "\n"
        )
        self.log_file.write(line)
        self.log_file.flush()

        # Publish compensated values
        for i in range(N_SENSORS):
            msg = Range()
            msg.header.stamp = now.to_msg()
            msg.range = float(comp_full[i])
            msg.radiation_type = Range.ULTRASOUND
            msg.field_of_view = 0.1
            msg.min_range = 0.0
            msg.max_range = 100000.0
            self.comp_pubs[i].publish(msg)

    def destroy_node(self):
        """Cleanup on node destruction."""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = RealtimeMonitorMLP()

    if hasattr(node, '_model_load_failed'):
        node.destroy_node()
        rclpy.shutdown()
        return

    executor = MultiThreadedExecutor()
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
