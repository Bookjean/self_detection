#!/usr/bin/env python3
"""
Real-time Self Detection Monitoring Node - Two-Stage Version
with 100 Hz compensated value logging.

Saved value:
    compensated raw1 raw2 raw3 raw4  (Stage1 + Stage2 if available)

Logging:
    ~/rb10_Proximity/logs/compensated_100hz_*.txt
"""

import os
import sys
import glob
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState, Range

# Add self_detection module path
sys.path.insert(0, os.path.expanduser('~/rb10_Proximity/ml/self_detection'))

try:
    import torch
    from self_detection_mlp.model import (
        Stage1StaticOffsetMLP,
        Stage2ResidualTCN,
        Stage2ResidualMemoryMLP,
        SimpleMLP,
        DeepMLP,
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"[WARN] PyTorch not available: {e}")


class RealtimeMonitorTwoStage(Node):
    def __init__(self):
        super().__init__('realtime_monitor_two_stage')

        # Parameters
        self.declare_parameter('config_file', '')
        self.declare_parameter('stage1_file', '')
        self.declare_parameter('stage2_file', '')
        self.declare_parameter('seq_len', 10)

        self.cb_group = ReentrantCallbackGroup()

        # Data
        self.raw_data = [0.0] * 4
        self.joint_states = None
        self.joint_names = None
        self.raw_received = False

        # Models
        self.stage1_model = None
        self.stage2_model = None
        self.stage1_type = None
        self.stage2_type = None

        # Normalization
        self.norm_params = None
        self.residual_norm_params = None

        # Residual buffer
        self.seq_len = self.get_parameter('seq_len').value
        self.residual_buffer = deque(maxlen=self.seq_len)

        # Load models
        if TORCH_AVAILABLE:
            self._load_models()

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

        # ===============================
        # 100 Hz logging
        # ===============================
        log_dir = os.path.expanduser('~/rb10_Proximity/logs')
        os.makedirs(log_dir, exist_ok=True)

        ts = self.get_clock().now().nanoseconds
        self.log_path = os.path.join(log_dir, f'compensated_100hz_{ts}.txt')
        self.log_file = open(self.log_path, 'w')
        self.log_file.write('# raw1 raw2 raw3 raw4 (compensated)\n')

        self.log_timer = self.create_timer(
            0.01,  # 100 Hz
            self.log_callback,
            callback_group=self.cb_group
        )

        self.get_logger().info('=' * 60)
        self.get_logger().info('Two-Stage Self Detection Realtime Monitor')
        self.get_logger().info(f'Logging @100Hz → {self.log_path}')
        self.get_logger().info('=' * 60)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        model_dir = os.path.expanduser('~/rb10_Proximity/ml/self_detection/models')

        config_file = self.get_parameter('config_file').value
        if config_file:
            path = os.path.join(model_dir, config_file)
            if os.path.exists(path):
                self._load_from_config(path, model_dir)
                return

        cfgs = sorted(glob.glob(os.path.join(model_dir, 'two_stage_config_*.pt')))
        if cfgs:
            self._load_from_config(cfgs[-1], model_dir)

    def _load_from_config(self, path, model_dir):
        cfg = torch.load(path, map_location='cpu', weights_only=False)

        self.norm_params = cfg.get('norm_params', None)
        self.residual_norm_params = cfg.get('residual_norm_params', None)

        if 'seq_len' in cfg:
            self.seq_len = cfg['seq_len']
            self.residual_buffer = deque(maxlen=self.seq_len)

        self._load_stage1(os.path.join(model_dir, cfg['stage1_model_file']))
        self._load_stage2(os.path.join(model_dir, cfg['stage2_model_file']))

    def _load_stage1(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt['model_state_dict']
        cfg = ckpt.get('model_config', {})

        self.stage1_model = Stage1StaticOffsetMLP(
            input_dim=cfg.get('input_dim', 6),
            output_dim=cfg.get('output_dim', 4),
            hidden_dim=cfg.get('hidden_dim', 32),
            num_layers=cfg.get('num_layers', 3),
        )
        self.stage1_model.load_state_dict(state, strict=False)
        self.stage1_model.eval()
        self.stage1_type = 'Stage1MLP'

    def _load_stage2(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt['model_state_dict']
        cfg = ckpt.get('model_config', {})

        self.stage2_model = Stage2ResidualTCN(
            input_dim=4,
            output_dim=4,
            seq_len=self.seq_len,
            channels=tuple(cfg.get('channels', (32, 32))),
            kernel_size=cfg.get('kernel_size', 3),
        )
        self.stage2_model.load_state_dict(state, strict=False)
        self.stage2_model.eval()
        self.stage2_type = 'Stage2TCN'

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def raw_callback(self, msg, idx):
        self.raw_data[idx] = msg.range
        self.raw_received = True

    def joint_callback(self, msg):
        self.joint_states = np.array(msg.position)
        self.joint_names = msg.name

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_stage1(self):
        if self.stage1_model is None or self.joint_states is None:
            return None

        joints = self.joint_states[:6]
        if self.norm_params:
            joints = (joints - self.norm_params['X_mean']) / (self.norm_params['X_std'] + 1e-8)

        with torch.no_grad():
            y = self.stage1_model(torch.tensor(joints).float().unsqueeze(0)).squeeze().numpy()

        if self.norm_params:
            y = y * self.norm_params['y_std'] + self.norm_params['y_mean']
        return y

    def _run_stage2(self):
        if self.stage2_model is None or len(self.residual_buffer) < self.seq_len:
            return None

        seq = np.array(self.residual_buffer)
        if self.residual_norm_params:
            seq = (seq - self.residual_norm_params['mean']) / (self.residual_norm_params['std'] + 1e-8)

        with torch.no_grad():
            y = self.stage2_model(torch.tensor(seq).float().unsqueeze(0)).squeeze().numpy()

        if self.residual_norm_params:
            y = y * self.residual_norm_params['std'] + self.residual_norm_params['mean']
        return y

    def _run_inference(self):
        if not self.raw_received:
            return None

        raw = np.array(self.raw_data)
        stage1 = self._run_stage1()
        if stage1 is None:
            return None

        residual = raw - stage1
        self.residual_buffer.append(residual.copy())

        stage2 = self._run_stage2()
        if stage2 is not None:
            comp = raw - stage1 - stage2
        else:
            comp = raw - stage1

        return comp

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_callback(self):
        result = self._run_inference()
        if result is None:
            return

        self.log_file.write(
            f"{result[0]:.6f} {result[1]:.6f} {result[2]:.6f} {result[3]:.6f}\n"
        )


def main(args=None):
    rclpy.init(args=args)
    node = RealtimeMonitorTwoStage()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node.log_file:
            node.log_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
