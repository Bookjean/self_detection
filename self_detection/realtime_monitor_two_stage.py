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
        self.raw_timestamps = [None] * 4
        self.joint_timestamp = None

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
            self.get_logger().info('PyTorch available, loading models...')
            self._load_models()
        else:
            self.get_logger().error('PyTorch NOT available! Models cannot be loaded.')

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

        # Publishers
        self.compensated_publishers = []
        for i in range(1, 5):
            self.compensated_publishers.append(
                self.create_publisher(Range, f'/compensated_distance{i}', 10)
            )

        # Logging
        log_dir = os.path.expanduser('~/rb10_Proximity/logs')
        os.makedirs(log_dir, exist_ok=True)

        ts = self.get_clock().now().nanoseconds
        self.log_path = os.path.join(log_dir, f'all_topics_100hz_{ts}.txt')

        try:
            self.log_file = open(self.log_path, 'w')
            header = (
                '# timestamp j1 j2 j3 j4 j5 j6 '
                'raw1 raw2 raw3 raw4 '
                'comp1 comp2 comp3 comp4 '
                'stage1_1 stage1_2 stage1_3 stage1_4 '
                'stage2_1 stage2_2 stage2_3 stage2_4\n'
            )
            self.log_file.write(header)
            self.log_file.flush()
        except Exception as e:
            self.get_logger().error(f'Failed to open log file: {e}')
            self.log_file = None

        self.log_timer = self.create_timer(0.01, self.log_callback, callback_group=self.cb_group)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        # 패키지 기준 상대경로로 models 폴더 탐색
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(current_dir, 'models')

        self.get_logger().info(f'Loading models from: {model_dir}')

        if not os.path.exists(model_dir):
            self.get_logger().error(f'Model directory does not exist: {model_dir}')
            return

        config_file = self.get_parameter('config_file').value
        if config_file:
            cfg_path = os.path.join(model_dir, config_file)
            if os.path.exists(cfg_path):
                self._load_from_config(cfg_path, model_dir)
                return

        cfgs = sorted(glob.glob(os.path.join(model_dir, 'two_stage_config_*.pt')))
        if cfgs:
            self._load_from_config(cfgs[-1], model_dir)
        else:
            self.get_logger().error('No config file found for two-stage model')

    def _load_from_config(self, path, model_dir):
        cfg = torch.load(path, map_location='cpu', weights_only=False)

        self.norm_params = cfg.get('norm_params')
        self.residual_norm_params = cfg.get('residual_norm_params')

        self.seq_len = cfg.get('seq_len', self.seq_len)
        self.residual_buffer = deque(maxlen=self.seq_len)

        stage1_file = cfg.get('stage1_model_file', '')
        stage2_file = cfg.get('stage2_model_file', '')

        if stage1_file:
            self._load_stage1(os.path.join(model_dir, stage1_file))
        if stage2_file:
            self._load_stage2(os.path.join(model_dir, stage2_file))

    def _load_stage1(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('model_config', {})
        self.stage1_model = Stage1StaticOffsetMLP(
            input_dim=cfg.get('input_dim', 6),
            output_dim=cfg.get('output_dim', 4),
            hidden_dim=cfg.get('hidden_dim', 32),
            num_layers=cfg.get('num_layers', 3),
        )
        self.stage1_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.stage1_model.eval()

    def _load_stage2(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('model_config', {})
        self.stage2_model = Stage2ResidualTCN(
            input_dim=4,
            output_dim=4,
            seq_len=self.seq_len,
            channels=tuple(cfg.get('channels', (32, 32))),
            kernel_size=cfg.get('kernel_size', 3),
        )
        self.stage2_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.stage2_model.eval()

    # ------------------------------------------------------------------
    # Callbacks & inference
    # ------------------------------------------------------------------

    def raw_callback(self, msg, idx):
        self.raw_data[idx] = msg.range
        self.raw_received = True

    def joint_callback(self, msg):
        self.joint_states = np.array(msg.position)

    def _run_stage1(self):
        if self.stage1_model is None or self.joint_states is None:
            return None
        x = self.joint_states[:6]
        if self.norm_params:
            x = (x - self.norm_params['X_mean']) / (self.norm_params['X_std'] + 1e-8)
        with torch.no_grad():
            y = self.stage1_model(torch.tensor(x).float().unsqueeze(0)).squeeze().numpy()
        if self.norm_params:
            y = y * self.norm_params['y_std'] + self.norm_params['y_mean']
        return y

    def _run_stage2(self):
        if self.stage2_model is None or len(self.residual_buffer) < self.seq_len:
            return None
        seq = np.array(self.residual_buffer)
        with torch.no_grad():
            return self.stage2_model(torch.tensor(seq).float().unsqueeze(0)).squeeze().numpy()

    def log_callback(self):
        if not self.raw_received:
            return

        stage1 = self._run_stage1()
        if stage1 is None:
            return

        raw = np.array(self.raw_data)
        residual = raw - stage1
        self.residual_buffer.append(residual)

        stage2 = self._run_stage2()
        comp = raw - stage1 - stage2 if stage2 is not None else raw - stage1

        now = self.get_clock().now().nanoseconds / 1e9
        joints = self.joint_states[:6] if self.joint_states is not None else [0.0] * 6

        if self.log_file:
            line = (
                f"{now:.6f} "
                + " ".join(f"{j:.6f}" for j in joints) + " "
                + " ".join(f"{r:.6f}" for r in raw) + " "
                + " ".join(f"{c:.6f}" for c in comp) + "\n"
            )
            self.log_file.write(line)
            self.log_file.flush()


def main(args=None):
    rclpy.init(args=args)
    node = RealtimeMonitorTwoStage()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

