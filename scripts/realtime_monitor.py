#!/usr/bin/env python3
"""
Real-time Self Detection Monitoring Node - MLP Version

- Input: 6 joint angles (j1-j6)
- Target: raw_distance1~4
- Compensation: compensated = raw - predicted

- Logging: compensated raw (4ch) @ 100 Hz

Usage:
    ros2 run ecan_driver realtime_monitor.py
    ros2 run ecan_driver realtime_monitor.py --ros-args -p model_file:=stage1_mlp_standard_20260121.pth
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

# self_detection module path
sys.path.insert(0, os.path.expanduser('~/rb10_Proximity/ml/self_detection'))

try:
    import torch
    from self_detection_mlp.model import (
        Stage1StaticOffsetMLP,
        Stage2ResidualMemoryMLP,
        SimpleMLP,
        DeepMLP
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"[WARN] Torch not available: {e}")


class RealtimeMonitor(Node):
    def __init__(self):
        super().__init__('realtime_monitor')

        # Declare parameters
        self.declare_parameter('model_file', '')

        self.cb_group = ReentrantCallbackGroup()

        # Data buffers
        self.raw_data = [0.0] * 4
        self.raw_received = False
        self.joint_states = None
        self.joint_names = None

        # ML
        self.model = None
        self.model_type = None
        self.norm_params = None

        if TORCH_AVAILABLE:
            self._load_model()

        # Subscribers
        self.raw_subs = []
        for i in range(1, 5):
            self.raw_subs.append(
                self.create_subscription(
                    Range,
                    f'/raw_distance{i}',
                    lambda msg, idx=i-1: self.raw_callback(msg, idx),
                    10,
                    callback_group=self.cb_group
                )
            )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10,
            callback_group=self.cb_group
        )

        # Logging (100 Hz)
        log_dir = os.path.expanduser('~/rb10_Proximity/logs')
        os.makedirs(log_dir, exist_ok=True)

        self.log_path = os.path.join(log_dir, 'compensated_raw_100hz.txt')
        self.log_file = open(self.log_path, 'w')
        self.log_file.write('# time[s] comp1 comp2 comp3 comp4\n')

        self.start_time = self.get_clock().now()

        self.inference_timer = self.create_timer(
            0.01,  # 100 Hz
            self.inference_and_log_callback,
            callback_group=self.cb_group
        )

        # Display (10 Hz)
        self.display_timer = self.create_timer(0.1, self.display_callback)

        self.get_logger().info('=' * 60)
        self.get_logger().info(' Self Detection Realtime Monitor - MLP Version')
        self.get_logger().info(' Logging compensated raw @ 100 Hz')
        self.get_logger().info(f' Log file: {self.log_path}')
        self.get_logger().info('=' * 60)

    def _load_model(self):
        """Load MLP model with auto-detection."""
        model_dir = os.path.expanduser('~/rb10_Proximity/ml/self_detection/models')
        model_file = self.get_parameter('model_file').value

        # Find model file
        if model_file:
            model_path = os.path.join(model_dir, model_file)
        else:
            # Auto-find MLP models (exclude tcn)
            patterns = [
                'stage1_mlp_*.pth',
                'simple_mlp_*.pth', 
                'deep_mlp_*.pth',
                'stage1_static_offset_model.pth',  # Legacy
            ]
            model_path = None
            for pattern in patterns:
                files = sorted(glob.glob(os.path.join(model_dir, pattern)))
                if files:
                    model_path = files[-1]
                    break

            if model_path:
                self.get_logger().info(f'Auto-selected: {os.path.basename(model_path)}')
            else:
                self.get_logger().warn('No MLP model found')
                return

        if not os.path.exists(model_path):
            self.get_logger().error(f'Model not found: {model_path}')
            return

        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Extract state dict and config
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                config = checkpoint.get('config', {})
                model_config = checkpoint.get('model_config', {})

                # Load norm params from checkpoint
                norm_raw = checkpoint.get('norm_params')
                if norm_raw:
                    self.norm_params = {
                        'y_mean': np.array(norm_raw.get('y_mean', norm_raw.get('stage1_means', [0]*4))),
                        'y_std': np.array(norm_raw.get('y_std', norm_raw.get('stage1_stds', [1]*4))),
                    }
            else:
                state_dict = checkpoint
                config = {}
                model_config = {}

            # Get model type and dimensions
            model_name = config.get('model_name', model_config.get('model', 'stage1_mlp'))
            
            first_layer = 'net.0.weight'
            input_dim = state_dict[first_layer].shape[1] if first_layer in state_dict else 6
            output_dim = model_config.get('output_dim', 4)
            hidden_dim = model_config.get('hidden_dim', 32)
            num_layers = model_config.get('num_layers', 3)

            # Create model
            if 'deep' in model_name.lower():
                self.model = DeepMLP(input_dim=input_dim, output_dim=output_dim,
                                     hidden_dim=hidden_dim, num_layers=num_layers)
                self.model_type = 'DeepMLP'
            elif 'simple' in model_name.lower():
                hidden_dims = model_config.get('hidden_dims', [64, 32])
                self.model = SimpleMLP(input_dim=input_dim, output_dim=output_dim,
                                       hidden_dims=hidden_dims)
                self.model_type = 'SimpleMLP'
            else:
                self.model = Stage1StaticOffsetMLP(input_dim=input_dim, output_dim=output_dim,
                                                   hidden_dim=hidden_dim, num_layers=num_layers)
                self.model_type = 'Stage1MLP'

            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.get_logger().info(f'Model loaded: {self.model_type} (in:{input_dim}, out:{output_dim})')

        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return

        # Load norm params from file if not in checkpoint
        if self.norm_params is None:
            norm_path = os.path.join(model_dir, 'normalization_params.pt')
            if os.path.exists(norm_path):
                try:
                    norm_data = torch.load(norm_path, map_location='cpu')
                    y_mean = norm_data.get('stage1_means', norm_data.get('y_mean'))
                    y_std = norm_data.get('stage1_stds', norm_data.get('y_std'))
                    
                    if isinstance(y_mean, torch.Tensor):
                        y_mean = y_mean.numpy()
                    if isinstance(y_std, torch.Tensor):
                        y_std = y_std.numpy()
                    
                    self.norm_params = {
                        'y_mean': np.array(y_mean) if y_mean is not None else np.zeros(4),
                        'y_std': np.array(y_std) if y_std is not None else np.ones(4),
                    }
                    self.get_logger().info('Norm params loaded')
                except Exception as e:
                    self.get_logger().warn(f'Norm params load failed: {e}')

    def raw_callback(self, msg, index):
        self.raw_data[index] = msg.range
        self.raw_received = True

    def joint_callback(self, msg):
        self.joint_states = np.array(msg.position)
        self.joint_names = msg.name

    def inference_and_log_callback(self):
        result = self._run_inference()
        if result is None:
            return
        predicted, compensated = result
        
        t = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        self.log_file.write(f'{t:.6f} {compensated[0]:.4f} {compensated[1]:.4f} '
                           f'{compensated[2]:.4f} {compensated[3]:.4f}\n')

    def _run_inference(self):
        """Run MLP inference. Returns (predicted, compensated) or None."""
        if not TORCH_AVAILABLE or self.model is None:
            return None
        if self.joint_states is None or not self.raw_received:
            return None
        if len(self.joint_states) < 6:
            return None

        try:
            x = torch.tensor(self.joint_states[:6], dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                pred_norm = self.model(x).squeeze().numpy()

            if self.norm_params:
                predicted = pred_norm * self.norm_params['y_std'] + self.norm_params['y_mean']
            else:
                predicted = pred_norm

            raw = np.array(self.raw_data)
            compensated = raw - predicted
            return predicted, compensated

        except Exception as e:
            self.get_logger().warn(f'Inference error: {e}')
            return None

    def display_callback(self):
        print('\033[2J\033[H', end='')
        print('=' * 60)
        print(' SELF DETECTION REALTIME MONITOR - MLP')
        print('=' * 60)

        print(f'Model: {self.model_type if self.model else "[Not loaded]"}')

        print('\n[RAW SENSOR]')
        if self.raw_received:
            for i, v in enumerate(self.raw_data):
                print(f'  raw{i+1}: {v:12.1f}')
        else:
            print('  [Waiting...]')

        print('\n[JOINT STATES]')
        if self.joint_states is not None:
            for i, rad in enumerate(self.joint_states[:6]):
                print(f'  j{i+1}: {np.degrees(rad):8.2f} deg')
        else:
            print('  [Waiting...]')

        print('\n[COMPENSATION]')
        result = self._run_inference()
        if result:
            predicted, compensated = result
            for i in range(4):
                print(f'  comp{i+1}: {compensated[i]:+12.1f}  '
                      f'(raw:{self.raw_data[i]:.1f} - pred:{predicted[i]:.1f})')
        else:
            print('  [Waiting for data...]')

        print('\n' + '-' * 60)
        print(f'Logging @ 100 Hz -> {os.path.basename(self.log_path)}')
        print('=' * 60)

    def destroy_node(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealtimeMonitor()
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
