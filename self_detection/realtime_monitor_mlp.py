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
from datetime import datetime

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
        self.model_name = 'unknown'  # 모델 이름 저장
        self.model_path = None  # 로드된 모델 경로

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

        # ===============================
        # Publishers for compensated values
        # ===============================
        self.compensated_publishers = []
        for i in range(1, 5):
            pub = self.create_publisher(
                Range,
                f'/compensated_mlp_distance{i}',
                10
            )
            self.compensated_publishers.append(pub)

        # ===============================
        # Logging - 모든 토픽 데이터 저장
        # ===============================
        log_rate = self.get_parameter('log_rate').value
        log_dir = os.path.expanduser('~/rb10_Proximity/logs')
        os.makedirs(log_dir, exist_ok=True)

        # 파일명: 모델명_날짜_시간_분.txt
        now = datetime.now()
        date_str = now.strftime('%Y%m%d_%H%M')  # YYYYMMDD_HHMM
        model_name_clean = self.model_name.replace(' ', '_').replace('/', '_')
        filename = f'compensated_mlp_{model_name_clean}_{date_str}.txt'
        self.log_path = os.path.join(log_dir, filename)
        
        try:
            self.log_file = open(self.log_path, 'w')
            # 헤더: timestamp, j1~j6, raw1~4, compensated1~4, mlp_out1~4
            header = '# timestamp j1 j2 j3 j4 j5 j6 raw1 raw2 raw3 raw4 compensated1 compensated2 compensated3 compensated4 mlp_out1 mlp_out2 mlp_out3 mlp_out4\n'
            self.log_file.write(header)
            self.log_file.flush()
            self.get_logger().info(f'Log file opened: {self.log_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to open log file: {e}')
            self.log_file = None

        self.log_timer = self.create_timer(
            1.0 / log_rate,
            self.log_callback,
            callback_group=self.cb_group
        )

        self.get_logger().info('=' * 60)
        self.get_logger().info('MLP-Only Self Detection Realtime Monitor')
        self.get_logger().info(f'Model loaded: {self.model is not None}')
        if self.model_path:
            self.get_logger().info(f'Model: {os.path.basename(self.model_path)}')
        self.get_logger().info(f'Logging all topics @{log_rate}Hz → {self.log_path}')
        self.get_logger().info('Saved data: timestamp, j1~j6, raw1~4, compensated1~4, mlp_out1~4')
        self.get_logger().info('Publishing compensated values:')
        for i in range(1, 5):
            self.get_logger().info(f'  /compensated_mlp_distance{i}')
        self.get_logger().info('=' * 60)

    def _load_model(self):
        """Load MLP model from file."""
        # 현재 프로젝트의 models 폴더 사용
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(current_dir, 'models')

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
        self.model_path = path

        # 모델 이름 추출 (파일명에서)
        # 예: stage1_stage1_mlp_20260121_211733.pth -> stage1_mlp
        basename = os.path.basename(path)
        # stage1_로 시작하는 경우 stage1_mlp 추출
        if 'stage1_' in basename:
            parts = basename.replace('.pth', '').replace('.pt', '').split('_')
            if len(parts) >= 3:
                # stage1_stage1_mlp_20260121_211733 -> stage1_mlp
                self.model_name = f"{parts[0]}_{parts[2]}" if len(parts) > 2 else f"{parts[0]}_{parts[1]}"
            else:
                self.model_name = 'stage1_mlp'
        else:
            # 파일명에서 모델 타입 추출 시도
            self.model_name = basename.split('_')[0] if '_' in basename else 'mlp'

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

        self.get_logger().info(f'Model loaded successfully: {self.model_name}')
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
            return None, None

        if self.model is None:
            return None, None

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

        return compensated, y_pred

    def log_callback(self):
        """Timer callback for logging compensated values."""
        result, mlp_out = self._run_inference()
        if result is None:
            # 디버깅: 왜 None인지 확인 (처음 몇 번만 출력)
            if not hasattr(self, '_skip_count'):
                self._skip_count = 0
            if self._skip_count < 3:
                if not self.raw_received:
                    self.get_logger().warn(f'Inference skipped: raw_received=False (count: {self._skip_count})')
                elif self.model is None:
                    self.get_logger().warn(f'Inference skipped: model is None (count: {self._skip_count})')
                elif self.joint_states is None:
                    self.get_logger().warn(f'Inference skipped: joint_states is None (count: {self._skip_count})')
                self._skip_count += 1
            return

        # 현재 타임스탬프
        now = self.get_clock().now()
        timestamp_sec = now.nanoseconds / 1e9

        # 관절 상태 (없으면 0으로 채움)
        joints = self.joint_states[:6] if self.joint_states is not None else [0.0] * 6
        if len(joints) < 6:
            joints = list(joints) + [0.0] * (6 - len(joints))

        # 원본 센서값
        raw = np.array(self.raw_data)

        # MLP 출력 (없으면 0으로 채움)
        if mlp_out is None:
            mlp_out = np.array([0.0] * 4)
        else:
            mlp_out = np.array(mlp_out)

        # 파일에 모든 데이터 저장
        # timestamp, j1~j6, raw1~4, compensated1~4, mlp_out1~4
        line = f"{timestamp_sec:.9f} "  # timestamp
        line += " ".join([f"{j:.6f}" for j in joints]) + " "  # j1~j6
        line += " ".join([f"{r:.6f}" for r in raw]) + " "  # raw1~4
        line += " ".join([f"{c:.6f}" for c in result]) + " "  # compensated1~4
        line += " ".join([f"{m:.6f}" for m in mlp_out]) + "\n"  # mlp_out1~4
        
        if self.log_file is not None:
            try:
                self.log_file.write(line)
                self.log_file.flush()  # 실시간 저장을 위해 즉시 디스크에 쓰기
            except Exception as e:
                self.get_logger().error(f'Failed to write to log file: {e}')
        else:
            self.get_logger().warn('Log file is None, cannot write data')

        # 토픽으로 publish
        for i in range(4):
            msg = Range()
            msg.header.stamp = now.to_msg()
            msg.header.frame_id = f'sensor{i+1}'
            msg.range = float(result[i])
            msg.min_range = 0.0
            msg.max_range = 1000.0
            self.compensated_publishers[i].publish(msg)
        
        # 처음 몇 번만 디버그 로그 출력 (너무 많이 찍히지 않도록)
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 5:
            self.get_logger().info(
                f'Published compensated_mlp: [{result[0]:.2f}, {result[1]:.2f}, {result[2]:.2f}, {result[3]:.2f}] '
                f'(raw: [{raw[0]:.2f}, {raw[1]:.2f}, {raw[2]:.2f}, {raw[3]:.2f}])'
            )
            self._debug_count += 1

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
