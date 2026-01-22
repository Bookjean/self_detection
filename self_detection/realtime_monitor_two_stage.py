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
        self.raw_timestamps = [None] * 4  # 각 센서의 타임스탬프
        self.joint_timestamp = None  # 관절 상태 타임스탬프

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

        # ===============================
        # Publishers for compensated values
        # ===============================
        self.compensated_publishers = []
        for i in range(1, 5):
            pub = self.create_publisher(
                Range,
                f'/compensated_distance{i}',
                10
            )
            self.compensated_publishers.append(pub)

        # ===============================
        # 100 Hz logging - 모든 토픽 데이터 저장
        # ===============================
        log_dir = os.path.expanduser('~/rb10_Proximity/logs')
        os.makedirs(log_dir, exist_ok=True)

        ts = self.get_clock().now().nanoseconds
        self.log_path = os.path.join(log_dir, f'all_topics_100hz_{ts}.txt')
        try:
            self.log_file = open(self.log_path, 'w')
            # 헤더: timestamp, j1~j6, raw1~4, compensated1~4, stage1_out1~4, stage2_out1~4
            header = '# timestamp j1 j2 j3 j4 j5 j6 raw1 raw2 raw3 raw4 compensated1 compensated2 compensated3 compensated4 stage1_out1 stage1_out2 stage1_out3 stage1_out4 stage2_out1 stage2_out2 stage2_out3 stage2_out4\n'
            self.log_file.write(header)
            self.log_file.flush()  # 헤더도 즉시 저장
            self.get_logger().info(f'Log file opened and header written: {self.log_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to open log file: {e}')
            self.log_file = None

        self.log_timer = self.create_timer(
            0.01,  # 100 Hz
            self.log_callback,
            callback_group=self.cb_group
        )

        self.get_logger().info('=' * 60)
        self.get_logger().info('Two-Stage Self Detection Realtime Monitor')
        self.get_logger().info(f'Logging all topics @100Hz → {self.log_path}')
        self.get_logger().info('Saved data: timestamp, j1~j6, raw1~4, compensated1~4, stage1_out1~4, stage2_out1~4')
        self.get_logger().info('Publishing compensated values:')
        for i in range(1, 5):
            self.get_logger().info(f'  /compensated_distance{i}')
        
        # 모델 로드 상태 확인
        if TORCH_AVAILABLE:
            if self.stage1_model is not None:
                self.get_logger().info(f'✓ Stage1 model loaded: {self.stage1_type}')
            else:
                self.get_logger().warn('✗ Stage1 model NOT loaded')
            if self.stage2_model is not None:
                self.get_logger().info(f'✓ Stage2 model loaded: {self.stage2_type}')
            else:
                self.get_logger().warn('✗ Stage2 model NOT loaded')
        else:
            self.get_logger().warn('✗ PyTorch not available')
        
        self.get_logger().info('=' * 60)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        # 현재 프로젝트의 models 폴더 사용
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(current_dir, 'models')

        self.get_logger().info(f'Loading models from: {model_dir}')
        self.get_logger().info(f'Model directory exists: {os.path.exists(model_dir)}')
        
        if not os.path.exists(model_dir):
            self.get_logger().error(f'Model directory does not exist: {model_dir}')
            return

        # 디렉토리 내용 확인
        files = os.listdir(model_dir) if os.path.exists(model_dir) else []
        self.get_logger().info(f'Files in model directory: {files}')

        config_file = self.get_parameter('config_file').value
        if config_file:
            path = os.path.join(model_dir, config_file)
            if os.path.exists(path):
                self.get_logger().info(f'Loading config: {config_file}')
                self._load_from_config(path, model_dir)
                return
            else:
                self.get_logger().warn(f'Config file not found: {path}')

        cfgs = sorted(glob.glob(os.path.join(model_dir, 'two_stage_config_*.pt')))
        if cfgs:
            self.get_logger().info(f'Auto-loading latest config: {os.path.basename(cfgs[-1])}')
            self._load_from_config(cfgs[-1], model_dir)
        else:
            self.get_logger().error(f'No config files found in {model_dir}')

    def _load_from_config(self, path, model_dir):
        try:
            cfg = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            self.get_logger().error(f'Failed to load config: {e}')
            return

        self.norm_params = cfg.get('norm_params', None)
        self.residual_norm_params = cfg.get('residual_norm_params', None)
        if self.norm_params:
            self.get_logger().info('✓ Normalization params loaded')
        else:
            self.get_logger().warn('✗ Normalization params not found in config')

        if 'seq_len' in cfg:
            self.seq_len = cfg['seq_len']
            self.residual_buffer = deque(maxlen=self.seq_len)
            self.get_logger().info(f'Sequence length: {self.seq_len}')

        stage1_file = cfg.get('stage1_model_file', '')
        stage2_file = cfg.get('stage2_model_file', '')
        
        self.get_logger().info(f'Stage1 model file from config: {stage1_file}')
        self.get_logger().info(f'Stage2 model file from config: {stage2_file}')
        
        if stage1_file:
            stage1_path = os.path.join(model_dir, stage1_file)
            self.get_logger().info(f'Loading Stage1 from: {stage1_path}')
            self._load_stage1(stage1_path)
        else:
            self.get_logger().error('stage1_model_file not found in config')
        
        if stage2_file:
            stage2_path = os.path.join(model_dir, stage2_file)
            self.get_logger().info(f'Loading Stage2 from: {stage2_path}')
            self._load_stage2(stage2_path)
        else:
            self.get_logger().warn('stage2_model_file not found in config (Stage2 will be skipped)')

    def _load_stage1(self, path):
        try:
            if not os.path.exists(path):
                self.get_logger().error(f'Stage1 model file not found: {path}')
                return
            
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
            self.get_logger().info(f'✓ Stage1 model loaded from {os.path.basename(path)}')
        except Exception as e:
            self.get_logger().error(f'Failed to load Stage1 model: {e}')
            self.stage1_model = None

    def _load_stage2(self, path):
        try:
            if not os.path.exists(path):
                self.get_logger().warn(f'Stage2 model file not found: {path}')
                return
            
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
            self.get_logger().info(f'✓ Stage2 model loaded from {os.path.basename(path)}')
        except Exception as e:
            self.get_logger().error(f'Failed to load Stage2 model: {e}')
            self.stage2_model = None

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def raw_callback(self, msg, idx):
        self.raw_data[idx] = msg.range
        self.raw_timestamps[idx] = msg.header.stamp
        self.raw_received = True

    def joint_callback(self, msg):
        self.joint_states = np.array(msg.position)
        self.joint_names = msg.name
        self.joint_timestamp = msg.header.stamp

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
        # 보상 계산
        result = self._run_inference()
        if result is None:
            # 디버깅: 왜 None인지 확인 (처음 몇 번만 출력)
            if not hasattr(self, '_skip_count'):
                self._skip_count = 0
            if self._skip_count < 3:
                if not self.raw_received:
                    self.get_logger().warn(f'Inference skipped: raw_received=False (count: {self._skip_count})')
                elif self.stage1_model is None:
                    self.get_logger().warn(f'Inference skipped: stage1_model is None (count: {self._skip_count})')
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

        # Stage1 출력
        stage1_out = self._run_stage1()
        if stage1_out is None:
            stage1_out = np.array([0.0] * 4)
        else:
            stage1_out = np.array(stage1_out)

        # Stage2 출력
        stage2_out = self._run_stage2()
        if stage2_out is None:
            stage2_out = np.array([0.0] * 4)
        else:
            stage2_out = np.array(stage2_out)

        # 파일에 모든 데이터 저장
        # timestamp, j1~j6, raw1~4, compensated1~4, stage1_out1~4, stage2_out1~4
        line = f"{timestamp_sec:.9f} "  # timestamp
        line += " ".join([f"{j:.6f}" for j in joints]) + " "  # j1~j6
        line += " ".join([f"{r:.6f}" for r in raw]) + " "  # raw1~4
        line += " ".join([f"{c:.6f}" for c in result]) + " "  # compensated1~4
        line += " ".join([f"{s1:.6f}" for s1 in stage1_out]) + " "  # stage1_out1~4
        line += " ".join([f"{s2:.6f}" for s2 in stage2_out]) + "\n"  # stage2_out1~4
        
        if self.log_file is not None:
            try:
                self.log_file.write(line)
                self.log_file.flush()  # 실시간 저장을 위해 즉시 디스크에 쓰기
                # 처음 몇 번만 저장 확인 로그
                if not hasattr(self, '_write_count'):
                    self._write_count = 0
                if self._write_count < 3:
                    self.get_logger().info(f'Data written to log file (count: {self._write_count + 1})')
                    self._write_count += 1
            except Exception as e:
                self.get_logger().error(f'Failed to write to log file: {e}')
        else:
            if not hasattr(self, '_log_file_warn_count'):
                self._log_file_warn_count = 0
            if self._log_file_warn_count < 3:
                self.get_logger().warn('Log file is None, cannot write data')
                self._log_file_warn_count += 1

        # 토픽으로 publish
        for i in range(4):
            msg = Range()
            msg.header.stamp = now.to_msg()
            msg.header.frame_id = f'sensor{i+1}'
            msg.range = float(result[i])
            msg.min_range = 0.0
            msg.max_range = 1000.0  # 적절한 값으로 설정
            self.compensated_publishers[i].publish(msg)
        
        # 처음 몇 번만 디버그 로그 출력 (너무 많이 찍히지 않도록)
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 5:
            self.get_logger().info(
                f'Published compensated: [{result[0]:.2f}, {result[1]:.2f}, {result[2]:.2f}, {result[3]:.2f}] '
                f'(raw: [{raw[0]:.2f}, {raw[1]:.2f}, {raw[2]:.2f}, {raw[3]:.2f}])'
            )
            self._debug_count += 1


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
