"""MLP-only self detection launch file.

Usage:
    ros2 launch self_detection mlp.launch.py
    ros2 launch self_detection mlp.launch.py model_file:=stage1_stage1_mlp_20260122.pth
    ros2 launch self_detection mlp.launch.py log_rate:=50.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # 패키지 share 디렉토리에서 소스 디렉토리로 올라가기
    share_dir = get_package_share_directory('self_detection')
    # share_dir에서 rb_ws까지 올라간 후 src/self_detection으로
    rb_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(share_dir))))
    package_dir = os.path.join(rb_ws_dir, 'src', 'self_detection')
    
    ybvenv_python = os.path.join(package_dir, 'ybvenv', 'bin', 'python3')
    script_path = os.path.join(package_dir, 'self_detection', 'realtime_monitor_mlp.py')
    
    def launch_setup(context):
        actions = []
        
        # Launch arguments 값 가져오기
        model_file = context.launch_configurations.get('model_file', '')
        log_rate = context.launch_configurations.get('log_rate', '100.0')
        
        # ybvenv Python이 있으면 사용
        if os.path.exists(ybvenv_python) and os.path.exists(script_path):
            # PYTHONPATH에 ybvenv site-packages와 패키지 경로 추가
            site_packages = os.path.join(package_dir, 'ybvenv', 'lib', 'python3.10', 'site-packages')
            package_path = package_dir
            
            current_pythonpath = os.environ.get('PYTHONPATH', '')
            new_pythonpath = f"{site_packages}:{package_path}"
            if current_pythonpath:
                new_pythonpath = f"{new_pythonpath}:{current_pythonpath}"
            
            actions.append(SetEnvironmentVariable('PYTHONPATH', new_pythonpath))
            
            # ybvenv Python으로 직접 실행
            cmd = [
                ybvenv_python,
                script_path,
                '--ros-args',
                '-r', '__node:=self_detection_mlp',
            ]
            
            if model_file:
                cmd.extend(['-p', f'model_file:={model_file}'])
            if log_rate:
                cmd.extend(['-p', f'log_rate:={log_rate}'])
            
            node = ExecuteProcess(
                cmd=cmd,
                output='screen',
            )
        else:
            # 시스템 Python 사용 (fallback)
            from launch_ros.actions import Node
            if not os.path.exists(ybvenv_python):
                print(f"[WARN] ybvenv Python not found at {ybvenv_python}, using system Python")
            if not os.path.exists(script_path):
                print(f"[WARN] Script not found at {script_path}, using entry_point")
            
            node = Node(
                package='self_detection',
                executable='realtime_monitor_mlp',
                name='self_detection_mlp',
                output='screen',
                parameters=[{
                    'model_file': model_file,
                    'log_rate': log_rate,
                }]
            )
        
        actions.append(node)
        return actions
    
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'model_file',
            default_value='',
            description='Model file name (auto-detect if empty)'
        ),
        DeclareLaunchArgument(
            'log_rate',
            default_value='100.0',
            description='Logging rate in Hz'
        ),
        # OpaqueFunction을 사용하여 동적으로 노드 생성
        OpaqueFunction(function=launch_setup),
    ])
