from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 패키지 share 디렉토리에서 소스 디렉토리로 올라가기
    # install/self_detection/share/self_detection -> src/self_detection
    share_dir = get_package_share_directory('self_detection')
    # share_dir에서 rb_ws까지 올라간 후 src/self_detection으로
    # share_dir: /home/son_rb/rb_ws/install/self_detection/share/self_detection
    # dirname 4번: /home/son_rb/rb_ws
    rb_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(share_dir))))
    package_dir = os.path.join(rb_ws_dir, 'src', 'self_detection')
    
    ybvenv_python = os.path.join(package_dir, 'ybvenv', 'bin', 'python3')
    script_path = os.path.join(package_dir, 'self_detection', 'realtime_monitor_two_stage.py')
    
    actions = []
    
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
        node = ExecuteProcess(
            cmd=[
                ybvenv_python,
                script_path,
                '--ros-args',
                '-r', '__node:=self_detection_node',
                '-p', 'seq_len:=10',
            ],
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
            executable='realtime_monitor_two_stage',
            name='self_detection_node',
            output='screen',
            parameters=[
                {'seq_len': 10},
            ]
        )
    
    actions.append(node)
    return LaunchDescription(actions)

