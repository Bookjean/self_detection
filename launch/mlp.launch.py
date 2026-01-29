"""MLP-only self detection launch file.

Supports 8 sensors (raw1-raw8) with backward compatibility for 4 sensors.
The node automatically detects the number of sensors from the model checkpoint.

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
    # 절대 경로 사용
    package_dir = '/home/son_rb/rb_ws/src/self_detection'
    
    ybvenv_python = os.path.join(package_dir, 'ybvenv', 'bin', 'python3')
    script_path = os.path.join(package_dir, 'self_detection', 'realtime_monitor_mlp.py')
    
    # 디버깅: 경로 확인
    print(f"[DEBUG] package_dir: {package_dir}")
    print(f"[DEBUG] ybvenv_python: {ybvenv_python}, exists: {os.path.exists(ybvenv_python)}")
    print(f"[DEBUG] script_path: {script_path}, exists: {os.path.exists(script_path)}")
    
    # 모델 파일이 명령줄 인자로 지정되지 않았으면 사용자에게 선택하게 함
    # sys.argv를 확인하여 model_file 파라미터가 있는지 확인
    import sys
    model_file_from_args = ''
    for i, arg in enumerate(sys.argv):
        if arg.startswith('model_file:='):
            model_file_from_args = arg.split(':=', 1)[1]
            break
    
    # 모델 파일이 지정되지 않았으면 사용자에게 선택하게 함
    if not model_file_from_args:
        import glob
        model_dir = os.path.join(package_dir, 'models')
        # Stage1 모델만 찾기
        stage1_files = sorted(
            [f for f in glob.glob(os.path.join(model_dir, 'stage1_*.pth')) + 
                    glob.glob(os.path.join(model_dir, 'stage1_*.pt'))
             if 'stage2' not in os.path.basename(f).lower() and 
                'config' not in os.path.basename(f).lower()],
            reverse=True
        )
        
        if not stage1_files:
            print("\n" + "=" * 60)
            print(f"[ERROR] No Stage1 models found in {model_dir}")
            print("[ERROR] Please train a model first.")
            print("=" * 60 + "\n")
            # 빈 LaunchDescription 반환 (노드 없음)
            return LaunchDescription([])
        
        # 사용자에게 모델 선택하게 함
        print("\n" + "=" * 60)
        print("[INFO] Model file not specified. Please select a model:")
        print("=" * 60)
        print(f"\nAvailable Stage1 models in {model_dir}:")
        for i, f in enumerate(stage1_files):
            print(f"  [{i}] {os.path.basename(f)}")
        print(f"  [{len(stage1_files)}] Cancel (exit)")
        
        while True:
            try:
                choice = input(f"\nSelect model [0-{len(stage1_files)}] (default: 0): ").strip()
                if choice == '':
                    choice = '0'
                
                choice_idx = int(choice)
                
                if choice_idx == len(stage1_files):
                    print("[INFO] Launch cancelled by user.")
                    return LaunchDescription([])
                
                if 0 <= choice_idx < len(stage1_files):
                    model_file_from_args = os.path.basename(stage1_files[choice_idx])
                    print(f"[INFO] Selected model: {model_file_from_args}")
                    print("=" * 60 + "\n")
                    break
                else:
                    print(f"[ERROR] Invalid choice. Please enter a number between 0 and {len(stage1_files)}.")
            except ValueError:
                print(f"[ERROR] Invalid input. Please enter a number between 0 and {len(stage1_files)}.")
            except KeyboardInterrupt:
                print("\n[INFO] Launch cancelled by user (Ctrl+C).")
                return LaunchDescription([])
    
    # 선택한 모델 파일을 환경 변수에 저장하여 launch_setup에서 사용
    os.environ['SELECTED_MODEL_FILE'] = model_file_from_args
    
    def launch_setup(context):
        actions = []
        
        # Launch arguments 값 가져오기
        model_file = context.launch_configurations.get('model_file', '')
        # 환경 변수에서 선택한 모델 파일 가져오기 (명령줄 인자가 없으면)
        if not model_file or model_file.strip() == '':
            model_file = os.environ.get('SELECTED_MODEL_FILE', '')
        
        log_rate_str = context.launch_configurations.get('log_rate', '100.0')
        # float로 변환 (ROS 2 파라미터는 문자열로 전달되므로)
        try:
            log_rate = float(log_rate_str)
        except (ValueError, TypeError):
            log_rate = 100.0
        
        # 모델 파일이 여전히 없으면 에러
        if not model_file or model_file.strip() == '':
            print("\n" + "=" * 60)
            print("[ERROR] Model file not specified!")
            print("=" * 60 + "\n")
            return actions
        
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
            # log_rate는 float로 전달 (ROS 2 파라미터는 문자열로 전달되지만, 노드에서 자동 변환됨)
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
                    'log_rate': log_rate,  # float 타입
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
