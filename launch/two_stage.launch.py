from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='self_detection',
            executable='realtime_monitor_two_stage',
            name='self_detection_node',
            output='screen',
            parameters=[
                {'seq_len': 10},
                # {'config_file': 'two_stage_config_20260121_211733.pt'},
            ]
        )
    ])

