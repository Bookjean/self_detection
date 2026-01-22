"""MLP-only self detection launch file.

Usage:
    ros2 launch self_detection mlp.launch.py
    ros2 launch self_detection mlp.launch.py model_file:=stage1_stage1_mlp_20260122.pth
    ros2 launch self_detection mlp.launch.py log_rate:=50.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
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

        # MLP inference node
        Node(
            package='self_detection',
            executable='realtime_monitor_mlp',
            name='self_detection_mlp',
            output='screen',
            parameters=[{
                'model_file': LaunchConfiguration('model_file'),
                'log_rate': LaunchConfiguration('log_rate'),
            }]
        )
    ])
