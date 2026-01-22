from setuptools import setup

package_name = 'self_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[
        'self_detection',
        'self_detection_mlp',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/two_stage.launch.py',
            'launch/mlp.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='song',
    maintainer_email='song@todo.todo',
    description='Two-stage self detection with MLP + TCN',
    license='MIT',
    tests_require=['pytest'],

    ### 🔴 이게 없어서 지금 에러 난 거다
    entry_points={
        'console_scripts': [
            'realtime_monitor_two_stage = self_detection.realtime_monitor_two_stage:main',
            'realtime_monitor_mlp = self_detection.realtime_monitor_mlp:main',
        ],
    },
)

