from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'ros2_fortress_ws'

# Helper function to recursively get all files in a directory
def get_all_files(directory):
    """Get all files in a directory recursively."""
    result = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            relpath = os.path.relpath(filepath, '.')
            result.append(relpath)
    return result

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'models'), get_all_files('models')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='johan',
    maintainer_email='johan@example.com',
    description='ROS2 Gazebo Fortress workspace',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'pose_frame_publisher=ros2_fortress_ws.pose_frame_publisher:main',
        ],
    },
)
