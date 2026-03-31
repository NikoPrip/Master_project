import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for Gazebo Fortress with empty world and Grader model.
    Includes ros_gz bridge to connect Gazebo to ROS2.
    """
    
    pkg_dir = get_package_share_directory("ros2_fortress_ws")
    worlds_dir = os.path.join(pkg_dir, "worlds")
    
    # Point to the source models directory to preserve directory structure
    # The workspace root is three levels up from pkg_dir (/install/ros2_fortress_ws/share/ros2_fortress_ws)
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pkg_dir))))
    models_dir = os.path.join(workspace_root, "models")
    world_file = "empty.sdf"
    empty_world = os.path.join(worlds_dir, world_file)
    
    # Start Gazebo Fortress 
    gazebo = ExecuteProcess(
        cmd=["ign", "gazebo", "-v", "4", "-r", empty_world],
        output="screen",
        env={
            **os.environ,
            "IGN_GAZEBO_RESOURCE_PATH": models_dir,
        }
    )
    
    # Start ros_gz bridge for blade rotation joint control
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            # Grader topics
            "/blade_rotation_joint/position@std_msgs/msg/Float64]ignition.msgs.Double",
            "/blade_rotation_joint/state@sensor_msgs/msg/JointState[ignition.msgs.Model",
            "/grader/camera@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/grader/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",


            # Callibration topics
            #"/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image",
            #"/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
            #"/tilt_joint/cmd_pos@std_msgs/msg/Float64]ignition.msgs.Double",
            #"/tilt_joint/state@sensor_msgs/msg/JointState[ignition.msgs.Model",
            #"/pan_joint/cmd_pos@std_msgs/msg/Float64]ignition.msgs.Double",
            #"/pan_joint/state@sensor_msgs/msg/JointState[ignition.msgs.Model",
        ],
        output="screen",
    )
    
    return LaunchDescription([gazebo, bridge])