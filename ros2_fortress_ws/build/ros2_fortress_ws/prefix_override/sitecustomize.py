import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/johan/Documents/masters/ros2_fortress_ws/install/ros2_fortress_ws'
