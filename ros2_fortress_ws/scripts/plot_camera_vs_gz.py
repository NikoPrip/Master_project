

#import the helper functions
from helper_functions import *
import matplotlib.pyplot as plt
import argparse

def handle_csv_data(csv_file):
    """
    Reads a merged CSV file and returns three arrays:
      - position_rad: list of position values in radians
      - time:         list of timestamps (time_s column)
      - poses:        list of [tx, ty, tz, qx, qy, qz, qw] per row
    """
    import csv

    position_rad = []
    time = []
    poses = []

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            position_rad.append(float(row['position_rad']))
            time.append(float(row['time_s']))
            poses.append([
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw']),
            ])

    return position_rad, time, poses


def argument_passer():
    parser = argparse.ArgumentParser(description="Calculate the marker frame in world coordinates for a given blade rotation angle.")
    parser.add_argument("--csv_file", type=str, help="path to csv file containing data from both gazebo and pose tracker")
    parser.add_argument("--is_displaced", type=lambda x: x.lower() in ('true', '1', 'yes'), help="flag to indicate if the marker center is displaced from the paper center", default=False)
    return parser.parse_args()

def blade_corners_in_world_from_joint_state(position_rad_list):
    corners = blade_corners_in_marker_frame()
    corners_list = []
    for position_rad in position_rad_list:
        T_world_marker_direct = marker_frame_in_world(position_rad)
        left_corner = T_world_marker_direct @ np.append(corners["left"], 1)
        right_corner = T_world_marker_direct @ np.append(corners["right"], 1)
        corners_list.append((left_corner, right_corner))
    return corners_list 

def calculate_camera_marker_transform(joint_state):
    T_world_marker = marker_frame_in_world(joint_state)
    T_world_camera = transform_matrix(
        translation=np.array([initial_link_poses.CAMERA_FRAME_TRANSFORM["translation"]["x"],
                            initial_link_poses.CAMERA_FRAME_TRANSFORM["translation"]["y"],
                            initial_link_poses.CAMERA_FRAME_TRANSFORM["translation"]["z"]]),
        fixed_angles=np.array([initial_link_poses.CAMERA_FRAME_TRANSFORM["rotation"]["roll"],
                                initial_link_poses.CAMERA_FRAME_TRANSFORM["rotation"]["pitch"],
                                initial_link_poses.CAMERA_FRAME_TRANSFORM["rotation"]["yaw"]]))
    T_camera_marker = np.linalg.inv(T_world_camera) @ T_world_marker
    return T_camera_marker

def blade_corners_in_world_from_vision(poses, is_displaced=False):
    corners_list = []
    T_world_camera = transform_matrix(
        translation=np.array([initial_link_poses.CAMERA_FRAME_TRANSFORM["translation"]["x"],
                            initial_link_poses.CAMERA_FRAME_TRANSFORM["translation"]["y"],
                            initial_link_poses.CAMERA_FRAME_TRANSFORM["translation"]["z"]]),
        fixed_angles=np.array([initial_link_poses.CAMERA_FRAME_TRANSFORM["rotation"]["roll"],
                                initial_link_poses.CAMERA_FRAME_TRANSFORM["rotation"]["pitch"],
                                initial_link_poses.CAMERA_FRAME_TRANSFORM["rotation"]["yaw"]]))
    for pose_in_mm in poses:
        pose_in_m = tuple(val / 1000.0 for val in pose_in_mm)
        T_camera_marker = marker_camera_transform(*pose_in_m, marker_offset=is_displaced)

        # temporary fix
        #T_camera_marker[:, 0] *= -1
        #T_camera_marker[:, 2] *= -1
        T_world_marker_from_vision = T_world_camera @ T_camera_marker


        corners = blade_corners_in_marker_frame()
        left_corner = T_world_marker_from_vision @ np.append(corners["left"], 1)
        right_corner = T_world_marker_from_vision @ np.append(corners["right"], 1)
        corners_list.append((left_corner, right_corner))
    return corners_list

def plot_corners(corners_from_joint_state, corners_from_vision, time):
    js_left  = np.array([left[:3]  for left, right in corners_from_joint_state])
    js_right = np.array([right[:3] for left, right in corners_from_joint_state])
    vis_left  = np.array([left[:3]  for left, right in corners_from_vision])
    vis_right = np.array([right[:3] for left, right in corners_from_vision])

    # rmse
    print("RMSE Left Corner:", rmse(js_left, vis_left))
    print("RMSE Right Corner:", rmse(js_right, vis_right))

    axis_labels = ['X (m)', 'Y (m)', 'Z (m)']
    # 3 rows (X, Y, Z) x 2 columns (left, right)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    for i, label in enumerate(axis_labels):
        # Left corner column
        axes[i, 0].plot(time, js_left[:, i],  color='blue', label='Joint State')
        axes[i, 0].plot(time, vis_left[:, i], color='red',  label='Vision')
        axes[i, 0].set_ylabel(label)
        axes[i, 0].set_title(f'Left Corner {label} vs Time')
        axes[i, 0].legend()
        axes[i, 0].grid(True)

        # Right corner column
        axes[i, 1].plot(time, js_right[:, i],  color='blue', label='Joint State')
        axes[i, 1].plot(time, vis_right[:, i], color='red',  label='Vision')
        axes[i, 1].set_ylabel(label)
        axes[i, 1].set_title(f'Right Corner {label} vs Time')
        axes[i, 1].legend()
        axes[i, 1].grid(True)

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    fig.suptitle('Blade Corners in World Frame')
    plt.tight_layout()
    plt.show()

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


if __name__ == "__main__":
    # Calculate the marker frame in world coordinates for the given blade rotation angle
    args = argument_passer()
    position_rad, time, poses = handle_csv_data(args.csv_file)

    corners_from_world_to_marker = blade_corners_in_world_from_joint_state(position_rad)
    corners_from_vision = blade_corners_in_world_from_vision(poses, is_displaced=args.is_displaced)

    plot_corners(corners_from_world_to_marker, corners_from_vision, time)

    # print transform from camera to marker for the 500th data point as an example
    T_camera_marker_example = calculate_camera_marker_transform(position_rad[500])
    print("Transform from camera to marker for the 500th data point:")
    print(T_camera_marker_example)

    print("Transform from camera to marker from vision for the 500th data point:")
    poses_500 = tuple(val / 1000.0 for val in poses[500])  # convert from mm to m
    T_camera_marker_vision_example = marker_camera_transform(*poses_500, marker_offset=args.is_displaced) 
    # multiply first and third columns of T_camera_marker_vision_example by -1 to account for the 180 deg rotation around the camera's x-axis
    T_camera_marker_vision_example_copy = T_camera_marker_vision_example @ np.diag([-1, 1, -1, 1])
    print(T_camera_marker_vision_example_copy)

    #T_camera_marker_vision_example[:, 0] *= -1
    #T_camera_marker_vision_example[:, 2] *= -1
    print("Transform from camera to marker from vision for the 500th data point (after accounting for 180 deg rotation):")
    print(T_camera_marker_vision_example)

