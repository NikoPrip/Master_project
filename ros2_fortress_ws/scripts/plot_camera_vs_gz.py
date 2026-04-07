

#import the helper functions
from helper_functions import *
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

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

        T_world_marker_from_vision = T_world_camera @ T_camera_marker


        corners = blade_corners_in_marker_frame()
        left_corner = T_world_marker_from_vision @ np.append(corners["left"], 1)
        right_corner = T_world_marker_from_vision @ np.append(corners["right"], 1)
        corners_list.append((left_corner, right_corner))
    return corners_list

def plot_corners(corners_from_joint_state, corners_from_vision, time, marker_type='unknown', save_dir=None):
    js_left  = np.array([left[:3]  for left, right in corners_from_joint_state])
    js_right = np.array([right[:3] for left, right in corners_from_joint_state])
    vis_left  = np.array([left[:3]  for left, right in corners_from_vision])
    vis_right = np.array([right[:3] for left, right in corners_from_vision])

    # rmse
    print("RMSE Left Corner:", combined_rmse(*[rmse(js_left[:, i], vis_left[:, i]) for i in range(3)]))
    print("RMSE Right Corner:", combined_rmse(*[rmse(js_right[:, i], vis_right[:, i]) for i in range(3)]))

    #print("RMSE Left Corner (no z):", rmse_no_z(js_left, vis_left))
    #print("RMSE Right Corner (no z):", rmse_no_z(js_right, vis_right))

    axis_labels = ['X (m)', 'Y (m)', 'Z (m)']
    left_data  = [(js_left[:, i],  vis_left[:, i])  for i in range(3)]
    right_data = [(js_right[:, i], vis_right[:, i]) for i in range(3)]

    # --- Left corner plot ---
    fig_l, axes_l = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axis_errs_l = []
    for i, (label, (js, vis)) in enumerate(zip(axis_labels, left_data)):
        err = np.sqrt(np.mean((js - vis) ** 2))
        axis_errs_l.append(err)
        axes_l[i].plot(time, js,  color='blue', label='Joint State')
        axes_l[i].plot(time, vis, color='red',  label='Vision')
        axes_l[i].set_ylabel(label)
        axes_l[i].set_title(f'Left Corner {label} vs Time  |  RMSE: {err:.4f}')
        axes_l[i].legend()
        axes_l[i].grid(True)
    axes_l[-1].set_xlabel('Time (s)')
    err_combined = combined_rmse(*axis_errs_l)
    fig_l.suptitle(f'[{marker_type}] Blade Left Corner in World Frame: Joint State vs Vision  |  RMSE: {err_combined:.4f}')
    plt.tight_layout()
    if save_dir is not None:
        fig_l.savefig(Path(save_dir) / f'{marker_type}_left_corner_in_world.png', dpi=150)

    # --- Right corner plot ---
    fig_r, axes_r = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axis_errs_r = []
    for i, (label, (js, vis)) in enumerate(zip(axis_labels, right_data)):
        err = np.sqrt(np.mean((js - vis) ** 2))
        axis_errs_r.append(err)
        axes_r[i].plot(time, js,  color='blue', label='Joint State')
        axes_r[i].plot(time, vis, color='red',  label='Vision')
        axes_r[i].set_ylabel(label)
        axes_r[i].set_title(f'Right Corner {label} vs Time  |  RMSE: {err:.4f}')
        axes_r[i].legend()
        axes_r[i].grid(True)
    axes_r[-1].set_xlabel('Time (s)')
    err_combined = combined_rmse(*axis_errs_r)
    fig_r.suptitle(f'[{marker_type}] Blade Right Corner in World Frame: Joint State vs Vision  |  RMSE: {err_combined:.4f}')
    plt.tight_layout()
    if save_dir is not None:
        fig_r.savefig(Path(save_dir) / f'{marker_type}_right_corner_in_world.png', dpi=150)

    plt.show()

def plot_rot_matrix_elements(position_rad, time, poses, is_displaced=False, marker_type='unknown', save_dir=None):
    from scipy.spatial.transform import Rotation

    js_tx, js_ty, js_tz = [], [], []
    js_rx, js_ry, js_rz = [], [], []
    vis_tx, vis_ty, vis_tz = [], [], []
    vis_rx, vis_ry, vis_rz = [], [], []

    for joint, pose_in_mm in zip(position_rad, poses):
        # Joint-state based T_camera_marker
        T_js = calculate_camera_marker_transform(joint)
        js_tx.append(T_js[0, 3])
        js_ty.append(T_js[1, 3])
        js_tz.append(T_js[2, 3])
        r_js = Rotation.from_matrix(T_js[:3, :3]).as_euler('xyz', degrees=True)
        js_rx.append(r_js[0])
        js_ry.append(r_js[1])
        js_rz.append(r_js[2])

        # Vision-based T_camera_marker
        pose_in_m = tuple(val / 1000.0 for val in pose_in_mm)
        T_vis = marker_camera_transform(*pose_in_m, marker_offset=is_displaced)
        vis_tx.append(T_vis[0, 3])
        vis_ty.append(T_vis[1, 3])
        vis_tz.append(T_vis[2, 3])
        r_vis = Rotation.from_matrix(T_vis[:3, :3]).as_euler('xyz', degrees=True)
        vis_rx.append(r_vis[0])
        vis_ry.append(r_vis[1])
        vis_rz.append(r_vis[2])

    time = np.array(time)
    t_labels  = ['tx (m)', 'ty (m)', 'tz (m)']
    t_js_data = [js_tx,  js_ty,  js_tz]
    t_vis_data = [vis_tx, vis_ty, vis_tz]

    r_labels  = ['rx (deg)', 'ry (deg)', 'rz (deg)']
    r_js_data = [js_rx,  js_ry,  js_rz]
    r_vis_data = [vis_rx, vis_ry, vis_rz]

    # --- Translation plot ---
    fig_t, axes_t = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    t_axis_errs = []
    for i, (label, js, vis) in enumerate(zip(t_labels, t_js_data, t_vis_data)):
        err = np.sqrt(np.mean((np.array(js) - np.array(vis)) ** 2))
        t_axis_errs.append(err)
        axes_t[i].plot(time, js,  color='blue', label='Joint State')
        axes_t[i].plot(time, vis, color='red',  label='Vision')
        axes_t[i].set_ylabel(label)
        axes_t[i].set_title(f'T_camera_marker {label} vs Time  |  RMSE: {err:.4f}')
        axes_t[i].legend()
        axes_t[i].grid(True)
    axes_t[-1].set_xlabel('Time (s)')
    err_combined = combined_rmse(*t_axis_errs)
    fig_t.suptitle(f'[{marker_type}] T_camera_marker Translations: Joint State vs Vision  |  RMSE: {err_combined:.4f}')
    plt.tight_layout()
    if save_dir is not None:
        fig_t.savefig(Path(save_dir) / f'{marker_type}_camera_marker_translation.png', dpi=150)

    # --- Rotation plot ---
    fig_r, axes_r = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    r_axis_errs = []
    for i, (label, js, vis) in enumerate(zip(r_labels, r_js_data, r_vis_data)):
        err = np.sqrt(np.mean((np.array(js) - np.array(vis)) ** 2))
        r_axis_errs.append(err)
        axes_r[i].plot(time, js,  color='blue', label='Joint State')
        axes_r[i].plot(time, vis, color='red',  label='Vision')
        axes_r[i].set_ylabel(label)
        axes_r[i].set_title(f'T_camera_marker {label} vs Time  |  RMSE: {err:.4f}')
        axes_r[i].legend()
        axes_r[i].grid(True)
    axes_r[-1].set_xlabel('Time (s)')
    err_combined = combined_rmse(*r_axis_errs)
    fig_r.suptitle(f'[{marker_type}] T_camera_marker Rotations: Joint State vs Vision  |  RMSE: {err_combined:.4f}')
    plt.tight_layout()
    if save_dir is not None:
        fig_r.savefig(Path(save_dir) / f'{marker_type}_camera_marker_rotation.png', dpi=150)

    plt.show()


def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

def combined_rmse(rmse_x, rmse_y, rmse_z):
    return np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2)

def rmse_no_z(predictions, targets):
    return np.sqrt(np.mean((predictions[:, :2] - targets[:, :2]) ** 2))


if __name__ == "__main__":
    # Calculate the marker frame in world coordinates for the given blade rotation angle
    args = argument_passer()
    position_rad, time, poses = handle_csv_data(args.csv_file)

    marker_type = Path(args.csv_file).stem.replace('_merged', '')
    save_dir = Path(args.csv_file).parent

    corners_from_world_to_marker = blade_corners_in_world_from_joint_state(position_rad)
    corners_from_vision = blade_corners_in_world_from_vision(poses, is_displaced=args.is_displaced)

    plot_corners(corners_from_world_to_marker, corners_from_vision, time, marker_type=marker_type, save_dir=save_dir)
    plot_rot_matrix_elements(position_rad, time, poses, is_displaced=args.is_displaced, marker_type=marker_type, save_dir=save_dir)

    # print transform from camera to marker for the 500th data point as an example
    #T_camera_marker_example = calculate_camera_marker_transform(position_rad[500])
    #print("Transform from camera to marker for the 500th data point:")
    #print(T_camera_marker_example)

    #print("Transform from camera to marker from vision for the 500th data point:")
    #poses_500 = tuple(val / 1000.0 for val in poses[500])  # convert from mm to m
    #T_camera_marker_vision_example = marker_camera_transform(*poses_500, marker_offset=args.is_displaced) 

    #print("Transform from camera to marker from vision for the 500th data point:")
    #print(T_camera_marker_vision_example)

