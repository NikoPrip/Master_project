import os
import sys
import numpy as np
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config.initial_link_poses as initial_link_poses

def fixed_rotation_matrix(roll_rad, pitch_rad, yaw_rad):
    # Compute the fixed rotation matrix from roll, pitch, yaw angles (in radians)
    cr = np.cos(roll_rad)
    sr = np.sin(roll_rad)
    cp = np.cos(pitch_rad)
    sp = np.sin(pitch_rad)
    cy = np.cos(yaw_rad)
    sy = np.sin(yaw_rad)

    # Rotation matrix for roll
    R_roll = np.array([[1, 0, 0],
                       [0, cr, -sr],
                       [0, sr, cr]])

    # Rotation matrix for pitch
    R_pitch = np.array([[cp, 0, sp],
                        [0, 1, 0],
                        [-sp, 0, cp]])

    # Rotation matrix for yaw
    R_yaw = np.array([[cy, -sy, 0],
                      [sy, cy, 0],
                      [0, 0, 1]])

    # Combined rotation matrix (R = R_yaw * R_pitch * R_roll)
    R_combined = R_yaw @ R_pitch @ R_roll
    return R_combined

def transform_matrix(translation: np.ndarray, fixed_angles: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transformation matrix from translation and rotation.

    Args:
        translation: 3-element array (tx, ty, tz) in metres.
        fixed_angles: 3-element array (roll, pitch, yaw) in radians.

    Returns:
        4x4 homogeneous transformation matrix.
    """
    rotation_matrix = fixed_rotation_matrix(fixed_angles[0], fixed_angles[1], fixed_angles[2])
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform

def blade_corners_in_world_from_transform_matrix(grader_blade_measurements, blade_rotation_rad:float):
    translation_vector = [initial_link_poses.Grader_blade_measurements["blade_origin_in_world"]["x"],
                            initial_link_poses.Grader_blade_measurements["blade_origin_in_world"]["y"],
                            initial_link_poses.Grader_blade_measurements["blade_origin_in_world"]["z"]]
    fixed_angles = [0,0,blade_rotation_rad]
    transform = transform_matrix(translation_vector, fixed_angles)
    left_blade_corner_in_blade_frame = np.array([0,grader_blade_measurements["blade_width"]/2, grader_blade_measurements["blade_height"]/2, 1])
    right_blade_corner_in_blade_frame = np.array([0,-grader_blade_measurements["blade_width"]/2, grader_blade_measurements["blade_height"]/2, 1])
    left_blade_corner_in_world = transform @ left_blade_corner_in_blade_frame
    right_blade_corner_in_world = transform @ right_blade_corner_in_blade_frame
    corners = {
        "left":  (left_blade_corner_in_world[0], left_blade_corner_in_world[1], left_blade_corner_in_world[2]),
        "right": (right_blade_corner_in_world[0], right_blade_corner_in_world[1], right_blade_corner_in_world[2]),
    }
    return corners

def blade_corners_in_marker_frame():
    # calculate the fixed position of the blade corners in the marker frame (x forward, y left, z up)
    blade_width = initial_link_poses.Grader_blade_measurements["blade_width"]
    blade_height = initial_link_poses.Grader_blade_measurements["blade_height"]
    # Corners in world frame relative to marker center
    right_corner_world = [initial_link_poses.Grader_blade_measurements["blade_origin_in_world"]["x"],
                          initial_link_poses.Grader_blade_measurements["blade_origin_in_world"]["y"] - blade_width/2,
                          initial_link_poses.Grader_blade_measurements["blade_origin_in_world"]["z"] + blade_height/2]
    left_corner_world = [initial_link_poses.Grader_blade_measurements["blade_origin_in_world"]["x"],
                         initial_link_poses.Grader_blade_measurements["blade_origin_in_world"]["y"] + blade_width/2,
                         initial_link_poses.Grader_blade_measurements["blade_origin_in_world"]["z"] + blade_height/2]


    grader_blade_right_corner = np.array(right_corner_world) - np.array([initial_link_poses.MARKER_CENTER_POSE["x"],
                                                                      initial_link_poses.MARKER_CENTER_POSE["y"],
                                                                      initial_link_poses.MARKER_CENTER_POSE["z"]])
    grader_blade_left_corner = np.array(left_corner_world) - np.array([initial_link_poses.MARKER_CENTER_POSE["x"],
                                                                     initial_link_poses.MARKER_CENTER_POSE["y"],
                                                                     initial_link_poses.MARKER_CENTER_POSE["z"]])
    world_frame_to_marker_frame_rotation = fixed_rotation_matrix(initial_link_poses.MARKER_CENTER_POSE["roll"],
                                                              initial_link_poses.MARKER_CENTER_POSE["pitch"],
                                                              initial_link_poses.MARKER_CENTER_POSE["yaw"])
    grader_blade_right_corner = world_frame_to_marker_frame_rotation.T @ grader_blade_right_corner
    grader_blade_left_corner = world_frame_to_marker_frame_rotation.T @ grader_blade_left_corner
    corners = {
        "left":  grader_blade_left_corner,
        "right": grader_blade_right_corner,
    }
    return corners

def marker_camera_transform(tx: float, ty: float, tz: float,
                            qx: float, qy: float, qz: float, qw: float,
                            marker_offset: bool = False) -> np.ndarray:
    """Build a 4x4 homogeneous transformation matrix from a translation and quaternion.

    Args:
        tx, ty, tz: Translation components (metres).
        qx, qy, qz, qw: Quaternion components (unit quaternion).
        marker_offset: If True, apply marker offset correction (default False).

    Returns:
        4x4 numpy array representing the transformation matrix T = [R | t; 0 0 0 1].
    """
    # Normalise quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [tx, ty, tz]

    if marker_offset:
        T_copy = T @ transform_matrix(
            translation=np.array([initial_link_poses.NFold_Center_Offset["x"],
                                initial_link_poses.NFold_Center_Offset["y"],
                                initial_link_poses.NFold_Center_Offset["z"]]),
            fixed_angles=np.array([0, 0, 0])  # No rotation for the marker offset
        )
        return T_copy
    else:
        return T

def marker_frame_in_world(joint_angle_rad: float):
    # Calculate the marker frame in world coordinates for a given blade rotation angle
    marker_rotation_axis_length = initial_link_poses.MARKER_CENTER_POSE["y"]
    marker_new_x = initial_link_poses.MARKER_CENTER_POSE["x"] - marker_rotation_axis_length * np.sin(joint_angle_rad)
    marker_new_y = marker_rotation_axis_length * np.cos(joint_angle_rad)
    marker_new_z = initial_link_poses.MARKER_CENTER_POSE["z"]
    T_marker_to_world = transform_matrix(
        translation=np.array([marker_new_x, marker_new_y, marker_new_z]),
        fixed_angles=np.array([initial_link_poses.MARKER_CENTER_POSE["roll"], initial_link_poses.MARKER_CENTER_POSE["pitch"], initial_link_poses.MARKER_CENTER_POSE["yaw"]+joint_angle_rad])
    )
    return T_marker_to_world

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate blade corner positions in world frame for a given rotation angle.")
    parser.add_argument("--rotation_deg", type=float, default=0.0, help="Blade rotation angle in degrees (0 = flat, positive CCW)")
    args = parser.parse_args()
    #blade_rotation_rad = np.deg2rad(args.rotation_deg)
    blade_rotation_rad = -3.7e-05  # rad, from Gazebo model

    # Calculate the marker frame in world coordinates for the given blade rotation angle
    T_world_marker_direct = marker_frame_in_world(blade_rotation_rad)

    # calculate the blade corners in world frame from the transform matrix approach (using the known blade origin and rotation)
    corners_from_transform = blade_corners_in_world_from_transform_matrix(initial_link_poses.Grader_blade_measurements, blade_rotation_rad)
    print("Blade corners in world frame from transform matrix (x, y):")
    for side, corner in corners_from_transform.items():
        print(f"  {side}: {corner}")

    blade_corners_in_marker_frame = blade_corners_in_marker_frame()
    print("Blade corners in marker frame (x forward, y left, z up):")
    for side, corner in blade_corners_in_marker_frame.items():
        print(f"  {side}: {corner}")
    transform_data_in_mm = (24.509,100.1904,1174.0572,0.924046,-0.000273,0.000212,0.382282)
    transform_data_in_m = tuple(val/1000 if i < 3 else val for i, val in enumerate(transform_data_in_mm))
    T_camera_marker = marker_camera_transform(*transform_data_in_m)
    print("Camera to marker transform from csv:\n", T_camera_marker)

    T_camera_marker_direct = transform_matrix([initial_link_poses.MARKER_CENTER_POSE["x"]-initial_link_poses.CAMERA_FRAME_TRANSFORM["translation"]["x"],
                                              -initial_link_poses.MARKER_CENTER_POSE["z"]+initial_link_poses.CAMERA_FRAME_TRANSFORM["translation"]["z"],
                                              initial_link_poses.MARKER_CENTER_POSE["y"]-initial_link_poses.CAMERA_FRAME_TRANSFORM["translation"]["y"]],
                                              [initial_link_poses.MARKER_CENTER_POSE["roll"]-initial_link_poses.CAMERA_FRAME_TRANSFORM["rotation"]["roll"],
                                              initial_link_poses.MARKER_CENTER_POSE["pitch"]-initial_link_poses.CAMERA_FRAME_TRANSFORM["rotation"]["pitch"],
                                              initial_link_poses.MARKER_CENTER_POSE["yaw"]-initial_link_poses.CAMERA_FRAME_TRANSFORM["rotation"]["yaw"]])
    print("Camera to marker transform from direct calculation:\n", T_camera_marker_direct)


    T_world_camera = transform_matrix(
        translation=np.array([initial_link_poses.CAMERA_FRAME_TRANSFORM["translation"]["x"],
                            initial_link_poses.CAMERA_FRAME_TRANSFORM["translation"]["y"],
                            initial_link_poses.CAMERA_FRAME_TRANSFORM["translation"]["z"]]),
        fixed_angles=np.array([initial_link_poses.CAMERA_FRAME_TRANSFORM["rotation"]["roll"],
                                initial_link_poses.CAMERA_FRAME_TRANSFORM["rotation"]["pitch"],
                                initial_link_poses.CAMERA_FRAME_TRANSFORM["rotation"]["yaw"]]))
    print("World to camera transform:\n", T_world_camera)

    T_world_marker_chain = T_world_camera @ T_camera_marker
    print("World to marker transform from chain (world to camera to marker):\n", T_world_marker_chain)

    T_world_marker_chain_truth = T_world_camera @ T_camera_marker_direct
    print("World to marker transform from chain with direct camera to marker (world to camera to marker):\n", T_world_marker_chain_truth)

    print("world maker direct\n", T_world_marker_direct)

    corners_from_chain = {
        "left": T_world_marker_chain @ np.append(blade_corners_in_marker_frame["left"], 1),
        "right": T_world_marker_chain @ np.append(blade_corners_in_marker_frame["right"], 1),
    }
    print("Blade corners in world frame from chain (x, y):")
    for side, corner in corners_from_chain.items():
        print(f"  {side}: {corner[:3]}")

    corners_from_direct = {
        "left": T_world_marker_direct @ np.append(blade_corners_in_marker_frame["left"], 1),
        "right": T_world_marker_direct @ np.append(blade_corners_in_marker_frame["right"], 1),
    }
    print("Blade corners in world frame from direct marker transform (x, y):")
    for side, corner in corners_from_direct.items():
        print(f"  {side}: {corner[:3]}")

    corners_from_chain_truth = {
        "left": T_world_marker_chain_truth @ np.append(blade_corners_in_marker_frame["left"], 1),
        "right": T_world_marker_chain_truth @ np.append(blade_corners_in_marker_frame["right"], 1),
    }
    print("Blade corners in world frame from chain with direct camera to marker (x, y):")
    for side, corner in corners_from_chain_truth.items():
        print(f"  {side}: {corner[:3]}")

   