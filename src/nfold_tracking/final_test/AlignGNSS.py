"""
AlignGNSS.py — Align ArUco tracker output with GNSS ground truth.

Time alignment
--------------
aruco.csv        : time_s          (video seconds from frame 0)
gnss_3730117.csv : video_time_s    (same video clock, 1 Hz)
gnss_3730072.csv : timestamp_utc   (HHMMSS.SS GPS UTC → video_time_s via meta)

Spatial offsets
---------------
CAM_T_ANT2_17:  position of 3730117 ANT2 in camera frame (mm)
                Physically: 100 mm left, 135 mm up, 50 mm behind camera
                OpenCV convention: x right, y down, z forward

ANT2 (3730072):
  ARUCO_REF_ANT2:  board top, 28 cm behind board face  [3, 135, -280] mm
  WORLD_VERT_ANT2: 2300 mm straight up in world frame

ANT1 (3730072):
  ARUCO_REF_ANT1:  board top, 2.95 m in front of board face (toward camera)  [3, 135, +2950] mm
  WORLD_VERT_ANT1: 1590 mm straight up in world frame
  Note: ANT1 ends up geometrically behind the camera — the math is still valid.

Board frame convention: x right, y up, z out of board (toward camera).
World vertical in camera frame: [0, -cos(α), -sin(α)] × height_mm
  where α is the camera tilt downward angle.

Comparison
----------
For each tracker frame the 6DOF pose (tx,ty,tz + quaternion) is used to
project the ANT offsets into camera frame and compute distances from
3730117 ANT2.  GNSS distances are computed from UTM coordinates.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_TILT_DEG = 25.0
_alpha = np.radians(CAMERA_TILT_DEG)

# World-up unit vector in camera frame (camera tilts DOWN by alpha around X)
_world_up_cam = np.array([0.0, -np.cos(_alpha), -np.sin(_alpha)])

def world_vert(height_mm):
    """height_mm straight up in world → vector in camera frame (mm)."""
    return height_mm * _world_up_cam

# 3730117 ANT2 in camera frame (mm)  [x right, y down, z forward]
CAM_T_ANT2_17 = np.array([-100.0, -135.0, -50.0], dtype=np.float64)

# Fixed physical offset between GNSS dual-antenna heading and camera pointing direction.
# Calibrated from aruco_2 and aruco_dirt runs (both gave -5.4° to -5.8°).
CAMERA_HEADING_OFFSET_DEG = -5.6

# 3730072 ANT2: board-frame reference + world vertical
ARUCO_REF_ANT2   = np.array([3.0, 135.0, -280.0],   dtype=np.float64)  # board frame
WORLD_VERT_ANT2  = world_vert(2300.0)                                    # camera frame

# 3730072 ANT1: board-frame reference + world vertical
# Board face tilts upward at β=25° (measured physically, matches Wahba camera tilt).
# Board Z has sin(25°)≈0.423 upward component per mm, so a large Z offset would
# add spurious world-height.  Compensate with delta_Y = -3233*sin(β), delta_Z = 3233*cos(β)
# where 3233 mm is derived from horizontal GNSS blade extent (≈3241mm) with azimuth correction.
#   delta_Z = 3233 * cos(25°) = 2930 mm
#   delta_Y = -3233 * sin(25°) = -1366 mm
#   ARUCO_REF_ANT1 = [3+217, 135-1366, -280+2930] = [220, -1231, 2650]
ARUCO_REF_ANT1  = np.array([3.0, -1231.0, 2650.0], dtype=np.float64)  # board frame
WORLD_VERT_ANT1 = world_vert(1660.0)                                     # camera frame


# ---------------------------------------------------------------------------
# Paths  (resolved in main() from --name / --mode)
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nmea_utc_to_s(ts) -> float:
    ts = f"{float(ts):09.2f}"   # restore leading zero dropped by pandas float parsing
    return int(ts[0:2]) * 3600 + int(ts[2:4]) * 60 + float(ts[4:])


def make_R_ENU_cam(heading_deg, tilt_deg):
    """Build cam→ENU rotation matrix from camera heading and downward tilt."""
    h = np.radians(heading_deg)
    a = np.radians(tilt_deg)
    cam_x = np.array([np.cos(h), -np.sin(h), 0.0])
    cam_z = np.array([np.sin(h) * np.cos(a), np.cos(h) * np.cos(a), -np.sin(a)])
    cam_y = np.cross(cam_z, cam_x)
    return np.column_stack([cam_x, cam_y, cam_z])


_to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)


def latlon_to_utm(lat_arr, lon_arr):
    e, n = _to_utm.transform(lon_arr, lat_arr)
    return np.asarray(e), np.asarray(n)


def interp_to(t_target, t_src, values):
    return np.interp(t_target, t_src, values, left=np.nan, right=np.nan)


def quat_to_R(qw, qx, qy, qz):
    """Quaternion [qw, qx, qy, qz] → 3×3 rotation matrix (board → camera)."""
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ], dtype=np.float64)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Align tracker CSV with GNSS ground truth.')
    parser.add_argument('--name', default='output',
                        help="Session name used to locate all files "
                             "(e.g. --name aruco_1 reads aruco_aruco_1.csv, "
                             "gnss_log_3730117_aruco_1.csv, gnss_log_3730072_aruco_1.csv)")
    parser.add_argument('--mode', default='aruco',
                        choices=['aruco', 'hybrid', 'nfold'],
                        help='Tracker mode; determines which results CSV to read (default: aruco)')
    args = parser.parse_args()

    name = args.name
    mode = args.mode

    tracker_csv = BASE / 'results'   / f'{mode}_{name}.csv'
    gnss17_csv  = BASE / 'csv_files' / f'gnss_log_3730117_{name}.csv'
    gnss72_csv  = BASE / 'csv_files' / f'gnss_log_3730072_{name}.csv'
    out_csv     = BASE / 'results'   / f'aligned_{mode}_{name}.csv'

    print(f"Mode        : {mode}")
    print(f"Name        : {name}")
    print(f"Tracker CSV : {tracker_csv}")
    print(f"GNSS 117    : {gnss17_csv}")
    print(f"GNSS 072    : {gnss72_csv}")

    # ---------------------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------------------

    aruco = pd.read_csv(tracker_csv)
    g17   = pd.read_csv(gnss17_csv)
    g72   = pd.read_csv(gnss72_csv)

    # Read PIPELINE_GPS_UTC_S from meta file written by VideoGnssCapture.py
    meta_path = gnss17_csv.with_suffix('.meta.txt')
    pipeline_gps_utc_s = None
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if line.startswith('pipeline_gps_utc_s='):
                pipeline_gps_utc_s = float(line.split('=', 1)[1])
                break
    if pipeline_gps_utc_s is None:
        raise RuntimeError(f"Could not read pipeline_gps_utc_s from {meta_path}")
    print(f"Pipeline GPS UTC start: {pipeline_gps_utc_s:.3f}s  (from {meta_path.name})")

    t   = aruco['time_s'].values
    t17 = g17['video_time_s'].values
    g72['video_time_s'] = g72['timestamp_utc'].apply(nmea_utc_to_s) - pipeline_gps_utc_s
    t72 = g72['video_time_s'].values

    print(f"\nTracker     : {len(aruco)} rows  {t[0]:.2f}s → {t[-1]:.2f}s")
    print(f"GNSS 3730117: {len(g17)} rows  {t17[0]:.2f}s → {t17[-1]:.2f}s")
    print(f"GNSS 3730072: {len(g72)} rows  {t72[0]:.2f}s → {t72[-1]:.2f}s")

    # ---------------------------------------------------------------------------
    # Convert to UTM
    # ---------------------------------------------------------------------------

    g17_ant2_e, g17_ant2_n = latlon_to_utm(
        g17['ant2_latitude_deg'].values, g17['ant2_longitude_deg'].values)

    g72_ant2_e, g72_ant2_n = latlon_to_utm(
        g72['ant2_latitude_deg'].values, g72['ant2_longitude_deg'].values)

    g72_ant1_e, g72_ant1_n = latlon_to_utm(
        g72['latitude_deg'].values, g72['longitude_deg'].values)

    # ---------------------------------------------------------------------------
    # Interpolate GNSS → tracker timestamps
    # ---------------------------------------------------------------------------

    aruco['g17_ant2_e_m'] = interp_to(t, t17, g17_ant2_e)
    aruco['g17_ant2_n_m'] = interp_to(t, t17, g17_ant2_n)
    aruco['g17_ant2_h_m'] = interp_to(t, t17, g17['ant2_altitude_m'].values)

    aruco['g72_ant2_e_m'] = interp_to(t, t72, g72_ant2_e)
    aruco['g72_ant2_n_m'] = interp_to(t, t72, g72_ant2_n)
    aruco['g72_ant2_h_m'] = interp_to(t, t72, g72['ant2_height_m'].values)

    aruco['g72_ant1_e_m'] = interp_to(t, t72, g72_ant1_e)
    aruco['g72_ant1_n_m'] = interp_to(t, t72, g72_ant1_n)
    aruco['g72_ant1_h_m'] = interp_to(t, t72, g72['height_m'].values)

    aruco['g17_heading_deg'] = interp_to(t, t17, g17['heading_deg'].values)

    # ---------------------------------------------------------------------------
    # GNSS distances from 3730117 ANT2  (mm)
    # ---------------------------------------------------------------------------

    aruco['gnss_dE_mm']    = (aruco['g72_ant2_e_m'] - aruco['g17_ant2_e_m']) * 1000
    aruco['gnss_dN_mm']    = (aruco['g72_ant2_n_m'] - aruco['g17_ant2_n_m']) * 1000
    aruco['gnss_dU_mm']    = (aruco['g72_ant2_h_m'] - aruco['g17_ant2_h_m']) * 1000
    aruco['gnss_dist_mm']  = np.sqrt(
        aruco['gnss_dE_mm']**2 + aruco['gnss_dN_mm']**2 + aruco['gnss_dU_mm']**2)

    aruco['gnss_ant1_dE_mm']   = (aruco['g72_ant1_e_m'] - aruco['g17_ant2_e_m']) * 1000
    aruco['gnss_ant1_dN_mm']   = (aruco['g72_ant1_n_m'] - aruco['g17_ant2_n_m']) * 1000
    aruco['gnss_ant1_dU_mm']   = (aruco['g72_ant1_h_m'] - aruco['g17_ant2_h_m']) * 1000
    aruco['gnss_ant1_dist_mm'] = np.sqrt(
        aruco['gnss_ant1_dE_mm']**2 + aruco['gnss_ant1_dN_mm']**2 + aruco['gnss_ant1_dU_mm']**2)

    # ---------------------------------------------------------------------------
    # Tracker: project ANT2 and ANT1 into camera frame
    # ---------------------------------------------------------------------------

    tracker_ant2_x, tracker_ant2_y, tracker_ant2_z, tracker_dist    = [], [], [], []
    tracker_ant1_x, tracker_ant1_y, tracker_ant1_z, tracker_ant1_dist = [], [], [], []

    for _, row in aruco.iterrows():
        tx, ty, tz = row['tx'], row['ty'], row['tz']
        qw, qx = row['qw'], row['qx']
        qy, qz = row['qy'], row['qz']

        R    = quat_to_R(qw, qx, qy, qz)
        tvec = np.array([tx, ty, tz])

        # ANT2
        ant2_cam = R @ ARUCO_REF_ANT2 + tvec + WORLD_VERT_ANT2
        tracker_ant2_x.append(ant2_cam[0])
        tracker_ant2_y.append(ant2_cam[1])
        tracker_ant2_z.append(ant2_cam[2])
        tracker_dist.append(np.linalg.norm(ant2_cam - CAM_T_ANT2_17))

        # ANT1
        ant1_cam = R @ ARUCO_REF_ANT1 + tvec + WORLD_VERT_ANT1
        tracker_ant1_x.append(ant1_cam[0])
        tracker_ant1_y.append(ant1_cam[1])
        tracker_ant1_z.append(ant1_cam[2])
        tracker_ant1_dist.append(np.linalg.norm(ant1_cam - CAM_T_ANT2_17))

    aruco['tracker_ant2_x_mm']    = tracker_ant2_x
    aruco['tracker_ant2_y_mm']    = tracker_ant2_y
    aruco['tracker_ant2_z_mm']    = tracker_ant2_z
    aruco['tracker_dist_mm']      = tracker_dist

    aruco['tracker_ant1_x_mm']    = tracker_ant1_x
    aruco['tracker_ant1_y_mm']    = tracker_ant1_y
    aruco['tracker_ant1_z_mm']    = tracker_ant1_z
    aruco['tracker_ant1_dist_mm'] = tracker_ant1_dist

    # ---------------------------------------------------------------------------
    # Wahba rotation: fit mean R_cam_ENU from ANT2 data → extract tilt + heading offset
    # Per-frame heading is then taken from the GNSS 3730117 heading sensor.
    # ---------------------------------------------------------------------------

    valid_w = aruco.dropna(subset=['gnss_dist_mm', 'tracker_ant2_x_mm',
                                    'gnss_dE_mm', 'gnss_dN_mm', 'gnss_dU_mm'])

    G = valid_w[['gnss_dE_mm', 'gnss_dN_mm', 'gnss_dU_mm']].values          # N×3  ENU
    V = (valid_w[['tracker_ant2_x_mm', 'tracker_ant2_y_mm', 'tracker_ant2_z_mm']].values
         - CAM_T_ANT2_17)                                                     # N×3  cam

    H         = V.T @ G
    U, _, Vt  = np.linalg.svd(H)
    R_cam_ENU = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    R_ENU_cam = R_cam_ENU.T

    wahba_tilt_deg = np.degrees(np.arcsin(-R_ENU_cam[2, 2]))
    wahba_heading_deg = np.degrees(np.arctan2(R_ENU_cam[0, 2], R_ENU_cam[1, 2]))

    print(f"\nWahba tilt: {wahba_tilt_deg:.1f}°  (heading from Wahba: {wahba_heading_deg:.1f}°, using fixed offset)")
    print(f"GNSS heading mean: {aruco['g17_heading_deg'].mean():.1f}°  "
          f"camera offset: {CAMERA_HEADING_OFFSET_DEG:+.1f}° (fixed)")

    # Per-frame ENU rotation: GNSS heading + fixed physical mount offset + Wahba tilt.
    ant2_cam_arr = (aruco[['tracker_ant2_x_mm', 'tracker_ant2_y_mm', 'tracker_ant2_z_mm']].values
                    - CAM_T_ANT2_17)
    ant1_cam_arr = (aruco[['tracker_ant1_x_mm', 'tracker_ant1_y_mm', 'tracker_ant1_z_mm']].values
                    - CAM_T_ANT2_17)

    ant2_enu_rows, ant1_enu_rows = [], []
    for i, h_gnss in enumerate(aruco['g17_heading_deg'].values):
        if np.isnan(h_gnss):
            ant2_enu_rows.append([np.nan, np.nan, np.nan])
            ant1_enu_rows.append([np.nan, np.nan, np.nan])
            continue
        R_f = make_R_ENU_cam(h_gnss + CAMERA_HEADING_OFFSET_DEG, wahba_tilt_deg)
        ant2_enu_rows.append(R_f @ ant2_cam_arr[i])
        ant1_enu_rows.append(R_f @ ant1_cam_arr[i])

    ant2_enu = np.array(ant2_enu_rows)   # N×3 ENU
    ant1_enu = np.array(ant1_enu_rows)   # N×3 ENU

    aruco['tracker_ant2_dE_mm'] = ant2_enu[:, 0]
    aruco['tracker_ant2_dN_mm'] = ant2_enu[:, 1]
    aruco['tracker_ant2_dU_mm'] = ant2_enu[:, 2]

    aruco['tracker_ant1_dE_mm'] = ant1_enu[:, 0]
    aruco['tracker_ant1_dN_mm'] = ant1_enu[:, 1]
    aruco['tracker_ant1_dU_mm'] = ant1_enu[:, 2]

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------

    valid = aruco.dropna(subset=['gnss_dist_mm', 'tracker_dist_mm'])

    def print_stats(label, tracker_col, gnss_col):
        err = valid[tracker_col] - valid[gnss_col]
        print(f"\n{label}")
        print(f"  GNSS   : mean={valid[gnss_col].mean():.0f} mm  std={valid[gnss_col].std():.0f} mm")
        print(f"  Tracker: mean={valid[tracker_col].mean():.0f} mm  std={valid[tracker_col].std():.0f} mm")
        print(f"  Error  : mean={err.mean():+.1f} mm  std={err.std():.1f} mm"
              f"  [{err.min():+.0f} … {err.max():+.0f}]")

    print(f"\nAligned rows: {len(valid)}")
    print_stats("3730117 ANT2 → 3730072 ANT2", 'tracker_dist_mm',     'gnss_dist_mm')
    print_stats("3730117 ANT2 → 3730072 ANT1", 'tracker_ant1_dist_mm', 'gnss_ant1_dist_mm')

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------

    out_cols = [
        'frame', 'time_s',
        'tx', 'ty', 'tz',
        'tracker_ant2_x_mm', 'tracker_ant2_y_mm', 'tracker_ant2_z_mm', 'tracker_dist_mm',
        'tracker_ant1_x_mm', 'tracker_ant1_y_mm', 'tracker_ant1_z_mm', 'tracker_ant1_dist_mm',
        'gnss_dE_mm', 'gnss_dN_mm', 'gnss_dU_mm', 'gnss_dist_mm',
        'tracker_ant2_dE_mm', 'tracker_ant2_dN_mm', 'tracker_ant2_dU_mm',
        'gnss_ant1_dE_mm', 'gnss_ant1_dN_mm', 'gnss_ant1_dU_mm', 'gnss_ant1_dist_mm',
        'tracker_ant1_dE_mm', 'tracker_ant1_dN_mm', 'tracker_ant1_dU_mm',
        'g17_heading_deg',
        'g17_ant2_e_m', 'g17_ant2_n_m', 'g17_ant2_h_m',
        'g72_ant2_e_m', 'g72_ant2_n_m', 'g72_ant2_h_m',
        'g72_ant1_e_m', 'g72_ant1_n_m', 'g72_ant1_h_m',
    ]
    aruco[out_cols].to_csv(out_csv, index=False, float_format='%.4f')
    print(f"\nSaved → {out_csv}")


if __name__ == '__main__':
    main()
