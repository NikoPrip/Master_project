import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import correlate

# ---------------------------------------------------------------------------
# Camera extrinsic  (same for all tracker types)
# ---------------------------------------------------------------------------
# Rx185: phone body Y+=up, Z+=world+Z; camera Y+=down, Z+=into scene.
# Empirically calibrated: camera is physically ~5.1° tilted relative to phone body
# (more than pure Rx180 = diag([1,-1,-1])). Fit from hybrid run 1 alignment:
#   sin(5.1°)×mean_tz ≈ 220mm  →  explains 222mm constant Ty offset
#   sin(5.1°)×ΔTz ≈ 57mm      →  explains Ty range mismatch (72mm opti vs 32mm tracker)
CAM_R_BODY = Rotation.from_euler('xy', [185.1, 2.6], degrees=True).as_matrix()

CAM_R_CORRECTION = np.eye(3)  # no correction applied
CAM_T_BODY = np.array([-42.6, -80.8, -3.9])  # mm, camera lens in phone body frame
                                             # computed from centroid-to-camera offset:
                                             # camera = marker4 - 20mm Y; centroid is rigid body origin

# Board-frame → tracker-model-frame convention correction.
# OptiTrack board body: X+ = right, Y+ = up, Z+ = toward camera.
# Tracker MARKER_3D model:  X+ = left,  Y+ = up, Z+ = toward camera.
# → flip X and Z  ⟹  Ry180 = diag([-1, 1, -1])
BOARD_R_MODEL = np.diag([-1.0, 1.0, -1.0])  # Ry180

# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_tracker(csv_path):
    """Load tracker CSV → DataFrame with columns:
       frame, time_s, tx, ty, tz, qx, qy, qz, qw  (quaternion scalar-last)
    """
    return pd.read_csv(csv_path)


def load_optitrack(csv_path):
    """Load OptiTrack CSV → DataFrame with board and phone rigid-body poses.

    Column layout (0-indexed, after skipping 7 header rows):
      Col 1:     time_s
      Cols 2-5:  board  qx, qy, qz, qw
      Cols 6-8:  board  tx, ty, tz  (mm)
      Cols 39-42: phone qx, qy, qz, qw
      Cols 43-45: phone tx, ty, tz  (mm)
    """
    raw = pd.read_csv(csv_path, skiprows=7, header=0, low_memory=False)
    raw = raw.apply(pd.to_numeric, errors="coerce").dropna(how="all").reset_index(drop=True)

    # Column layout (77-col format):
    #   Board rigid body: cols 2-5 (qx,qy,qz,qw), cols 6-8 (tx,ty,tz)
    #   Board markers (10×3): cols 9-38
    #   Rigid Body 001 (fixed calibration reference at origin): cols 39-45  ← NOT the phone
    #   Rigid Body 001 markers (4×3): cols 46-57
    #   Rigid Body 002 (phone/camera, fixed on tripod): cols 58-61 (qxyzw), 62-64 (txyz)
    #   Rigid Body 002 markers (4×3): cols 65-76
    # Column layout depends on number of rigid bodies recorded:
    #   58-col (2 rigid bodies): Board (2-8), Board markers (9-38), Phone (39-45), Phone markers (46-57)
    #   77-col (3 rigid bodies): Board (2-8), Board markers (9-38), RB001 fixed ref (39-45),
    #                            RB001 markers (46-57), Phone/RB002 (58-64), Phone markers (65-76)
    ncols = len(raw.columns)
    if ncols >= 77:
        phone_start = 58
    else:
        phone_start = 39

    df = pd.DataFrame({
        "time_s":   raw.iloc[:, 1].astype(float),
        "board_qx": raw.iloc[:, 2].astype(float),
        "board_qy": raw.iloc[:, 3].astype(float),
        "board_qz": raw.iloc[:, 4].astype(float),
        "board_qw": raw.iloc[:, 5].astype(float),
        "board_tx": raw.iloc[:, 6].astype(float),
        "board_ty": raw.iloc[:, 7].astype(float),
        "board_tz": raw.iloc[:, 8].astype(float),
        "phone_qx": raw.iloc[:, phone_start    ].astype(float),
        "phone_qy": raw.iloc[:, phone_start + 1].astype(float),
        "phone_qz": raw.iloc[:, phone_start + 2].astype(float),
        "phone_qw": raw.iloc[:, phone_start + 3].astype(float),
        "phone_tx": raw.iloc[:, phone_start + 4].astype(float),
        "phone_ty": raw.iloc[:, phone_start + 5].astype(float),
        "phone_tz": raw.iloc[:, phone_start + 6].astype(float),
    })

    # Drop rows where either rigid body has an invalid quaternion
    valid = (
        np.linalg.norm(df[["board_qx","board_qy","board_qz","board_qw"]].values, axis=1) > 0.5
    ) & (
        np.linalg.norm(df[["phone_qx","phone_qy","phone_qz","phone_qw"]].values, axis=1) > 0.5
    )
    return df[valid].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Relative pose computation
# ---------------------------------------------------------------------------

def compute_relative_pose(opti_df, board_t_marker, board_r_model=None, cam_r_body=None):
    """Compute board-in-camera pose for every OptiTrack row.

    Parameters
    ----------
    opti_df       : DataFrame from load_optitrack()
    board_t_marker: (3,) offset from OptiTrack board centroid to tracker
                    reference point (e.g. ArUco centre / nfold marker 0),
                    in OptiTrack board body frame (mm, Y-up)

    Returns
    -------
    opti_df with added columns: rel_tx, rel_ty, rel_tz, rel_qx, rel_qy, rel_qz, rel_qw
    """
    R_board = Rotation.from_quat(opti_df[["board_qx","board_qy","board_qz","board_qw"]].values)
    R_phone = Rotation.from_quat(opti_df[["phone_qx","phone_qy","phone_qz","phone_qw"]].values)
    t_board = opti_df[["board_tx","board_ty","board_tz"]].values
    t_phone = opti_df[["phone_tx","phone_ty","phone_tz"]].values

    # Shift board centroid to tracker marker origin in global frame
    t_marker = t_board + R_board.apply(board_t_marker)

    # Board in phone body frame
    R_rel = R_phone.inv() * R_board
    t_rel = R_phone.inv().apply(t_marker - t_phone)

    # Apply board convention correction then camera extrinsic.
    # R_rel is board-body→phone-body; multiply by BOARD_R_MODEL to account for
    # the axis mismatch between OptiTrack board frame and tracker MARKER_3D frame
    # (board X+/Z+ are opposite to model X+/Z+).
    R_cam_body  = Rotation.from_matrix(cam_r_body if cam_r_body is not None else CAM_R_BODY)
    R_correction = Rotation.from_matrix(CAM_R_CORRECTION)
    mdl = board_r_model if board_r_model is not None else BOARD_R_MODEL
    R_board_mdl = Rotation.from_matrix(mdl)
    R_cam = R_cam_body.inv() * R_rel * R_board_mdl * R_correction
    t_cam = R_cam_body.inv().apply(t_rel - CAM_T_BODY)

    xyzw = R_cam.as_quat()
    opti_df = opti_df.copy()
    opti_df["rel_tx"] = t_cam[:, 0]
    opti_df["rel_ty"] = t_cam[:, 1]
    opti_df["rel_tz"] = t_cam[:, 2]
    opti_df["rel_qx"] = xyzw[:, 0]
    opti_df["rel_qy"] = xyzw[:, 1]
    opti_df["rel_qz"] = xyzw[:, 2]
    opti_df["rel_qw"] = xyzw[:, 3]
    return opti_df


# ---------------------------------------------------------------------------
# Time alignment
# ---------------------------------------------------------------------------

def estimate_time_offset(tracker_df, opti_df, search_range=2.0, coarse_step=0.01, fine_step=0.001):
    """Estimate clock offset by minimising 3D translation RMSE over a grid search.

    Searches offsets in [-search_range, +search_range] seconds at coarse_step
    resolution, then refines around the best coarse estimate at fine_step.

    Returns
    -------
    float : offset such that  opti_time - offset ≈ tracker_time
    """
    tracker_t   = tracker_df["time_s"].values
    tracker_xyz = tracker_df[["tx","ty","tz"]].values
    opti_t_raw  = opti_df["time_s"].values
    opti_xyz    = opti_df[["rel_tx","rel_ty","rel_tz"]].values

    def rmse_at(offset):
        opti_t  = opti_t_raw - offset
        t_start = max(tracker_t[0], opti_t[0])
        t_end   = min(tracker_t[-1], opti_t[-1])
        mask    = (tracker_t >= t_start) & (tracker_t <= t_end)
        trk_t   = tracker_t[mask]
        if len(trk_t) < 30:
            return np.inf
        interped = np.stack([
            interp1d(opti_t, opti_xyz[:, i], bounds_error=False, fill_value=np.nan)(trk_t)
            for i in range(3)
        ], axis=1)
        valid = np.all(np.isfinite(interped), axis=1)
        if valid.sum() < 30:
            return np.inf
        diff = tracker_xyz[mask][valid] - interped[valid]
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    # Coarse pass
    coarse_offsets = np.arange(-search_range, search_range, coarse_step)
    coarse_rmses   = np.array([rmse_at(o) for o in coarse_offsets])
    best_coarse    = coarse_offsets[np.argmin(coarse_rmses)]

    # Fine pass around best coarse estimate
    fine_offsets = np.arange(best_coarse - coarse_step, best_coarse + coarse_step, fine_step)
    fine_rmses   = np.array([rmse_at(o) for o in fine_offsets])
    return float(fine_offsets[np.argmin(fine_rmses)])


# ---------------------------------------------------------------------------
# Resampling and merge
# ---------------------------------------------------------------------------

def align(tracker_df, opti_df, time_offset):
    """Resample OptiTrack to tracker timestamps and return merged DataFrame.

    Parameters
    ----------
    tracker_df  : DataFrame from load_tracker()
    opti_df     : DataFrame from compute_relative_pose()
    time_offset : float from estimate_time_offset()

    Returns
    -------
    DataFrame with columns:
        frame, time_s,
        tracker_tx/ty/tz, tracker_qx/qy/qz/qw,
        opti_tx/ty/tz,    opti_qx/qy/qz/qw
    """
    opti_t    = opti_df["time_s"].values - time_offset
    tracker_t = tracker_df["time_s"].values

    # Restrict to overlapping window
    t_start = max(tracker_t[0], opti_t[0])
    t_end   = min(tracker_t[-1], opti_t[-1])
    mask    = (tracker_t >= t_start) & (tracker_t <= t_end)
    trk     = tracker_df[mask].reset_index(drop=True)
    trk_t   = trk["time_s"].values

    # Interpolate translation
    opti_tx = interp1d(opti_t, opti_df["rel_tx"].values, bounds_error=False, fill_value=np.nan)(trk_t)
    opti_ty = interp1d(opti_t, opti_df["rel_ty"].values, bounds_error=False, fill_value=np.nan)(trk_t)
    opti_tz = interp1d(opti_t, opti_df["rel_tz"].values, bounds_error=False, fill_value=np.nan)(trk_t)

    # Slerp for rotation
    rots = Rotation.from_quat(opti_df[["rel_qx","rel_qy","rel_qz","rel_qw"]].values)
    xyzw = Slerp(opti_t, rots)(np.clip(trk_t, opti_t[0], opti_t[-1])).as_quat()

    return pd.DataFrame({
        "frame":      trk["frame"].values,
        "time_s":     trk_t,
        "tracker_tx": trk["tx"].values,
        "tracker_ty": trk["ty"].values,
        "tracker_tz": trk["tz"].values,
        "tracker_qx": trk["qx"].values,
        "tracker_qy": trk["qy"].values,
        "tracker_qz": trk["qz"].values,
        "tracker_qw": trk["qw"].values,
        "opti_tx":    opti_tx,
        "opti_ty":    opti_ty,
        "opti_tz":    opti_tz,
        "opti_qx":    xyzw[:, 0],
        "opti_qy":    xyzw[:, 1],
        "opti_qz":    xyzw[:, 2],
        "opti_qw":    xyzw[:, 3],
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    BASE     = os.path.join(os.path.dirname(__file__), "optitrack")
    RES_DIR  = os.path.join(BASE, "results")
    CSV_DIR  = os.path.join(BASE, "csv")
    OUT_DIR  = os.path.join(BASE, "aligned")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Offset from OptiTrack board origin (Marker 003 = top-left corner) to the
    # tracker reference point, in OptiTrack board body frame (X+=right, Y+=up, mm).
    # Z=-20: board surface is 20mm behind the reflective stalk tops.
    BOARD_T_HYBRID = np.array([175.0, -222.0, -20.0])  # → nfold marker 0
    BOARD_T_ARUCO  = np.array([306.0, -282.0, -20.0])   # → ArUco marker centre

    BOARD_T_NFOLD  = np.array([175.0, -222.0, -20.0])   # → nfold marker 0 (same board as hybrid)

    configs = [
        ("hybrid", BOARD_T_HYBRID, [1, 2, 3], "hybrid", None),              # BOARD_R_MODEL = Ry180
        ("aruco",  BOARD_T_ARUCO,  [1, 2, 3], "aruco",  np.eye(3)),        # ArUco: standard X+=right, no flip
        ("nfold",  BOARD_T_NFOLD,  [1, 2, 3], "hybrid", None),             # BOARD_R_MODEL = Ry180
    ]

    for name, board_t_ref, runs, opti_name, b_r_model in configs:
        for run in runs:
            trk_csv  = os.path.join(RES_DIR, f"{name}_{run}.csv")
            opti_csv = os.path.join(CSV_DIR,  f"optitrack_{opti_name}_{run:02d}.csv")
            out_csv  = os.path.join(OUT_DIR,  f"{name}_{run}_aligned.csv")

            if not (os.path.exists(trk_csv) and os.path.exists(opti_csv)):
                print(f"Skipping {name} run {run} — files not found")
                continue

            print(f"\n--- {name.capitalize()} run {run} ---")
            tracker_df = load_tracker(trk_csv)
            opti_df    = load_optitrack(opti_csv)
            opti_df    = compute_relative_pose(opti_df, board_t_ref, board_r_model=b_r_model)

            offset = estimate_time_offset(tracker_df, opti_df)
            print(f"  Time offset: {offset:+.4f} s")

            merged = align(tracker_df, opti_df, offset)
            merged.to_csv(out_csv, index=False)
            print(f"  Saved: {out_csv}  ({len(merged)} rows)")
