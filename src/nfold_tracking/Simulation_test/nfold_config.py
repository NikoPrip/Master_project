import numpy as np


# ============================================================================
# N-fold Marker Configuration
# ============================================================================

# N-fold marker 3D positions (relative to marker 0 at origin)
# X+ = right, Y+ = up
MARKER_3D = {
    0:  np.array([   0.0,    0.0, 0.0], dtype=np.float32),  # Origin
    1:  np.array([ 260.0,    0.0, 0.0], dtype=np.float32),  # 260mm right
    2:  np.array([   0.0, -132.5, 0.0], dtype=np.float32),  # 132.5mm below
    3:  np.array([ 260.0, -176.0, 0.0], dtype=np.float32),  # 260mm right, 176mm below
    4:  np.array([ 208.0, -176.0, 0.0], dtype=np.float32),  # 208mm right, 176mm below
    5:  np.array([ 104.0,    0.0, 0.0], dtype=np.float32),  # 104mm right
    6:  np.array([ 260.0,  -88.0, 0.0], dtype=np.float32),  # 260mm right, 88mm below
    7:  np.array([ 104.0, -176.0, 0.0], dtype=np.float32),  # 104mm right, 176mm below
    8:  np.array([   0.0,  -44.0, 0.0], dtype=np.float32),  # 44mm below
    9:  np.array([ 208.0,    0.0, 0.0], dtype=np.float32),  # 208mm right
    10: np.array([  52.0, -176.0, 0.0], dtype=np.float32),  # 52mm right, 176mm below
    11: np.array([ 260.0, -132.0, 0.0], dtype=np.float32),  # 260mm right, 132mm below
    12: np.array([ 156.0,    0.0, 0.0], dtype=np.float32),  # 156mm right
    13: np.array([   0.0,  -88.0, 0.0], dtype=np.float32),  # 88mm below
    14: np.array([ 260.0,  -44.0, 0.0], dtype=np.float32),  # 260mm right, 44mm below
    15: np.array([ 156.0, -176.0, 0.0], dtype=np.float32),  # 156mm right, 176mm below
}

NUM_MARKERS  = len(MARKER_3D)
NFOLD_ORDER  = 3
KERNEL_SIZE  = 18


# ============================================================================
# Board Geometry
# ============================================================================

# Board corner positions relative to marker 0.
# Measured: 175mm left, 430mm right, 222mm up, 383mm down.
BOARD_CORNERS_3D = np.array([
    [-20.0,  20.0, 0.0],  # Top-left     (20mm left, 20mm above marker 0)
    [ 277.0,  20.0, 0.0],  # Top-right    (277mm right, 20mm above marker 0)
    [ 277.0, -190.0, 0.0],  # Bottom-right (277mm right, 190mm below marker 0)
    [-20.0, -190.0, 0.0],  # Bottom-left  (20mm left, 190mm below marker 0)
], dtype=np.float32)


# ============================================================================
# OptiTrack alignment
# ============================================================================

# Offset from OptiTrack board origin (Marker 003 = top-left corner) to
# nfold marker 0, in OptiTrack board body frame (X+=right, Y+=up, mm).
# Top-left corner is 175mm left and 222mm above marker 0
# → marker 0 is 175mm right (+X) and 222mm below (-Y) the board origin.
# Z=-20mm: board surface is 20mm behind the reflective stalk tops.
BOARD_T_MARKER0 = np.array([175.0, -222.0, -20.0])