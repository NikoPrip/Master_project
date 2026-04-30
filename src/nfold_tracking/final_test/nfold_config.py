"""
N-fold Marker Configuration — Final Test Board

Same physical board as hybrid_config but tracked without an ArUco reference.
Coordinate system, units, and marker positions are identical to hybrid_config.

Coordinate System:
- Origin: Center of nfold marker 0
- X-axis: Increases to the right
- Y-axis: Increases upward
- Z-axis: Points away from board (toward camera)
- Units: Millimeters
"""

import numpy as np


# ============================================================================
# N-fold Marker Configuration
# ============================================================================

NFOLD_ORDER = 3
KERNEL_SIZE = 12

# Marker 3D positions as dict {id: np.array([x, y, z])} — required by NfoldPoseTracker
MARKER_3D = {
    0: np.array([   0.0,    0.0, 0.0], dtype=np.float32),  # Origin
    1: np.array([  74.5,    0.0, 0.0], dtype=np.float32),  # 74.5mm right
    2: np.array([  74.5,  -74.5, 0.0], dtype=np.float32),  # 74.5mm right, 74.5mm down
    3: np.array([  74.5, -149.0, 0.0], dtype=np.float32),  # 74.5mm right, 149mm down
    4: np.array([   0.0, -149.0, 0.0], dtype=np.float32),  # 149mm down
    5: np.array([ -74.5, -149.0, 0.0], dtype=np.float32),  # 74.5mm left, 149mm down
    6: np.array([ -74.5,  -74.5, 0.0], dtype=np.float32),  # 74.5mm left, 74.5mm down
    7: np.array([ -74.5,    0.0, 0.0], dtype=np.float32),  # 74.5mm left
}

MARKER_IDS  = list(MARKER_3D.keys())
NUM_MARKERS = len(MARKER_IDS)


# ============================================================================
# Board Geometry
# ============================================================================

BOARD_CORNERS_3D = np.array([
    [-136.0,   57.5, 0.0],  # Top-left
    [ 134.0,   57.5, 0.0],  # Top-right
    [ 134.0, -212.5, 0.0],  # Bottom-right
    [-136.0, -212.5, 0.0],  # Bottom-left
], dtype=np.float32)


# ============================================================================
# Detection Parameters
# ============================================================================

DEPTH_RANGE   = (300, 3000)
EDGE_MARGIN   = 50
DISPLAY_SCALE = 0.7
MAX_MARKERS   = 40
