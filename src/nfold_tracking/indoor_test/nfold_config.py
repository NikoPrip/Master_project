"""
N-fold Marker Configuration

Defines the geometry for 3-fold markers with geometric matching.

Coordinate System:
- Origin: Center of marker 0
- X-axis: Increases to the right
- Y-axis: Increases upward (same as ArUco convention)
- Z-axis: Points away from board (right-hand rule)
- Units: Millimeters

Author: Academic project
"""

import numpy as np


# ============================================================================
# N-fold Marker Configuration
# ============================================================================

# Valid marker IDs (sequential IDs for geometric matching)
MARKER_IDS = [0, 1, 2, 3, 4, 5]

# Number of markers
NUM_MARKERS = len(MARKER_IDS)

# N-fold symmetry order (number of wedges)
NFOLD_ORDER = 3

# Kernel size for marker detection (pixels)
KERNEL_SIZE = 12

# Marker 3D positions (in mm, relative to marker 0)
MARKER_3D = {
    0: np.array([  0.0,   0.0, 0.0], dtype=np.float32),  # Origin
    1: np.array([ 70.0,   0.0, 0.0], dtype=np.float32),  # 70mm right
    2: np.array([  0.0, -40.0, 0.0], dtype=np.float32),  # 40mm below
    3: np.array([ 70.0, -40.0, 0.0], dtype=np.float32),  # 70mm right, 40mm below
    4: np.array([ 35.0,  25.0, 0.0], dtype=np.float32),  # 35mm right, 25mm above
    5: np.array([100.0, -30.0, 0.0], dtype=np.float32)   # 100mm right, 30mm below
}


# ============================================================================
# Board Geometry
# ============================================================================

# Board corner positions for visualization (boundary box)
# Measured from marker 0 center to each board corner
BOARD_CORNERS_3D = np.array([
    [-635.0,  104.0, 0.0],  # Top-left:    635mm left,  104mm above marker 0
    [ 165.0,  104.0, 0.0],  # Top-right:   165mm right, 104mm above marker 0
    [ 165.0, -496.0, 0.0],  # Bottom-right: 165mm right, 496mm below marker 0
    [-635.0, -496.0, 0.0]   # Bottom-left:  635mm left,  496mm below marker 0
], dtype=np.float32)

# Board dimensions (calculated from corners)
BOARD_WIDTH  = BOARD_CORNERS_3D[1, 0] - BOARD_CORNERS_3D[0, 0]   # mm
BOARD_HEIGHT = BOARD_CORNERS_3D[2, 1] - BOARD_CORNERS_3D[1, 1]   # mm


# ============================================================================
# Physical Marker Properties
# ============================================================================

MARKER_PHYSICAL_SIZE = 30.0  # mm (diameter)
MARKER_DIGITAL_SIZE  = 750   # pixels


# ============================================================================
# Helper Functions
# ============================================================================

def get_markers():
    return MARKER_3D.copy(), MARKER_IDS.copy()


def get_marker_position(marker_id):
    return MARKER_3D.get(marker_id, None)


def get_board_model():
    return BOARD_CORNERS_3D.copy(), BOARD_WIDTH, BOARD_HEIGHT


def get_marker_info():
    return {
        'marker_ids': MARKER_IDS,
        'num_markers': NUM_MARKERS,
        'nfold_order': NFOLD_ORDER,
        'positions_3d': MARKER_3D.copy(),
        'physical_size_mm': MARKER_PHYSICAL_SIZE,
        'digital_size_px': MARKER_DIGITAL_SIZE,
        'board_corners': BOARD_CORNERS_3D.copy(),
        'board_size_mm': (BOARD_WIDTH, BOARD_HEIGHT)
    }


def validate_marker_id(marker_id):
    return marker_id in MARKER_IDS


def get_markers_as_arrays():
    ids = np.array(MARKER_IDS, dtype=np.int32)
    positions = np.array([MARKER_3D[mid] for mid in MARKER_IDS], dtype=np.float32)
    return ids, positions
