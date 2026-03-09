"""
N-fold Marker Configuration - Outdoor Setup

Defines the geometry for 3-fold markers with geometric matching.

This configuration includes:
- 6 nfold markers (3-fold symmetry)
- Calibration board: 270 x 270 mm
- Rectangular target object: 260 x 650 mm (board attached to this object)

The rectangular object is the actual target being tracked. The board with
markers is attached to this object, 14mm in front of it.

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
# Measured from marker 0 center (outdoor setup)
BOARD_CORNERS_3D = np.array([
    [-120.0,  134.0, 0.0],  # Top-left
    [ 150.0,  134.0, 0.0],  # Top-right
    [ 150.0, -136.0, 0.0],  # Bottom-right
    [-120.0, -136.0, 0.0]   # Bottom-left
], dtype=np.float32)

# Board dimensions
BOARD_WIDTH  = 270.0  # mm (150 + 120)
BOARD_HEIGHT = 270.0  # mm (134 + 136)


# ============================================================================
# Rectangular Object Geometry (Target object the board is attached to)
# ============================================================================

# Rectangular object dimensions
RECT_WIDTH  =  260.0  # mm
RECT_HEIGHT =  650.0  # mm
RECT_DEPTH  =   14.0  # mm (Z flipped: rectangle in front of board)

# Rectangular object corner positions (relative to marker 0)
RECT_CORNERS_3D = np.array([
    [-110.0,  384.0, 14.0],  # Top-left corner
    [ 150.0,  384.0, 14.0],  # Top-right corner
    [ 150.0, -266.0, 14.0],  # Bottom-right corner
    [-110.0, -266.0, 14.0]   # Bottom-left corner
], dtype=np.float32)


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


def get_rect_object_model():
    return RECT_CORNERS_3D.copy(), RECT_WIDTH, RECT_HEIGHT, RECT_DEPTH


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
