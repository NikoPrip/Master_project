"""
ArUco Marker ID 0 Configuration

Defines the geometry for the larger ArUco marker (ID 0) used in pure ArUco tracking.
This is separate from the hybrid tracking marker (ID 1) which is 40mm.

Physical marker: 151 x 151 mm

Author: Academic project
"""

import numpy as np


# ============================================================================
# ArUco Marker ID 0 Configuration
# ============================================================================

# ArUco marker ID
ARUCO_ID = 0

# ArUco marker size (one side of the square marker)
ARUCO_SIZE = 158  # mm (measured precisely)

# ArUco 3D corner positions (centered at origin)
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
# Using standard OpenCV ArUco coordinate system (Y-up)
ARUCO_CORNERS_3D = np.array([
    [-ARUCO_SIZE/2,  ARUCO_SIZE/2, 0.0],  # TL: (-75.5,  75.5, 0)
    [ ARUCO_SIZE/2,  ARUCO_SIZE/2, 0.0],  # TR: ( 75.5,  75.5, 0)
    [ ARUCO_SIZE/2, -ARUCO_SIZE/2, 0.0],  # BR: ( 75.5, -75.5, 0)
    [-ARUCO_SIZE/2, -ARUCO_SIZE/2, 0.0]   # BL: (-75.5, -75.5, 0)
], dtype=np.float32)


# ============================================================================
# Board Geometry (relative to ArUco ID 0)
# ============================================================================

# Board dimensions
BOARD_WIDTH  = 297  # mm (A4 paper size for reference)
BOARD_HEIGHT = 210  # mm (A4 paper size for reference)

# Board corner positions (relative to ArUco marker center)
# - Top-right:    130.5mm right, 79mm up
# - Bottom-right: 130.5mm right, 521mm down
# - Top-left:     670mm left,    79mm up
# - Bottom-left:  670mm left,    521mm down
BOARD_CORNERS_3D = np.array([
    [-150.0,   106.0, 0.0],  # Top-left corner
    [ 147.0,   106.0, 0.0],  # Top-right corner
    [ 147.0, -104.0, 0.0],  # Bottom-right corner
    [-150.0, -104.0, 0.0]   # Bottom-left corner
], dtype=np.float32)


# ============================================================================
# Helper Functions
# ============================================================================

def get_aruco_model():
    return ARUCO_CORNERS_3D, ARUCO_SIZE, ARUCO_ID


def get_board_model():
    return BOARD_CORNERS_3D, BOARD_WIDTH, BOARD_HEIGHT


def get_marker_info():
    return {
        'id': ARUCO_ID,
        'size_mm': ARUCO_SIZE,
        'corners_3d': ARUCO_CORNERS_3D,
        'board_corners_3d': BOARD_CORNERS_3D,
        'board_size_mm': (BOARD_WIDTH, BOARD_HEIGHT),
        'description': f'ArUco marker ID {ARUCO_ID}, {ARUCO_SIZE}x{ARUCO_SIZE} mm'
    }
