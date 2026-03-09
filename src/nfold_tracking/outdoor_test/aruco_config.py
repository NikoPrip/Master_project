"""
ArUco Marker ID 0 Configuration - Outdoor Test Setup

Defines the geometry for ArUco marker (ID 0) used in pure ArUco tracking.
This configuration includes:
- ArUco marker: 106.5 x 106.5 mm
- Calibration board: 270 x 270 mm (marker mounted on board)
- Rectangular target object: 260 x 640 mm (board attached to this object)

The rectangular object is the actual target being tracked. The board with
the ArUco marker is attached to this object, 14mm in front of it.

Coordinate system origin: Center of ArUco marker
X-axis: Right, Y-axis: Down, Z-axis: Away from board

Author: Academic project
"""

import numpy as np


# ============================================================================
# ArUco Marker ID 0 Configuration
# ============================================================================

# ArUco marker ID
ARUCO_ID = 0

# ArUco marker size (one side of the square marker)
ARUCO_SIZE = 106.5  # mm (measured precisely)

# ArUco 3D corner positions (centered at origin)
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
# Using standard OpenCV ArUco coordinate system (Y-up)
ARUCO_CORNERS_3D = np.array([
    [-ARUCO_SIZE/2,  ARUCO_SIZE/2, 0.0],  # TL: (-53.25,  53.25, 0)
    [ ARUCO_SIZE/2,  ARUCO_SIZE/2, 0.0],  # TR: ( 53.25,  53.25, 0)
    [ ARUCO_SIZE/2, -ARUCO_SIZE/2, 0.0],  # BR: ( 53.25, -53.25, 0)
    [-ARUCO_SIZE/2, -ARUCO_SIZE/2, 0.0]   # BL: (-53.25, -53.25, 0)
], dtype=np.float32)


# ============================================================================
# Board Geometry (relative to ArUco ID 0)
# ============================================================================

# Board dimensions
BOARD_WIDTH  = 270.0  # mm
BOARD_HEIGHT = 270.0  # mm

# Board corner positions (relative to ArUco marker center)
BOARD_CORNERS_3D = np.array([
    [-110.0,  111.0, 0.0],  # Top-left corner
    [ 160.0,  111.0, 0.0],  # Top-right corner
    [ 160.0, -159.0, 0.0],  # Bottom-right corner
    [-110.0, -159.0, 0.0]   # Bottom-left corner
], dtype=np.float32)


# ============================================================================
# Rectangular Object Geometry (Target object the board is attached to)
# ============================================================================

# Rectangular object dimensions
RECT_WIDTH  =  260.0  # mm
RECT_HEIGHT =  640.0  # mm
RECT_DEPTH  =  -14.0  # mm (behind board, negative Z)

# Rectangular object corner positions (relative to ArUco marker center)
RECT_CORNERS_3D = np.array([
    [-100.0,  386.0, -14.0],  # Top-left corner
    [ 160.0,  386.0, -14.0],  # Top-right corner
    [ 160.0, -254.0, -14.0],  # Bottom-right corner
    [-100.0, -254.0, -14.0]   # Bottom-left corner
], dtype=np.float32)


# ============================================================================
# Helper Functions
# ============================================================================

def get_aruco_model():
    return ARUCO_CORNERS_3D, ARUCO_SIZE, ARUCO_ID


def get_board_model():
    return BOARD_CORNERS_3D, BOARD_WIDTH, BOARD_HEIGHT


def get_rect_object_model():
    return RECT_CORNERS_3D, RECT_WIDTH, RECT_HEIGHT, RECT_DEPTH


def get_marker_info():
    return {
        'id': ARUCO_ID,
        'size_mm': ARUCO_SIZE,
        'corners_3d': ARUCO_CORNERS_3D,
        'board_corners_3d': BOARD_CORNERS_3D,
        'board_size_mm': (BOARD_WIDTH, BOARD_HEIGHT),
        'rect_corners_3d': RECT_CORNERS_3D,
        'rect_size_mm': (RECT_WIDTH, RECT_HEIGHT),
        'description': f'ArUco marker ID {ARUCO_ID}, {ARUCO_SIZE}x{ARUCO_SIZE} mm'
    }
