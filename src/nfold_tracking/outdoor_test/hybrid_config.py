"""
Board Configuration and Marker Model - Outdoor Hybrid Setup

Defines the 3D geometry of markers and board for hybrid pose tracking (ArUco + nfold).
This configuration includes:
- ArUco marker: 40 x 40 mm (ID 1)
- 6 nfold markers (3-fold symmetry)
- Calibration board: 270 x 270 mm
- Rectangular target object: 260 x 650 mm (board attached to this object)

The rectangular object is the actual target being tracked. The board with
markers is attached to this object, 14mm in front of it.

Coordinate System:
- Origin: Centered at nfold marker 0
- X-axis: Increases to the right
- Y-axis: Increases upward (same as ArUco convention)
- Z-axis: Points away from board (right-hand rule)
- Units: Millimeters

Author: Academic project
"""

import numpy as np


# ============================================================================
# ArUco Marker Configuration
# ============================================================================

# ArUco marker size (one side of the square marker)
ARUCO_SIZE = 40.0  # mm

# Reference ArUco marker ID (which marker to use for hybrid tracking)
ARUCO_REFERENCE_ID = 1

# ArUco 3D corner positions (relative to marker 0)
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
ARUCO_CORNERS_3D = np.array([
    [-60.0,  0.0, 0.0],  # OpenCV corner 0: left-top  (60mm left of marker 0)
    [-20.0,  0.0, 0.0],  # OpenCV corner 1: right-top (20mm left of marker 0)
    [-20.0, -40.0, 0.0],  # OpenCV corner 2: right-bottom
    [-60.0, -40.0, 0.0],  # OpenCV corner 3: left-bottom
], dtype=np.float32)

# ArUco top-left corner position (OpenCV corner 0)
ARUCO_TL_3D = np.array([-60.0, 0.0])  # [X, Y]


# ============================================================================
# N-fold Marker Configuration
# ============================================================================

# N-fold marker 3D positions (relative to marker 0 at origin)
MARKER_3D = np.array([
    [  0.0,   0.0, 0.0],  # Marker 0: Origin (reference point)
    [ 70.0,   0.0, 0.0],  # Marker 1: 70mm right of marker 0
    [  0.0, -40.0, 0.0],  # Marker 2: 40mm below marker 0
    [ 70.0, -40.0, 0.0],  # Marker 3: 70mm right, 40mm below marker 0
    [ 35.0,  25.0, 0.0],  # Marker 4: 35mm right, 25mm above marker 0
    [100.0, -30.0, 0.0]   # Marker 5: 100mm right, 30mm below marker 0
], dtype=np.float32)

# Number of markers
NUM_MARKERS = MARKER_3D.shape[0]

# N-fold marker order (for convolution detection)
MARKER_ORDER = 3

# Kernel size for marker detection
KERNEL_SIZE = 18

# X-axis offsets from ArUco right edge to nfold markers (legacy, not currently used)
NFOLD_OFFSET_X = [0.0, 70.0]  # mm from ArUco right edge


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
# Detection Parameters
# ============================================================================

# Edge margin for marker detection (pixels from image edge)
EDGE_MARGIN = 50

# Display scale for visualization
DISPLAY_SCALE = 0.7


# ============================================================================
# Helper Functions
# ============================================================================

def get_marker_subset(marker_ids):
    return MARKER_3D[marker_ids]


def get_aruco_model():
    return ARUCO_CORNERS_3D, ARUCO_SIZE


def get_nfold_model():
    return MARKER_3D, NUM_MARKERS


def get_board_model():
    return BOARD_CORNERS_3D, BOARD_WIDTH, BOARD_HEIGHT


def get_rect_object_model():
    return RECT_CORNERS_3D, RECT_WIDTH, RECT_HEIGHT, RECT_DEPTH
