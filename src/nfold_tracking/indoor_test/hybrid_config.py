"""
Board Configuration and Marker Model

Defines the 3D geometry of markers and board for pose tracking.
This configuration is shared across all tracking methods (hybrid, ArUco-only, nfold-only).

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
# ArUco marker is left of marker 0, at same Y level (top edge at marker 0's Y)
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
# Measured from marker 0 center
BOARD_CORNERS_3D = np.array([
    [-636.0,  104.5, 0.0],  # Top-left:    636mm left,  104.5mm above marker 0
    [ 164.0,  104.5, 0.0],  # Top-right:   164mm right, 104.5mm above marker 0
    [ 164.0, -496.0, 0.0],  # Bottom-right: 164mm right, 496mm below marker 0
    [-636.0, -496.0, 0.0]   # Bottom-left:  636mm left,  496mm below marker 0
], dtype=np.float32)

# Board dimensions
BOARD_WIDTH  = 800.0   # mm (164 + 636)
BOARD_HEIGHT = 600.5   # mm (104.5 + 496)


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
