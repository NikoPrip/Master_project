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
ARUCO_SIZE = 50.0  # mm

# Reference ArUco marker ID (which marker to use for hybrid tracking)
ARUCO_REFERENCE_ID = 1

# ArUco 3D corner positions (relative to marker 0)
# Marker 0 is 74.5mm above ArUco center, so ArUco center is at Y=-74.5mm
# ArUco top edge: Y = -74.5 + 25 = -49.5mm, bottom: -74.5 - 25 = -99.5mm
ARUCO_CORNERS_3D = np.array([
    [-25.0, -49.5, 0.0],  # OpenCV corner 0: left-top
    [ 25.0, -49.5, 0.0],  # OpenCV corner 1: right-top
    [ 25.0, -99.5, 0.0],  # OpenCV corner 2: right-bottom
    [-25.0, -99.5, 0.0],  # OpenCV corner 3: left-bottom
], dtype=np.float32)

# ArUco top-left corner position (OpenCV corner 0)
ARUCO_TL_3D = np.array([-25.0, -49.5])  # [X, Y]


# ============================================================================
# N-fold Marker Configuration
# ============================================================================

# N-fold marker 3D positions (relative to marker 0 at origin)
MARKER_3D = np.array([
    [    0.0,    0.0, 0.0],  # Marker 0: Origin
    [  74.5,    0.0, 0.0],  # Marker 1: 74.5mm right
    [  74.5,  -74.5, 0.0], # Marker 2: 74.5mm right, 74.5mm down
    [  74.5, -149.0, 0.0],  # Marker 3: 74.5mm right, 148mm down
    [   0.0,  -149.0, 0.0],  # Marker 4: 148mm down
    [ -74.5, -149.0, 0.0],  # Marker 5: 74.5mm left, 148mm down
    [ -74.5,  -74.5, 0.0], # Marker 6: 74.5mm left, 74.5mm down
    [ -74.5,    0.0, 0.0],  # Marker 7: 74.5mm left
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
    [-136.0,  57.5, 0.0],  # Top-left:    136mm left,  57.5mm above marker 0
    [ 134.0,  57.5, 0.0],  # Top-right:   134mm right, 57.5mm above marker 0
    [ 134.0, -212.5, 0.0], # Bottom-right: 134mm right, 212.5mm below marker 0
    [-136.0, -212.5, 0.0]  # Bottom-left:  136mm left,  212.5mm below marker 0
], dtype=np.float32)

# Board dimensions
BOARD_WIDTH  = 270.0   # mm (134 + 136)
BOARD_HEIGHT = 270.0   # mm (57.5 + 212.5)


# ============================================================================
# Detection Parameters
# ============================================================================

# Minimum quality score for a detected nfold marker to be accepted.
# Quality is in [0, 1] — higher means the local patch better matches
# the expected bright/dark n-fold pattern. Raise to reject more false positives.
QUALITY_THRESHOLD = 0.5

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
