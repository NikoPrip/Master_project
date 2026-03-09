
import numpy as np


# ============================================================================
# ArUco Marker Configuration
# ============================================================================

# ArUco marker size (one side of the square marker)
ARUCO_SIZE = 79.0  # mm

# Reference ArUco marker ID (which marker to use for hybrid tracking)
ARUCO_REFERENCE_ID = 1

# ArUco 3D corner positions (relative to marker 0)
# Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
ARUCO_CORNERS_3D = np.array([
    [ -90.0,  -48.5, 0.0],  # OpenCV corner 0: right-top  (90mm left,  48.5mm below marker 0)
    [-169.0,  -48.5, 0.0],  # OpenCV corner 1: left-top  (169mm left, 48.5mm below marker 0)
    [-169.0, -127.5, 0.0],  # OpenCV corner 2: left-bottom
    [ -90.0, -127.5, 0.0],  # OpenCV corner 3: right-bottom
], dtype=np.float32)

# ArUco top-left corner position (for offset calculations)
ARUCO_TL_3D = np.array([-90.0, -48.5])  # [X, Y]


# ============================================================================
# N-fold Marker Configuration
# ============================================================================

# N-fold marker 3D positions (relative to marker 0 at origin)
MARKER_3D = np.array([
    [   0.0,    0.0, 0.0],  # Marker 0: Origin (reference point)
    [-260.0,    0.0, 0.0],  # Marker 1: 260mm left of marker 0
    [   0.0, -176.0, 0.0],  # Marker 2: 176mm below marker 0
    [-260.0, -176.0, 0.0],  # Marker 3: 260mm left, 176mm below marker 0
    [ -52.0,    0.0, 0.0],  # Marker 4: 52mm left of marker 0
    [-104.0, -176.0, 0.0],  # Marker 5: 104mm left, 176mm below marker 0
], dtype=np.float32)

# Number of markers
NUM_MARKERS = MARKER_3D.shape[0]

# N-fold marker order (for convolution detection)
MARKER_ORDER = 3

# Kernel size for marker detection
KERNEL_SIZE = 18

# ============================================================================
# Board Geometry
# ============================================================================

# Board corner positions for visualization (boundary box)
# Measured from marker 0 center (outdoor setup)
BOARD_CORNERS_3D = np.array([
    [ 173.0,  225.0, 0.0],  # Top-left
    [-430.0,  225.0, 0.0],  # Top-right
    [-430.0, -381.0, 0.0],  # Bottom-right
    [ 173.0, -381.0, 0.0],  # Bottom-left
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
    return BOARD_CORNERS_3D

