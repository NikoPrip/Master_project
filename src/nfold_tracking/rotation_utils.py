"""
Rotation Utility Functions

Conversions between Rodrigues vectors (used by OpenCV solvePnP / projectPoints)
and unit quaternions used throughout the rest of the pipeline.

Quaternion convention: scalar-first  [qw, qx, qy, qz]
"""
import numpy as np


def rvec_to_quat(rvec):
    """Convert Rodrigues rotation vector to unit quaternion [qw, qx, qy, qz]."""
    rvec = np.asarray(rvec, dtype=np.float64).flatten()
    angle = np.linalg.norm(rvec)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rvec / angle
    half = angle * 0.5
    s = np.sin(half)
    return np.array([np.cos(half), s * axis[0], s * axis[1], s * axis[2]])


def quat_to_rvec(q):
    """Convert unit quaternion [qw, qx, qy, qz] to Rodrigues rotation vector (3x1)."""
    q = quat_normalize(np.asarray(q, dtype=np.float64).flatten())
    qw, qx, qy, qz = q
    sin_half = np.sqrt(qx**2 + qy**2 + qz**2)
    if sin_half < 1e-10:
        return np.zeros((3, 1), dtype=np.float64)
    angle = 2.0 * np.arctan2(sin_half, qw)
    axis = np.array([qx, qy, qz]) / sin_half
    return (angle * axis).reshape(3, 1)


def quat_multiply(q1, q2):
    """Hamilton product q1 ⊗ q2, both in [qw, qx, qy, qz] format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_normalize(q):
    """Normalize quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_angular_distance(q1, q2):
    """Geodesic angular distance between two unit quaternions (radians)."""
    q1 = quat_normalize(np.asarray(q1, dtype=np.float64).flatten())
    q2 = quat_normalize(np.asarray(q2, dtype=np.float64).flatten())
    dot = min(abs(np.dot(q1, q2)), 1.0)  # abs handles quaternion double-cover
    return 2.0 * np.arccos(dot)
