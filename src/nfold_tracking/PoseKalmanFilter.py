"""
Kalman Filter for 6DOF Pose Tracking

Implements a Kalman filter for tracking 3D pose (position + orientation) using
a constant velocity motion model.

State representation:
- Translation: [x, y, z, vx, vy, vz]  (6D, constant-velocity model)
- Rotation:    [qw, qx, qy, qz, ωx, ωy, ωz]  (7D quaternion + angular velocity)

The rotation sub-filter uses a linearized quaternion kinematics model
(q_dot = 0.5 * Omega(ω) * q) with renormalization after each predict/update.

Public API uses quaternions throughout — callers should convert to/from
Rodrigues vectors only at OpenCV call sites (solvePnP, projectPoints).

Author: Nikolai Prip
"""

import numpy as np
from rotation_utils import rvec_to_quat, quat_to_rvec


class PoseKalmanFilter:
    """
    Kalman filter for 6DOF pose tracking.

    update(quat, tvec) — feed a new quaternion + translation measurement.
    predict()          — returns (quat, tvec) predicted pose.
    get_filtered_pose()— returns (quat, tvec) smoothed pose.
    """

    def __init__(self, dt=1/30.0, process_noise=0.1, measurement_noise=1.0,
                 gate_trans_mm=150.0, gate_rot_deg=20.0):
        self.dt = dt

        # ---- Translation sub-filter ----------------------------------------
        self.state_trans = np.zeros((6, 1), dtype=np.float64)
        self.P_trans = np.eye(6, dtype=np.float64) * 1000.0

        self.F_trans = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ],
        ], dtype=np.float64)

        self.H_trans = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        self.Q_trans = np.eye(6, dtype=np.float64) * process_noise
        self.Q_trans[3:, 3:] *= 2.0
        self.R_trans = np.eye(3, dtype=np.float64) * measurement_noise

        # ---- Rotation sub-filter (quaternion) --------------------------------
        # State: [qw, qx, qy, qz, ωx, ωy, ωz]
        self.state_rot = np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        ).reshape(7, 1)
        self.P_rot = np.eye(7, dtype=np.float64) * 1000.0

        # H_rot: observe quaternion components only
        self.H_rot = np.zeros((4, 7), dtype=np.float64)
        self.H_rot[:4, :4] = np.eye(4)

        self.Q_rot = np.eye(7, dtype=np.float64) * (process_noise * 0.1)
        self.Q_rot[4:, 4:] *= 2.0      # higher noise on angular velocity
        self.R_rot = np.eye(4, dtype=np.float64) * (measurement_noise * 0.1)

        # Outlier gating thresholds (set to None to disable)
        self.gate_trans_mm = gate_trans_mm
        self.gate_rot_deg  = gate_rot_deg

        self.initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rot_jacobian(self):
        """
        Linearized state-transition Jacobian for the rotation state.

        Derived from q_dot = 0.5 * Omega(ω) * q  (quaternion kinematics).
        """
        q  = self.state_rot[:4].flatten()
        w  = self.state_rot[4:].flatten()
        qw, qx, qy, qz = q
        wx, wy, wz = w
        dt = self.dt

        # Omega(ω) — the 4×4 skew-symmetric quaternion product matrix
        Omega = np.array([
            [ 0,  -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0 ],
        ], dtype=np.float64)

        F = np.eye(7, dtype=np.float64)
        # d(q_new)/d(q)
        F[:4, :4] = np.eye(4) + 0.5 * dt * Omega
        # d(q_new)/d(ω)
        F[:4, 4:] = 0.5 * dt * np.array([
            [-qx, -qy, -qz],
            [ qw, -qz,  qy],
            [ qz,  qw, -qx],
            [-qy,  qx,  qw],
        ], dtype=np.float64)
        # ω rows: F[4:, 4:] = I  (constant angular velocity model)
        return F

    @staticmethod
    def _normalize_quat(state_rot):
        q = state_rot[:4]
        n = np.linalg.norm(q)
        state_rot[:4] = q / n if n > 1e-10 else np.array([[1.], [0.], [0.], [0.]])
        return state_rot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self):
        """
        Prediction step.

        Returns:
            (quat, tvec): predicted pose as quaternion (4,) + translation (3,1),
                          or (None, None) if not yet initialised.
        """
        if not self.initialized:
            return None, None

        # Translation
        self.state_trans = self.F_trans @ self.state_trans
        self.P_trans = self.F_trans @ self.P_trans @ self.F_trans.T + self.Q_trans

        # Rotation — linearised quaternion kinematics
        F_rot = self._rot_jacobian()

        q  = self.state_rot[:4].flatten()
        w  = self.state_rot[4:].flatten()
        wx, wy, wz = w
        Omega = np.array([
            [ 0,  -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0 ],
        ], dtype=np.float64)
        q_new = q + 0.5 * self.dt * (Omega @ q)
        n = np.linalg.norm(q_new)
        self.state_rot[:4] = (q_new / n if n > 1e-10 else np.array([1., 0., 0., 0.])).reshape(4, 1)

        self.P_rot = F_rot @ self.P_rot @ F_rot.T + self.Q_rot

        quat = self.state_rot[:4].flatten().copy()
        tvec = self.state_trans[:3].reshape(3, 1).copy()
        return quat, tvec

    def update(self, quat, tvec):
        """
        Update step: fuse prediction with a new measurement.

        Args:
            quat: measured rotation as unit quaternion array-like (4,) [qw,qx,qy,qz]
            tvec: measured translation (3,1) or (3,)
        """
        if quat is None or tvec is None:
            return False

        tvec_meas = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        q_meas    = np.asarray(quat, dtype=np.float64).flatten().reshape(4, 1)

        if not self.initialized:
            self.state_trans[:3] = tvec_meas
            self.state_trans[3:] = 0.0
            self.state_rot[:4]   = q_meas
            self.state_rot[4:]   = 0.0
            self.initialized = True
            return True

        # Outlier gating — reject if measurement is too far from current state
        if self.gate_trans_mm is not None:
            trans_err = np.linalg.norm(tvec_meas.flatten() - self.state_trans[:3].flatten())
            if trans_err > self.gate_trans_mm:
                return False

        if self.gate_rot_deg is not None:
            q_state = self.state_rot[:4].flatten()
            q_m     = q_meas.flatten()
            if np.dot(q_m, q_state) < 0:
                q_m = -q_m
            dot = np.clip(np.abs(np.dot(q_m, q_state)), 0.0, 1.0)
            rot_err_deg = 2.0 * np.degrees(np.arccos(dot))
            if rot_err_deg > self.gate_rot_deg:
                return False

        # Sign consistency — quaternion double-cover
        if np.dot(q_meas.flatten(), self.state_rot[:4].flatten()) < 0:
            q_meas = -q_meas

        # Translation update
        y_t = tvec_meas - self.H_trans @ self.state_trans
        S_t = self.H_trans @ self.P_trans @ self.H_trans.T + self.R_trans
        K_t = self.P_trans @ self.H_trans.T @ np.linalg.inv(S_t)
        self.state_trans = self.state_trans + K_t @ y_t
        self.P_trans = (np.eye(6) - K_t @ self.H_trans) @ self.P_trans

        # Rotation update
        y_r = q_meas - self.H_rot @ self.state_rot
        S_r = self.H_rot @ self.P_rot @ self.H_rot.T + self.R_rot
        K_r = self.P_rot @ self.H_rot.T @ np.linalg.inv(S_r)
        self.state_rot = self.state_rot + K_r @ y_r
        self.state_rot = self._normalize_quat(self.state_rot)
        self.P_rot = (np.eye(7) - K_r @ self.H_rot) @ self.P_rot
        return True

    def get_filtered_pose(self):
        """
        Returns the current smoothed pose.

        Returns:
            (quat, tvec): quaternion (4,) + translation (3,1),
                          or (None, None) if not initialised.
        """
        if not self.initialized:
            return None, None
        quat = self.state_rot[:4].flatten().copy()
        tvec = self.state_trans[:3].reshape(3, 1).copy()
        return quat, tvec

    def get_velocity(self):
        """
        Returns:
            (angular_velocity (3,1), linear_velocity (3,1)) or (None, None)
        """
        if not self.initialized:
            return None, None
        return self.state_rot[4:].reshape(3, 1), self.state_trans[3:].reshape(3, 1)

    def get_covariance(self):
        """Returns (P_rot, P_trans) covariance matrices."""
        return self.P_rot.copy(), self.P_trans.copy()

    def reset(self):
        """Reset filter to uninitialised state."""
        self.initialized = False
        self.state_trans = np.zeros((6, 1), dtype=np.float64)
        self.state_rot   = np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        ).reshape(7, 1)
        self.P_trans = np.eye(6, dtype=np.float64) * 1000.0
        self.P_rot   = np.eye(7, dtype=np.float64) * 1000.0

    def is_initialized(self):
        """Check if filter has received at least one measurement."""
        return self.initialized
