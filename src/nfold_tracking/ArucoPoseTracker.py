"""
ArUco Pose Tracker Class

Tracks 6DOF pose using only ArUco marker ID 0.
Uses Kalman filter for temporal smoothing.

Supports both lab and outdoor configurations:
- Lab setup: Uses aruco_config (board only)
- Outdoor setup: Uses outdoor_test.aruco_config (board + target object)

Author: Nikolai Prip
"""
import os
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
import csv
import cv2
cv2.setLogLevel(0)
import numpy as np
from pathlib import Path

from PoseKalmanFilter import PoseKalmanFilter
from rotation_utils import rvec_to_quat, quat_to_rvec
import importlib


class ArucoPoseTracker:
    """
    Pure ArUco pose tracking using marker ID 0.
    """

    def __init__(self, calib_path, video_90_path, video_91_path=None, config_module='indoor_test.aruco_config', csv_path=None):
        """
        Initialize ArUco pose tracker.

        Args:
            calib_path: Path to calibration files directory
            video_90_path: Path to 90 camera video
            video_91_path: Path to 91 camera video (optional)
            config_module: Config module name ('aruco_config' for lab, 'outdoor_test.aruco_config' for outdoor)
        """
        self.calib_path = Path(calib_path)

        # Import configuration dynamically
        config = importlib.import_module(config_module)
        self.ARUCO_CORNERS_3D = config.ARUCO_CORNERS_3D
        self.ARUCO_ID = config.ARUCO_ID
        self.BOARD_CORNERS_3D = config.BOARD_CORNERS_3D

        # Import rectangular object if available (outdoor config)
        self.RECT_CORNERS_3D = getattr(config, 'RECT_CORNERS_3D', None)
        self.DEPTH_RANGE = getattr(config, 'DEPTH_RANGE', (300, 3000))

        # Import display scale (try outdoor_test.hybrid_config, fallback to 0.7)
        try:
            display_config = importlib.import_module('outdoor_test.hybrid_config')
            self.DISPLAY_SCALE = display_config.DISPLAY_SCALE
        except:
            self.DISPLAY_SCALE = 0.7

        self.aruco_reference_id = self.ARUCO_ID

        # Setup ArUco detector with subpixel refinement
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        aruco_params.cornerRefinementWinSize = 5
        aruco_params.cornerRefinementMaxIterations = 30
        aruco_params.cornerRefinementMinAccuracy = 0.1

        self.aruco_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            aruco_params)

        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        if csv_path:
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['frame', 'time_s', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

        self.frame_count = 0

        # Setup cameras
        self.cameras = self._setup_cameras(video_90_path, video_91_path)

        # Get frame dimensions
        h, w = self.cameras[0]['cap'].read()[1].shape[:2]
        self.cameras[0]['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_size = (w, h)

        # Setup undistortion maps
        for cam in self.cameras:
            cam['map1'], cam['map2'] = cv2.initUndistortRectifyMap(
                cam['K'], cam['dist'], None, cam['K'], self.frame_size, cv2.CV_32FC1)

    def _setup_cameras(self, video_90_path, video_91_path):
        """Setup camera configurations."""
        cameras = []
        sides = []
        if video_90_path:
            sides.append('90')
        if video_91_path:
            sides.append('91')

        for side in sides:
            data = np.load(str(self.calib_path / f'calib_data_{side}.npz'))
            cameras.append({
                'K': data['mtx'],
                'dist': data['dist'],
                'cap': cv2.VideoCapture(str(video_90_path if side == '90' else video_91_path)),
                'color': (0, 255, 255) if side == '90' else (255, 0, 255),
                'kalman': PoseKalmanFilter(dt=1/30.0, process_noise=100.0, measurement_noise=1.0,
                                           gate_trans_mm=300.0, gate_rot_deg=30.0)
            })
        return cameras

    def detect_aruco_marker(self, gray):
        """Detect specific ArUco marker."""
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is not None:
            ids = ids.flatten()
            marker_indices = np.where(ids == self.aruco_reference_id)[0]
            if len(marker_indices) > 0:
                return corners[marker_indices[0]], True

        return None, False

    def estimate_and_validate_pose(self, corners, K, quat_init=None, tvec_init=None):
        """Estimate pose from ArUco corners and validate.

        Returns (quat, tvec, error) where quat is [qw, qx, qy, qz], or
        (None, None, None) on failure.
        """
        if corners is None:
            return None, None, None

        img_pts = corners.reshape(4, 2).astype(np.float32)
        obj_pts = self.ARUCO_CORNERS_3D

        # Use IPPE solver (optimized for planar markers)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, None,
            flags=cv2.SOLVEPNP_IPPE)

        if not success:
            return None, None, None

        # If a Kalman prediction is available, refine from it instead of the
        # raw IPPE result — this prevents flipping between IPPE's two solutions
        if quat_init is not None and tvec_init is not None:
            rvec = quat_to_rvec(quat_init)
            tvec = tvec_init.copy()

        # Refine pose
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, None, rvec, tvec)

        # Calculate error (OpenCV boundary — needs rvec)
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
        error = np.sqrt(np.mean(np.sum((img_pts - proj.reshape(-1, 2))**2, axis=1)))

        # Validate
        depth_min, depth_max = self.DEPTH_RANGE
        if error > 3.0 or not (depth_min <= tvec[2, 0] <= depth_max):
            return None, None, None

        return rvec_to_quat(rvec), tvec, error

    def visualize(self, frame, aruco_corners, aruco_detected, rvec, tvec, error, color, K):
        """Visualize ArUco detection and pose."""
        # Draw ArUco marker
        if aruco_detected and aruco_corners is not None:
            corners_array = [aruco_corners]
            ids_array = np.array([[self.aruco_reference_id]])
            cv2.aruco.drawDetectedMarkers(frame, corners_array, ids_array)

        # Draw board boundary
        if rvec is not None and tvec is not None:
            board_proj = cv2.projectPoints(self.BOARD_CORNERS_3D, rvec, tvec, K, None)[0].reshape(-1, 2).astype(int)
            for i in range(4):
                cv2.line(frame, tuple(board_proj[i]), tuple(board_proj[(i+1)%4]), color, 3)

            # Draw rectangular object boundary (if configured)
            if self.RECT_CORNERS_3D is not None:
                rect_proj = cv2.projectPoints(self.RECT_CORNERS_3D, rvec, tvec, K, None)[0].reshape(-1, 2).astype(int)
                rect_color = (0, 255, 0)  # Green for target object
                for i in range(4):
                    cv2.line(frame, tuple(rect_proj[i]), tuple(rect_proj[(i+1)%4]), rect_color, 4)

            error_str = f"{error:.2f}px" if error is not None else "N/A"
            cv2.putText(frame, f"Pos:({tvec[0,0]:.1f},{tvec[1,0]:.1f},{tvec[2,0]:.1f})mm Err:{error_str}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Status
        status = "Tracking" if rvec is not None else "Lost"
        mode = " + Target Object" if self.RECT_CORNERS_3D is not None else ""
        cv2.putText(frame, f"ArUco ID {self.aruco_reference_id}{mode} | Status: {status}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def process_frame_pair(self):
        """Process one frame pair from both cameras."""
        frames = [(cam['cap'].get(cv2.CAP_PROP_POS_MSEC), *cam['cap'].read()) for cam in self.cameras]
        if not all(ret for _, ret, _ in frames):
            return False

        for cam, (timestamp_ms, _, frame) in zip(self.cameras, frames):
            frame = cv2.remap(frame, cam['map1'], cam['map2'], cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Kalman predict
            quat_pred, tvec_pred = cam['kalman'].predict()

            # Detect ArUco
            aruco_corners, aruco_detected = self.detect_aruco_marker(gray)

            # Estimate pose (use Kalman prediction as initial guess if available)
            quat_meas, tvec_meas, error = self.estimate_and_validate_pose(
                aruco_corners, cam['K'], quat_init=quat_pred, tvec_init=tvec_pred)

            # Kalman update
            if quat_meas is not None and tvec_meas is not None:
                cam['kalman'].update(quat_meas, tvec_meas)

            # Get filtered pose
            quat, tvec = cam['kalman'].get_filtered_pose()

            # Log pose to CSV
            if self.csv_writer and quat is not None and tvec is not None:
                self.csv_writer.writerow([
                    self.frame_count,
                    f'{timestamp_ms/1000.0:.6f}',
                    f'{tvec[0,0]:.4f}', f'{tvec[1,0]:.4f}', f'{tvec[2,0]:.4f}',
                    f'{quat[1]:.6f}', f'{quat[2]:.6f}', f'{quat[3]:.6f}', f'{quat[0]:.6f}'
                ])

            # Visualize — convert quat→rvec only at the OpenCV boundary
            rvec = quat_to_rvec(quat) if quat is not None else None
            self.visualize(frame, aruco_corners, aruco_detected, rvec, tvec, error, cam['color'], cam['K'])
            cam['frame'] = frame

        self.frame_count += 1
        return True

    def run(self):
        """Main tracking loop."""
        while True:
            if not self.process_frame_pair():
                break

            # Display
            display = np.hstack([cv2.resize(cam['frame'], None, fx=self.DISPLAY_SCALE, fy=self.DISPLAY_SCALE)
                               for cam in self.cameras])
            cv2.imshow('ArUco Stereo Tracking', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        """Release resources."""
        for cam in self.cameras:
            cam['cap'].release()
        if self.csv_file:
            self.csv_file.close()
        cv2.destroyAllWindows()
