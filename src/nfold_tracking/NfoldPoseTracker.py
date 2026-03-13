"""
N-fold Marker Detection and Pose Tracking

Supports both lab and outdoor configurations:
- Lab setup: Uses nfold_config (board only)
- Outdoor setup: Uses outdoor_test.nfold_config (board + target object)
"""
import os
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
import csv
import cv2
cv2.setLogLevel(0)
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import contextlib
import importlib

from MarkerTracker import MarkerTracker
from PoseKalmanFilter import PoseKalmanFilter
from rotation_utils import rvec_to_quat, quat_to_rvec, quat_angular_distance


class NfoldPoseTracker:
    """N-fold marker detector with geometric constellation matching."""

    def __init__(self, calib_path, video_90_path, video_91_path=None, config_module='nfold_config', csv_path=None):
        """
        Initialize N-fold pose tracker.

        Args:
            calib_path: Path to calibration files directory
            video_90_path: Path to 90 camera video
            video_91_path: Path to 91 camera video (optional)
            config_module: Config module name ('nfold_config' for lab, 'outdoor_test.nfold_config' for outdoor)
        """
        self.calib_path = Path(calib_path)

        # Import configuration dynamically
        config = importlib.import_module(config_module)
        self.NFOLD_ORDER = config.NFOLD_ORDER
        self.KERNEL_SIZE = config.KERNEL_SIZE
        self.MARKER_3D = config.MARKER_3D
        self.BOARD_CORNERS_3D = config.BOARD_CORNERS_3D

        # Import rectangular object if available (outdoor config)
        self.RECT_CORNERS_3D = getattr(config, 'RECT_CORNERS_3D', None)
        self.PROCESS_SCALE = getattr(config, 'PROCESS_SCALE', 1.0)
        self.DEPTH_RANGE = getattr(config, 'DEPTH_RANGE', (300, 3000))

        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        if csv_path:
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['frame', 'time_s', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

        self.frame_count = 0

        self.cameras = self._setup_cameras(video_90_path, video_91_path)

        # Setup undistortion maps
        ret, frame = self.cameras[0]['cap'].read()
        self.cameras[0]['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_size = (frame.shape[1], frame.shape[0])

        for cam in self.cameras:
            cam['map1'], cam['map2'] = cv2.initUndistortRectifyMap(
                cam['K'], cam['dist'], None, cam['K'], frame_size, cv2.CV_32FC1)
            cam['kalman'] = PoseKalmanFilter(dt=1/30.0, process_noise=100.0, measurement_noise=1.0, gate_trans_mm=300.0, gate_rot_deg=30.0)
            cam['frames_without_match'] = 0

    def _setup_cameras(self, video_90_path, video_91_path):
        cameras = []
        camera_configs = []
        if video_90_path:
            camera_configs.append(('90', video_90_path, (0, 255, 255)))
        if video_91_path:
            camera_configs.append(('91', video_91_path, (255, 0, 255)))

        for side, video_path, color in camera_configs:
            data = np.load(str(self.calib_path / f'calib_data_{side}.npz'))
            cameras.append({
                'K': data['mtx'],
                'dist': data['dist'],
                'tracker': MarkerTracker(order=self.NFOLD_ORDER, kernel_size=self.KERNEL_SIZE, scale_factor=0.1),
                'cap': cv2.VideoCapture(str(video_path)),
                'color': color
            })
        return cameras

    def detect_markers(self, gray, tracker):
        with open(os.devnull, 'w') as devnull, \
             contextlib.redirect_stderr(devnull), \
             contextlib.redirect_stdout(devnull):
            if self.PROCESS_SCALE < 1.0:
                gray_small = cv2.resize(gray, None, fx=self.PROCESS_SCALE, fy=self.PROCESS_SCALE)
            else:
                gray_small = gray
            tracker.locate_marker_init(gray_small)
            try:
                markers, _, _ = tracker.detect_multiple_markers(gray_small)
                if self.PROCESS_SCALE < 1.0:
                    inv = 1.0 / self.PROCESS_SCALE
                    return [(m.x * inv, m.y * inv, 1.0) for m in markers]
                return [(m.x, m.y, 1.0) for m in markers]
            except:
                return []

    def get_expected_positions(self, kalman_pose, K):
        if kalman_pose and K is not None:
            quat, tvec = kalman_pose
            if quat is not None:
                proj, _ = cv2.projectPoints(
                    np.array([self.MARKER_3D[i] for i in sorted(self.MARKER_3D.keys())], dtype=np.float32),
                    quat_to_rvec(quat), tvec, K, None
                )
                return {i: proj[i, 0] for i in range(6)}
        return None

    def match_with_expected(self, detected_positions, expected_positions, max_distance=30, recovery_mode=False):
        if not expected_positions:
            return None

        identified, used = [], set()
        for marker_id, exp_pos in expected_positions.items():
            best = min(
                ((i, np.linalg.norm([detected_positions[i][0] - exp_pos[0],
                                    detected_positions[i][1] - exp_pos[1]]))
                 for i in range(len(detected_positions)) if i not in used),
                key=lambda x: x[1], default=(None, max_distance + 1)
            )
            if best[1] < max_distance and best[0] is not None:
                identified.append((best[0], marker_id))
                used.add(best[0])

        if len(identified) >= 4 and self.validate_chirality(identified, detected_positions, recovery_mode):
            return identified
        return None

    def validate_chirality(self, matches, detected_positions, recovery_mode=False):
        match_dict = {mid: detected_positions[det_idx] for det_idx, mid in matches}

        if recovery_mode and len(match_dict) >= 3:
            return True

        # Check multiple marker triplets
        triplets = [(0, 1, 2), (0, 1, 3)]
        for m0, m1, m2 in triplets:
            if m0 in match_dict and m1 in match_dict and m2 in match_dict:
                p0 = np.array([match_dict[m0][0], match_dict[m0][1]])
                p1 = np.array([match_dict[m1][0], match_dict[m1][1]])
                p2 = np.array([match_dict[m2][0], match_dict[m2][1]])

                cross_z = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
                if cross_z <= 0:
                    return False

        return len(match_dict) >= 3 or recovery_mode

    def match_constellation(self, detected_positions, max_cost=0.85, recovery_mode=False):
        min_markers = 3 if recovery_mode else 4
        if len(detected_positions) < min_markers:
            return None

        if recovery_mode:
            max_cost = 1.2

        model_ids = sorted(self.MARKER_3D.keys())
        model_2d = np.array([(self.MARKER_3D[mid][0], self.MARKER_3D[mid][1]) for mid in model_ids])

        det_arr = np.array(detected_positions, dtype=np.float32)
        det_norm = (det_arr - det_arr.mean(axis=0)) / np.mean(np.linalg.norm(det_arr - det_arr.mean(axis=0), axis=1))
        model_norm = (model_2d - model_2d.mean(axis=0)) / np.mean(np.linalg.norm(model_2d - model_2d.mean(axis=0), axis=1))

        # Two-stage rotation search
        best_match, best_cost, best_angle = None, float('inf'), 0
        for angle_deg in range(-180, 180, 5):
            rot = np.radians(angle_deg)
            rot_mat = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            model_rot = model_norm @ rot_mat.T

            cost_matrix = np.linalg.norm(det_norm[:, None] - model_rot[None, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            avg_cost = cost_matrix[row_ind, col_ind].mean()

            if avg_cost < best_cost:
                best_cost = avg_cost
                best_match = list(zip(row_ind, col_ind))
                best_angle = angle_deg

        # Fine search around best angle
        for angle_deg in range(best_angle - 7, best_angle + 8, 1):
            rot = np.radians(angle_deg)
            rot_mat = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            model_rot = model_norm @ rot_mat.T

            cost_matrix = np.linalg.norm(det_norm[:, None] - model_rot[None, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            avg_cost = cost_matrix[row_ind, col_ind].mean()

            if avg_cost < best_cost:
                best_cost = avg_cost
                best_match = list(zip(row_ind, col_ind))

        matches = [(det_idx, model_ids[model_idx]) for det_idx, model_idx in best_match]
        if not self.validate_chirality(matches, detected_positions, recovery_mode):
            return None

        if best_cost > max_cost:
            return None

        return matches

    def estimate_pose(self, markers, matches, K, predicted_pose=None):
        """Estimate pose from matched markers.

        predicted_pose dict uses 'quat' key (quaternion).
        Returns dict with 'quat', 'tvec', 'error' keys, or None on failure.
        """
        if len(matches) < 4:
            return None

        img_pts = np.array([[markers[i][0], markers[i][1]] for i, _ in matches], dtype=np.float32)
        obj_pts = np.array([self.MARKER_3D[mid] for _, mid in matches], dtype=np.float32)

        try:
            if predicted_pose:
                # OpenCV boundary: convert quat→rvec for solvePnP initial guess
                rvec_init = quat_to_rvec(predicted_pose['quat'])
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, None,
                    rvec_init, predicted_pose['tvec'],
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_SQPNP)
            else:
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_SQPNP)

            if not success:
                return None

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-6)
            rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, None, rvec, tvec, criteria)

            # OpenCV boundary: projectPoints needs rvec
            proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
            error = np.sqrt(np.mean(np.sum((img_pts - proj_pts.reshape(-1, 2))**2, axis=1)))

            depth_min, depth_max = self.DEPTH_RANGE
            if error > 3.0 or not (depth_min <= tvec[2, 0] <= depth_max):
                return None

            return {'quat': rvec_to_quat(rvec), 'tvec': tvec, 'error': error}
        except:
            return None

    def visualize(self, frame, markers, K, color, matched_indices, marker_ids, pose):
        matched_set = set(matched_indices)

        if pose:
            # Convert quat→rvec only at the OpenCV boundary
            rvec = quat_to_rvec(pose['quat'])
            tvec = pose['tvec']

            # Draw board boundary
            board_proj = cv2.projectPoints(self.BOARD_CORNERS_3D, rvec, tvec, K, None)[0]
            pts = board_proj.reshape(-1, 2).astype(int)
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]), color, 3)

            # Draw rectangular object boundary (if configured)
            if self.RECT_CORNERS_3D is not None:
                rect_proj = cv2.projectPoints(self.RECT_CORNERS_3D, rvec, tvec, K, None)[0]
                rect_pts = rect_proj.reshape(-1, 2).astype(int)
                rect_color = (0, 255, 0)  # Green for target object
                for i in range(4):
                    cv2.line(frame, tuple(rect_pts[i]), tuple(rect_pts[(i+1)%4]), rect_color, 4)

        for i, (x, y, _) in enumerate(markers):
            if i in matched_set:
                cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 0), -1)
                cv2.putText(frame, str(marker_ids[matched_indices.index(i)]),
                          (int(x)+12, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), 2)

    def process_frame_pair(self):
        frames = [(cam['cap'].get(cv2.CAP_PROP_POS_MSEC), cam['cap'].read()[1]) for cam in self.cameras]
        if not all(f is not None for _, f in frames):
            return False

        for cam, (timestamp_ms, frame) in zip(self.cameras, frames):
            frame = cv2.remap(frame, cam['map1'], cam['map2'], cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            markers = self.detect_markers(gray, cam['tracker'])

            pred_quat, pred_tvec = cam['kalman'].predict()
            kalman_pose = (pred_quat, pred_tvec) if pred_quat is not None else None

            recovery_mode = cam['frames_without_match'] > 5
            if cam['frames_without_match'] > 30:
                cam['kalman'] = PoseKalmanFilter(dt=1/30.0, process_noise=100.0, measurement_noise=1.0, gate_trans_mm=300.0, gate_rot_deg=30.0)
                kalman_pose = None

            matched_indices, marker_ids, pose = [], [], None
            matches = None

            if kalman_pose and not recovery_mode:
                expected = self.get_expected_positions(kalman_pose, cam['K'])
                matches = self.match_with_expected([(m[0], m[1]) for m in markers], expected, recovery_mode=recovery_mode)

            if not matches:
                matches = self.match_constellation([(m[0], m[1]) for m in markers], recovery_mode=recovery_mode)

            if matches:
                predicted_pose_dict = None
                if kalman_pose and kalman_pose[0] is not None:
                    predicted_pose_dict = {'quat': kalman_pose[0], 'tvec': kalman_pose[1]}
                measured_pose = self.estimate_pose(markers, matches, cam['K'], predicted_pose_dict)

                if measured_pose and kalman_pose and kalman_pose[0] is not None:
                    ang_diff  = quat_angular_distance(measured_pose['quat'], pred_quat)
                    tvec_diff = np.linalg.norm(measured_pose['tvec'] - pred_tvec)

                    if ang_diff > 1.0 or tvec_diff > 200.0:
                        measured_pose = None

                if measured_pose:
                    matched_indices = [i for i, _ in matches]
                    marker_ids = [mid for _, mid in matches]
                    cam['kalman'].update(measured_pose['quat'], measured_pose['tvec'])
                    cam['frames_without_match'] = 0

                    # Only show pose when we have measurements (no temporal prediction)
                    filt_quat, filt_tvec = cam['kalman'].get_filtered_pose()
                    if filt_quat is not None:
                        pose = {'quat': filt_quat, 'tvec': filt_tvec}

            if not matches or not measured_pose:
                cam['frames_without_match'] += 1

            # Log pose to CSV
            if self.csv_writer and pose is not None:
                quat, tvec = pose['quat'], pose['tvec']
                self.csv_writer.writerow([
                    self.frame_count,
                    f'{timestamp_ms/1000.0:.6f}',
                    f'{tvec[0,0]:.4f}', f'{tvec[1,0]:.4f}', f'{tvec[2,0]:.4f}',
                    f'{quat[1]:.6f}', f'{quat[2]:.6f}', f'{quat[3]:.6f}', f'{quat[0]:.6f}'
                ])

            self.visualize(frame, markers, cam['K'], cam['color'], matched_indices, marker_ids, pose)
            cam['frame'] = frame

        self.frame_count += 1
        return True

    def run(self):
        while True:
            if not self.process_frame_pair():
                break

            display = np.hstack([cam['frame'] for cam in self.cameras])
            display = cv2.resize(display, None, fx=0.5, fy=0.5)
            cv2.imshow('N-fold Tracking', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for cam in self.cameras:
            cam['cap'].release()
        if self.csv_file:
            self.csv_file.close()
        cv2.destroyAllWindows()
