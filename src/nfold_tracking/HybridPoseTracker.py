"""
Hybrid Pose Tracker - ArUco reference + N-fold markers

Supports both lab and outdoor configurations:
- Lab setup: Uses hybrid_config (board only)
- Outdoor setup: Uses outdoor_test.hybrid_config (board + target object)
"""
import os
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
import csv
import cv2
cv2.setLogLevel(0)
import numpy as np
from pathlib import Path
import contextlib
import importlib
from scipy.optimize import linear_sum_assignment

from MarkerPose import MarkerPose
from MarkerTracker import MarkerTracker
from PoseKalmanFilter import PoseKalmanFilter
from rotation_utils import rvec_to_quat, quat_to_rvec, quat_angular_distance


class HybridPoseTracker:
    """Hybrid pose tracking using ArUco marker as reference and nfold markers for pose estimation."""

    def __init__(self, calib_path, video_90_path, video_91_path=None, config_module='indoor_test.hybrid_config', csv_path=None, num_markers=None):
        """
        Initialize hybrid pose tracker.

        Args:
            calib_path: Path to calibration files directory
            video_90_path: Path to 90 camera video
            video_91_path: Path to 91 camera video (optional)
            config_module: Config module name ('hybrid_config' for lab, 'outdoor_test.hybrid_config' for outdoor)
        """
        self.calib_path = Path(calib_path)

        # Import configuration dynamically
        config = importlib.import_module(config_module)
        self.MARKER_3D = getattr(config, 'MARKER_3D_ARRAY', config.MARKER_3D)
        if num_markers is not None:
            self.MARKER_3D = self.MARKER_3D[:num_markers]
        self.BOARD_CORNERS_3D = config.BOARD_CORNERS_3D
        self.ARUCO_SIZE = config.ARUCO_SIZE
        self.ARUCO_CORNERS_3D = config.ARUCO_CORNERS_3D
        self.ARUCO_REFERENCE_ID = config.ARUCO_REFERENCE_ID
        self.DEPTH_RANGE = getattr(config, 'DEPTH_RANGE', (300, 3000))
        self.MARKER_ORDER = config.MARKER_ORDER
        self.KERNEL_SIZE = config.KERNEL_SIZE
        self.DISPLAY_SCALE = config.DISPLAY_SCALE
        self.EDGE_MARGIN = config.EDGE_MARGIN
        self.PROCESS_SCALE = getattr(config, 'PROCESS_SCALE', 1.0)

        # Import rectangular object if available (outdoor config)
        self.RECT_CORNERS_3D = getattr(config, 'RECT_CORNERS_3D', None)
        self.QUALITY_THRESHOLD = getattr(config, 'QUALITY_THRESHOLD', 0.5)

        # Setup ArUco detector with subpixel refinement
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        aruco_params.cornerRefinementWinSize = 5
        aruco_params.cornerRefinementMaxIterations = 30
        aruco_params.cornerRefinementMinAccuracy = 0.1

        self.aruco_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), aruco_params)

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

        # Get frame dimensions and setup undistortion
        h, w = self.cameras[0]['cap'].read()[1].shape[:2]
        self.cameras[0]['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_size = (w, h)

        for cam in self.cameras:
            cam['map1'], cam['map2'] = cv2.initUndistortRectifyMap(
                cam['K'], cam['dist'], None, cam['K'], self.frame_size, cv2.CV_32FC1)

    def _setup_cameras(self, video_90_path, video_91_path):
        """Setup camera configurations."""
        cameras = []
        camera_configs = []
        if video_90_path:
            camera_configs.append(('90', video_90_path, (0, 255, 255)))
        if video_91_path:
            camera_configs.append(('91', video_91_path, (255, 0, 255)))

        for side, path, color in camera_configs:
            data = np.load(str(self.calib_path / f'calib_data_{side}.npz'))
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            dt = 1.0 / fps if fps > 0 else 1/30.0
            cameras.append({
                'K': data['mtx'],
                'dist': data['dist'],
                'tracker': MarkerTracker(order=self.MARKER_ORDER, kernel_size=self.KERNEL_SIZE, scale_factor=0.1),
                'cap': cap,
                'color': color,
                'last_pose': None,
                'kalman': PoseKalmanFilter(dt=dt, process_noise=100.0, measurement_noise=1.0, gate_trans_mm=300.0, gate_rot_deg=30.0),
                'frames_without_match': 0
            })
        return cameras

    def get_expected_positions(self, aruco_corners, aruco_ids, last_pose, K):
        """Get expected marker positions from ArUco or last pose."""
        # Try ArUco reference first
        if aruco_corners is not None and aruco_ids is not None:
            ref_idx = np.where(aruco_ids == self.ARUCO_REFERENCE_ID)[0]
            if len(ref_idx) > 0:
                corners_img = aruco_corners[ref_idx[0]].reshape(4, 2).astype(np.float32)
                corners_3d  = self.ARUCO_CORNERS_3D[:, :2].astype(np.float32)
                H, _ = cv2.findHomography(corners_3d, corners_img)
                if H is not None:
                    expected = {}
                    for i, pt in enumerate(self.MARKER_3D):
                        p = H @ np.array([pt[0], pt[1], 1.0], dtype=np.float64)
                        expected[i] = p[:2] / p[2]
                    return expected

        # Fallback to last pose
        if last_pose and K is not None:
            quat, tvec = last_pose
            if quat is not None:
                proj, _ = cv2.projectPoints(
                    self.MARKER_3D, quat_to_rvec(quat), tvec, K, None)
                return {i: proj[i, 0] for i in range(len(self.MARKER_3D))}

        return None

    def detect_and_identify_markers(self, gray, tracker, aruco_corners, aruco_ids, last_pose, K,
                                     frames_without_match=0):
        """Detect and identify nfold markers."""
        # Determine whether ArUco is providing the expected positions.
        # When it is, predictions are tight (homography); when coasting on Kalman,
        # widen the search as prediction confidence decreases with time.
        aruco_ref_visible = (
            aruco_corners is not None and aruco_ids is not None and
            len(np.where(aruco_ids == self.ARUCO_REFERENCE_ID)[0]) > 0
        )
        if aruco_ref_visible:
            search_radius = 30
        else:
            # When coasting on Kalman, widen search slightly for vibration/motion.
            # Cap at 60px — beyond that the masks overlap and cause misidentification.
            # If too many frames have been missed without ArUco, stop trying to
            # identify via stale Kalman predictions entirely.
            if frames_without_match > 10:
                return []
            search_radius = min(30 + frames_without_match * 3, 60)
        match_threshold = search_radius + 10

        with open(os.devnull, 'w') as devnull, \
             contextlib.redirect_stderr(devnull), \
             contextlib.redirect_stdout(devnull):
            if self.PROCESS_SCALE < 1.0:
                gray_small = cv2.resize(gray, None, fx=self.PROCESS_SCALE, fy=self.PROCESS_SCALE)
            else:
                gray_small = gray
            tracker.locate_marker_init(gray_small)

            # Zero out ArUco region in convolution response before peak detection
            # so it can't set the reference intensity or suppress real nfold markers
            if aruco_corners is not None and aruco_ids is not None:
                ref_idx = np.where(aruco_ids == self.ARUCO_REFERENCE_ID)[0]
                if len(ref_idx) > 0:
                    ac = aruco_corners[ref_idx[0]].reshape(4, 2)
                    if self.PROCESS_SCALE < 1.0:
                        ac = ac * self.PROCESS_SCALE
                    margin = 10
                    ax1 = max(0, int(ac[:, 0].min()) - margin)
                    ay1 = max(0, int(ac[:, 1].min()) - margin)
                    ax2 = min(tracker.frame_sum_squared.shape[1], int(ac[:, 0].max()) + margin)
                    ay2 = min(tracker.frame_sum_squared.shape[0], int(ac[:, 1].max()) + margin)
                    tracker.frame_sum_squared[ay1:ay2, ax1:ax2] = 0

            expected = self.get_expected_positions(aruco_corners, aruco_ids, last_pose, K)

            # When expected positions are known, zero out all frame_sum_squared regions
            # that are far from any expected marker position. This prevents strong
            # non-marker responses elsewhere in the image from dominating
            # reference_intensity in detect_multiple_markers.
            if expected:
                scale = self.PROCESS_SCALE
                h_s, w_s = tracker.frame_sum_squared.shape
                sr_s = max(1, int(search_radius * scale))
                keep_mask = np.zeros((h_s, w_s), dtype=np.float32)
                for exp_pos in expected.values():
                    ex_s = int(exp_pos[0] * scale)
                    ey_s = int(exp_pos[1] * scale)
                    x1 = max(0, ex_s - sr_s)
                    x2 = min(w_s, ex_s + sr_s + 1)
                    y1 = max(0, ey_s - sr_s)
                    y2 = min(h_s, ey_s + sr_s + 1)
                    keep_mask[y1:y2, x1:x2] = 1.0
                tracker.frame_sum_squared *= keep_mask

            try:
                nfold, _, _ = tracker.detect_multiple_markers(gray_small)
                if self.PROCESS_SCALE < 1.0:
                    for m in nfold:
                        m.scale_position(1.0 / self.PROCESS_SCALE)
                w, h = self.frame_size
                nfold = [m for m in nfold if self.EDGE_MARGIN <= m.x < w - self.EDGE_MARGIN
                        and self.EDGE_MARGIN <= m.y < h - self.EDGE_MARGIN]
                # Quality filter only when ArUco is absent — proximity to homography-
                # projected positions already rejects noise when ArUco is visible.
                if not aruco_ref_visible:
                    nfold = [m for m in nfold if m.quality >= self.QUALITY_THRESHOLD]

                # Remove detections inside the ArUco bounding box
                if aruco_corners is not None and aruco_ids is not None:
                    ref_idx = np.where(aruco_ids == self.ARUCO_REFERENCE_ID)[0]
                    if len(ref_idx) > 0:
                        ac = aruco_corners[ref_idx[0]].reshape(4, 2)
                        margin = 10
                        ax1, ay1 = ac[:, 0].min() - margin, ac[:, 1].min() - margin
                        ax2, ay2 = ac[:, 0].max() + margin, ac[:, 1].max() + margin
                        nfold = [m for m in nfold
                                 if not (ax1 <= m.x <= ax2 and ay1 <= m.y <= ay2)]

                if expected and nfold:
                    marker_ids   = list(expected.keys())
                    exp_pos_arr  = np.array([expected[mid] for mid in marker_ids], dtype=np.float64)
                    det_pos_arr  = np.array([[m.x, m.y] for m in nfold], dtype=np.float64)
                    cost = np.linalg.norm(
                        exp_pos_arr[:, None, :] - det_pos_arr[None, :, :], axis=2)
                    row_ind, col_ind = linear_sum_assignment(cost)
                    identified = []
                    for r, c in zip(row_ind, col_ind):
                        if cost[r, c] < match_threshold:
                            nfold[c].number = marker_ids[r]
                            identified.append(nfold[c])
                    return identified
            except Exception:
                pass
        return []

    def estimate_pose(self, markers, K, predicted_pose=None, aruco_img_pts=None):
        """Estimate pose from markers using SQPNP.

        Optionally augments the point set with ArUco corners (aruco_img_pts,
        shape (4,2)) for improved translation accuracy.

        Returns (quat, tvec, error) where quat is [qw, qx, qy, qz], or
        (None, None, None) on failure.
        """
        if len(markers) < 4:
            return None, None, None

        img_pts = np.array([[m.x, m.y] for m in markers], dtype=np.float32)
        obj_pts = np.array([self.MARKER_3D[m.number] for m in markers], dtype=np.float32)

        if aruco_img_pts is not None and len(aruco_img_pts) == 4:
            img_pts = np.vstack([img_pts, aruco_img_pts.astype(np.float32)])
            obj_pts = np.vstack([obj_pts, self.ARUCO_CORNERS_3D])

        try:
            if predicted_pose is not None:
                rvec_init = quat_to_rvec(predicted_pose[0])
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, None,
                    rvec_init, predicted_pose[1],
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_SQPNP)
            else:
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_SQPNP)

            if not success:
                return None, None, None

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-6)
            rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, None, rvec, tvec, criteria)

            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
            error = np.sqrt(np.mean(np.sum((img_pts - proj.reshape(-1, 2))**2, axis=1)))

            depth_min, depth_max = self.DEPTH_RANGE
            if error > 5.0 or not (depth_min <= tvec[2, 0] <= depth_max):
                return None, None, None

            return rvec_to_quat(rvec), tvec, error
        except Exception:
            return None, None, None

    def visualize(self, frame, aruco_corners, aruco_ids, identified, quat, tvec, color, K):
        """Draw visualization elements."""
        # Draw reference ArUco marker
        if aruco_corners is not None and aruco_ids is not None:
            ref_idx = np.where(aruco_ids == self.ARUCO_REFERENCE_ID)[0]
            if len(ref_idx) > 0:
                cv2.aruco.drawDetectedMarkers(frame, [aruco_corners[ref_idx[0]]],
                                             np.array([aruco_ids[ref_idx[0]]]))

        # Draw board boundary
        if quat is not None and tvec is not None:
            rvec = quat_to_rvec(quat)
            board_proj = cv2.projectPoints(self.BOARD_CORNERS_3D, rvec, tvec, K, None)[0]
            pts = board_proj.reshape(-1, 2).astype(int)
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]), color, 3)

            # Draw rectangular object boundary (if configured)
            if self.RECT_CORNERS_3D is not None:
                rect_proj = cv2.projectPoints(self.RECT_CORNERS_3D, rvec, tvec, K, None)[0].reshape(-1, 2).astype(int)
                rect_color = (0, 255, 0)
                for i in range(4):
                    cv2.line(frame, tuple(rect_proj[i]), tuple(rect_proj[(i+1)%4]), rect_color, 4)

        # Draw identified markers
        for m in identified:
            cv2.circle(frame, (int(m.x), int(m.y)), 8, (0, 255, 0), -1)
            cv2.putText(frame, str(m.number), (int(m.x) + 12, int(m.y) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Status text
        mode = " + Target" if self.RECT_CORNERS_3D is not None else ""
        cv2.putText(frame, f"Markers: {len(identified)}/{len(self.MARKER_3D)}{mode}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def process_frame_pair(self):
        """Process one frame pair from both cameras."""
        frames = [(cam['cap'].get(cv2.CAP_PROP_POS_MSEC), *cam['cap'].read()) for cam in self.cameras]
        if not all(ret for _, ret, _ in frames):
            return False

        for cam, (timestamp_ms, _, frame) in zip(self.cameras, frames):
            frame = cv2.remap(frame, cam['map1'], cam['map2'], cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco
            aruco_corners, aruco_ids, _ = self.aruco_detector.detectMarkers(gray)
            aruco_ids = aruco_ids.flatten() if aruco_ids is not None else None

            # Kalman prediction — provides expected pose for this frame
            pred_quat, pred_tvec = cam['kalman'].predict()
            kalman_pose = (pred_quat, pred_tvec) if pred_quat is not None else None

            # Reset filter after prolonged tracking loss
            if cam['frames_without_match'] > 30:
                cam['kalman'].reset()
                cam['last_pose'] = None
                cam['frames_without_match'] = 0
                kalman_pose = None

            # Detect and identify nfold markers (use Kalman prediction as fallback)
            identified = self.detect_and_identify_markers(
                gray, cam['tracker'], aruco_corners, aruco_ids, kalman_pose, cam['K'],
                frames_without_match=cam['frames_without_match'])

            # Extract reference ArUco corners for augmented PnP
            aruco_ref_corners = None
            if aruco_corners is not None and aruco_ids is not None:
                ref_idx = np.where(aruco_ids == self.ARUCO_REFERENCE_ID)[0]
                if len(ref_idx) > 0:
                    aruco_ref_corners = aruco_corners[ref_idx[0]].reshape(4, 2)

            # Estimate pose (ArUco corners augment the nfold point set when available)
            quat, tvec, error = self.estimate_pose(identified, cam['K'], predicted_pose=kalman_pose,
                                                   aruco_img_pts=aruco_ref_corners)

            # Reject outliers that diverge too far from the Kalman prediction.
            # Skip gate when ArUco is visible (identification used homography, not Kalman)
            # or during recovery (frames_without_match > 30) to allow self-correction.
            aruco_visible = aruco_ref_corners is not None
            if quat is not None and kalman_pose is not None and not aruco_visible and cam['frames_without_match'] <= 30:
                if (quat_angular_distance(quat, pred_quat) > 1.0 or
                        np.linalg.norm(tvec - pred_tvec) > 200.0):
                    quat, tvec, error = None, None, None

            if quat is not None:
                # When ArUco is visible and the new pose is far from the current
                # Kalman state, force-reset so the inner gate doesn't block recovery.
                if aruco_visible and cam['kalman'].is_initialized():
                    _, filt_t = cam['kalman'].get_filtered_pose()
                    if filt_t is not None and np.linalg.norm(tvec - filt_t) > 500.0:
                        cam['kalman'].reset()
                cam['kalman'].update(quat, tvec)
                cam['frames_without_match'] = 0
                filt_quat, filt_tvec = cam['kalman'].get_filtered_pose()
                cam['last_pose'] = (filt_quat, filt_tvec)
            else:
                cam['frames_without_match'] += 1
                filt_quat, filt_tvec = None, None

            # Log pose to CSV
            if self.csv_writer and filt_quat is not None and filt_tvec is not None:
                self.csv_writer.writerow([
                    self.frame_count,
                    f'{timestamp_ms/1000.0:.6f}',
                    f'{filt_tvec[0,0]:.4f}', f'{filt_tvec[1,0]:.4f}', f'{filt_tvec[2,0]:.4f}',
                    f'{filt_quat[1]:.6f}', f'{filt_quat[2]:.6f}', f'{filt_quat[3]:.6f}', f'{filt_quat[0]:.6f}'
                ])

            # Visualize using filtered pose
            self.visualize(frame, aruco_corners, aruco_ids, identified,
                         filt_quat, filt_tvec, cam['color'], cam['K'])
            cam['frame'] = frame

        self.frame_count += 1
        return True

    def run(self):
        """Main tracking loop."""
        while True:
            if not self.process_frame_pair():
                break

            display = np.hstack([cv2.resize(cam['frame'], None, fx=self.DISPLAY_SCALE, fy=self.DISPLAY_SCALE)
                               for cam in self.cameras])
            cv2.imshow('Hybrid Tracking', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for cam in self.cameras:
            cam['cap'].release()
        if self.csv_file:
            self.csv_file.close()
        cv2.destroyAllWindows()

    def run_headless(self):
        while True:
            if not self.process_frame_pair():
                break
        for cam in self.cameras:
            cam['cap'].release()
        if self.csv_file:
            self.csv_file.close()
