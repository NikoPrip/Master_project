import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import cv2
import time

class ObjectTracker:
    def __init__(self):
        # Load calibration data
        left_calib = np.load('src/camera_calibration/calib_files/calib_data_left.npz')
        right_calib = np.load('src/camera_calibration/calib_files/calib_data_right.npz')
        stereo_calib = np.load('src/camera_calibration/calib_files/stereo_calib_data.npz')
        
        # Calculate rectification and create maps
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left_calib['mtx'], left_calib['dist'],
            right_calib['mtx'], right_calib['dist'],
            (1280, 800), stereo_calib['R'], stereo_calib['T'])
        
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            left_calib['mtx'], left_calib['dist'], R1, P1, (1280, 800), cv2.CV_16SC2)
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            right_calib['mtx'], right_calib['dist'], R2, P2, (1280, 800), cv2.CV_16SC2)
        
        self.rect_mtx_left = P1[:3, :3]
        self.rect_mtx_right = P2[:3, :3]
        
        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_size = 0.16
        
        # Object points relative to marker
        self.object_points = np.array([[0, 0.105, 0], [0.70, 0.105, 0]], dtype=np.float32)

    def run(self):
        Gst.init(None)
        
        # Create pipelines
        pipeline1 = Gst.parse_launch("udpsrc port=58004 ! application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink1 emit-signals=false sync=false max-buffers=2 drop=true")
        pipeline2 = Gst.parse_launch("udpsrc port=58006 ! application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink2 emit-signals=false sync=false max-buffers=2 drop=true")
        
        appsink1, appsink2 = pipeline1.get_by_name("sink1"), pipeline2.get_by_name("sink2")
        pipeline1.set_state(Gst.State.PLAYING)
        pipeline2.set_state(Gst.State.PLAYING)
        time.sleep(2)
        
        cv2.namedWindow("Stereo ArUco Tracking", cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                frame_left = self.get_frame(appsink2, self.map1_left, self.map2_left)
                frame_right = self.get_frame(appsink1, self.map1_right, self.map2_right)
                
                if frame_left is not None and frame_right is not None:
                    self.process_frames(frame_left, frame_right)
                    cv2.imshow("Stereo ArUco Tracking", np.hstack((cv2.resize(frame_left, (640, 480)), cv2.resize(frame_right, (640, 480)))))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            pipeline1.set_state(Gst.State.NULL)
            pipeline2.set_state(Gst.State.NULL)

    def get_frame(self, appsink, map1, map2):
        sample = appsink.emit("try-pull-sample", 10000000)
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            h, w = caps.get_value('height'), caps.get_value('width')
            frame = np.ndarray((h, w, 3), buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8).copy()
            return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        return None

    def process_frames(self, frame_left, frame_right):
        corners_left, ids_left = cv2.aruco.detectMarkers(frame_left, self.aruco_dict, parameters=self.aruco_params)[:2]
        corners_right, ids_right = cv2.aruco.detectMarkers(frame_right, self.aruco_dict, parameters=self.aruco_params)[:2]
        
        self.draw_markers(frame_left, corners_left, ids_left, True)
        self.draw_markers(frame_right, corners_right, ids_right, False)
        
        # Print pose comparison for common markers
        if ids_left is not None and ids_right is not None:
            common_ids = np.intersect1d(ids_left.flatten(), ids_right.flatten())
            for marker_id in common_ids:
                pose_left = self.get_pose(corners_left, ids_left, marker_id, True)
                pose_right = self.get_pose(corners_right, ids_right, marker_id, False)
                if pose_left is not None and pose_right is not None:
                    diff = np.linalg.norm(pose_left - pose_right)
                    print(f"Marker {marker_id}: Left{pose_left}, Right{pose_right}")

    def draw_markers(self, frame, corners, ids, is_left):
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            for corner, marker_id in zip(corners, ids.flatten()):
                center = np.mean(corner[0], axis=0).astype(int)
                cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{marker_id}", tuple(center + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Calculate pose and draw axes + object points
                mtx = self.rect_mtx_left if is_left else self.rect_mtx_right
                marker_corners = np.array([[-self.marker_size/2, self.marker_size/2, 0], [self.marker_size/2, self.marker_size/2, 0], [self.marker_size/2, -self.marker_size/2, 0], [-self.marker_size/2, -self.marker_size/2, 0]], dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(marker_corners, corner[0], mtx, np.zeros(5))
                if success:
                    cv2.drawFrameAxes(frame, mtx, np.zeros(5), rvec, tvec, self.marker_size * 0.5)
                    self.draw_object_points(frame, rvec, tvec, mtx)

    def draw_object_points(self, frame, rvec, tvec, mtx):
        # Transform object points to camera coordinates
        R, _ = cv2.Rodrigues(rvec)
        object_points_3d = [(R @ point.reshape(3, 1) + tvec.reshape(3, 1)).flatten() for point in self.object_points]
        
        # Project to image coordinates and draw
        image_points, _ = cv2.projectPoints(np.array(object_points_3d), np.zeros(3), np.zeros(3), mtx, np.zeros(5))
        for i, point in enumerate(image_points.reshape(-1, 2).astype(int)):
            cv2.circle(frame, tuple(point), 8, (0, 255, 255), -1)
            cv2.putText(frame, f"P{i}", tuple(point + [10, 10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def get_pose(self, corners, ids, marker_id, is_left):
        if marker_id in ids.flatten():
            idx = np.where(ids.flatten() == marker_id)[0][0]
            mtx = self.rect_mtx_left if is_left else self.rect_mtx_right
            marker_corners = np.array([[-self.marker_size/2, self.marker_size/2, 0], [self.marker_size/2, self.marker_size/2, 0], [self.marker_size/2, -self.marker_size/2, 0], [-self.marker_size/2, -self.marker_size/2, 0]], dtype=np.float32)
            success, _, tvec = cv2.solvePnP(marker_corners, corners[idx][0], mtx, np.zeros(5))
            return tvec.flatten() if success else None
        return None

if __name__ == "__main__":
    ObjectTracker().run()
