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
        
        # Calculate rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left_calib['mtx'], left_calib['dist'], right_calib['mtx'], right_calib['dist'],
            (1280, 800), stereo_calib['R'], stereo_calib['T'])
        
        # Create rectification maps
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(left_calib['mtx'], left_calib['dist'], R1, P1, (1280, 800), cv2.CV_16SC2)
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(right_calib['mtx'], right_calib['dist'], R2, P2, (1280, 800), cv2.CV_16SC2)
        
        self.rect_mtx_left, self.rect_mtx_right = P1[:3, :3], P2[:3, :3]
        self.baseline = np.linalg.norm(stereo_calib['T'])
        
        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_size = 0.16
        self.object_points = np.array([[-0.105, 0.105, 0], [0.70, 0.105, 0]], dtype=np.float32)

    def run(self):
        Gst.init(None)
        
        # Create pipelines
        left_pipeline = Gst.parse_launch("udpsrc port=58006 ! application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false max-buffers=2 drop=true")
        right_pipeline = Gst.parse_launch("udpsrc port=58004 ! application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false max-buffers=2 drop=true")
        
        left_sink, right_sink = left_pipeline.get_by_name("sink"), right_pipeline.get_by_name("sink")
        left_pipeline.set_state(Gst.State.PLAYING)
        right_pipeline.set_state(Gst.State.PLAYING)
        time.sleep(2)
        
        try:
            while True:
                left_frame = self.get_frame(left_sink, self.map1_left, self.map2_left)
                right_frame = self.get_frame(right_sink, self.map1_right, self.map2_right)
                
                if left_frame is not None and right_frame is not None:
                    self.process_frames(left_frame, right_frame)
                    cv2.imshow("Stereo ArUco Tracking", np.hstack((cv2.resize(left_frame, (640, 480)), cv2.resize(right_frame, (640, 480)))))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            left_pipeline.set_state(Gst.State.NULL)
            right_pipeline.set_state(Gst.State.NULL)

    def get_frame(self, sink, map1, map2):
        sample = sink.emit("try-pull-sample", 10000000)
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            h, w = caps.get_value('height'), caps.get_value('width')
            frame = np.ndarray((h, w, 3), buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8).copy()
            return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    def process_frames(self, left_frame, right_frame):
        left_corners, left_ids = cv2.aruco.detectMarkers(left_frame, self.aruco_dict, parameters=self.aruco_params)[:2]
        right_corners, right_ids = cv2.aruco.detectMarkers(right_frame, self.aruco_dict, parameters=self.aruco_params)[:2]
        
        self.draw_markers(left_frame, left_corners, left_ids)
        self.draw_markers(right_frame, right_corners, right_ids)
        
        if left_ids is not None and right_ids is not None:
            for marker_id in np.intersect1d(left_ids.flatten(), right_ids.flatten()):
                self.calculate_distances(left_corners, left_ids, right_corners, right_ids, marker_id)

    def calculate_distances(self, left_corners, left_ids, right_corners, right_ids, marker_id):
        left_idx = np.where(left_ids.flatten() == marker_id)[0][0]
        right_idx = np.where(right_ids.flatten() == marker_id)[0][0]
        
        left_marker = left_corners[left_idx][0]
        right_marker = right_corners[right_idx][0]
        
        # Triangulate marker center
        left_center = np.mean(left_marker, axis=0)
        right_center = np.mean(right_marker, axis=0)
        P1 = np.hstack([self.rect_mtx_left, np.zeros((3, 1))])
        P2 = np.hstack([self.rect_mtx_right, np.array([[-self.baseline * self.rect_mtx_right[0, 0]], [0], [0]])])
        point_4d = cv2.triangulatePoints(P1, P2, left_center, right_center)
        marker_center_3d = (point_4d[:3] / point_4d[3]).flatten()
        
        # Get object points
        marker_3d = np.array([[-self.marker_size/2, self.marker_size/2, 0], [self.marker_size/2, self.marker_size/2, 0], 
                             [self.marker_size/2, -self.marker_size/2, 0], [-self.marker_size/2, -self.marker_size/2, 0]], dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(marker_3d, left_marker, self.rect_mtx_left, np.zeros(5))
        
        if success:
            R, _ = cv2.Rodrigues(rvec)
            object_points_3d = [(R @ point.reshape(3, 1) + tvec.reshape(3, 1)).flatten() for point in self.object_points]
            
            # Print distances
            print(f"\nMarker {marker_id} distances from left camera:")
            print(f"  Marker center: {np.linalg.norm(marker_center_3d):.3f}m")
            for i, obj_point in enumerate(object_points_3d):
                print(f"  Object point {i}: {np.linalg.norm(obj_point):.3f}m")

    def draw_markers(self, frame, corners, ids):
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            mtx = self.rect_mtx_left if frame.shape[1] == 1280 else self.rect_mtx_right
            marker_3d = np.array([[-self.marker_size/2, self.marker_size/2, 0], [self.marker_size/2, self.marker_size/2, 0], 
                                 [self.marker_size/2, -self.marker_size/2, 0], [-self.marker_size/2, -self.marker_size/2, 0]], dtype=np.float32)
            
            for corner, marker_id in zip(corners, ids.flatten()):
                center = np.mean(corner[0], axis=0).astype(int)
                cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{marker_id}", tuple(center + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                success, rvec, tvec = cv2.solvePnP(marker_3d, corner[0], mtx, np.zeros(5))
                if success:
                    cv2.drawFrameAxes(frame, mtx, np.zeros(5), rvec, tvec, self.marker_size * 0.5)
                    R, _ = cv2.Rodrigues(rvec)
                    object_points_3d = [(R @ point.reshape(3, 1) + tvec.reshape(3, 1)).flatten() for point in self.object_points]
                    image_points, _ = cv2.projectPoints(np.array(object_points_3d), np.zeros(3), np.zeros(3), mtx, np.zeros(5))
                    
                    for i, point in enumerate(image_points.reshape(-1, 2).astype(int)):
                        cv2.circle(frame, tuple(point), 8, (0, 255, 255), -1)
                        cv2.putText(frame, f"P{i}", tuple(point + [10, 10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

if __name__ == "__main__":
    ObjectTracker().run()
