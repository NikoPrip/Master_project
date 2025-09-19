import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import cv2
import cv2.aruco as aruco

class VideoUndistorter:
    def __init__(self, calib_file='calib_data.npz', save_video=False, marker_type=aruco.DICT_5X5_250):
        self.calib = np.load(calib_file)
        self.mtx = self.calib['mtx']
        self.dist = self.calib['dist']
        self.save_video = save_video
        self.marker_type = marker_type
        Gst.init(None)
        self.pipeline_90 = self._create_pipeline_90()
        self.pipeline_91 = self._create_pipeline_91()
        self.appsink_90 = self.pipeline_90.get_by_name("sink")
        self.appsink_91 = self.pipeline_91.get_by_name("sink")

        # Choose a dictionary, e.g., 5x5_250
        self.aruco_dict = aruco.getPredefinedDictionary(self.marker_type)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_center = 0

    def _create_pipeline_90(self):
        if self.save_video:
            pipeline_str = (
                "udpsrc port=58004 buffer-size=212992 do-timestamp=true ! "
                "application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! "
                "rtph264depay ! h264parse ! tee name=t "
                "t. ! queue ! qtmux fragment-duration=1000 ! filesink location=output.mp4 "
                "t. ! queue ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false"
            )
        else:
            pipeline_str = (
                "udpsrc port=58004 buffer-size=212992 do-timestamp=true ! "
                "application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! "
                "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false"
            )
        return Gst.parse_launch(pipeline_str)
    
    def _create_pipeline_91(self):
        if self.save_video:
            pipeline_str = (
                "udpsrc port=58006 buffer-size=212992 do-timestamp=true ! "
                "application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! "
                "rtph264depay ! h264parse ! tee name=t "
                "t. ! queue ! qtmux fragment-duration=1000 ! filesink location=output.mp4 "
                "t. ! queue ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false"
            )
        else:
            pipeline_str = (
                "udpsrc port=58006 buffer-size=212992 do-timestamp=true ! "
                "application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! "
                "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false"
            )
        return Gst.parse_launch(pipeline_str)

    def run(self):
        self.pipeline_90.set_state(Gst.State.PLAYING)
        self.pipeline_91.set_state(Gst.State.PLAYING)
        try:
            while True:
                sample_90 = self.appsink_90.emit("try-pull-sample", 100000000)
                sample_91 = self.appsink_91.emit("try-pull-sample", 100000000)
                if sample_90:
                    buf_90 = sample_90.get_buffer()
                    caps_90 = sample_90.get_caps()
                    buf_91 = sample_91.get_buffer()
                    caps_91 = sample_91.get_caps()
                    width_90 = caps_90.get_structure(0).get_value('width')
                    height_90 = caps_90.get_structure(0).get_value('height')
                    width = caps_91.get_structure(0).get_value('width')
                    height = caps_91.get_structure(0).get_value('height')
                    arr_90 = np.ndarray(
                        (height_90, width_90, 3),
                        buffer=buf_90.extract_dup(0, buf_90.get_size()),
                        dtype=np.uint8
                    )
                    arr_91 = np.ndarray(
                        (height, width, 3),
                        buffer=buf_91.extract_dup(0, buf_91.get_size()),
                        dtype=np.uint8
                    )
                    undistorted_90 = cv2.undistort(arr_90, self.mtx, self.dist, None, self.mtx)
                    undistorted_91 = cv2.undistort(arr_91, self.mtx, self.dist, None, self.mtx)

                    # --- ArUco marker detection ---
                    marker_size = 16.8  # cm
                    gray_90 = cv2.cvtColor(undistorted_90, cv2.COLOR_BGR2GRAY)
                    corners_90, ids_90, rejected_90 = aruco.detectMarkers(
                        gray_90, self.aruco_dict, parameters=self.aruco_params
                    )
                    gray_91 = cv2.cvtColor(undistorted_91, cv2.COLOR_BGR2GRAY)
                    corners_91, ids_91, rejected_91 = aruco.detectMarkers(
                        gray_91, self.aruco_dict, parameters=self.aruco_params
                    )
                    if ids_90 is not None:
                        aruco.drawDetectedMarkers(undistorted_90, corners_90, ids_90)
                        for marker_corners, marker_id in zip(corners_90, ids_90):
                            # marker_corners shape: (1, 4, 2)
                            self.aruco_center = marker_corners[0].mean(axis=0).astype(int)
                            print("Corners:", corners_90)
                            print(f"Marker ID {marker_id[0]} center: ({self.aruco_center[0]}, {self.aruco_center[1]})")
                            # Optionally, draw the center on the image
                            cv2.circle(undistorted_90, tuple(self.aruco_center), 5, (0, 0, 255), -1)
                            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                                marker_corners, marker_size, self.mtx, self.dist
                            )
                            print("Distance (tvec):", tvec)
                    if ids_91 is not None:
                        aruco.drawDetectedMarkers(undistorted_91, corners_91, ids_91)
                        for marker_corners, marker_id in zip(corners_91, ids_91):
                            # marker_corners shape: (1, 4, 2)
                            self.aruco_center = marker_corners[0].mean(axis=0).astype(int)
                            print("Corners:", corners_91)
                            print(f"Marker ID {marker_id[0]} center: ({self.aruco_center[0]}, {self.aruco_center[1]})")
                            # Optionally, draw the center on the image
                            cv2.circle(undistorted_91, tuple(self.aruco_center), 5, (0, 0, 255), -1)
                            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                                marker_corners, marker_size, self.mtx, self.dist
                            )
                            print("Distance (tvec):", tvec)
                    cv2.imshow("Video 90", undistorted_90)
                    cv2.imshow("Video 91", undistorted_91)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    continue
        finally:
            self.pipeline_90.set_state(Gst.State.NULL)
            self.pipeline_91.set_state(Gst.State.NULL)
