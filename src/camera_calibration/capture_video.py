import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import cv2
import cv2.aruco as aruco
import time

class VideoUndistorter:
    def __init__(self, camera_configs, save_video=False, marker_type=aruco.DICT_5X5_250):
        """
        camera_configs: list of dicts with 'calib_file' and 'port' keys
        Example: [
            {'calib_file': 'calib_data_90.npz', 'port': 58004},
            {'calib_file': 'calib_data_91.npz', 'port': 58006}
        ]
        """
        self.camera_configs = camera_configs
        self.save_video = save_video
        self.marker_type = marker_type
        
        # Load calibration data for each camera
        self.cameras = []
        for i, config in enumerate(camera_configs):
            calib_data = np.load(config['calib_file'])
            camera_info = {
                'id': i,
                'port': config['port'],
                'mtx': calib_data['mtx'],
                'dist': calib_data['dist'],
                'aruco_center': None
            }
            self.cameras.append(camera_info)
        
        Gst.init(None)
        
        # Create pipelines and appsinks for each camera
        self.pipelines = []
        self.appsinks = []
        for camera in self.cameras:
            pipeline = self._create_pipeline(camera['port'])
            appsink = pipeline.get_by_name("sink")
            self.pipelines.append(pipeline)
            self.appsinks.append(appsink)

        # ArUco setup
        self.aruco_dict = aruco.getPredefinedDictionary(self.marker_type)
        self.aruco_params = aruco.DetectorParameters()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

    def _create_pipeline(self, port):
        # Reduced buffer size and added more efficient pipeline
        if self.save_video:
            pipeline_str = (
                f"udpsrc port={port} buffer-size=65536 do-timestamp=true ! "
                "application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! "
                "rtph264depay ! h264parse ! tee name=t "
                f"t. ! queue max-size-buffers=10 ! qtmux fragment-duration=1000 ! filesink location=output_{port}.mp4 "
                "t. ! queue max-size-buffers=2 ! avdec_h264 ! videoconvert ! "
                "video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
            )
        else:
            pipeline_str = (
                f"udpsrc port={port} buffer-size=65536 do-timestamp=true ! "
                "application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! "
                "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
                "video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
            )
        return Gst.parse_launch(pipeline_str)

    def _process_frame(self, camera, frame):
        """Process a single frame for ArUco detection"""
        marker_size = 4.5  # cm
        
        # Undistort using stored calibration data
        undistorted = cv2.undistort(frame, camera['mtx'], camera['dist'], None, camera['mtx'])
        
        # ArUco marker detection on smaller image if possible
        # Resize for detection if frame is large
        detection_frame = undistorted
        scale_factor = 1.0
        if undistorted.shape[0] > 480:  # If height > 480, resize for detection
            scale_factor = 480.0 / undistorted.shape[0]
            new_width = int(undistorted.shape[1] * scale_factor)
            detection_frame = cv2.resize(undistorted, (new_width, 480))
        
        gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
            # Scale corners back to original size if we resized
            if scale_factor != 1.0:
                corners = [corner / scale_factor for corner in corners]
            
            aruco.drawDetectedMarkers(undistorted, corners, ids)
            for marker_corners, marker_id in zip(corners, ids):
                # Calculate marker center
                camera['aruco_center'] = marker_corners[0].mean(axis=0).astype(int)
                
                # Draw center on image
                cv2.circle(undistorted, tuple(camera['aruco_center']), 5, (0, 0, 255), -1)
                
                # Only estimate pose for the first marker to save computation
                if marker_id[0] == ids[0][0]:  # Process only first detected marker
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                        marker_corners, marker_size, camera['mtx'], camera['dist']
                    )
                print(f"Camera {camera['id']} Marker ID: {marker_id[0]} Position (tvec): {tvec[0][0]}")
        
        return undistorted

    def run(self):
        # Start all pipelines
        for pipeline in self.pipelines:
            pipeline.set_state(Gst.State.PLAYING)
        
        try:
            while True:
                frames_processed = 0
                
                # Process each camera
                for i, (camera, appsink) in enumerate(zip(self.cameras, self.appsinks)):
                    # Reduced timeout for faster frame dropping
                    sample = appsink.emit("try-pull-sample", 10000000)  # 10ms timeout
                    if sample:
                        buf = sample.get_buffer()
                        caps = sample.get_caps()
                        width = caps.get_structure(0).get_value('width')
                        height = caps.get_structure(0).get_value('height')
                        
                        # Create numpy array from buffer
                        arr = np.ndarray(
                            (height, width, 3),
                            buffer=buf.extract_dup(0, buf.get_size()),
                            dtype=np.uint8
                        )
                        
                        # Process frame
                        processed_frame = self._process_frame(camera, arr)
                        
                        # Display frame
                        cv2.imshow(f"Camera {camera['id']} (Port {camera['port']})", processed_frame)
                        frames_processed += 1
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                        
        finally:
            # Stop all pipelines
            for pipeline in self.pipelines:
                pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()

