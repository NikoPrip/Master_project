import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import os
import glob
import cv2
from gi.repository import Gst
import gi
gi.require_version('Gst', '1.0')

class StereoCalibrator:
    def __init__(self, calib_90_file, calib_91_file, chessboard_size=(8, 11), marker_type=aruco.DICT_5X5_250):
        # Load individual camera calibrations
        self.calib_90 = np.load(calib_90_file)
        self.calib_91 = np.load(calib_91_file)
        self.mtx_90 = self.calib_90['mtx']
        self.dist_90 = self.calib_90['dist']
        self.mtx_91 = self.calib_91['mtx']
        self.dist_91 = self.calib_91['dist']
        
        # Checkerboard setup
        self.chessboard_size = chessboard_size
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # ArUco setup
        self.aruco_dict = aruco.getPredefinedDictionary(marker_type)
        self.aruco_params = aruco.DetectorParameters()
        
        # Storage for stereo calibration
        self.objpoints = []  # 3D points in real world space
        self.imgpoints_90 = []  # 2D points in camera 90
        self.imgpoints_91 = []  # 2D points in camera 91
        
        # Stereo calibration results
        self.R = None  # Rotation matrix between cameras
        self.T = None  # Translation vector between cameras
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix

    def collect_stereo_image_pairs(self, images_dir_90, images_dir_91, print_images=False):
        """Collect synchronized image pairs from both cameras"""
        images_90 = sorted(glob.glob(os.path.join(images_dir_90, '*.jpeg')))
        images_91 = sorted(glob.glob(os.path.join(images_dir_91, '*.jpeg')))
        
        if len(images_90) == 0:
            images_90 = sorted(glob.glob(os.path.join(images_dir_90, '*.jpg')))
        if len(images_91) == 0:
            images_91 = sorted(glob.glob(os.path.join(images_dir_91, '*.jpg')))  # Fixed: use images_dir_91 not images_91
            
        print(f"Found {len(images_90)} images in camera 90 dir")
        print(f"Found {len(images_91)} images in camera 91 dir")
        
        # Ensure we have the same number of images
        min_images = min(len(images_90), len(images_91))
        successful_pairs = 0
        
        for i in range(min_images):
            img_90 = cv.imread(images_90[i])
            img_91 = cv.imread(images_91[i])
            
            if img_90 is None or img_91 is None:
                continue
                
            gray_90 = cv.cvtColor(img_90, cv.COLOR_BGR2GRAY)
            gray_91 = cv.cvtColor(img_91, cv.COLOR_BGR2GRAY)
            
            # Find chessboard corners in both images
            ret_90, corners_90 = cv.findChessboardCorners(gray_90, self.chessboard_size, None)
            ret_91, corners_91 = cv.findChessboardCorners(gray_91, self.chessboard_size, None)
            
            if ret_90 and ret_91:
                # Refine corners
                corners_90 = cv.cornerSubPix(gray_90, corners_90, (11, 11), (-1, -1), self.criteria)
                corners_91 = cv.cornerSubPix(gray_91, corners_91, (11, 11), (-1, -1), self.criteria)
                
                # Store the points
                self.objpoints.append(self.objp)
                self.imgpoints_90.append(corners_90)
                self.imgpoints_91.append(corners_91)
                successful_pairs += 1
                
                if print_images:
                    cv.drawChessboardCorners(img_90, self.chessboard_size, corners_90, ret_90)
                    cv.drawChessboardCorners(img_91, self.chessboard_size, corners_91, ret_91)
                    
                    # Also detect and draw ArUco markers
                    aruco_corners_90, aruco_ids_90, _ = aruco.detectMarkers(gray_90, self.aruco_dict, parameters=self.aruco_params)
                    aruco_corners_91, aruco_ids_91, _ = aruco.detectMarkers(gray_91, self.aruco_dict, parameters=self.aruco_params)
                    
                    if aruco_ids_90 is not None:
                        aruco.drawDetectedMarkers(img_90, aruco_corners_90, aruco_ids_90)
                    if aruco_ids_91 is not None:
                        aruco.drawDetectedMarkers(img_91, aruco_corners_91, aruco_ids_91)
                    
                    # Show images side by side
                    combined = np.hstack((img_90, img_91))
                    cv.imshow('Stereo Calibration Images', cv.resize(combined, (1200, 400)))
                    cv.waitKey(500)
                    
                print(f"Processed pair {i+1}/{min_images} - Success: {ret_90 and ret_91}")
            else:
                print(f"Processed pair {i+1}/{min_images} - Failed to find chessboard in both images")
        
        print(f"Successfully collected {successful_pairs} stereo image pairs")
        cv.destroyAllWindows()
        return successful_pairs > 0

    def calibrate_stereo(self, image_size=(768, 480), print_vals=False):
        """Perform stereo calibration"""
        if len(self.objpoints) == 0:
            print("No valid image pairs found for stereo calibration!")
            return False
            
        # Stereo calibration
        ret, mtx1, dist1, mtx2, dist2, self.R, self.T, self.E, self.F = cv.stereoCalibrate(
            self.objpoints, self.imgpoints_90, self.imgpoints_91,
            self.mtx_90, self.dist_90, self.mtx_91, self.dist_91,
            image_size, criteria=self.criteria, flags=cv.CALIB_FIX_INTRINSIC
        )
        
        if print_vals:
            print(f"Stereo calibration RMS error: {ret}")
            print(f"Rotation matrix R:\n{self.R}")
            print(f"Translation vector T:\n{self.T}")
            print(f"Distance between cameras: {np.linalg.norm(self.T):.2f} units")
            
            # Manual rotation matrix to Euler angles conversion
            # Using OpenCV's Rodrigues instead of scipy
            rvec, _ = cv.Rodrigues(self.R)
            print(f"Rotation vector: {rvec.flatten()}")
            
            # Simple approximation for small angles
            angle_magnitude = np.linalg.norm(rvec) * 180 / np.pi
            print(f"Rotation angle magnitude: {angle_magnitude:.2f} degrees")
        
        # Save stereo calibration results
        np.savez(os.path.join(os.getcwd(), 'stereo_calib_data.npz'), 
                 R=self.R, T=self.T, E=self.E, F=self.F,
                 mtx_90=self.mtx_90, dist_90=self.dist_90,
                 mtx_91=self.mtx_91, dist_91=self.dist_91)
        
        return True

    def compute_stereo_reprojection_error(self):
        """Compute reprojection error for stereo calibration"""
        if self.R is None or self.T is None:
            print("Stereo calibration not performed yet!")
            return None
            
        total_error = 0
        total_points = 0
        
        for i in range(len(self.objpoints)):
            # Project points using stereo calibration
            imgpoints_90_proj, _ = cv.projectPoints(self.objpoints[i], np.zeros((3,1)), np.zeros((3,1)), self.mtx_90, self.dist_90)
            imgpoints_91_proj, _ = cv.projectPoints(self.objpoints[i], self.R, self.T, self.mtx_91, self.dist_91)
            
            error_90 = cv.norm(self.imgpoints_90[i], imgpoints_90_proj, cv.NORM_L2) / len(imgpoints_90_proj)
            error_91 = cv.norm(self.imgpoints_91[i], imgpoints_91_proj, cv.NORM_L2) / len(imgpoints_91_proj)
            
            total_error += error_90 + error_91
            total_points += 2
            
        mean_error = total_error / total_points
        print(f"Stereo reprojection error: {mean_error}")
        return mean_error

class StereoPairCapture:
    def __init__(self):
        Gst.init(None)
        self.pipeline_90 = self._create_pipeline(58004)
        self.pipeline_91 = self._create_pipeline(58006)
        self.appsink_90 = self.pipeline_90.get_by_name("sink")
        self.appsink_91 = self.pipeline_91.get_by_name("sink")
        self.pair_count = 0
        
    def _create_pipeline(self, port):
        pipeline_str = (
            f"udpsrc port={port} buffer-size=212992 do-timestamp=true ! "
            "application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false"
        )
        return Gst.parse_launch(pipeline_str)
    
    def capture_synchronized_pairs(self, output_dir_90, output_dir_91):
        """Capture synchronized image pairs for stereo calibration"""
        os.makedirs(output_dir_90, exist_ok=True)
        os.makedirs(output_dir_91, exist_ok=True)
        
        self.pipeline_90.set_state(Gst.State.PLAYING)
        self.pipeline_91.set_state(Gst.State.PLAYING)
        
        print("Press SPACE to capture synchronized pair, 'q' to quit")
        
        try:
            while True:
                sample_90 = self.appsink_90.emit("try-pull-sample", 100000000)
                sample_91 = self.appsink_91.emit("try-pull-sample", 100000000)
                
                if sample_90 and sample_91:
                    # Get frames from both cameras
                    buf_90 = sample_90.get_buffer()
                    caps_90 = sample_90.get_caps()
                    buf_91 = sample_91.get_buffer()
                    caps_91 = sample_91.get_caps()
                    
                    width_90 = caps_90.get_structure(0).get_value('width')
                    height_90 = caps_90.get_structure(0).get_value('height')
                    width_91 = caps_91.get_structure(0).get_value('width')
                    height_91 = caps_91.get_structure(0).get_value('height')
                    
                    arr_90 = np.ndarray(
                        (height_90, width_90, 3),
                        buffer=buf_90.extract_dup(0, buf_90.get_size()),
                        dtype=np.uint8
                    )
                    arr_91 = np.ndarray(
                        (height_91, width_91, 3),
                        buffer=buf_91.extract_dup(0, buf_91.get_size()),
                        dtype=np.uint8
                    )
                    
                    # Show both feeds side by side
                    combined = np.hstack((arr_90, arr_91))
                    cv2.imshow('Stereo Pair Capture - Press SPACE to capture', 
                              cv2.resize(combined, (1200, 400)))
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # Space bar to capture
                        # Save synchronized pair
                        filename = f"stereo_pair_{self.pair_count:04d}.jpg"
                        cv2.imwrite(os.path.join(output_dir_90, filename), arr_90)
                        cv2.imwrite(os.path.join(output_dir_91, filename), arr_91)
                        self.pair_count += 1
                        print(f"Captured stereo pair {self.pair_count}")
                        
                    elif key == ord('q'):
                        break
                        
        finally:
            self.pipeline_90.set_state(Gst.State.NULL)
            self.pipeline_91.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()
            print(f"Captured {self.pair_count} stereo pairs")