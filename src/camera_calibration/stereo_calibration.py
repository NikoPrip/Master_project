import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import os
import glob

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
        
        # ArUco setup (optional - can be removed if not needed)
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
        self.rms_error = None

    def collect_stereo_image_pairs(self, images_dir_90, images_dir_91, print_images=False):
        """Collect synchronized image pairs from both cameras"""
        images_90 = sorted(glob.glob(os.path.join(images_dir_90, '*.jpeg')))
        images_91 = sorted(glob.glob(os.path.join(images_dir_91, '*.jpeg')))
        
        if len(images_90) == 0:
            images_90 = sorted(glob.glob(os.path.join(images_dir_90, '*.jpg')))
        if len(images_91) == 0:
            images_91 = sorted(glob.glob(os.path.join(images_dir_91, '*.jpg')))
            
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
            
        # Stereo calibration with fixed intrinsics
        flags = cv.CALIB_FIX_INTRINSIC
        
        ret, mtx1, dist1, mtx2, dist2, self.R, self.T, self.E, self.F = cv.stereoCalibrate(
            self.objpoints, self.imgpoints_90, self.imgpoints_91,
            self.mtx_90, self.dist_90, self.mtx_91, self.dist_91,
            image_size, 
            criteria=self.criteria, 
            flags=flags
        )
        
        # Store the RMS error
        self.rms_error = ret
        
        if print_vals:
            print(f"Stereo calibration RMS error: {ret}")
            print(f"Rotation matrix R:\n{self.R}")
            print(f"Translation vector T:\n{self.T}")
            print(f"Distance between cameras: {np.linalg.norm(self.T):.2f} units")
            
            # Convert rotation matrix to rotation vector
            rvec, _ = cv.Rodrigues(self.R)
            print(f"Rotation vector: {rvec.flatten()}")
            
            # Calculate rotation angle magnitude
            angle_magnitude = np.linalg.norm(rvec) * 180 / np.pi
            print(f"Rotation angle magnitude: {angle_magnitude:.2f} degrees")
        
        # Save stereo calibration results
        np.savez(os.path.join(os.getcwd(), 'stereo_calib_data.npz'), 
                 R=self.R, T=self.T, E=self.E, F=self.F,
                 mtx_90=self.mtx_90, dist_90=self.dist_90,
                 mtx_91=self.mtx_91, dist_91=self.dist_91,
                 rms_error=self.rms_error)
        
        return True

    def compute_stereo_reprojection_error(self):
        """Compute reprojection error for stereo calibration"""
        if self.R is None or self.T is None:
            print("Stereo calibration not performed yet!")
            return
            
        total_error = 0
        total_points = 0
        
        for i in range(len(self.objpoints)):
            # Project 3D points to camera 90 (reference camera)
            imgpoints_90_proj, _ = cv.projectPoints(
                self.objpoints[i], 
                np.zeros((3,1)), np.zeros((3,1)),  # Identity pose for reference camera
                self.mtx_90, self.dist_90
            )
            
            # Project 3D points to camera 91 (using stereo transformation)
            imgpoints_91_proj, _ = cv.projectPoints(
                self.objpoints[i], 
                cv.Rodrigues(self.R)[0], self.T,  # Stereo transformation
                self.mtx_91, self.dist_91
            )
            
            # Calculate reprojection errors
            error_90 = cv.norm(self.imgpoints_90[i], imgpoints_90_proj, cv.NORM_L2) / len(imgpoints_90_proj)
            error_91 = cv.norm(self.imgpoints_91[i], imgpoints_91_proj, cv.NORM_L2) / len(imgpoints_91_proj)
            
            total_error += error_90 + error_91
            total_points += 2
        
        mean_error = total_error / total_points
        print(f"Stereo reprojection error: {mean_error}")
        
        # Interpretation
        if mean_error < 1.0:
            print("✓ Excellent stereo calibration quality")
        elif mean_error < 2.0:
            print("✓ Good stereo calibration quality")
        elif mean_error < 5.0:
            print("⚠ Acceptable stereo calibration quality")
        else:
            print("✗ Poor stereo calibration quality")
            
        return mean_error
    