import cv2 as cv
import cv2.aruco as aruco
import glob
import os
import numpy as np

class Calibrator:
    def __init__(self, image_dir, charuco_board_size=(12, 9), marker_type=aruco.DICT_5X5_250, square_size=0.06, marker_size=0.045, legacy_pattern=False):
        self.image_dir = image_dir
        self.aruco_dict = aruco.getPredefinedDictionary(marker_type)
        self.charuco_board = aruco.CharucoBoard(
            charuco_board_size,
            square_size,
            marker_size,
            self.aruco_dict
        )
        self.charuco_board.setLegacyPattern(legacy_pattern)
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.charuco_corners = []  # Store charuco corners
        self.charuco_ids = []     # Store charuco IDs
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.rms_error = None

    def detect_images(self):
        images = 0
        markers_found = 0
        no_arucos = 0
        insufficient_corners = 0
        
        for fname in glob.glob(os.path.join(self.image_dir, '*.jpeg')):
            images += 1
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict)
            if len(corners) > 0:
                markers_found += 1
                _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, self.charuco_board)
                if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) >= 30:
                    self.charuco_corners.append(charuco_corners)
                    self.charuco_ids.append(charuco_ids)
                elif charuco_corners is not None and charuco_ids is not None:
                    insufficient_corners += 1
                else:
                    no_arucos += 1

        print(f"Total images processed: {images}")
        print(f"Images with detected markers: {markers_found}")
        print(f"Images with valid ChArUco corners (>=30): {len(self.charuco_corners)}")
        print(f"Images with insufficient corners (<30): {insufficient_corners}")
        print(f"Images without valid ChArUco corners: {no_arucos}")

    def calibrate_camera(self, file_name=None):
        if len(self.charuco_corners) == 0:
            print("No valid ChArUco corners found for calibration!")
            return
        
        print(f"Starting calibration with {len(self.charuco_corners)} images...")
        
        # Get image size from the first image
        first_image = glob.glob(os.path.join(self.image_dir, '*.jpeg'))[0]
        img = cv.imread(first_image)
        image_size = (img.shape[1], img.shape[0])  # (width, height)
        print(f"Using image size: {image_size}")
        
        # Calibrate camera using ChArUco corners
        ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
            self.charuco_corners,
            self.charuco_ids,
            self.charuco_board,
            image_size,
            None,
            None
        )
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.rms_error = ret
        
        print(f"Calibration successful!")
        print(f"Calibration RMS error: {ret:.6f}")
        print("Camera matrix:")
        print(mtx)
        print("Distortion coefficients:")
        print(dist)
        
        return ret, mtx, dist, rvecs, tvecs
    
    def save_calibration(self, output_file=None):
        np.savez(output_file, 
                 mtx=self.mtx, 
                 dist=self.dist, 
                 rms_error=self.rms_error,
                 rvecs=self.rvecs,
                 tvecs=self.tvecs)
        print(f"Calibration data saved to {output_file}")

class StereoCalibrator:
    def __init__(self, left_calib_data, right_calib_data, image_size, charuco_board_size=(12, 9), marker_type=aruco.DICT_5X5_250, square_size=0.06, marker_size=0.045):
        self.left_mtx = left_calib_data['mtx']
        self.left_dist = left_calib_data['dist']
        self.right_mtx = right_calib_data['mtx']
        self.right_dist = right_calib_data['dist']
        self.image_size = image_size
        self.aruco_dict = aruco.getPredefinedDictionary(marker_type)
        self.charuco_board = aruco.CharucoBoard(
            charuco_board_size,
            square_size,
            marker_size,
            self.aruco_dict
        )
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []

    def detect_stereo_corners(self, left_image_dir, right_image_dir):
        left_images = sorted(glob.glob(os.path.join(left_image_dir, '*.jpeg')))
        right_images = sorted(glob.glob(os.path.join(right_image_dir, '*.jpeg')))
        for left_img_path, right_img_path in zip(left_images, right_images):
            img_left = cv.imread(left_img_path)
            img_right = cv.imread(right_img_path)
            gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
            gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

            corners_left, ids_left, _ = aruco.detectMarkers(gray_left, self.aruco_dict)
            corners_right, ids_right, _ = aruco.detectMarkers(gray_right, self.aruco_dict)

            if len(corners_left) > 0 and len(corners_right) > 0:
                _, charuco_corners_left, charuco_ids_left = aruco.interpolateCornersCharuco(corners_left, ids_left, gray_left, self.charuco_board)
                _, charuco_corners_right, charuco_ids_right = aruco.interpolateCornersCharuco(corners_right, ids_right, gray_right, self.charuco_board)

                if (charuco_corners_left is not None and charuco_ids_left is not None and
                    charuco_corners_right is not None and charuco_ids_right is not None and
                    len(charuco_corners_left) >= 30 and len(charuco_corners_right) >= 30):

                    # Find common IDs
                    ids_left_set = set(charuco_ids_left.flatten())
                    ids_right_set = set(charuco_ids_right.flatten())
                    common_ids = np.array(list(ids_left_set & ids_right_set))

                    if len(common_ids) >= 10:
                        objp = []
                        imgp_left = []
                        imgp_right = []
                        
                        # Get all chessboard corners first
                        chessboard_corners = self.charuco_board.getChessboardCorners()
                        
                        for cid in common_ids:
                            idx_left = np.where(charuco_ids_left == cid)[0][0]
                            idx_right = np.where(charuco_ids_right == cid)[0][0]
                            objp.append(chessboard_corners[cid])  # Index the returned array
                            imgp_left.append(charuco_corners_left[idx_left][0])
                            imgp_right.append(charuco_corners_right[idx_right][0])
                        self.objpoints.append(np.array(objp, dtype=np.float32))
                        self.imgpoints_left.append(np.array(imgp_left, dtype=np.float32))
                        self.imgpoints_right.append(np.array(imgp_right, dtype=np.float32))

    def stereo_calibrate(self):
        flags = cv.CALIB_FIX_INTRINSIC
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        retval, left_mtx, left_dist, right_mtx, right_dist, R, T, E, F = cv.stereoCalibrate(
            self.objpoints,
            self.imgpoints_left,
            self.imgpoints_right,
            self.left_mtx,
            self.left_dist,
            self.right_mtx,
            self.right_dist,
            self.image_size,
            criteria=criteria,
            flags=flags
        )
        self.R = R
        self.T = T
        self.E = E
        self.F = F
        self.stereo_error = retval
        print(f"Stereo calibration RMS error: {retval:.6f}")
        print("Rotation matrix (R):")
        print(R)
        print("Translation vector (T):")
        print(T)
        return retval, R, T, E, F

    def save_stereo_calibration(self, output_file):
        np.savez(output_file,
                 left_mtx=self.left_mtx,
                 left_dist=self.left_dist,
                 right_mtx=self.right_mtx,
                 right_dist=self.right_dist,
                 R=self.R,
                 T=self.T,
                 E=self.E,
                 F=self.F,
                 stereo_error=self.stereo_error)
        print(f"Stereo calibration data saved to {output_file}")
