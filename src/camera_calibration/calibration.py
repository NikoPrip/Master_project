import numpy as np
import cv2 as cv
import glob
import os

class CameraCalibrator:
    def __init__(self, images_dir='Images', chessboard_size=(8, 11), crop_percent=0.125):
        self.images_dir = images_dir
        self.chessboard_size = chessboard_size
        self.crop_percent = crop_percent
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objpoints = []
        self.imgpoints = []
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.rms_error = None

    def collect_image_points(self, print_images=False):
        images = glob.glob(os.path.join(self.images_dir, '*.jpeg'))
        print(f"Found {len(images)} images in {self.images_dir}.")
        im_count = 0
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, self.chessboard_size, None)
            if ret:
                im_count += 1
                self.objpoints.append(self.objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)
                cv.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                if print_images:
                    cv.imshow('img', img)
                    cv.waitKey(50)
        print(f"Successfully processed {im_count} images with detected chessboards.")
        cv.destroyAllWindows()

    def calibrate(self, print_vals=False):
        if len(self.objpoints) == 0:
            print("ERROR: No chessboard corners found! Run collect_image_points() first.")
            return False
            
        # Use fixed image shape (width=768, height=480)
        image_shape = (768, 480)
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(
            self.objpoints, self.imgpoints, image_shape, None, None)
        
        # Store the RMS error
        self.rms_error = ret
        
        if print_vals:
            print(f"RMS error: {self.rms_error:.6f}")
            print("Camera matrix:")
            print(self.mtx)
            print("Distortion coefficients:")
            print(self.dist)
            
            # Print calibration quality assessment
            if self.rms_error < 0.5:
                print("✓ Excellent calibration quality")
            elif self.rms_error < 1.0:
                print("✓ Good calibration quality")
            elif self.rms_error < 2.0:
                print("⚠ Acceptable calibration quality")
            else:
                print("✗ Poor calibration quality - consider recapturing images")
        
        # Determine output file based on directory name
        if self.images_dir[-1] == '0':
            output_file = os.path.join(os.getcwd(), 'src/camera_calibration/calib_files/calib_data_90.npz')
        else:
            output_file = os.path.join(os.getcwd(), 'src/camera_calibration/calib_files/calib_data_91.npz')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save calibration data including RMS error
        np.savez(output_file, 
                 mtx=self.mtx, 
                 dist=self.dist, 
                 rms_error=self.rms_error,
                 rvecs=self.rvecs,
                 tvecs=self.tvecs)
        
        print(f"Calibration data saved to: {output_file}")
        return True

    def undistort_and_crop(self, image_path, output_path='calibresult.png'):
        img = cv.imread(image_path)
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv.undistort(img, self.mtx, self.dist, None, newcameramtx)
        crop_x = int(w * self.crop_percent)
        crop_y = int(h * self.crop_percent)
        dst_cropped = dst[crop_y:h-crop_y, crop_x:w-crop_x]
        cv.imwrite(output_path, dst_cropped)
        #print(f"Saved undistorted and cropped image to {output_path}")

    def compute_reprojection_error(self, print_vals=False):
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        total_error = mean_error / len(self.objpoints)
        if print_vals:
            print(f"Total reprojection error: {total_error}")
        return total_error

    def calibrate_both_cameras(self, images_dir_90, images_dir_91, print_vals=False):
        """Calibrate both cameras and return success status"""
        print("=== CALIBRATING CAMERA 90 ===")
        self.images_dir = images_dir_90
        self.objpoints = []  # Reset
        self.imgpoints = []  # Reset
        self.collect_image_points(print_images=False)
        success_90 = self.calibrate(print_vals=print_vals)
        rms_90 = self.rms_error if success_90 else None
        
        print("\n=== CALIBRATING CAMERA 91 ===")
        self.images_dir = images_dir_91
        self.objpoints = []  # Reset
        self.imgpoints = []  # Reset
        self.collect_image_points(print_images=False)
        success_91 = self.calibrate(print_vals=print_vals)
        rms_91 = self.rms_error if success_91 else None
        
        print(f"\n=== CALIBRATION SUMMARY ===")
        print(f"Camera 90 success: {success_90}")
        print(f"Camera 91 success: {success_91}")
        
        if success_90 and success_91:
            print("Both cameras calibrated successfully!")
            print(f"Camera 90 RMS: {rms_90:.6f}")
            print(f"Camera 91 RMS: {rms_91:.6f}")
            
            # Quality assessment
            if rms_90 > 1.0 or rms_91 > 1.0:
                print("⚠ WARNING: One or both cameras have high RMS errors")
                print("Consider:")
                print("- Using more/better images")
                print("- Ensuring chessboard is flat and well-lit")
                print("- Checking for motion blur")
            else:
                print("✓ Both cameras have good calibration quality")
                
            return True, rms_90, rms_91
        else:
            print("❌ One or both calibrations failed!")
            return False, rms_90, rms_91
