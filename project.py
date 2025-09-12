import os
import glob
import cv2 as cv
from src.camera_calibration.calibration import CameraCalibrator

calibrator = CameraCalibrator(images_dir = os.path.join(os.path.dirname(__file__), 'src/camera_calibration/Images'), chessboard_size=(8, 11), crop_percent=0.125)
calibrator.collect_image_points()
# Use the shape of the first image for calibration
first_image = glob.glob(os.path.join(calibrator.images_dir, '*.jpeg'))[0]
img = cv.imread(first_image)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
calibrator.calibrate(gray.shape[::-1])
#calibrator.undistort_and_crop(os.path.join(calibrator.images_dir, 'frame_00570.jpeg'))
calibrator.compute_reprojection_error()