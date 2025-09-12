import os
import glob
import cv2 as cv
from src.camera_calibration.calibration import CameraCalibrator

print_vals = True
print_images = False
calibrator = CameraCalibrator(images_dir = os.path.join(os.path.dirname(__file__), 'src/camera_calibration/Images'), chessboard_size=(8, 11), crop_percent=0.125)
calibrator.collect_image_points(print_images)
calibrator.calibrate(print_vals)
#calibrator.undistort_and_crop(os.path.join(calibrator.images_dir, 'frame_00570.jpeg'))
calibrator.compute_reprojection_error(print_vals)