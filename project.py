import os
import cv2.aruco as aruco
from src.camera_calibration.calibration import CameraCalibrator
from src.camera_calibration.capture_video import VideoUndistorter

def camera_calibration():
    pass
    print_vals = True
    print_images = False
    calibrator = CameraCalibrator(images_dir = os.path.join(os.path.dirname(__file__), 'src/camera_calibration/Images'), chessboard_size=(8, 11), crop_percent=0.125)
    calibrator.collect_image_points(print_images)
    calibrator.calibrate(print_vals)
    #calibrator.undistort_and_crop(os.path.join(calibrator.images_dir, 'frame_00570.jpeg'))
    #calibrator.compute_reprojection_error(print_vals)

def video_undistortion():
    undistorter = VideoUndistorter(calib_file=os.path.join(os.path.dirname(__file__), 'src/camera_calibration/calib_data.npz'), save_video=False, marker_type=aruco.DICT_6X6_250)
    undistorter.run()

if __name__ == "__main__":
    #camera_calibration()
    video_undistortion()
