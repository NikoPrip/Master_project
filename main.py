import os
import numpy as np
import cv2.aruco as aruco
from src.camera_calibration.video_processing import VideoProcessor
from src.camera_calibration.calibration import StereoCalibrator
from src.camera_calibration.calibration import Calibrator

def camera_calibration(camera_calib = False, stereo_calib = False):

    detector_left = Calibrator(
        image_dir='src/camera_calibration/Images_cam_left',
        charuco_board_size=(12, 9),
        marker_type=aruco.DICT_5X5_250
    )
    detector_right = Calibrator(
        image_dir='src/camera_calibration/Images_cam_right',
        charuco_board_size=(12, 9),
        marker_type=aruco.DICT_5X5_250
    )
    if camera_calib:
        detector_left.detect_images()
        calibration_results_left = detector_left.calibrate_camera()
        detector_right.detect_images()
        calibration_results_right = detector_right.calibrate_camera()
        detector_left.save_calibration('src/camera_calibration/calib_files/calib_data_left.npz')
        detector_right.save_calibration('src/camera_calibration/calib_files/calib_data_right.npz')
    if stereo_calib:
        left_data = np.load('src/camera_calibration/calib_files/calib_data_left.npz')
        right_data = np.load('src/camera_calibration/calib_files/calib_data_right.npz')
        image_size = (1280, 800)  # Replace with actual image size used during calibration
        stereo_calibrator = StereoCalibrator(
            left_calib_data=left_data,
            right_calib_data=right_data,
            image_size=image_size,
            charuco_board_size=(12, 9),
            marker_type=aruco.DICT_5X5_250
        )
        stereo_calibrator.detect_stereo_corners(
            left_image_dir='src/camera_calibration/Stereo_Images_left',
            right_image_dir='src/camera_calibration/Stereo_Images_right'
        )
        stereo_calibrator.stereo_calibrate()
        stereo_calibrator.save_stereo_calibration('src/camera_calibration/calib_files/stereo_calib_data.npz')

def video_processing(camera_calib = False, stereo_calib = False):

    if camera_calib:
        video_path_right = 'src/camera_calibration/camera_calib_video/input_video_right.mp4'
        output_dir_right = 'src/camera_calibration/Images_cam_right'
        video_path_left = 'src/camera_calibration/camera_calib_video/input_video_left.mp4'
        output_dir_left = 'src/camera_calibration/Images_cam_left'

        processor_right = VideoProcessor(video_path_right, output_dir_right, 
                                aruco_dict_type=cv.aruco.DICT_5X5_250,
                                board_size=(12, 9),
                                square_length=0.06,
                                marker_length=0.045)
        processor_left = VideoProcessor(video_path_left, output_dir_left, 
                                aruco_dict_type=cv.aruco.DICT_5X5_250,
                                board_size=(12, 9),
                                square_length=0.06,
                                marker_length=0.045)
        processor_right.extract_frames()
        processor_left.extract_frames()

    if stereo_calib:
        video_path_right = 'src/camera_calibration/stereo_calib_video/cam_right.mp4'
        output_dir_right = 'src/camera_calibration/Stereo_Images_right'
        video_path_left = 'src/camera_calibration/stereo_calib_video/cam_left.mp4'
        output_dir_left = 'src/camera_calibration/Stereo_Images_left'

        processor_right = VideoProcessor(video_path_right, output_dir_right, 
                                aruco_dict_type=cv.aruco.DICT_5X5_250,
                                board_size=(12, 9),
                                square_length=0.06,
                                marker_length=0.045)
        processor_left = VideoProcessor(video_path_left, output_dir_left, 
                                aruco_dict_type=cv.aruco.DICT_5X5_250,
                                board_size=(12, 9),
                                square_length=0.06,
                                marker_length=0.045)
        processor_right.extract_frames()
        processor_left.extract_frames()

if __name__ == "__main__":
    video_processing(camera_calib = False, stereo_calib = False)
    camera_calibration(camera_calib = True, stereo_calib = True)

