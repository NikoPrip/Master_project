import os
import numpy as np
import cv2.aruco as aruco
from src.camera_calibration.video_processing import VideoProcessor
from src.camera_calibration.calibration import StereoCalibrator
from src.camera_calibration.calibration import Calibrator

def camera_calibration(camera_calib = False, stereo_calib = False, data = 'indoor'):
    path = os.path.join(os.path.dirname(__file__), 'src/camera_calibration/')

    if data == 'indoor':
        detector_90 = Calibrator(
            image_dir=path + 'indoor/Images_cam_90',
            charuco_board_size=(12, 9),
            marker_type=aruco.DICT_5X5_250
        )
        detector_91 = Calibrator(
            image_dir=path + 'indoor/Images_cam_91',
            charuco_board_size=(12, 9),
            marker_type=aruco.DICT_5X5_250
        )
    elif data == 'outdoor':
        detector_90 = Calibrator(
            image_dir=path + 'outdoor/Images_cam_90',
            charuco_board_size=(12, 9),
            marker_type=aruco.DICT_5X5_250
        )
        detector_91 = Calibrator(
            image_dir=path + 'outdoor/Images_cam_91',
            charuco_board_size=(12, 9),
            marker_type=aruco.DICT_5X5_250
        )
    elif data == 'phone':
        detector_90 = Calibrator(
            image_dir=path + 'phone/Images_cam_90',
            charuco_board_size=(12, 9),
            marker_type=aruco.DICT_5X5_250
        )

    elif data == 'sim':
        detector_90 = Calibrator(
            image_dir=path + 'sim/Images_cam_90',
            charuco_board_size=(11, 8),
            marker_type=aruco.DICT_4X4_250,
            legacy_pattern=True
        )
    elif data == 'final':
        detector_90 = Calibrator(
            image_dir=path + 'final_test/Images_cam_90',
            charuco_board_size=(12, 9),
            marker_type=aruco.DICT_5X5_250
        )
    
    if camera_calib:
        detector_90.detect_images()
        calibration_results_90 = detector_90.calibrate_camera()
        if data == 'indoor':
            detector_91.detect_images()
            calibration_results_91 = detector_91.calibrate_camera()
            detector_90.save_calibration(path + 'indoor/calib_files/calib_data_90.npz')
            detector_91.save_calibration(path + 'indoor/calib_files/calib_data_91.npz')
        elif data == 'outdoor':
            detector_91.detect_images()
            calibration_results_91 = detector_91.calibrate_camera()
            detector_90.save_calibration(path + 'outdoor/calib_files/calib_data_90.npz')
            detector_91.save_calibration(path + 'outdoor/calib_files/calib_data_91.npz')
        elif data == 'phone':
            detector_90.save_calibration(path + 'phone/calib_files/calib_data_90.npz')
        elif data == 'sim':
            detector_90.save_calibration(path + 'sim/calib_files/calib_data_90.npz')
        elif data == 'final':
            detector_90.save_calibration(path + 'final_test/calib_files/calib_data_90.npz')
    if stereo_calib:
        if data == 'indoor':
            data_90 = np.load(path + 'indoor/calib_files/calib_data_90.npz')
            data_91 = np.load(path + 'indoor/calib_files/calib_data_91.npz')
        elif data == 'outdoor':
            data_90 = np.load(path + 'outdoor/calib_files/calib_data_90.npz')
            data_91 = np.load(path + 'outdoor/calib_files/calib_data_91.npz')
        elif data == 'phone':
            data_90 = np.load(path + 'phone/calib_files/calib_data_90.npz')
        image_size = (1280, 800)  # Replace with actual image size used during calibration
        stereo_calibrator = StereoCalibrator(
            calib_data_90=data_90,
            calib_data_91=data_91,
            image_size=image_size,
            charuco_board_size=(12, 9),
            marker_type=aruco.DICT_5X5_250
        )
        if data == 'indoor':
            stereo_calibrator.detect_stereo_corners(
                image_dir_90=path + 'indoor/Stereo_Images_90',
                image_dir_91=path + 'indoor/Stereo_Images_91'
            )
        elif data == 'outdoor':
            stereo_calibrator.detect_stereo_corners(
                image_dir_90=path + 'outdoor/Stereo_Images_90_sync',
                image_dir_91=path + 'outdoor/Stereo_Images_91_sync'
            )
        elif data == 'phone':
            stereo_calibrator.detect_stereo_corners(
                image_dir_90=path + 'phone/Stereo_Images_90_sync',
                image_dir_91=None
            )

        stereo_calibrator.stereo_calibrate()
        if data == 'indoor':
            stereo_calibrator.save_stereo_calibration(path + 'indoor/calib_files/stereo_calib_data.npz')
        elif data == 'phone':
            stereo_calibrator.save_stereo_calibration(path + 'phone/calib_files/stereo_calib_data.npz')
        else:
            stereo_calibrator.save_stereo_calibration(path + 'outdoor/calib_files/stereo_calib_data.npz')

def video_processing(camera_calib = False, stereo_calib = False, data = 'indoor'):
    path = os.path.join(os.path.dirname(__file__), 'src/camera_calibration/')

    if camera_calib:
        if data == 'indoor':
            video_path_91 = path + 'indoor/camera_calib_video/input_video_91.mp4'
            output_dir_91 = path + 'indoor/Images_cam_91'
            video_path_90 = path + 'indoor/camera_calib_video/input_video_90.mp4'
            output_dir_90 = path + 'indoor/Images_cam_90'
        elif data == 'outdoor':
            video_path_91 = path + 'outdoor/camera_calib_video/input_video_91.mp4'
            output_dir_91 = path + 'outdoor/Images_cam_91'
            video_path_90 = path + 'outdoor/camera_calib_video/input_video_90.mp4'
            output_dir_90 = path + 'outdoor/Images_cam_90'
        elif data == 'phone':
            video_path_91 = path + 'phone/camera_calib_video/input_video_91.mp4'
            output_dir_91 = path + 'phone/Images_cam_91'
            video_path_90 = path + 'phone/camera_calib_video/input_video_90.mp4'
            output_dir_90 = path + 'phone/Images_cam_90'
        elif data == 'sim':
            video_path_91 = path + 'sim/camera_calib_video/input_video_91.mp4'
            output_dir_91 = path + 'sim/Images_cam_91'
            video_path_90 = path + 'sim/camera_calib_video/input_video_90.mp4'
            output_dir_90 = path + 'sim/Images_cam_90'
        elif data == 'final':
            video_path_91 = path + 'final_test/camera_calib_video/input_video_91.mp4'
            output_dir_91 = path + 'final_test/Images_cam_91'
            video_path_90 = path + 'final_test/camera_calib_video/input_video_90.mp4'
            output_dir_90 = path + 'final_test/Images_cam_90'
        
        if data == 'sim':
            processor_90 = VideoProcessor(video_path_90, output_dir_90,
                                aruco_dict_type=aruco.DICT_4X4_250,
                                board_size=(11, 8),
                                square_length=0.06,
                                marker_length=0.045,
                                legacy_pattern=True)
        else:
            processor_91 = VideoProcessor(video_path_91, output_dir_91, 
                                aruco_dict_type=aruco.DICT_5X5_250,
                                board_size=(12, 9),
                                square_length=0.06,
                                marker_length=0.045)
            processor_90 = VideoProcessor(video_path_90, output_dir_90, 
                                aruco_dict_type=aruco.DICT_5X5_250,
                                board_size=(12, 9),
                                square_length=0.06,
                                marker_length=0.045)
        if data != 'sim':
            processor_91.extract_frames()
        processor_90.extract_frames()

    if stereo_calib:
        if data == 'indoor':
            video_path_91 = path + 'indoor/stereo_calib_video/cam_91.mp4'
            output_dir_91 = path + 'indoor/Stereo_Images_91'
            video_path_90 = path + 'indoor/stereo_calib_video/cam_90.mp4'
            output_dir_90 = path + 'indoor/Stereo_Images_90'
        elif data == 'outdoor':
            video_path_91 = path + 'outdoor/stereo_calib_video/cam_91.mp4'
            output_dir_91 = path + 'outdoor/Stereo_Images_91'
            video_path_90 = path + 'outdoor/stereo_calib_video/cam_90.mp4'
            output_dir_90 = path + 'outdoor/Stereo_Images_90'
        elif data == 'phone':
            video_path_90 = path + 'phone/stereo_calib_video/cam_90.mp4'
            output_dir_90 = path + 'phone/Stereo_Images_90'
        elif data == 'sim':
            video_path_91 = path + 'sim/stereo_calib_video/cam_91.mp4'
            output_dir_91 = path + 'sim/Stereo_Images_91'
            video_path_90 = path + 'sim/stereo_calib_video/cam_90.mp4'
            output_dir_90 = path + 'sim/Stereo_Images_90'

        processor_91 = VideoProcessor(video_path_91, output_dir_91, 
                                aruco_dict_type=aruco.DICT_5X5_250,
                                board_size=(12, 9),
                                square_length=0.06,
                                marker_length=0.045)
        processor_90 = VideoProcessor(video_path_90, output_dir_90, 
                                aruco_dict_type=aruco.DICT_5X5_250,
                                board_size=(12, 9),
                                square_length=0.06,
                                marker_length=0.045)
        processor_91.extract_frames()
        processor_90.extract_frames()

if __name__ == "__main__":
    video_processing(camera_calib = True, stereo_calib = False, data = 'final')
    camera_calibration(camera_calib = True, stereo_calib = False, data = 'final')

