import os
import cv2.aruco as aruco
import numpy as np
from src.camera_calibration.calibration import CameraCalibrator
from src.camera_calibration.capture_video import VideoUndistorter
from src.camera_calibration.stereo_calibration import StereoCalibrator, StereoPairCapture


def camera_calibration():
    calibrator = CameraCalibrator(chessboard_size=(8, 11), crop_percent=0.125)
    
    images_dir_90 = os.path.join(os.path.dirname(__file__), 'src/camera_calibration/Images_cam_90')
    images_dir_91 = os.path.join(os.path.dirname(__file__), 'src/camera_calibration/Images_cam_91')
    
    success, rms_90, rms_91 = calibrator.calibrate_both_cameras(
        images_dir_90=images_dir_90,
        images_dir_91=images_dir_91,
        print_vals=True
    )
    
    return success

def video_undistortion():
    undistorter = VideoUndistorter(calib_90=os.path.join(os.path.dirname(__file__), 'src/camera_calibration/calib_files/calib_data_90.npz'), calib_91=os.path.join(os.path.dirname(__file__), 'src/camera_calibration/calib_files/calib_data_91.npz'), save_video=False, marker_type=aruco.DICT_6X6_250)
    undistorter.run()

def stereo_calibration():
    # First, let's check the individual camera calibrations
    print("=== CHECKING INDIVIDUAL CAMERA CALIBRATIONS ===")
    
    # Load and check camera 90 calibration
    calib_90_file = os.path.join(os.path.dirname(__file__), 'src/camera_calibration/calib_files/calib_data_90.npz')
    calib_91_file = os.path.join(os.path.dirname(__file__), 'src/camera_calibration/calib_files/calib_data_91.npz')
    
    if os.path.exists(calib_90_file):
        calib_90 = np.load(calib_90_file)
        print(f"Camera 90 available keys: {list(calib_90.keys())}")
        # Try different possible key names for RMS error
        rms_90 = calib_90.get('rms_error', calib_90.get('ret', calib_90.get('rms', 'Not found')))
        print(f"Camera 90 - RMS error: {rms_90}")
    else:
        print("Camera 90 calibration file not found!")
        
    if os.path.exists(calib_91_file):
        calib_91 = np.load(calib_91_file)
        print(f"Camera 91 available keys: {list(calib_91.keys())}")
        # Try different possible key names for RMS error
        rms_91 = calib_91.get('rms_error', calib_91.get('ret', calib_91.get('rms', 'Not found')))
        print(f"Camera 91 - RMS error: {rms_91}")
    else:
        print("Camera 91 calibration file not found!")
    
    print("\n=== STARTING STEREO CALIBRATION ===")
    
    stereo_calibrator = StereoCalibrator(
        calib_90_file=calib_90_file,
        calib_91_file=calib_91_file,
        chessboard_size=(8, 11),
        marker_type=aruco.DICT_6X6_250
    )
    
    # Add more verbose debugging
    print("Collecting stereo image pairs...")
    success = stereo_calibrator.collect_stereo_image_pairs(
        images_dir_90='src/camera_calibration/Stereo_Images_90',
        images_dir_91='src/camera_calibration/Stereo_Images_91',
        print_images=True
    )
    
    if success:
        print(f"\n=== CALIBRATION INPUT SUMMARY ===")
        # Use the correct attribute name from the stereo_calibrator class
        print(f"Number of stereo pairs: {len(stereo_calibrator.objpoints)}")
        print(f"Chessboard size: {stereo_calibrator.chessboard_size}")
        print(f"Expected corners per image: {stereo_calibrator.chessboard_size[0] * stereo_calibrator.chessboard_size[1]}")
        
        if len(stereo_calibrator.objpoints) < 5:
            print("ERROR: Too few valid stereo pairs for calibration!")
            print("This suggests chessboard detection is failing.")
            print("Check that:")
            print("- Chessboard is clearly visible and not blurry")
            print("- Chessboard size (8,11) matches your actual board")
            print("- Images are not corrupted")
            return
            
        print("\nProceeding with stereo calibration...")
        
        # Check individual camera calibration quality first
        if rms_90 != 'Not found' and isinstance(rms_90, (int, float)):
            if rms_90 > 1.0:
                print(f"WARNING: Camera 90 has high RMS error ({rms_90:.3f})")
        if rms_91 != 'Not found' and isinstance(rms_91, (int, float)):
            if rms_91 > 1.0:
                print(f"WARNING: Camera 91 has high RMS error ({rms_91:.3f})")
        
        # Perform stereo calibration with correct image size
        # Your individual calibrations used (768, 480), so use the same
        result = stereo_calibrator.calibrate_stereo(image_size=(768, 480), print_vals=True)
        
        if result:
            # Print interpretation of results
            print("\n=== RESULT INTERPRETATION ===")
            # The RMS error is returned by calibrate_stereo, not stored as an attribute
            # We need to modify the stereo calibrator to store it
            
            stereo_calibrator.compute_stereo_reprojection_error()
        else:
            print("Stereo calibration failed!")
    else:
        print("Failed to collect stereo image pairs")
        print("\nDebugging steps:")
        print("1. Verify images exist in both directories")
        print("2. Check if chessboard is clearly visible")
        print("3. Verify chessboard size (8,11) is correct")
        print("4. Try viewing a few images manually")

if __name__ == "__main__":
    # Calibrate both cameras
    calibration_success = camera_calibration()
    
    # Only proceed to stereo calibration if individual calibrations are successful
    if calibration_success:
        print("\n" + "="*60)
        print("PROCEEDING TO STEREO CALIBRATION")
        print("="*60)
        stereo_calibration()
    else:
        print("❌ Skipping stereo calibration due to failed individual calibrations")

    #camera_configs = [
    #    {'calib_file': os.path.join(os.path.dirname(__file__), 'src/camera_calibration/calib_files/calib_data_90.npz'), 'port': 58004},
    #    {'calib_file': os.path.join(os.path.dirname(__file__), 'src/camera_calibration/calib_files/calib_data_91.npz'), 'port': 58006}
    #]
    #undistorter = VideoUndistorter(camera_configs, save_video=False)
    #undistorter.run()
    #capturer = StereoPairCapture()
    #capturer.capture_synchronized_pairs(
    #    'src/camera_calibration/Stereo_Images_90',
    #    'src/camera_calibration/Stereo_Images_91'
    #)
    #stereo_calibration()
