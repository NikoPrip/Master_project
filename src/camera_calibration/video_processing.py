import cv2 as cv
import glob
import os

class VideoProcessor:
    def __init__(self, video_path, output_dir, aruco_dict_type=cv.aruco.DICT_5X5_250, board_size=(12, 9), square_length=0.06, marker_length=0.045):
        self.video_path = video_path
        self.output_dir = output_dir
        self.aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict_type)
        self.board = cv.aruco.CharucoBoard(board_size, square_length, marker_length, self.aruco_dict)
        self.detector = cv.aruco.ArucoDetector(self.aruco_dict)
        
        if not os.path.exists(output_dir):
            print("Output directory does not exist.")
        else:
            self.clear_output_directory()

    def clear_output_directory(self):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        removed_count = 0
        
        for extension in image_extensions:
            files = glob.glob(os.path.join(self.output_dir, extension))
            for file in files:
                try:
                    os.remove(file)
                    removed_count += 1
                except OSError as e:
                    print(f"Error removing {file}: {e}")
        
        if removed_count > 0:
            print(f"Removed {removed_count} existing image files from {self.output_dir}")

    def detect_charuco_board(self, frame):
        """Detect ChArUco board in frame and return True if found"""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)
        
        if len(marker_corners) > 0:
            # Interpolate ChArUco corners
            charuco_retval, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.board
            )
            
            # Return True if we found enough ChArUco corners (you can adjust this threshold)
            return charuco_retval is not None and charuco_retval >= 15
        
        return False

    def extract_frames(self):
        cap = cv.VideoCapture(self.video_path)
        frame_count = 0
        saved_count = 0
        frame_interval = 30  # Save every 30th frame (1 second at 30fps)

        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check both: interval AND ChArUco board detection
            if frame_count % frame_interval == 0 and self.detect_charuco_board(frame):
                frame_filename = os.path.join(self.output_dir, f"frame_{saved_count:04d}.jpeg")
                cv.imwrite(frame_filename, frame)
                saved_count += 1
                print(f"Saved frame {frame_count} with ChArUco board detected")
            
            frame_count += 1

        cap.release()
        print(f"Extracted {saved_count} frames with ChArUco boards from {self.video_path} to {self.output_dir}")

if __name__ == "__main__":
    camera_calib = False
    stereo_calib = True

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
