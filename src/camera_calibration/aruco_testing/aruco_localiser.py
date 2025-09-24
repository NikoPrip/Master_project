import cv2
import os
import sys
import numpy as np
import time

# Change this global variable to use a different video file
FILENAME = "video_with_aruco_markers_dict_4x4_250.mov"


def fallback_mode(video_path):
    """A simplified mode that tries to process video frame-by-frame with minimal OpenCV dependencies"""
    print("\nTrying FALLBACK MODE with minimal dependencies")
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video even in fallback mode")
            return False
            
        # Try to read just one frame to test
        ret, frame = cap.read()
        if not ret:
            print("Failed to read first frame in fallback mode")
            cap.release()
            return False
            
        # Show the first frame as a test
        print("Successfully read first frame in fallback mode")
        height, width = frame.shape[:2]
        print(f"Frame dimensions: {width}x{height}")
        
        # Try to display it
        try:
            cv2.imshow("First Frame", frame)
            cv2.waitKey(1000)  # Wait 1 second
            cv2.destroyAllWindows()
            print("Successfully displayed first frame")
        except Exception as e:
            print(f"Cannot display frame: {e}")
        
        # Release resources
        cap.release()
        return True
        
    except Exception as e:
        print(f"Fallback mode error: {e}")
        return False


def main():
    # Use the global filename variable
    global FILENAME
    
    # Get the absolute path to ensure the file is found
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, FILENAME)
    
    # Print the path to verify it's correct
    print(f"Attempting to open video from: {video_path}")
    
    # Check if the file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
    
    # Print OpenCV version for debugging
    print(f"OpenCV version: {cv2.__version__}")
    
    # Try different backends for video capture
    backends = [cv2.CAP_FFMPEG, cv2.CAP_AVFOUNDATION, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    cap = None
    
    for backend in backends:
        try:
            print(f"Trying video backend: {backend}")
            cap = cv2.VideoCapture(video_path, backend)
            if cap.isOpened():
                print(f"Successfully opened video with backend {backend}")
                break
        except Exception as e:
            print(f"Failed to open with backend {backend}: {e}")
    
    # If no backend worked, try without specifying one
    if cap is None or not cap.isOpened():
        print("Trying default backend")
        cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        print("This could be due to missing codecs or an incompatible video format")
        return False
    
    # Get video properties for debugging
    try:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {frame_count} frames")
    except Exception as e:
        print(f"Could not retrieve video properties: {e}")
    
    # Determine ArUco API version
    print("Setting up ArUco detector")
    aruco_detector = None
    use_new_api = False
    
    # Try with newer OpenCV API first
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        use_new_api = True
        print("Using new ArUco API")
    except (AttributeError, cv2.error, TypeError) as e:
        print(f"New ArUco API not available: {e}")
        
        # Try with older OpenCV API
        try:
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
            parameters = cv2.aruco.DetectorParameters_create()
            use_new_api = False
            print("Using older ArUco API")
        except Exception as e:
            print(f"Error setting up ArUco detector: {e}")
            # Release resources and exit
            cap.release()
            cv2.destroyAllWindows()
            return False

    # Analyse each frame individually
    frame_count = 0
    success = False
    
    try:
        while cap.isOpened():
            # Read a frame with timeout to avoid hanging
            ret, frame = cap.read()
            
            if not ret:
                print("End of video or failed to read frame")
                break
            
            frame_count += 1
            if frame_count % 10 == 0:  # Print every 10th frame to reduce output
                print(f"Processing frame {frame_count}")
            
            # Set success to True if we've successfully processed at least one frame
            success = True
            
            # Check frame is valid before processing
            if frame is None or frame.size == 0:
                print(f"Warning: Empty frame received at position {frame_count}")
                continue
            
            # Process frame based on API version
            try:
                if use_new_api and aruco_detector is not None:
                    # New API (OpenCV 4.7.0+)
                    markerCorners, markerIds, rejectedCandidates = aruco_detector.detectMarkers(frame)
                    
                    if markerIds is not None and len(markerIds) > 0:
                        print(f"Detected {len(markerIds)} markers")
                        # Draw detected markers
                        frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), markerCorners, markerIds)
                    else:
                        print("No markers detected")
                        frame_markers = frame.copy()
                else:
                    # Old API
                    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
                        frame, dictionary, parameters=parameters
                    )
                    
                    if markerIds is not None and len(markerIds) > 0:
                        print(f"Detected {len(markerIds)} markers")
                        # Draw detected markers
                        frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), markerCorners, markerIds)
                    else:
                        print("No markers detected")
                        frame_markers = frame.copy()
                
                # Show frame with markers
                cv2.imshow("ArUco Markers", frame_markers)
                
                # Check for key presses (timeout of 1ms for smooth video)
                k = cv2.waitKey(1)
                if k == ord("q"):  # Quit on 'q'
                    print("User interrupted with 'q'")
                    break
                elif k == ord("p"):  # Pause on 'p'
                    print("Paused - press any key to continue")
                    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Continue with next frame instead of crashing
                continue
                
    except Exception as e:
        print(f"Error in video processing loop: {e}")
    finally:
        # Always release resources
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")
        print("Program ended")
        
    return success


if __name__ == "__main__":
    try:
        print("Starting ArUco Marker Detection")
        success = main()
        
        # If main failed, try fallback mode
        if not success:
            print("Main processing failed, trying fallback mode")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            video_path = os.path.join(script_dir, FILENAME)
            fallback_mode(video_path)
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try fallback mode after exception
        try:
            print("Trying fallback mode after exception")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            video_path = os.path.join(script_dir, FILENAME)
            fallback_mode(video_path)
        except:
            pass
            
    finally:
        print("Exiting program")
        # Ensure all windows are closed
        cv2.destroyAllWindows()