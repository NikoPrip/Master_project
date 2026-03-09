import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import threading
import time
import os
import numpy as np
import cv2
import queue

class DualVideoCapture:
    def __init__(self, camera_configs, output_dir="video_captures"):
        self.camera_configs = camera_configs
        self.output_dir = output_dir
        self.recording = False
        self.preview_active = False
        self.preview_pipelines = []  # For preview during countdown
        self.recording_pipelines = []  # For actual recording
        self.appsinks = []
        self.frame_queues = {}
        self.record_start_time = None
        self.sync_lock = threading.Lock()  # For frame synchronization
        self.frame_buffers = {}  # Store frames with timestamps
        self.sync_tolerance = 0.033  # 33ms tolerance (for 30fps)
        
        os.makedirs(self.output_dir, exist_ok=True)
        Gst.init(None)

    def _create_preview_pipeline(self, port, camera_name):
        """Create preview-only pipeline for countdown"""
        pipeline_str = (
            f"udpsrc port={port} ! "
            "application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink name=preview_{camera_name} "
            "emit-signals=false sync=false max-buffers=2 drop=true"
        )
        return Gst.parse_launch(pipeline_str)

    def _create_recording_pipeline(self, port, output_filename, camera_name):
        """Create pipeline with both preview and recording"""
        pipeline_str = (
            f"udpsrc port={port} ! "
            "application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! "
            "rtph264depay ! h264parse ! "
            "tee name=t "

            # Preview branch
            "t. ! queue max-size-buffers=2 leaky=downstream ! avdec_h264 ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink name=preview_{camera_name} "
            "emit-signals=false sync=false max-buffers=2 drop=true "

            # Recording branch
            f"t. ! queue ! mp4mux ! filesink location={output_filename}"
        )
        return Gst.parse_launch(pipeline_str)

    def _capture_frames(self, appsink, camera_name):
        """Capture frames with precise timestamps"""
        print(f"[{camera_name}] Frame capture thread started")
        frame_count = 0
        while self.preview_active:
            try:
                # Shorter timeout for more responsive pulling (1 second)
                sample = appsink.emit("try-pull-sample", 1000000000)
                if sample:
                    frame_count += 1
                    buf = sample.get_buffer()
                    caps = sample.get_caps()

                    # Get precise timestamp
                    timestamp = time.time()

                    struct = caps.get_structure(0)
                    width = struct.get_value('width')
                    height = struct.get_value('height')

                    frame = np.ndarray(
                        (height, width, 3),
                        buffer=buf.extract_dup(0, buf.get_size()),
                        dtype=np.uint8
                    ).copy()

                    # Debug: Print first few frames
                    if frame_count <= 3:
                        print(f"[{camera_name}] Captured frame {frame_count}: {width}x{height}")

                    # Store frame with timestamp
                    with self.sync_lock:
                        if camera_name not in self.frame_buffers:
                            self.frame_buffers[camera_name] = queue.Queue(maxsize=10)

                        try:
                            self.frame_buffers[camera_name].put_nowait((timestamp, frame))
                        except queue.Full:
                            # Remove oldest frame and add new one
                            try:
                                self.frame_buffers[camera_name].get_nowait()
                                self.frame_buffers[camera_name].put_nowait((timestamp, frame))
                            except queue.Empty:
                                pass
                else:
                    time.sleep(0.01)

            except Exception as e:
                print(f"Error capturing {camera_name}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        print(f"[{camera_name}] Frame capture thread stopped. Total frames: {frame_count}")

    def _get_synchronized_frames(self, strict=False):
        """Get synchronized frames from all cameras

        Args:
            strict: If True, enforce sync tolerance. If False, return latest frames regardless.
        """
        with self.sync_lock:
            if len(self.frame_buffers) != len(self.camera_configs):
                return None

            # Get latest frames from each camera (drain queue to get newest)
            frames_with_timestamps = {}
            for camera_name in self.frame_buffers:
                latest_frame = None
                # Drain queue to get the newest frame
                while True:
                    try:
                        latest_frame = self.frame_buffers[camera_name].get_nowait()
                    except queue.Empty:
                        break

                if latest_frame:
                    timestamp, frame = latest_frame
                    frames_with_timestamps[camera_name] = (timestamp, frame)
                else:
                    # No frames available for this camera
                    return None

            if not frames_with_timestamps:
                return None

            # Find reference timestamp (latest)
            timestamps = [ts for ts, _ in frames_with_timestamps.values()]
            ref_timestamp = max(timestamps)

            # Check if all frames are within sync tolerance
            synchronized_frames = {}
            all_synced = True
            for camera_name, (timestamp, frame) in frames_with_timestamps.items():
                synchronized_frames[camera_name] = frame
                if abs(timestamp - ref_timestamp) > self.sync_tolerance:
                    all_synced = False

            # If strict mode and not synced, return None
            if strict and not all_synced:
                return None

            return synchronized_frames if len(synchronized_frames) == len(self.camera_configs) else None

    def start_recording(self, duration_seconds=None, countdown_seconds=3):
        """Start synchronized recording from all cameras"""
        timestamp = int(time.time())
        
        print(f"Starting synchronized capture from {len(self.camera_configs)} cameras...")
        if duration_seconds:
            print(f"Recording for {duration_seconds} seconds")
        
        # Initialize frame buffers
        for config in self.camera_configs:
            self.frame_buffers[config['name']] = queue.Queue(maxsize=10)

        # Create PREVIEW-ONLY pipelines for countdown
        print("Creating preview pipelines for countdown...")

        for config in self.camera_configs:
            pipeline = self._create_preview_pipeline(config['port'], config['name'])
            self.preview_pipelines.append(pipeline)

            appsink = pipeline.get_by_name(f"preview_{config['name']}")
            if appsink:
                self.appsinks.append(appsink)
                print(f"✓ Preview pipeline created for {config['name']}")
            else:
                print(f"✗ Failed to create preview pipeline for {config['name']}")

        # Start all preview pipelines
        print("Starting preview pipelines...")
        self.preview_active = True

        for pipeline in self.preview_pipelines:
            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print(f"✗ Failed to start preview pipeline")
            else:
                print(f"✓ Started preview pipeline")
        
        print("Waiting for synchronized streams...")
        time.sleep(4)  # Allow time for sync
        
        # Start capture threads
        for i, config in enumerate(self.camera_configs):
            if i < len(self.appsinks):
                thread = threading.Thread(
                    target=self._capture_frames, 
                    args=(self.appsinks[i], config['name']),
                    daemon=True
                )
                thread.start()
        
        # Create windows
        for config in self.camera_configs:
            cv2.namedWindow(f"Camera {config['name']}", cv2.WINDOW_AUTOSIZE)

        # Note: Recording to file will start AFTER countdown
        self.preview_active = True
        print("📹 Preview active, countdown starting...")

        # Countdown with synchronized frames
        for i in range(countdown_seconds, 0, -1):
            print(f"Recording... {i}s")

            countdown_start = time.time()
            while time.time() - countdown_start < 1.0:
                # Use non-strict mode for preview (show even if not perfectly synced)
                sync_frames = self._get_synchronized_frames(strict=False)
                if sync_frames:
                    for camera_name, frame in sync_frames.items():
                        cv2.putText(frame, f"SYNC Recording: {i}s", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow(f"Camera {camera_name}", frame)
                else:
                    print(f"[DEBUG] No frames available during countdown")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_recording()
                    return

                time.sleep(0.016)  # ~60fps display

        # Stop preview pipelines
        print("Stopping preview-only pipelines...")
        self.preview_active = False
        time.sleep(0.5)  # Let capture threads finish

        for pipeline in self.preview_pipelines:
            pipeline.set_state(Gst.State.NULL)

        # Clear old appsinks
        self.appsinks = []

        # NOW create and start RECORDING pipelines
        print("🔴 Creating recording pipelines...")
        shared_clock = Gst.SystemClock.obtain()

        for config in self.camera_configs:
            output_filename = os.path.join(
                self.output_dir,
                f"{config['name']}_{timestamp}.mp4"
            )

            pipeline = self._create_recording_pipeline(config['port'], output_filename, config['name'])
            pipeline.use_clock(shared_clock)
            self.recording_pipelines.append(pipeline)

            appsink = pipeline.get_by_name(f"preview_{config['name']}")
            if appsink:
                self.appsinks.append(appsink)
                print(f"✓ Recording pipeline created for {config['name']}")
            else:
                print(f"✗ Failed to create recording pipeline for {config['name']}")

        # Start all recording pipelines simultaneously
        print("Starting recording pipelines...")
        self.preview_active = True

        start_time = shared_clock.get_time() + Gst.SECOND
        for pipeline in self.recording_pipelines:
            pipeline.set_start_time(start_time)
            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print(f"✗ Failed to start recording pipeline")
            else:
                print(f"✓ Started recording pipeline")

        print("Waiting for recording streams to start...")
        time.sleep(2)

        # Restart capture threads for new appsinks
        for i, config in enumerate(self.camera_configs):
            if i < len(self.appsinks):
                thread = threading.Thread(
                    target=self._capture_frames,
                    args=(self.appsinks[i], config['name']),
                    daemon=True
                )
                thread.start()

        self.recording = True
        self.record_start_time = time.time()
        print("🔴 RECORDING TO FILE STARTED!")

        try:
            # Main synchronized recording loop
            target_end_time = time.time() + duration_seconds if duration_seconds else None
            sync_frame_count = 0
            dropped_frame_count = 0
            
            while self.preview_active and self.recording:
                current_time = time.time()

                # Use non-strict mode for preview
                sync_frames = self._get_synchronized_frames(strict=False)
                if sync_frames:
                    sync_frame_count += 1
                    elapsed = current_time - self.record_start_time

                    for camera_name, frame in sync_frames.items():
                        cv2.putText(frame, f"● SYNC REC {elapsed:.1f}s", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Sync: {sync_frame_count}", (50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow(f"Camera {camera_name}", frame)
                else:
                    dropped_frame_count += 1
                    if dropped_frame_count % 100 == 0:
                        print(f"[WARNING] Dropped {dropped_frame_count} frames")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if target_end_time and current_time >= target_end_time:
                    print(f"Synchronized recording completed: {current_time - self.record_start_time:.1f}s")
                    break
                
                time.sleep(0.016)  # ~60fps
                    
        except KeyboardInterrupt:
            print("Recording stopped by user")
        finally:
            print(f"Synchronized frames: {sync_frame_count}, Dropped: {dropped_frame_count}")
            self.stop_recording()

    def stop_recording(self):
        """Stop recording and cleanup"""
        print("Stopping recording...")

        self.recording = False
        self.preview_active = False
        cv2.destroyAllWindows()

        # Send EOS to recording pipelines
        for pipeline in self.recording_pipelines:
            pipeline.send_event(Gst.Event.new_eos())

        time.sleep(1)  # Allow EOS to propagate

        # Stop all recording pipelines
        for i, pipeline in enumerate(self.recording_pipelines):
            try:
                pipeline.set_state(Gst.State.NULL)
                print(f"✓ Stopped recording pipeline {i+1}")
            except Exception as e:
                print(f"✗ Error stopping pipeline {i+1}: {e}")

        # Stop preview pipelines if still running
        for pipeline in self.preview_pipelines:
            try:
                pipeline.set_state(Gst.State.NULL)
            except:
                pass

        if self.record_start_time:
            duration = time.time() - self.record_start_time
            print(f"Final recording duration: {duration:.1f}s")

        print(f"Files saved to: {self.output_dir}")

# Usage
if __name__ == "__main__":
    cameras = [
        {'port': 58004, 'name': 'cam_90'},
        {'port': 58006, 'name': 'cam_91'}
    ]
    
    capture = DualVideoCapture(cameras, output_dir="src/camera_calibration/outdoor/camera_calib_video")
    capture.start_recording(duration_seconds=60, countdown_seconds=10)