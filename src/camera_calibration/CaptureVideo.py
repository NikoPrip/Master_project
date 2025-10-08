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
        self.pipelines = []
        self.appsinks = []
        self.frame_queues = {}
        self.record_start_time = None
        self.sync_lock = threading.Lock()  # For frame synchronization
        self.frame_buffers = {}  # Store frames with timestamps
        self.sync_tolerance = 0.033  # 33ms tolerance (for 30fps)
        
        os.makedirs(self.output_dir, exist_ok=True)
        Gst.init(None)

    def _create_pipeline(self, port, output_filename, camera_name):
        """Create pipeline with preview and recording - with sync timestamps"""
        pipeline_str = (
            f"udpsrc port={port} ! "
            "application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! "
            "rtph264depay ! h264parse ! "
            "tee name=t "
            
            # Preview branch - with timestamp
            "t. ! queue ! avdec_h264 ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink name=preview_{camera_name} "
            "emit-signals=false sync=true max-buffers=2 drop=false "  # sync=true, drop=false
            
            # Recording branch - with timestamp
            f"t. ! queue ! mp4mux ! filesink location={output_filename} sync=true"  # sync=true
        )
        return Gst.parse_launch(pipeline_str)

    def _capture_frames(self, appsink, camera_name):
        """Capture frames with precise timestamps"""
        while self.preview_active:
            try:
                sample = appsink.emit("try-pull-sample", 100000000)
                if sample:
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
                    )
                    
                    # Store frame with timestamp
                    with self.sync_lock:
                        if camera_name not in self.frame_buffers:
                            self.frame_buffers[camera_name] = queue.Queue(maxsize=10)
                        
                        try:
                            self.frame_buffers[camera_name].put_nowait((timestamp, frame.copy()))
                        except queue.Full:
                            # Remove oldest frame and add new one
                            try:
                                self.frame_buffers[camera_name].get_nowait()
                                self.frame_buffers[camera_name].put_nowait((timestamp, frame.copy()))
                            except queue.Empty:
                                pass
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Error capturing {camera_name}: {e}")
                time.sleep(0.1)

    def _get_synchronized_frames(self):
        """Get synchronized frames from all cameras"""
        with self.sync_lock:
            if len(self.frame_buffers) != len(self.camera_configs):
                return None
            
            # Get latest frames from each camera
            frames_with_timestamps = {}
            for camera_name in self.frame_buffers:
                try:
                    timestamp, frame = self.frame_buffers[camera_name].get_nowait()
                    frames_with_timestamps[camera_name] = (timestamp, frame)
                except queue.Empty:
                    return None
            
            # Find reference timestamp (latest)
            timestamps = [ts for ts, _ in frames_with_timestamps.values()]
            ref_timestamp = max(timestamps)
            
            # Check if all frames are within sync tolerance
            synchronized_frames = {}
            for camera_name, (timestamp, frame) in frames_with_timestamps.items():
                if abs(timestamp - ref_timestamp) <= self.sync_tolerance:
                    synchronized_frames[camera_name] = frame
                else:
                    # Frame too old, discard and return None
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
        
        # Create pipelines with shared clock
        print("Creating synchronized pipelines...")
        shared_clock = Gst.SystemClock.obtain()
        
        for config in self.camera_configs:
            output_filename = os.path.join(
                self.output_dir, 
                f"{config['name']}_{timestamp}.mp4"
            )
            
            pipeline = self._create_pipeline(config['port'], output_filename, config['name'])
            pipeline.use_clock(shared_clock)  # Use shared clock for sync
            self.pipelines.append(pipeline)
            
            appsink = pipeline.get_by_name(f"preview_{config['name']}")
            if appsink:
                self.appsinks.append(appsink)
                print(f"✓ Synchronized pipeline created for {config['name']}")
            else:
                print(f"✗ Failed to create pipeline for {config['name']}")
        
        # Start all pipelines simultaneously
        print("Starting synchronized pipelines...")
        self.preview_active = True
        
        # Start all pipelines at once for better sync
        start_time = shared_clock.get_time() + Gst.SECOND  # Start 1 second from now
        for pipeline in self.pipelines:
            pipeline.set_start_time(start_time)
            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print(f"✗ Failed to start pipeline")
            else:
                print(f"✓ Started synchronized pipeline")
        
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
        
        # Start recording
        self.recording = True
        self.record_start_time = time.time()
        print("🔴 SYNCHRONIZED RECORDING STARTED!")
        
        # Countdown with synchronized frames
        for i in range(countdown_seconds, 0, -1):
            print(f"Recording... {i}s")
            
            countdown_start = time.time()
            while time.time() - countdown_start < 1.0:
                sync_frames = self._get_synchronized_frames()
                if sync_frames:
                    for camera_name, frame in sync_frames.items():
                        cv2.putText(frame, f"SYNC Recording: {i}s", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow(f"Camera {camera_name}", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_recording()
                    return
                
                time.sleep(0.016)  # ~60fps display
        
        try:
            # Main synchronized recording loop
            target_end_time = time.time() + duration_seconds if duration_seconds else None
            sync_frame_count = 0
            dropped_frame_count = 0
            
            while self.preview_active and self.recording:
                current_time = time.time()
                
                sync_frames = self._get_synchronized_frames()
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
        print("Stopping synchronized recording...")
        
        self.recording = False
        self.preview_active = False
        cv2.destroyAllWindows()
        
        # Send EOS to all pipelines simultaneously
        for pipeline in self.pipelines:
            pipeline.send_event(Gst.Event.new_eos())
        
        time.sleep(1)  # Allow EOS to propagate
        
        # Stop all pipelines
        for i, pipeline in enumerate(self.pipelines):
            try:
                pipeline.set_state(Gst.State.NULL)
                print(f"✓ Stopped synchronized pipeline {i+1}")
            except Exception as e:
                print(f"✗ Error stopping pipeline {i+1}: {e}")
        
        if self.record_start_time:
            duration = time.time() - self.record_start_time
            print(f"Final synchronized duration: {duration:.1f}s")
        
        print(f"Synchronized files saved to: {self.output_dir}")

# Usage
if __name__ == "__main__":
    cameras = [
        {'port': 58004, 'name': 'cam_90'},
        {'port': 58006, 'name': 'cam_91'}
    ]
    
    capture = DualVideoCapture(cameras, output_dir="src/camera_calibration/stereo_calib_video")
    capture.start_recording(duration_seconds=60, countdown_seconds=10)