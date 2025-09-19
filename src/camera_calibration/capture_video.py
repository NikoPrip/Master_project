import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import cv2

class VideoUndistorter:
    def __init__(self, calib_file='calib_data.npz', save_video=False):
        self.calib = np.load(calib_file)
        self.mtx = self.calib['mtx']
        self.dist = self.calib['dist']
        self.save_video = save_video
        Gst.init(None)
        self.pipeline = self._create_pipeline()
        self.appsink = self.pipeline.get_by_name("sink")

    def _create_pipeline(self):
        if self.save_video:
            pipeline_str = (
                "udpsrc port=58006 buffer-size=212992 do-timestamp=true ! "
                "application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! "
                "rtph264depay ! h264parse ! tee name=t "
                "t. ! queue ! qtmux fragment-duration=1000 ! filesink location=output.mp4 "
                "t. ! queue ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false"
            )
        else:
            pipeline_str = (
                "udpsrc port=58006 buffer-size=212992 do-timestamp=true ! "
                "application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 ! "
                "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false"
            )
        return Gst.parse_launch(pipeline_str)

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            while True:
                sample = self.appsink.emit("try-pull-sample", 100000000)
                if sample:
                    buf = sample.get_buffer()
                    caps = sample.get_caps()
                    width = caps.get_structure(0).get_value('width')
                    height = caps.get_structure(0).get_value('height')
                    arr = np.ndarray(
                        (height, width, 3),
                        buffer=buf.extract_dup(0, buf.get_size()),
                        dtype=np.uint8
                    )
                    undistorted = cv2.undistort(arr, self.mtx, self.dist, None, self.mtx)
                    cv2.imshow("Video", undistorted)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    continue
        finally:
            self.pipeline.set_state(Gst.State.NULL)
