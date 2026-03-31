#!/usr/bin/env python3
"""
Record a video from the /camera/image_raw topic published by the calibration world camera.

Usage (after sourcing the workspace):
    python3 callibration_video_recorder.py [--output <file.mp4>] [--fps <fps>] [--duration <seconds>]

Press Ctrl+C to stop recording early.
"""

import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


TOPIC = '/camera/image_raw'


class VideoRecorderNode(Node):
    def __init__(self, output_path: str, fps: float, duration: float | None):
        super().__init__('calibration_video_recorder')

        self.bridge = CvBridge()
        self.writer: cv2.VideoWriter | None = None
        self.output_path = output_path
        self.fps = fps
        self.duration = duration
        self.frame_count = 0
        self.start_time: float | None = None

        self.sub = self.create_subscription(
            Image,
            TOPIC,
            self._image_callback,
            10,
        )

        self.get_logger().info(
            f"Subscribing to {TOPIC} – recording to '{output_path}'"
            + (f" for {duration}s" if duration else " (press Ctrl+C to stop)")
        )

    # ------------------------------------------------------------------
    def _image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Initialise the VideoWriter on the first frame so we know the size.
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
            self.start_time = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info(f"First frame received ({w}x{h}). Writing video…")

        self.writer.write(frame)
        self.frame_count += 1

        # Stop automatically when the requested duration has elapsed.
        if self.duration is not None:
            elapsed = self.get_clock().now().nanoseconds * 1e-9 - self.start_time
            if elapsed >= self.duration:
                self.get_logger().info(
                    f"Duration of {self.duration}s reached after {self.frame_count} frames."
                )
                self._finish()
                raise SystemExit(0)

    # ------------------------------------------------------------------
    def _finish(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info(
                f"Saved {self.frame_count} frames to '{self.output_path}'"
            )
            self.writer = None


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Record /camera/image_raw to a video file.")
    default_output = f"calibration_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    parser.add_argument(
        '--output', '-o',
        default=default_output,
        help=f"Output video file path (default: {default_output})",
    )
    parser.add_argument(
        '--fps', '-f',
        type=float,
        default=30.0,
        help="Frames per second for the output video (default: 30)",
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=None,
        help="Stop recording after this many seconds (default: record until Ctrl+C)",
    )
    args = parser.parse_args()

    rclpy.init(args=None)
    node = VideoRecorderNode(
        output_path=os.path.abspath(args.output),
        fps=args.fps,
        duration=args.duration,
    )

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node._finish()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
