#!/usr/bin/env python3
"""
Record blade joint state and camera images from the Gazebo simulation.

Outputs:
  <output_dir>/joint_states.csv  - timestamp, joint angle (rad), velocity, effort
  <output_dir>/images/           - PNG frames named by timestamp
"""

import os
import csv
import time
import argparse
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64

# cv_bridge converts ROS Image messages to numpy/OpenCV arrays
from cv_bridge import CvBridge
import cv2


class RecorderNode(Node):

    def __init__(self, output_dir: str):
        super().__init__("state_camera_recorder")

        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        # ── CSV setup ────────────────────────────────────────────────────────
        csv_path = os.path.join(output_dir, "joint_states.csv")
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["ros_time_s", "wall_time_s", "test_time_s", "joint_name",
                                   "position_rad", "velocity_rad_s", "effort_nm",
                                   "commanded_position_rad", "img_filename"])
        self.get_logger().info(f"Saving data to: {csv_path}")

        # ── cv_bridge ────────────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._image_count = 0

        # Latest joint state cache (written on each camera frame)
        self._latest_joint: dict = {}

        # Latest commanded position cache (initialised to 0, nan values are ignored)
        self._latest_commanded_position: float = 0.0

        # Test time: wall time at recording start (used to compute test_time_s)
        self.has_wall_time = False
        self._start_wall_time: float = 0.0

        # Track frame timestamps for accurate FPS calculation
        self._frame_timestamps: list = []
        self._frame_size: tuple = (0, 0)  # (width, height)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            JointState,
            "/blade_rotation_joint/state",
            self._joint_state_callback,
            10,
        )

        self.create_subscription(
            Image,
            "/grader/camera",
            self._camera_callback,
            10,
        )

        self.create_subscription(
            Float64,
            "/blade_rotation_joint/position",
            self._commanded_position_callback,
            10,
        )

        self.get_logger().info("Recorder node started. Waiting for messages...")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _commanded_position_callback(self, msg: Float64):
        """Cache the latest commanded position; ignore nan, keep last valid value."""
        import math
        if not math.isnan(msg.data):
            self._latest_commanded_position = msg.data

    def _joint_state_callback(self, msg: JointState):
        """Cache the latest joint state; writing is driven by the camera."""
        ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._latest_joint = {
            "ros_time": ros_time,
            "wall_time": time.time(),
            "names":     list(msg.name),
            "positions": list(msg.position),
            "velocities": list(msg.velocity),
            "efforts":   list(msg.effort),
        }

    def _camera_callback(self, msg: Image):
        """One camera frame → one image file + one CSV row (with latest joint state)."""
        img_ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge conversion failed: {e}")
            return

        # Save image
        filename = f"{self._image_count:06d}.png"
        filepath = os.path.join(self.images_dir, filename)
        cv2.imwrite(filepath, cv_image)
        self._image_count += 1
        self._frame_timestamps.append(img_ros_time)
        if self._frame_size == (0, 0):
            h, w = cv_image.shape[:2]
            self._frame_size = (w, h)
        
        # Write one CSV row using the latest cached joint state
        j = self._latest_joint
        if not self.has_wall_time:
            self._start_wall_time = j["ros_time"] if j else time.time()
            self.has_wall_time = True
        if j:
            names      = j["names"]
            positions  = j["positions"]
            velocities = j["velocities"]
            efforts    = j["efforts"]
            # Use first joint entry (blade_rotation_joint is the only one)
            name     = names[0]      if names      else ""
            position = positions[0]  if positions  else float("nan")
            velocity = velocities[0] if velocities else float("nan")
            effort   = efforts[0]    if efforts    else float("nan")
            self._csv_writer.writerow([
                f"{j['ros_time']:.6f}",
                f"{j['wall_time']:.6f}",
                f"{j['ros_time'] - self._start_wall_time:.6f}",
                name,
                f"{position:.6f}",
                f"{velocity:.6f}",
                f"{effort:.6f}",
                f"{self._latest_commanded_position:.6f}",
                filename,
            ])
        else:
            # No joint state received yet — write placeholders
            wall_now = time.time()
            self._csv_writer.writerow([
                f"{img_ros_time:.6f}", f"{wall_now:.6f}",
                f"{wall_now - self._start_wall_time:.6f}",
                "", "nan", "nan", "nan",
                f"{self._latest_commanded_position:.6f}", filename,
            ])

        self._csv_file.flush()

        if self._image_count % 100 == 0:
            self.get_logger().info(f"Recorded {self._image_count} frames so far.")

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def _save_video(self):
        """Assemble all saved PNG frames into an MP4 in chronological order."""
        if self._image_count == 0:
            self.get_logger().warn("No frames recorded — skipping video creation.")
            return

        # Calculate FPS from actual frame timestamps
        if len(self._frame_timestamps) >= 2:
            duration = self._frame_timestamps[-1] - self._frame_timestamps[0]
            fps = (len(self._frame_timestamps) - 1) / duration if duration > 0 else 30.0
        else:
            fps = 30.0
        fps = round(fps, 2)

        video_path = os.path.join(self.output_dir, "recording.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, self._frame_size)

        # Sort frames by filename (which sorts chronologically)
        frames = sorted(os.listdir(self.images_dir))
        for fname in frames:
            if not fname.endswith(".png"):
                continue
            frame = cv2.imread(os.path.join(self.images_dir, fname))
            if frame is not None:
                writer.write(frame)

        writer.release()
        self.get_logger().info(
            f"Video saved to: {video_path}  ({self._image_count} frames @ {fps:.2f} fps)"
        )

    def destroy_node(self):
        self._csv_file.close()
        self.get_logger().info(
            f"Recording stopped. "
            f"Saved {self._image_count} frames and joint state data to: {self.output_dir}"
        )
        self._save_video()
        super().destroy_node()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    _test_data_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")
    parser = argparse.ArgumentParser(description="Record joint state and camera from simulation.")
    parser.add_argument(
        "--output-dir",
        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        help=(
            "Sub-directory name inside scripts/test_data/ to write CSV and images into "
            "(default: <timestamp>)"
        ),
    )
    args, remaining = parser.parse_known_args()
    args.output_dir = os.path.join(_test_data_base, args.output_dir)

    rclpy.init(args=remaining)
    node = RecorderNode(output_dir=args.output_dir)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
