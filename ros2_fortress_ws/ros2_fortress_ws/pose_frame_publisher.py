#!/usr/bin/env python3
"""
Republish PoseArray as markers for RViz visualization.
Converts Gazebo poses to RViz-compatible marker messages.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class PoseFramePublisher(Node):
    def __init__(self):
        super().__init__('pose_frame_publisher')
        
        # QoS profile for reliability
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribe to the PoseArray topic
        self.subscription = self.create_subscription(
            PoseArray,
            '/world/empty_world/pose/info',
            self.pose_array_callback,
            qos_profile
        )
        
        # Publisher for grader pose (PoseStamped with frame ID)
        self.grader_pub = self.create_publisher(PoseStamped, '/grader/pose', qos_profile)
        
        # Publisher for markers (for RViz visualization)
        self.marker_pub = self.create_publisher(MarkerArray, '/grader/markers', qos_profile)
        
        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Publish a static root frame for RViz to use
        static_tf_broadcaster = StaticTransformBroadcaster(self)
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = ''  # Root/empty frame
        static_transform.child_frame_id = 'world'
        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.0
        static_transform.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        static_tf_broadcaster.sendTransform(static_transform)
        
        self.get_logger().info('Pose frame publisher initialized')
    
    def pose_array_callback(self, msg: PoseArray):
        """Process pose array and republish with proper frame IDs."""
        if not msg.poses or len(msg.poses) < 2:
            return
        
        # The grader is typically at index 1 in the pose array
        grader_pose = msg.poses[1]
        
        # Use wall clock time for RViz2 compatibility (not Gazebo sim time)
        now = self.get_clock().now().to_msg()
        
        # 1. Republish as PoseStamped with proper frame
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = now
        pose_stamped.header.frame_id = 'world'
        pose_stamped.pose = grader_pose
        self.grader_pub.publish(pose_stamped)
        
        # 2. Create marker for visualization
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = now
        marker.ns = 'grader'
        marker.id = 0
        marker.type = Marker.SPHERE  # Use a sphere to represent the grader
        marker.action = Marker.ADD
        marker.pose = grader_pose
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 0.7
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)
        
        # 3. Broadcast TF transform
        transform = TransformStamped()
        transform.header.stamp = now
        transform.header.frame_id = 'world'
        transform.child_frame_id = 'grader'
        transform.transform.translation.x = grader_pose.position.x
        transform.transform.translation.y = grader_pose.position.y
        transform.transform.translation.z = grader_pose.position.z
        transform.transform.rotation = grader_pose.orientation
        
        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = PoseFramePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


