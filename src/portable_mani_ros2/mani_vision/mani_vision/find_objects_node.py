"""ROS node for module 2: depth image in, candidate boxes out.

In:  aligned depth + CameraInfo (+ TF camera -> robot base)
Out: /mani/vision/candidates  (mani_interfaces/SceneObjectArray)
     uuid "cand-<n>", class_id "object", bbox_2d, pose (base frame), size
"""

from __future__ import annotations

from threading import Lock

import numpy as np
import rclpy
from mani_interfaces.msg import SceneObject, SceneObjectArray
from mani_perception.rgbd_localizer import decode_depth_m, quaternion_matrix_xyzw
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from mani_vision.proposals import FindConfig, find_objects


class FindObjects(Node):
    def __init__(self) -> None:
        super().__init__("mani_find_objects")
        p = self.declare_parameter
        p("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        p("camera_info_topic", "/camera/camera/color/camera_info")
        p("output_topic", "/mani/vision/candidates")
        p("target_frame", "magician_base_link")
        p("rate_hz", 5.0)
        p("table_z_m", -1.0)  # < -0.5 means: estimate from depth
        p("min_height_m", 0.008)
        p("max_height_m", 0.20)
        p("ring_min_m", 0.12)
        p("ring_max_m", 0.34)
        p("min_area_px", 150)
        self._lock = Lock()
        self._depth: Image | None = None
        self._info: CameraInfo | None = None
        self._last_stamp = None
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._pub = self.create_publisher(
            SceneObjectArray, str(self.get_parameter("output_topic").value), 10
        )
        self.create_subscription(
            Image, str(self.get_parameter("depth_topic").value), self._on_depth,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo, str(self.get_parameter("camera_info_topic").value), self._on_info,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)
        self.get_logger().info("find_objects ready: depth -> /mani/vision/candidates")

    def _on_depth(self, message: Image) -> None:
        with self._lock:
            self._depth = message

    def _on_info(self, message: CameraInfo) -> None:
        with self._lock:
            self._info = message

    def _config(self) -> FindConfig:
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        table = float(g("table_z_m"))
        return FindConfig(
            min_height_m=float(g("min_height_m")),
            max_height_m=float(g("max_height_m")),
            ring_min_m=float(g("ring_min_m")),
            ring_max_m=float(g("ring_max_m")),
            min_area_px=int(g("min_area_px")),
            table_z_m=None if table < -0.5 else table,
        )

    def _tick(self) -> None:
        with self._lock:
            depth, info = self._depth, self._info
        if depth is None or info is None:
            return
        stamp = (depth.header.stamp.sec, depth.header.stamp.nanosec)
        if stamp == self._last_stamp:
            return
        target = str(self.get_parameter("target_frame").value)
        source = depth.header.frame_id or info.header.frame_id
        try:
            transform = self._tf.lookup_transform(target, source, Time())
        except TransformException as exc:
            self.get_logger().warning(f"camera TF not available yet: {exc}", throttle_duration_sec=5.0)
            return
        t = transform.transform.translation
        q = transform.transform.rotation
        rotation = quaternion_matrix_xyzw(np.array([q.x, q.y, q.z, q.w]))
        try:
            table_z, candidates = find_objects(
                decode_depth_m(depth),
                float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5]),
                rotation, np.array([t.x, t.y, t.z]), self._config(),
            )
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
            return
        out = SceneObjectArray()
        out.header.stamp = depth.header.stamp
        out.header.frame_id = target
        for item in candidates:
            obj = SceneObject()
            obj.header = out.header
            obj.uuid = f"cand-{item.index}"
            obj.class_id = "object"
            obj.confidence = 1.0
            obj.pose.pose.position.x, obj.pose.pose.position.y, obj.pose.pose.position.z = item.center_m
            obj.pose.pose.orientation.z = float(np.sin(0.5 * item.yaw_rad))
            obj.pose.pose.orientation.w = float(np.cos(0.5 * item.yaw_rad))
            obj.size.x, obj.size.y, obj.size.z = item.size_m
            x, y, w, h = item.bbox_xywh
            obj.bbox_2d.x_offset, obj.bbox_2d.y_offset = x, y
            obj.bbox_2d.width, obj.bbox_2d.height = w, h
            obj.source = f"depth_above_table table_z={table_z:.3f}"
            out.objects.append(obj)
        self._pub.publish(out)
        self._last_stamp = stamp


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FindObjects()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
