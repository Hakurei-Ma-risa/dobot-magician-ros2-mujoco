"""Turn portable 2-D detections plus aligned depth into 3-D scene objects."""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Lock

import numpy as np
import rclpy
from mani_interfaces.msg import SceneObject, SceneObjectArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass(frozen=True)
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def decode_depth_m(message: Image) -> np.ndarray:
    if message.encoding not in ("16UC1", "mono16"):
        raise ValueError(f"expected 16-bit depth image, got {message.encoding!r}")
    byte_order = ">u2" if message.is_bigendian else "<u2"
    row_values = message.step // 2
    depth_mm = np.frombuffer(message.data, dtype=byte_order).reshape(
        message.height, row_values
    )[:, : message.width]
    return depth_mm.astype(np.float64) * 0.001


def robust_roi_depth(
    depth_m: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    *,
    inner_fraction: float = 0.5,
) -> float | None:
    x, y, width, height = bbox_xywh
    if width <= 0 or height <= 0:
        return None
    inset_x = int(round(width * (1.0 - inner_fraction) * 0.5))
    inset_y = int(round(height * (1.0 - inner_fraction) * 0.5))
    x0 = max(0, x + inset_x)
    y0 = max(0, y + inset_y)
    x1 = min(depth_m.shape[1], x + width - inset_x)
    y1 = min(depth_m.shape[0], y + height - inset_y)
    if x0 >= x1 or y0 >= y1:
        return None
    samples = depth_m[y0:y1, x0:x1]
    valid = samples[np.isfinite(samples) & (samples > 0.05) & (samples < 5.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def optical_point(
    u: float, v: float, depth_z_m: float, intrinsics: PinholeIntrinsics
) -> np.ndarray:
    return np.array(
        [
            (u - intrinsics.cx) * depth_z_m / intrinsics.fx,
            (v - intrinsics.cy) * depth_z_m / intrinsics.fy,
            depth_z_m,
        ],
        dtype=np.float64,
    )


def quaternion_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("transform quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def localize_bbox(
    bbox_xywh: tuple[int, int, int, int],
    depth_z_m: float,
    intrinsics: PinholeIntrinsics,
    translation_target_from_camera: np.ndarray,
    rotation_target_from_camera: np.ndarray,
    *,
    support_plane_z_m: float | None,
    object_height_m: float,
    support_anchor: str = "center",
    support_footprint_radius_m: float = 0.0,
) -> np.ndarray:
    x, y, width, height = bbox_xywh
    u = x + (width - 1) * 0.5
    v = y + (height - 1) * 0.5
    measured_camera = optical_point(u, v, depth_z_m, intrinsics)
    measured_target = (
        rotation_target_from_camera @ measured_camera
        + translation_target_from_camera
    )
    if support_plane_z_m is None or object_height_m <= 0.0:
        return measured_target

    center_z = support_plane_z_m + object_height_m * 0.5
    # Use the support-plane prior only while the object is plausibly on the
    # table. Once depth observes it clearly above the tabletop, preserve the
    # measured 3-D point so lift verification remains vision-based.
    release_height = support_plane_z_m + object_height_m + 0.040
    if measured_target[2] > release_height:
        return measured_target
    if support_anchor == "bbox_bottom":
        # If the upper object is hidden by the tool, the visible ROI center is
        # below the physical center. Its bottom edge approximates the contact
        # point on the known tabletop instead.
        ray_v = y + height - 1
        ray_plane_z = support_plane_z_m
    elif support_anchor == "center":
        ray_v = v
        ray_plane_z = center_z
    else:
        raise ValueError(f"unknown support anchor: {support_anchor}")
    ray_camera = optical_point(u, ray_v, 1.0, intrinsics)
    ray_target = rotation_target_from_camera @ ray_camera
    if abs(ray_target[2]) < 1e-9:
        return measured_target
    scale = (ray_plane_z - translation_target_from_camera[2]) / ray_target[2]
    if scale <= 0.0:
        return measured_target
    center = translation_target_from_camera + scale * ray_target
    center[2] = center_z
    if support_anchor == "bbox_bottom" and support_footprint_radius_m > 0.0:
        # The lowest visible pixel is typically the near rim of a tabletop
        # footprint. Move one footprint radius away from the camera to recover
        # the object's center in the support plane.
        away_from_camera = center[:2] - translation_target_from_camera[:2]
        distance = float(np.linalg.norm(away_from_camera))
        if distance > 1e-9:
            center[:2] += (
                away_from_camera / distance * support_footprint_radius_m
            )
    return center


def localize_roi_points(
    depth_m: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    intrinsics: PinholeIntrinsics,
    translation_target_from_camera: np.ndarray,
    rotation_target_from_camera: np.ndarray,
    *,
    support_plane_z_m: float,
    object_height_m: float,
) -> np.ndarray | None:
    """Estimate a tabletop object's center from its visible depth footprint.

    The bbox may include background. Project the ROI, retain points in the
    expected object-height band, then find the horizontal footprint midpoint.
    This is especially useful for blocks viewed obliquely by a base camera.
    """
    x, y, width, height = bbox_xywh
    x0, y0 = max(0, x), max(0, y)
    x1 = min(depth_m.shape[1], x + width)
    y1 = min(depth_m.shape[0], y + height)
    if x0 >= x1 or y0 >= y1 or object_height_m <= 0.0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    zz = depth_m[y0:y1, x0:x1]
    valid = np.isfinite(zz) & (zz > 0.05) & (zz < 5.0)
    if not np.any(valid):
        return None
    zz = zz[valid]
    optical = np.stack(
        (
            (xx[valid] - intrinsics.cx) * zz / intrinsics.fx,
            (yy[valid] - intrinsics.cy) * zz / intrinsics.fy,
            zz,
        ),
        axis=0,
    )
    points = (rotation_target_from_camera @ optical).T + translation_target_from_camera
    object_points = points[
        (points[:, 2] >= support_plane_z_m + 0.005)
        & (points[:, 2] <= support_plane_z_m + object_height_m + 0.015)
    ]
    if object_points.shape[0] < 20:
        return None
    bounds = np.quantile(object_points[:, :2], [0.01, 0.99], axis=0)
    return np.array(
        [
            float(np.mean(bounds[:, 0])),
            float(np.mean(bounds[:, 1])),
            support_plane_z_m + object_height_m * 0.5,
        ]
    )


class RgbdLocalizer(Node):
    def __init__(self) -> None:
        super().__init__("mani_rgbd_localizer")
        self.declare_parameter(
            "detections_topic", "/mani/perception/detections_2d"
        )
        self.declare_parameter(
            "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info"
        )
        self.declare_parameter("output_topic", "/mani/scene_objects")
        self.declare_parameter("target_frame", "magician_base_link")
        self.declare_parameter("support_plane_z_m", 0.05)
        self.declare_parameter("use_support_plane", True)
        self.declare_parameter("support_anchor", "center")
        self.declare_parameter("localization_mode", "bbox")

        self._lock = Lock()
        self._depth: Image | None = None
        self._info: CameraInfo | None = None
        self._detections: SceneObjectArray | None = None
        self._last_key: tuple[int, int, int, int] | None = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._publisher = self.create_publisher(
            SceneObjectArray, str(self.get_parameter("output_topic").value), 10
        )
        self._depth_sub = self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._on_depth,
            qos_profile_sensor_data,
        )
        self._info_sub = self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._on_info,
            qos_profile_sensor_data,
        )
        self._detections_sub = self.create_subscription(
            SceneObjectArray,
            str(self.get_parameter("detections_topic").value),
            self._on_detections,
            10,
        )
        self._timer = self.create_timer(0.05, self._process)
        self.get_logger().info("Portable RGB-D localizer is ready")

    def _on_depth(self, message: Image) -> None:
        with self._lock:
            self._depth = message

    def _on_info(self, message: CameraInfo) -> None:
        with self._lock:
            self._info = message

    def _on_detections(self, message: SceneObjectArray) -> None:
        with self._lock:
            self._detections = message

    def _process(self) -> None:
        with self._lock:
            depth = self._depth
            info = self._info
            detections = self._detections
        if depth is None or info is None or detections is None:
            return
        key = (
            depth.header.stamp.sec,
            depth.header.stamp.nanosec,
            detections.header.stamp.sec,
            detections.header.stamp.nanosec,
        )
        if key == self._last_key:
            return

        source_frame = depth.header.frame_id or info.header.frame_id
        target_frame = str(self.get_parameter("target_frame").value)
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame, source_frame, Time()
            )
        except TransformException as exc:
            self.get_logger().warning(f"RGB-D transform unavailable: {exc}")
            return

        try:
            depth_m = decode_depth_m(depth)
        except (ValueError, TypeError) as exc:
            self.get_logger().error(str(exc))
            return
        intrinsics = PinholeIntrinsics(
            fx=float(info.k[0]),
            fy=float(info.k[4]),
            cx=float(info.k[2]),
            cy=float(info.k[5]),
        )
        translation_message = transform.transform.translation
        translation = np.array(
            [translation_message.x, translation_message.y, translation_message.z],
            dtype=np.float64,
        )
        rotation_message = transform.transform.rotation
        rotation = quaternion_matrix_xyzw(
            np.array(
                [
                    rotation_message.x,
                    rotation_message.y,
                    rotation_message.z,
                    rotation_message.w,
                ]
            )
        )
        use_plane = bool(self.get_parameter("use_support_plane").value)
        support_plane = (
            float(self.get_parameter("support_plane_z_m").value)
            if use_plane
            else None
        )

        output = SceneObjectArray()
        output.header.stamp = depth.header.stamp
        output.header.frame_id = target_frame
        for detection in detections.objects:
            roi = detection.bbox_2d
            bbox = (
                int(roi.x_offset),
                int(roi.y_offset),
                int(roi.width),
                int(roi.height),
            )
            depth_value = robust_roi_depth(depth_m, bbox)
            if depth_value is None:
                continue
            position = None
            if (
                str(self.get_parameter("localization_mode").value) == "roi_points"
                and support_plane is not None
            ):
                position = localize_roi_points(
                    depth_m,
                    bbox,
                    intrinsics,
                    translation,
                    rotation,
                    support_plane_z_m=support_plane,
                    object_height_m=float(detection.size.z),
                )
            if position is None:
                position = localize_bbox(
                    bbox,
                    depth_value,
                    intrinsics,
                    translation,
                    rotation,
                    support_plane_z_m=support_plane,
                    object_height_m=float(detection.size.z),
                    support_anchor=str(self.get_parameter("support_anchor").value),
                    support_footprint_radius_m=(
                        0.5 * min(float(detection.size.x), float(detection.size.y))
                        if str(self.get_parameter("support_anchor").value)
                        == "bbox_bottom"
                        else 0.0
                    ),
                )
            item = SceneObject()
            item.header = output.header
            item.uuid = detection.uuid
            item.class_id = detection.class_id
            item.confidence = detection.confidence
            item.pose.pose.position.x = float(position[0])
            item.pose.pose.position.y = float(position[1])
            item.pose.pose.position.z = float(position[2])
            item.pose.pose.orientation.w = 1.0
            item.size = detection.size
            item.bbox_2d = detection.bbox_2d
            item.source = "rgbd_bbox_depth"
            output.objects.append(item)
        self._publisher.publish(output)
        self._last_key = key


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RgbdLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
