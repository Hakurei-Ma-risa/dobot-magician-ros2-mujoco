"""Localize one Gemini ER target through the live MuJoCo ROS RGB-D stream."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image as PilImage
import rclpy
from mani_interfaces.msg import SceneObject, SceneObjectArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .api import query_image
from .er_client import normalized_bbox_to_xywh, normalized_yx_to_pixel


TARGET_UUID = "gemini-er-target-0"


def decode_rgb(message: Image) -> np.ndarray:
    """Decode an RGB8/BGR8 ROS image, honoring padded row strides."""

    if message.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported color encoding: {message.encoding}")
    if message.step < message.width * 3:
        raise ValueError("image row stride is shorter than width * 3")
    expected_bytes = message.height * message.step
    if len(message.data) < expected_bytes:
        raise ValueError("image data is shorter than height * step")
    rows = np.frombuffer(message.data, dtype=np.uint8, count=expected_bytes).reshape(
        message.height, message.step
    )
    rgb = rows[:, : message.width * 3].reshape(message.height, message.width, 3)
    if message.encoding == "bgr8":
        rgb = rgb[:, :, ::-1]
    return np.ascontiguousarray(rgb)


class GeminiSimProbe(Node):
    def __init__(self) -> None:
        super().__init__("mani_gemini_sim_probe")
        self.image: Image | None = None
        self.localized: SceneObject | None = None
        self.ground_truth: SceneObject | None = None
        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self._on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            SceneObjectArray,
            "/mani/scene_objects",
            self._on_scene,
            10,
        )
        self.create_subscription(
            SceneObjectArray,
            "/mani/sim/ground_truth",
            self._on_truth,
            10,
        )
        self.detections = self.create_publisher(
            SceneObjectArray, "/mani/perception/detections_2d", 10
        )

    def _on_image(self, message: Image) -> None:
        self.image = message

    def _on_scene(self, message: SceneObjectArray) -> None:
        self.localized = next(
            (item for item in message.objects if item.uuid == TARGET_UUID), None
        )

    def _on_truth(self, message: SceneObjectArray) -> None:
        self.ground_truth = next(iter(message.objects), None)

    def wait_for_sim_image(self, timeout_sec: float) -> Image:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.image is None:
                continue
            publishers = self.get_publishers_info_by_topic(
                "/camera/camera/color/image_raw"
            )
            if publishers and any(
                endpoint.node_name != "mani_mujoco_server"
                for endpoint in publishers
            ):
                raise RuntimeError("camera topic has a non-MuJoCo publisher")
            if publishers:
                detection_publishers = self.get_publishers_info_by_topic(
                    "/mani/perception/detections_2d"
                )
                if any(
                    endpoint.node_name != self.get_name()
                    for endpoint in detection_publishers
                ):
                    raise RuntimeError(
                        "another detector is publishing; launch MuJoCo with "
                        "publish_perfect_detections:=false"
                    )
                return self.image
        raise TimeoutError("MuJoCo RGB image was not received")

    def publish_detection(
        self,
        bbox_xywh: tuple[int, int, int, int],
        confidence: float,
        *,
        frame_id: str,
        object_height_m: float,
    ) -> None:
        x, y, width, height = bbox_xywh
        detections = SceneObjectArray()
        detections.header.stamp = self.get_clock().now().to_msg()
        detections.header.frame_id = frame_id
        item = SceneObject()
        item.header = detections.header
        item.uuid = TARGET_UUID
        item.class_id = "cylinder"
        item.confidence = confidence
        item.pose.pose.orientation.w = 1.0
        item.size.x = 0.024
        item.size.y = 0.024
        item.size.z = object_height_m
        item.bbox_2d.x_offset = x
        item.bbox_2d.y_offset = y
        item.bbox_2d.width = width
        item.bbox_2d.height = height
        item.source = "gemini_robotics_er_2"
        detections.objects = [item]
        self.detections.publish(detections)

    def clear_detection(self, frame_id: str) -> None:
        empty = SceneObjectArray()
        empty.header.stamp = self.get_clock().now().to_msg()
        empty.header.frame_id = frame_id
        self.detections.publish(empty)


def _position(item: SceneObject) -> np.ndarray:
    p = item.pose.pose.position
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="Locate the yellow vertical cylinder on the table, even if the gripper partly hides it. Return a tight box around the visible yellow pixels and their center.",
    )
    parser.add_argument("--model", default="gemini-robotics-er-2-preview")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--object-height-m", type=float, default=0.05)
    parser.add_argument("--wait-sec", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = _args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")
    rclpy.init()
    node = GeminiSimProbe()
    frame_id = "camera_color_optical_frame"
    detection_published = False
    try:
        image = node.wait_for_sim_image(args.wait_sec)
        frame_id = image.header.frame_id or frame_id
        rgb = decode_rgb(image)
        with tempfile.TemporaryDirectory(prefix="gemini_er_sim_") as directory:
            image_path = Path(directory) / "frame.png"
            PilImage.fromarray(rgb).save(image_path)
            try:
                proposal = query_image(
                    image_path, args.task, model=args.model, api_key=api_key
                )
            except Exception as exc:
                status = getattr(exc, "code", "unknown")
                raise RuntimeError(
                    f"Gemini API failed: {type(exc).__name__} (status {status})"
                ) from None

        if proposal.label.lower() == "none":
            raise ValueError("Gemini did not find a target")
        if proposal.confidence is None or proposal.confidence < args.min_confidence:
            raise ValueError("Gemini confidence is below the requested minimum")
        if proposal.point_yx_norm is None or proposal.bbox_yxyx_norm is None:
            raise ValueError("Gemini response is missing a point or bounding box")
        pixel_xy = normalized_yx_to_pixel(
            proposal.point_yx_norm, image.width, image.height
        )
        bbox = normalized_bbox_to_xywh(
            proposal.bbox_yxyx_norm, image.width, image.height
        )
        x, y, width, height = bbox
        if not (
            x - 2 <= pixel_xy[0] <= x + width + 2
            and y - 2 <= pixel_xy[1] <= y + height + 2
        ):
            raise ValueError("Gemini point falls outside its bounding box")

        deadline = time.monotonic() + args.wait_sec
        while rclpy.ok() and time.monotonic() < deadline:
            node.publish_detection(
                bbox,
                proposal.confidence,
                frame_id=frame_id,
                object_height_m=args.object_height_m,
            )
            detection_published = True
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.localized is not None:
                break
        if node.localized is None:
            raise TimeoutError("RGB-D localizer did not produce the ER target")
        estimate = _position(node.localized)
        report = {
            "success": True,
            "model": args.model,
            "label": proposal.label,
            "confidence": proposal.confidence,
            "image_size": [image.width, image.height],
            "point_pixel_xy": pixel_xy,
            "bbox_pixel_xywh": bbox,
            "estimated_xyz_m": estimate.round(5).tolist(),
        }
        if node.ground_truth is not None:
            truth = _position(node.ground_truth)
            report["sim_truth_xyz_m"] = truth.round(5).tolist()
            report["localization_error_mm"] = round(
                float(np.linalg.norm(estimate - truth) * 1000.0), 2
            )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    finally:
        if detection_published:
            node.clear_detection(frame_id)
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
