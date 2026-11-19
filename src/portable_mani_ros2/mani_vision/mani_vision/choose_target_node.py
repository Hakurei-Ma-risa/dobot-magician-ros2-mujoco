"""ROS node for module 3: instruction + candidates in, ONE target out.

In:  /mani/vision/candidates   (from find_objects)
     /camera/camera/color/image_raw
     /mani/vision/instruction  (std_msgs/String, e.g. "the red can")
Out: /mani/perception/detections_2d  (the chosen box, uuid "target"; this is
                                      the input of rgbd_localizer, module 4)
     /mani/vision/choice        (std_msgs/String, JSON: index, reason, ...)
     /mani/vision/marked_image  (numbered boxes; view with rqt_image_view)

After choosing, the node keeps following the target from frame to frame so
the pick task can check that the object really went up.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import rclpy
from mani_interfaces.msg import SceneObject, SceneObjectArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2

from mani_vision.selectors import Box, Choice, RuleSelector, draw_marks, hue_pixel_count, make_selector, object_hue


def image_to_rgb(message: Image) -> np.ndarray:
    channels = 4 if message.encoding in ("rgba8", "bgra8") else 3
    data = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
    image = data[:, : message.width * channels].reshape(message.height, message.width, channels)
    if message.encoding.startswith("bgr"):
        image = image[..., [2, 1, 0]]
    return np.ascontiguousarray(image[..., :3])


def to_boxes(candidates: SceneObjectArray) -> list[Box]:
    boxes = []
    for obj in candidates.objects:
        index = int(obj.uuid.split("-")[-1]) if obj.uuid.startswith("cand-") else len(boxes)
        p, s, r = obj.pose.pose.position, obj.size, obj.bbox_2d
        boxes.append(
            Box(index, (int(r.x_offset), int(r.y_offset), int(r.width), int(r.height)),
                (p.x, p.y, p.z), (s.x, s.y, s.z))
        )
    return boxes


def bottom_z(obj: SceneObject) -> float:
    return float(obj.pose.pose.position.z - 0.5 * obj.size.z)


def follow(candidates: SceneObjectArray, last_xy: tuple[float, float], radius_m: float) -> SceneObject | None:
    """Same object in a new frame: nearby in x/y, and the LOWEST one there.

    Lowest matters: if the gripper rises without the object, the object (still
    on the table) is lower than the gripper, so we keep tracking the object
    and the pick check correctly fails.
    """

    near = [
        obj for obj in candidates.objects
        if np.hypot(obj.pose.pose.position.x - last_xy[0], obj.pose.pose.position.y - last_xy[1]) < radius_m
    ]
    return min(near, key=bottom_z) if near else None


class ChooseTarget(Node):
    def __init__(self) -> None:
        super().__init__("mani_choose_target")
        p = self.declare_parameter
        p("selector", "rule")  # rule | deepseek
        p("candidates_topic", "/mani/vision/candidates")
        p("image_topic", "/camera/camera/color/image_raw")
        p("output_topic", "/mani/perception/detections_2d")
        p("target_uuid", "target")
        p("follow_radius_m", 0.035)
        self.selector = make_selector(str(self.get_parameter("selector").value))
        self._lock = threading.Lock()
        self._image: Image | None = None
        self._candidates: SceneObjectArray | None = None
        self._pending: tuple[str, float] | None = None  # (instruction, wall time)
        self._busy = False
        self._target: SceneObject | None = None  # what we send downstream
        self._last_xy: tuple[float, float] | None = None
        self._hue: float | None = None  # target hue, checked while following
        self._detections = self.create_publisher(
            SceneObjectArray, str(self.get_parameter("output_topic").value), 10
        )
        self._choice = self.create_publisher(String, "/mani/vision/choice", 10)
        self._marked = self.create_publisher(Image, "/mani/vision/marked_image", 2)
        self.create_subscription(
            Image, str(self.get_parameter("image_topic").value), self._on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            SceneObjectArray, str(self.get_parameter("candidates_topic").value),
            self._on_candidates, 10,
        )
        self.create_subscription(String, "/mani/vision/instruction", self._on_instruction, 10)
        self.create_timer(0.1, self._maybe_choose)
        self.get_logger().info(
            f"choose_target ready (selector={self.selector.name}); "
            "publish an instruction on /mani/vision/instruction"
        )

    # ------------------------------------------------------------ inputs
    def _on_image(self, message: Image) -> None:
        with self._lock:
            self._image = message

    def _on_instruction(self, message: String) -> None:
        text = message.data.strip()
        with self._lock:
            if text.lower() in ("", "stop", "clear"):
                self._pending, self._target, self._last_xy = None, None, None
                self.get_logger().info("target cleared")
                return
            self._pending = (text, time.time())
            # Forget the old target at once, so nobody picks it by mistake.
            self._target, self._last_xy = None, None
            empty = SceneObjectArray()
            empty.header.stamp = self.get_clock().now().to_msg()
            self._detections.publish(empty)
        self.get_logger().info(f"instruction: {text!r} (waiting for a fresh frame)")

    def _on_candidates(self, message: SceneObjectArray) -> None:
        with self._lock:
            self._candidates = message
            if self._target is None or self._last_xy is None:
                return
            seen = follow(message, self._last_xy, float(self.get_parameter("follow_radius_m").value))
            if seen is None:
                return  # lost for this frame; keep the last box
            if self._hue is not None and self._image is not None:
                # A coloured target must still show its colour in the box. If
                # only the (grey) gripper is there, the object was not picked
                # up. Count pixels of the target's hue: a small object held in
                # a big gripper still gives dozens of them, the gripper none.
                r = seen.bbox_2d
                count = hue_pixel_count(
                    image_to_rgb(self._image), (r.x_offset, r.y_offset, r.width, r.height), self._hue
                )
                if count < 30:
                    return
            self._last_xy = (seen.pose.pose.position.x, seen.pose.pose.position.y)
            self._target.bbox_2d = seen.bbox_2d
            self._target.header = message.header
            self._publish_target()

    def _publish_target(self) -> None:
        out = SceneObjectArray()
        out.header = self._target.header
        out.objects.append(self._target)
        self._detections.publish(out)

    # ------------------------------------------------------------ choosing
    def _maybe_choose(self) -> None:
        with self._lock:
            if self._pending is None or self._busy or self._candidates is None or self._image is None:
                return
            text, asked_at = self._pending
            stamp = self._candidates.header.stamp
            if stamp.sec + stamp.nanosec * 1e-9 < asked_at + 0.2:
                return  # wait for a frame taken after the instruction
            candidates, image = self._candidates, self._image
            self._pending, self._busy = None, True
        threading.Thread(target=self._choose, args=(text, candidates, image), daemon=True).start()

    def _choose(self, text: str, candidates: SceneObjectArray, image_msg: Image) -> None:
        try:
            image = image_to_rgb(image_msg)
            boxes = to_boxes(candidates)
            try:
                choice = self.selector.select(text, image, boxes)
            except Exception as exc:  # network, key, quota ...
                # Never leave the participant stuck: fall back to the free,
                # offline rule chooser and say so.
                self.get_logger().error(f"{self.selector.name} failed ({exc}); using the rule chooser")
                choice = RuleSelector().select(text, image, boxes)
                choice = Choice(choice.index, f"[rule fallback] {choice.reason}", choice.seconds)
            self._report(text, choice.index, choice.reason, boxes, image,
                         choice.seconds, choice.input_tokens, choice.output_tokens)
            if choice.index < 0:
                return
            chosen = next(obj for obj in candidates.objects if obj.uuid == f"cand-{choice.index}")
            target = SceneObject()
            target.header = candidates.header
            target.uuid = str(self.get_parameter("target_uuid").value)
            target.class_id = text
            target.confidence = 1.0
            target.size = chosen.size  # object size measured while resting
            target.pose.pose.orientation = chosen.pose.pose.orientation  # footprint yaw
            target.bbox_2d = chosen.bbox_2d
            target.source = f"choose_target/{self.selector.name}"
            r = chosen.bbox_2d
            hue = object_hue(image, (r.x_offset, r.y_offset, r.width, r.height))
            with self._lock:
                self._hue = hue
                self._target = target
                self._last_xy = (chosen.pose.pose.position.x, chosen.pose.pose.position.y)
                self._publish_target()
        finally:
            with self._lock:
                self._busy = False

    def _report(self, text, index, reason, boxes, image, seconds, tokens_in, tokens_out) -> None:
        payload = {
            "instruction": text,
            "selector": self.selector.name,
            "index": index,
            "uuid": str(self.get_parameter("target_uuid").value) if index >= 0 else "",
            "reason": reason,
            "candidates": len(boxes),
            "seconds": round(seconds, 2),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
        }
        self._choice.publish(String(data=json.dumps(payload)))
        level = self.get_logger().info if index >= 0 else self.get_logger().warning
        level(f"choice {index} of {len(boxes)}: {reason}")
        marked = draw_marks(image, boxes)
        for box in boxes:
            if box.index == index:
                x, y, w, h = box.bbox_xywh
                cv2.rectangle(marked, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 0, 0), 3)
        out = Image()
        out.header.frame_id = "camera_color_optical_frame"
        out.height, out.width = marked.shape[:2]
        out.encoding, out.step = "rgb8", marked.shape[1] * 3
        out.data = marked.tobytes()
        self._marked.publish(out)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ChooseTarget()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
