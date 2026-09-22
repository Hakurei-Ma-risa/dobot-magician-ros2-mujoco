"""Interactive, camera-only Gemini ER commands for the MuJoCo clutter scene."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image as PilImage
import rclpy
from mani_interfaces.action import MoveToPose, PickObject
from mani_interfaces.msg import SceneObject, SceneObjectArray
from mani_interfaces.srv import CommandEndEffector, ResetScene
from mani_perception.rgbd_localizer import (
    PinholeIntrinsics,
    decode_depth_m,
    optical_point,
    quaternion_matrix_xyzw,
)
from mani_core import Pose
from mani_dobot.kinematics import inverse_tcp_pose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from .api import describe_image, query_image
from .er_client import normalized_bbox_to_xywh, normalized_yx_to_pixel
from .ros_sim_probe import decode_rgb


TABLE_Z_M = 0.05


def size_from_label(label: str) -> tuple[str, tuple[float, float, float]]:
    """Known tabletop object dimensions, not simulator pose or identity."""

    name = label.lower()
    if any(token in name for token in ("cube", "block", "方块", "立方")):
        return "block", (0.036, 0.036, 0.036)
    if any(token in name for token in ("can", "罐")):
        return "can", (0.032, 0.032, 0.090)
    if any(token in name for token in ("cylinder", "圆柱")):
        return "cylinder", (0.024, 0.024, 0.050)
    raise ValueError(f"unsupported object type in Gemini label: {label!r}")


class SimChat(Node):
    def __init__(self, *, api_key: str, model: str) -> None:
        super().__init__("mani_gemini_sim_chat")
        self.api_key = api_key
        self.model = model
        self.image: Image | None = None
        self.depth: Image | None = None
        self.camera_info: CameraInfo | None = None
        self.localized: SceneObject | None = None
        self.target_uuid: str | None = None
        self.target_class = ""
        self.target_size = (0.0, 0.0, 0.0)
        self.target_confidence = 0.0
        self.tracker = None
        self.track_failures = 0
        self.held_label: str | None = None
        self.held_size: tuple[float, float, float] | None = None
        self.detections = self.create_publisher(
            SceneObjectArray, "/mani/perception/detections_2d", 10
        )
        self.create_subscription(
            Image, "/camera/camera/color/image_raw", self._on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw",
            self._on_depth, qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info",
            self._on_camera_info, qos_profile_sensor_data,
        )
        self.create_subscription(
            SceneObjectArray, "/mani/scene_objects", self._on_scene, 10
        )
        self._pick = ActionClient(self, PickObject, "/mani/pick_object")
        self._move = ActionClient(self, MoveToPose, "/mani/move_to_pose")
        self._gripper = self.create_client(
            CommandEndEffector, "/mani/command_end_effector"
        )
        self._reset = self.create_client(ResetScene, "/mani/sim/reset_scene")
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)

    def _on_image(self, message: Image) -> None:
        self.image = message
        if self.tracker is None or self.target_uuid is None:
            return
        bgr = cv2.cvtColor(decode_rgb(message), cv2.COLOR_RGB2BGR)
        found, box = self.tracker.update(bgr)
        if not found:
            self.track_failures += 1
            if self.track_failures >= 3:
                self.tracker = None
                self._clear_detection()
            return
        self.track_failures = 0
        x, y, width, height = (int(round(value)) for value in box)
        if width > 1 and height > 1:
            self._publish_detection((x, y, width, height), message.header.frame_id)

    def _on_depth(self, message: Image) -> None:
        self.depth = message

    def _on_camera_info(self, message: CameraInfo) -> None:
        self.camera_info = message

    def _on_scene(self, message: SceneObjectArray) -> None:
        if self.target_uuid is not None:
            self.localized = next(
                (item for item in message.objects if item.uuid == self.target_uuid),
                None,
            )

    def _publish_detection(
        self, bbox: tuple[int, int, int, int], frame_id: str
    ) -> None:
        if self.target_uuid is None:
            return
        message = SceneObjectArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = frame_id or "camera_color_optical_frame"
        item = SceneObject()
        item.header = message.header
        item.uuid = self.target_uuid
        item.class_id = self.target_class
        item.confidence = self.target_confidence
        item.pose.pose.orientation.w = 1.0
        item.size.x, item.size.y, item.size.z = self.target_size
        item.bbox_2d.x_offset = max(0, bbox[0])
        item.bbox_2d.y_offset = max(0, bbox[1])
        item.bbox_2d.width = bbox[2]
        item.bbox_2d.height = bbox[3]
        item.source = "gemini_er_rgb_tracker"
        message.objects = [item]
        self.detections.publish(message)

    def _clear_detection(self) -> None:
        empty = SceneObjectArray()
        empty.header.stamp = self.get_clock().now().to_msg()
        empty.header.frame_id = "camera_color_optical_frame"
        self.detections.publish(empty)

    def clear_target(self) -> None:
        self.tracker = None
        self.target_uuid = None
        self.localized = None
        self._clear_detection()

    def _fresh_image(self, timeout_sec: float = 5.0) -> Image:
        self.image = None
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.image is None:
                continue
            publishers = self.get_publishers_info_by_topic(
                "/camera/camera/color/image_raw"
            )
            if len(publishers) != 1 or publishers[0].node_name != "mani_mujoco_server":
                raise RuntimeError("RGB image must come only from the MuJoCo server")
            competing = [
                item for item in self.get_publishers_info_by_topic(
                    "/mani/perception/detections_2d"
                )
                if item.node_name != self.get_name()
            ]
            if competing:
                raise RuntimeError("disable the MuJoCo perfect detector")
            return self.image
        raise TimeoutError("MuJoCo camera image unavailable")

    def _ask_target(self, instruction: str, *, keep_tracking: bool) -> SceneObject:
        image = self._fresh_image()
        rgb = decode_rgb(image)
        with tempfile.TemporaryDirectory(prefix="gemini_sim_chat_") as directory:
            image_path = Path(directory) / "camera.png"
            PilImage.fromarray(rgb).save(image_path)
            print("Gemini 正在分析基座相机图像…", flush=True)
            try:
                proposal = query_image(
                    image_path,
                    "Locate the single object specified by the user. "
                    "Return its visible center and tight bounding box. User: "
                    + instruction,
                    model=self.model,
                    api_key=self.api_key,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Gemini request failed: {type(exc).__name__} "
                    f"(status {getattr(exc, 'code', 'unknown')})"
                ) from None
        if proposal.label.lower() == "none" or proposal.confidence is None:
            raise ValueError("Gemini 没有找到明确目标")
        if proposal.confidence < 0.6:
            raise ValueError("Gemini 目标置信度低于 0.6")
        if proposal.point_yx_norm is None or proposal.bbox_yxyx_norm is None:
            raise ValueError("Gemini 未返回目标点和检测框")
        point = normalized_yx_to_pixel(
            proposal.point_yx_norm, image.width, image.height
        )
        bbox = normalized_bbox_to_xywh(
            proposal.bbox_yxyx_norm, image.width, image.height
        )
        if not (
            bbox[0] - 3 <= point[0] <= bbox[0] + bbox[2] + 3
            and bbox[1] - 3 <= point[1] <= bbox[1] + bbox[3] + 3
        ):
            raise ValueError("Gemini 目标点不在检测框内")
        object_class, size = size_from_label(proposal.label)
        self.clear_target()
        self.target_uuid = "gemini-" + uuid4().hex[:12]
        self.target_class = object_class
        self.target_size = size
        self.target_confidence = proposal.confidence
        self.localized = None
        if keep_tracking:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), bbox)
            self.tracker = tracker
            self.track_failures = 0
        deadline = time.monotonic() + 4.0
        while rclpy.ok() and time.monotonic() < deadline:
            self._publish_detection(bbox, image.header.frame_id)
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.localized is not None:
                break
        if self.localized is None:
            self.clear_target()
            raise TimeoutError("RGB-D 定位器没有返回目标位置")
        p = self.localized.pose.pose.position
        if not (0.12 <= p.x <= 0.35 and abs(p.y) <= 0.18 and 0.04 <= p.z <= 0.26):
            self.clear_target()
            raise ValueError("目标落在仿真操作区外")
        print(
            f"目标：{proposal.label}；框 {bbox}；"
            f"基座坐标 ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m"
        )
        result = self.localized
        if not keep_tracking:
            self.clear_target()
        return result

    def describe(self, question: str) -> None:
        image = self._fresh_image()
        with tempfile.TemporaryDirectory(prefix="gemini_sim_chat_") as directory:
            image_path = Path(directory) / "camera.png"
            PilImage.fromarray(decode_rgb(image)).save(image_path)
            print("Gemini 正在分析基座相机图像…", flush=True)
            try:
                answer = describe_image(
                    image_path, question, model=self.model, api_key=self.api_key
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Gemini request failed: {type(exc).__name__} "
                    f"(status {getattr(exc, 'code', 'unknown')})"
                ) from None
        print(answer or "Gemini 没有返回描述")

    def locate(self, instruction: str) -> None:
        self._ask_target(instruction, keep_tracking=False)

    def pick(self, instruction: str) -> None:
        if self.held_label is not None:
            raise ValueError("夹爪已持有物体；请先 /place")
        target = self._ask_target(instruction, keep_tracking=True)
        if not self._pick.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("仿真抓取服务不可用")
        goal = PickObject.Goal()
        goal.object_uuid = target.uuid
        goal.group_name = "arm"
        goal.end_effector_name = "gripper"
        goal.approach_distance_m = 0.08
        goal.lift_distance_m = 0.09
        goal.velocity_scale = 0.4
        goal.acceleration_scale = 0.4
        sent = self._pick.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=5.0)
        handle = sent.result()
        if handle is None or not handle.accepted:
            self.clear_target()
            raise RuntimeError("抓取目标被任务服务拒绝")
        finished = handle.get_result_async()
        rclpy.spin_until_future_complete(self, finished, timeout_sec=50.0)
        wrapped = finished.result()
        self.clear_target()
        if wrapped is not None and not wrapped.result.success and wrapped.result.failure_stage == "verify":
            # The gripper can occlude a continuously tracked object. Reacquire
            # it from a fresh camera frame before declaring the physical pick
            # a failure. This still uses only RGB-D, never simulator truth.
            print("连续追踪未能确认抬升，重新拍照复核…", flush=True)
            reacquired = self._ask_target(instruction, keep_tracking=False)
            start_z = target.pose.pose.position.z
            lifted = reacquired.pose.pose.position.z - start_z
            if reacquired.class_id == target.class_id and lifted >= 0.035:
                self.held_label = target.class_id
                self.held_size = (
                    float(target.size.x), float(target.size.y), float(target.size.z)
                )
                print(f"抓取成功；重新定位确认抬升 {lifted * 1000:.1f} mm")
                return
        if wrapped is None or not wrapped.result.success:
            message = "没有收到抓取结果" if wrapped is None else wrapped.result.message
            raise RuntimeError("抓取未通过视觉验证：" + message)
        self.held_label = target.class_id
        self.held_size = (float(target.size.x), float(target.size.y), float(target.size.z))
        print(f"抓取成功；视觉抬升 {wrapped.result.object_lift_m * 1000:.1f} mm")

    def _table_destination(self, pixel_xy: tuple[int, int]) -> tuple[float, float]:
        if self.depth is None or self.camera_info is None:
            raise RuntimeError("对齐深度或 CameraInfo 尚未收到")
        tf = self._tf.lookup_transform(
            "magician_base_link", "camera_color_optical_frame", Time()
        )
        origin = np.array(
            [tf.transform.translation.x, tf.transform.translation.y,
             tf.transform.translation.z], dtype=np.float64
        )
        q = tf.transform.rotation
        rotation = quaternion_matrix_xyzw(
            np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        )
        intrinsics = PinholeIntrinsics(
            fx=float(self.camera_info.k[0]), fy=float(self.camera_info.k[4]),
            cx=float(self.camera_info.k[2]), cy=float(self.camera_info.k[5]),
        )
        ray = rotation @ optical_point(
            float(pixel_xy[0]), float(pixel_xy[1]), 1.0, intrinsics
        )
        if abs(ray[2]) < 1e-9:
            raise ValueError("目的地点视线与桌面平行")
        depth_scale = (TABLE_Z_M - origin[2]) / ray[2]
        if depth_scale <= 0:
            raise ValueError("目的地点位于相机背面")
        point = origin + depth_scale * ray
        x, y = float(point[0]), float(point[1])
        if not (0.13 <= x <= 0.34 and abs(y) <= 0.16):
            raise ValueError("目的地点超出已验证桌面区域")
        depths = decode_depth_m(self.depth)
        u, v = pixel_xy
        radius = 16
        patch = depths[max(0, v-radius):v+radius+1, max(0, u-radius):u+radius+1]
        if patch.size == 0:
            raise ValueError("目的地点没有有效深度像素")
        table_fraction = float(np.mean(np.abs(patch - depth_scale) < 0.025))
        if table_fraction < 0.95:
            raise ValueError("目的地点附近不是足够空旷的桌面")
        return x, y

    def _move_pose(self, x: float, y: float, z: float, yaw: float) -> None:
        if not self._move.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("仿真运动服务不可用")
        goal = MoveToPose.Goal()
        goal.group_name = "arm"
        goal.target.header.frame_id = "magician_base_link"
        goal.target.pose.position.x = x
        goal.target.pose.position.y = y
        goal.target.pose.position.z = z
        goal.target.pose.orientation.z = math.sin(yaw * 0.5)
        goal.target.pose.orientation.w = math.cos(yaw * 0.5)
        goal.controlled_axes = [True, True, True, False, False, True]
        goal.position_tolerance_m = 0.005
        goal.orientation_tolerance_rad = math.radians(2.0)
        goal.velocity_scale = 0.4
        goal.acceleration_scale = 0.4
        sent = self._move.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=5.0)
        handle = sent.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("仿真位姿目标被拒绝")
        finished = handle.get_result_async()
        rclpy.spin_until_future_complete(self, finished, timeout_sec=20.0)
        result = finished.result()
        if result is None or not result.result.success:
            raise RuntimeError(
                "仿真移动失败：" + (
                    "无结果" if result is None else result.result.message
                )
            )

    def _reachable_destination(
        self, requested_pixel: tuple[int, int], hover_z: float
    ) -> tuple[tuple[int, int], float, float]:
        """Search nearby camera-observed free table for a reachable pose."""
        offsets = [(0, 0)]
        for radius in (24, 48, 72, 96):
            offsets.extend(
                (du, dv)
                for du, dv in (
                    (0, -radius), (-radius, 0), (radius, 0), (0, radius),
                    (-radius, -radius), (radius, -radius),
                    (-radius, radius), (radius, radius),
                )
            )
        for du, dv in offsets:
            pixel = (requested_pixel[0] + du, requested_pixel[1] + dv)
            try:
                x, y = self._table_destination(pixel)
                yaw = math.atan2(y, x)
                for z in (hover_z, TABLE_Z_M + 0.004):
                    inverse_tcp_pose(
                        Pose.from_xyz_yaw(
                            x, y, z, yaw, frame_id="magician_base_link"
                        )
                    )
            except (ValueError, IndexError):
                continue
            return pixel, x, y
        raise ValueError("Gemini 选择区域附近没有同时空旷且可达的桌面点")

    def place(self, instruction: str) -> None:
        if self.held_label is None or self.held_size is None:
            raise ValueError("当前没有已验证抓起的物体")
        image = self._fresh_image()
        with tempfile.TemporaryDirectory(prefix="gemini_sim_chat_") as directory:
            image_path = Path(directory) / "camera.png"
            PilImage.fromarray(decode_rgb(image)).save(image_path)
            print("Gemini 正在寻找空闲放置区…", flush=True)
            try:
                proposal = query_image(
                    image_path,
                    "Point to a clear, empty tabletop patch where the held "
                    f"{self.held_label} can be placed. User: {instruction}. "
                    "The point must be on visible free table surface.",
                    model=self.model,
                    api_key=self.api_key,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Gemini request failed: {type(exc).__name__} "
                    f"(status {getattr(exc, 'code', 'unknown')})"
                ) from None
        if proposal.point_yx_norm is None:
            raise ValueError("Gemini 未给出放置点")
        pixel = normalized_yx_to_pixel(
            proposal.point_yx_norm, image.width, image.height
        )
        hover_z = max(0.18, TABLE_Z_M + self.held_size[2] + 0.04)
        pixel, x, y = self._reachable_destination(pixel, hover_z)
        print(f"放置点：像素 {pixel}，基座坐标 ({x:.3f}, {y:.3f}) m")
        yaw = math.atan2(y, x)
        self._move_pose(x, y, hover_z, yaw)
        self._move_pose(x, y, TABLE_Z_M + 0.004, yaw)
        if not self._gripper.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("夹爪服务不可用")
        request = CommandEndEffector.Request()
        request.end_effector_name = "gripper"
        request.command = CommandEndEffector.Request.OPEN
        request.hold = False
        future = self._gripper.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError("夹爪未能打开")
        self._move_pose(x, y, hover_z, yaw)
        self.held_label = None
        self.held_size = None
        print("已在仿真中放下物体")

    def reset(self, seed: int) -> None:
        self.clear_target()
        self.held_label = None
        self.held_size = None
        if not self._reset.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("仿真重置服务不可用")
        request = ResetScene.Request()
        request.randomize = True
        request.seed = seed
        future = self._reset.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=8.0)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError("仿真重置失败")
        print(f"已重置多物体场景，seed={seed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-robotics-er-2-preview")
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")
    rclpy.init()
    node = SimChat(api_key=api_key, model=args.model)
    print("MuJoCo Gemini Terminal 已就绪。/help 查看命令。")
    try:
        while rclpy.ok():
            try:
                line = input("Gemini(sim)> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            if line == "/help":
                print(
                    "直接输入问题：让 Gemini 描述当前相机画面\n"
                    "/find 紫色方块：定位物体\n"
                    "/pick 紫色方块：在仿真中抓取\n"
                    "/place 右侧空旷位置：放下已抓物体\n"
                    "/reset 42：重新摆放杂物；/quit：退出"
                )
                continue
            try:
                if line.startswith("/find "):
                    node.locate(line[6:].strip())
                elif line.startswith("/pick "):
                    node.pick(line[6:].strip())
                elif line.startswith("/place "):
                    node.place(line[7:].strip())
                elif line.startswith("/reset"):
                    parts = line.split()
                    node.reset(int(parts[1]) if len(parts) > 1 else 0)
                else:
                    node.describe(line)
            except (RuntimeError, ValueError, TimeoutError) as exc:
                print(f"未执行：{exc}")
    except KeyboardInterrupt:
        pass
    finally:
        node.clear_target()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
