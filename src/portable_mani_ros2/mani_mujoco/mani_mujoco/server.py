"""Expose the Dobot MuJoCo backend through the portable ROS 2 contract."""

from __future__ import annotations

import math
from threading import Event, RLock, Thread
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from mani_core import Pose, TaskAxes
from mani_dobot.backends.mujoco_backend import DobotMujocoBackend
from mani_interfaces.action import MoveToPose
from mani_interfaces.msg import (
    EndEffectorCapability,
    ManipulatorCapability,
    RobotCapabilities,
    SceneObject,
    SceneObjectArray,
)
from mani_interfaces.srv import CommandEndEffector, ResetScene
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from .rgbd_camera import MujocoRgbdCamera


class MujocoServer(Node):
    """Simulation embodiment with the same public endpoints as real hardware."""

    def __init__(self) -> None:
        super().__init__("mani_mujoco_server")
        self._callbacks = ReentrantCallbackGroup()
        self._lock = RLock()
        self.declare_parameter("viewer", False)
        self.declare_parameter("publish_camera", True)
        self.declare_parameter("publish_perfect_detections", True)
        self.declare_parameter("camera_rate_hz", 10.0)
        self.declare_parameter("camera_width", 320)
        self.declare_parameter("camera_height", 240)
        viewer_enabled = bool(self.get_parameter("viewer").value)
        camera_enabled = bool(self.get_parameter("publish_camera").value)
        if viewer_enabled and camera_enabled:
            raise ValueError(
                "viewer and offscreen RGB-D rendering use incompatible GL "
                "contexts; set either viewer:=false or publish_camera:=false"
            )
        self._backend = DobotMujocoBackend()
        self._backend.reset_scene(0.22, 0.0)
        self._latest_object_bbox: tuple[int, int, int, int] | None = None
        self._viewer = None
        if viewer_enabled:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(
                self._backend.model, self._backend.data
            )
            self._backend.set_step_callback(self._sync_viewer_realtime)

        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._capabilities_pub = self.create_publisher(
            RobotCapabilities, "/mani/capabilities", latched
        )
        self._ground_truth_pub = self.create_publisher(
            SceneObjectArray, "/mani/sim/ground_truth", 10
        )
        self._scene_bypass_pub = (
            None
            if camera_enabled
            else self.create_publisher(SceneObjectArray, "/mani/scene_objects", 10)
        )
        self._detections_pub = (
            self.create_publisher(
                SceneObjectArray, "/mani/perception/detections_2d", 10
            )
            if bool(self.get_parameter("publish_perfect_detections").value)
            else None
        )
        self._tcp_pub = self.create_publisher(PoseStamped, "/mani/tcp_pose", 10)
        self._joints_pub = self.create_publisher(JointState, "/joint_states", 10)

        self._camera: MujocoRgbdCamera | None = None
        self._camera_thread: Thread | None = None
        self._camera_stop = Event()
        self._camera_ready = Event()
        self._camera_error: BaseException | None = None
        self._camera_width = 0
        self._camera_height = 0
        self._camera_rate_hz = 0.0
        self._tf_static = StaticTransformBroadcaster(self)
        if camera_enabled:
            self._camera_width = int(self.get_parameter("camera_width").value)
            self._camera_height = int(self.get_parameter("camera_height").value)
            self._camera_rate_hz = float(
                self.get_parameter("camera_rate_hz").value
            )
            if self._camera_rate_hz <= 0.0:
                raise ValueError("camera_rate_hz must be positive")
            sensor_qos = QoSProfile(depth=5)
            sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
            sensor_qos.durability = DurabilityPolicy.VOLATILE
            self._color_pub = self.create_publisher(
                Image, "/camera/camera/color/image_raw", sensor_qos
            )
            self._color_info_pub = self.create_publisher(
                CameraInfo, "/camera/camera/color/camera_info", sensor_qos
            )
            self._depth_pub = self.create_publisher(
                Image, "/camera/camera/depth/image_rect_raw", sensor_qos
            )
            self._depth_info_pub = self.create_publisher(
                CameraInfo, "/camera/camera/depth/camera_info", sensor_qos
            )
            self._aligned_depth_pub = self.create_publisher(
                Image,
                "/camera/camera/aligned_depth_to_color/image_raw",
                sensor_qos,
            )
            self._aligned_depth_info_pub = self.create_publisher(
                CameraInfo,
                "/camera/camera/aligned_depth_to_color/camera_info",
                sensor_qos,
            )

        self._move_server = ActionServer(
            self,
            MoveToPose,
            "/mani/move_to_pose",
            execute_callback=self._execute_move,
            goal_callback=self._goal_move,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=self._callbacks,
        )
        self._gripper_service = self.create_service(
            CommandEndEffector,
            "/mani/command_end_effector",
            self._command_end_effector,
            callback_group=self._callbacks,
        )
        self._reset_service = self.create_service(
            ResetScene,
            "/mani/sim/reset_scene",
            self._reset_scene,
            callback_group=self._callbacks,
        )
        self._timer = self.create_timer(
            0.05, self._publish_state, callback_group=self._callbacks
        )
        self._publish_capabilities()
        self._publish_state()
        if self._camera_rate_hz > 0.0:
            self._camera_thread = Thread(
                target=self._camera_loop,
                name="mani_mujoco_rgbd",
                daemon=True,
            )
            self._camera_thread.start()
            if not self._camera_ready.wait(timeout=10.0):
                raise RuntimeError("MuJoCo RGB-D renderer did not initialize")
            if self._camera_error is not None:
                raise RuntimeError("MuJoCo RGB-D renderer failed") from self._camera_error
        self.get_logger().info("Portable Mani MuJoCo backend is ready")

    def _sync_viewer_realtime(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
            time.sleep(1.0 / self._backend.CONTROL_RATE_HZ)

    @staticmethod
    def _fill_pose(message, pose: Pose) -> None:
        message.position.x, message.position.y, message.position.z = (
            float(value) for value in pose.position
        )
        (
            message.orientation.x,
            message.orientation.y,
            message.orientation.z,
            message.orientation.w,
        ) = (float(value) for value in pose.quaternion_xyzw)

    def _pose_stamped(self, pose: Pose) -> PoseStamped:
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = pose.frame_id
        self._fill_pose(message.pose, pose)
        return message

    def _scene_object_message(self) -> SceneObject:
        state = self._backend.scene_object()
        message = SceneObject()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = state.pose.frame_id
        message.uuid = state.uuid
        message.class_id = state.class_id
        message.confidence = state.confidence
        self._fill_pose(message.pose.pose, state.pose)
        message.size.x, message.size.y, message.size.z = (
            float(value) for value in state.size
        )
        message.source = state.source
        if self._latest_object_bbox is not None:
            x, y, width, height = self._latest_object_bbox
            message.bbox_2d.x_offset = x
            message.bbox_2d.y_offset = y
            message.bbox_2d.width = width
            message.bbox_2d.height = height
            message.bbox_2d.do_rectify = False
        return message

    def _detection_message(self, truth: SceneObject) -> SceneObject | None:
        if self._latest_object_bbox is None:
            return None
        message = SceneObject()
        message.header.stamp = truth.header.stamp
        message.header.frame_id = "camera_color_optical_frame"
        message.uuid = truth.uuid
        message.class_id = truth.class_id
        message.confidence = truth.confidence
        message.pose.pose.orientation.w = 1.0
        message.size = truth.size
        message.bbox_2d = truth.bbox_2d
        message.source = "mujoco_segmentation"
        return message

    def _camera_info(self, stamp, frame_id: str) -> CameraInfo:
        assert self._camera is not None
        calibration = self._camera.calibration
        message = CameraInfo()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.width = calibration.width
        message.height = calibration.height
        message.distortion_model = "plumb_bob"
        message.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        message.k = [
            calibration.fx,
            0.0,
            calibration.cx,
            0.0,
            calibration.fy,
            calibration.cy,
            0.0,
            0.0,
            1.0,
        ]
        message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        message.p = [
            calibration.fx,
            0.0,
            calibration.cx,
            0.0,
            0.0,
            calibration.fy,
            calibration.cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return message

    @staticmethod
    def _image_message(array: np.ndarray, stamp, frame_id: str, encoding: str) -> Image:
        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height, message.width = array.shape[:2]
        message.encoding = encoding
        message.is_bigendian = False
        message.step = int(array.strides[0])
        message.data = array.tobytes()
        return message

    def _publish_camera(self) -> None:
        if self._camera is None:
            return
        with self._lock:
            frame = self._camera.render(self._backend.data)
            self._latest_object_bbox = frame.object_bbox_xywh
        stamp = self.get_clock().now().to_msg()
        color_frame = "camera_color_optical_frame"
        depth_frame = "camera_depth_optical_frame"
        depth_mm = self._camera.depth_millimeters(frame.depth_m)
        color = self._image_message(frame.rgb, stamp, color_frame, "rgb8")
        depth = self._image_message(depth_mm, stamp, depth_frame, "16UC1")
        aligned = self._image_message(depth_mm, stamp, color_frame, "16UC1")
        self._color_pub.publish(color)
        self._depth_pub.publish(depth)
        self._aligned_depth_pub.publish(aligned)
        self._color_info_pub.publish(self._camera_info(stamp, color_frame))
        self._depth_info_pub.publish(self._camera_info(stamp, depth_frame))
        self._aligned_depth_info_pub.publish(self._camera_info(stamp, color_frame))

    def _camera_loop(self) -> None:
        """Own the OpenGL context on one fixed thread for its entire lifetime."""

        camera: MujocoRgbdCamera | None = None
        try:
            camera = MujocoRgbdCamera(
                self._backend.model,
                width=self._camera_width,
                height=self._camera_height,
            )
            self._camera = camera
            self._publish_camera_transforms()
            self._camera_ready.set()
            period = 1.0 / self._camera_rate_hz
            while not self._camera_stop.is_set() and rclpy.ok():
                started = time.monotonic()
                self._publish_camera()
                remaining = period - (time.monotonic() - started)
                if remaining > 0.0:
                    self._camera_stop.wait(remaining)
        except BaseException as exc:
            self._camera_error = exc
            self._camera_ready.set()
            if rclpy.ok():
                self.get_logger().error(f"MuJoCo RGB-D renderer stopped: {exc!r}")
        finally:
            if camera is not None:
                camera.close()
            self._camera = None

    def _publish_camera_transforms(self) -> None:
        assert self._camera is not None
        stamp = self.get_clock().now().to_msg()
        position, quaternion = self._camera.optical_pose(self._backend.data)
        base_to_color = TransformStamped()
        base_to_color.header.stamp = stamp
        base_to_color.header.frame_id = "magician_base_link"
        base_to_color.child_frame_id = "camera_color_optical_frame"
        (
            base_to_color.transform.translation.x,
            base_to_color.transform.translation.y,
            base_to_color.transform.translation.z,
        ) = (float(value) for value in position)
        (
            base_to_color.transform.rotation.x,
            base_to_color.transform.rotation.y,
            base_to_color.transform.rotation.z,
            base_to_color.transform.rotation.w,
        ) = (float(value) for value in quaternion)

        color_to_depth = TransformStamped()
        color_to_depth.header.stamp = stamp
        color_to_depth.header.frame_id = "camera_color_optical_frame"
        color_to_depth.child_frame_id = "camera_depth_optical_frame"
        color_to_depth.transform.rotation.w = 1.0
        self._tf_static.sendTransform([base_to_color, color_to_depth])

    def _publish_capabilities(self) -> None:
        capability = self._backend.capabilities
        source_arm = capability.manipulators[0]
        arm = ManipulatorCapability()
        arm.name = source_arm.name
        arm.root_frame = source_arm.root_frame
        arm.tip_frame = source_arm.tip_frame
        arm.joint_names = list(source_arm.joint_names)
        arm.task_axes = list(source_arm.task_axes.as_tuple())
        arm.control_modes = list(source_arm.control_modes)
        arm.end_effector = source_arm.end_effector or ""
        arm.command_rate_hz = source_arm.command_rate_hz

        source_gripper = capability.end_effectors[0]
        gripper = EndEffectorCapability()
        gripper.name = source_gripper.name
        gripper.kind = source_gripper.kind
        gripper.command_modes = list(source_gripper.command_modes)
        gripper.max_opening_m = source_gripper.max_opening_m

        message = RobotCapabilities()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = source_arm.root_frame
        message.embodiment_name = capability.name + "_mujoco"
        message.manipulators = [arm]
        message.end_effectors = [gripper]
        message.max_payload_kg = capability.max_payload_kg
        self._capabilities_pub.publish(message)

    def _publish_state(self) -> None:
        with self._lock:
            state = self._backend.state()
            scene_object = self._scene_object_message()
            if self._viewer is not None and self._viewer.is_running():
                self._viewer.sync()
        stamp = self.get_clock().now().to_msg()
        self._tcp_pub.publish(self._pose_stamped(state.tcp_pose))

        joints = JointState()
        joints.header.stamp = stamp
        joints.header.frame_id = state.tcp_pose.frame_id
        joints.name = list(state.joint_names)
        joints.position = state.position.tolist()
        joints.velocity = state.velocity.tolist()
        self._joints_pub.publish(joints)

        ground_truth = SceneObjectArray()
        ground_truth.header.stamp = stamp
        ground_truth.header.frame_id = state.tcp_pose.frame_id
        ground_truth.objects = [scene_object]
        self._ground_truth_pub.publish(ground_truth)
        if self._scene_bypass_pub is not None:
            self._scene_bypass_pub.publish(ground_truth)

        detection = self._detection_message(scene_object)
        detections = SceneObjectArray()
        detections.header.stamp = stamp
        detections.header.frame_id = "camera_color_optical_frame"
        detections.objects = [] if detection is None else [detection]
        if self._detections_pub is not None:
            self._detections_pub.publish(detections)

    def _goal_move(self, request: MoveToPose.Goal) -> GoalResponse:
        if request.group_name != "arm":
            self.get_logger().warning(f"Rejecting unknown group {request.group_name!r}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_move(self, goal_handle) -> MoveToPose.Result:
        request = goal_handle.request
        result_message = MoveToPose.Result()
        feedback = MoveToPose.Feedback()
        feedback.progress = 0.0
        with self._lock:
            feedback.current_pose = self._pose_stamped(self._backend.state().tcp_pose)
        goal_handle.publish_feedback(feedback)

        target_message = request.target
        q = target_message.pose.orientation
        try:
            target = Pose(
                position=np.array(
                    [
                        target_message.pose.position.x,
                        target_message.pose.position.y,
                        target_message.pose.position.z,
                    ]
                ),
                quaternion_xyzw=np.array([q.x, q.y, q.z, q.w]),
                frame_id=target_message.header.frame_id or "magician_base_link",
            )
            axes = TaskAxes.from_iterable(request.controlled_axes)
            duration = max(0.25, 1.5 / max(float(request.velocity_scale), 0.05))
            with self._lock:
                result = self._backend.move_to_pose(
                    request.group_name,
                    target,
                    controlled_axes=axes,
                    duration_s=duration,
                )
                achieved = self._backend.state().tcp_pose
        except (ValueError, RuntimeError) as exc:
            result_message.success = False
            result_message.message = str(exc)
            with self._lock:
                achieved = self._backend.state().tcp_pose
            result_message.achieved_pose = self._pose_stamped(achieved)
            goal_handle.abort()
            return result_message

        result_message.success = result.success
        result_message.message = result.message
        result_message.achieved_pose = self._pose_stamped(achieved)
        feedback.progress = 1.0
        feedback.current_pose = result_message.achieved_pose
        goal_handle.publish_feedback(feedback)
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self._publish_state()
        return result_message

    def _command_end_effector(self, request, response):
        command = {
            CommandEndEffector.Request.OPEN: "open",
            CommandEndEffector.Request.CLOSE: "close",
        }.get(request.command)
        if command is None:
            response.success = False
            response.message = "MuJoCo backend currently supports OPEN and CLOSE only"
            return response
        with self._lock:
            result = self._backend.command_end_effector(
                request.end_effector_name, command
            )
            opening = self._backend.state().end_effector_position
        response.success = result.success
        response.message = result.message
        response.achieved_position_m = opening
        self._publish_state()
        return response

    def _reset_scene(self, request, response):
        if request.randomize:
            rng = np.random.default_rng(request.seed)
            x = float(rng.uniform(0.20, 0.28))
            y = float(rng.uniform(-0.12, 0.12))
            yaw = float(rng.uniform(-math.pi, math.pi))
        else:
            x = float(request.object_x_m)
            y = float(request.object_y_m)
            yaw = float(request.object_yaw_rad)
        if not (0.20 <= x <= 0.28 and -0.12 <= y <= 0.12):
            response.success = False
            response.message = "object position outside validated pick workspace"
            return response
        with self._lock:
            self._backend.reset_scene(x, y, yaw=yaw)
            response.object = self._scene_object_message()
        response.success = True
        response.message = "scene reset"
        self._publish_state()
        return response

    def destroy_node(self) -> bool:
        self._move_server.destroy()
        self._camera_stop.set()
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=5.0)
        if self._viewer is not None:
            self._viewer.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MujocoServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
