"""Robot-independent ROS 2 façade over the existing magician_ros2 driver."""

from __future__ import annotations

import asyncio
import math
from time import monotonic
from typing import Sequence

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from dobot_msgs.action import PointToPoint
from dobot_msgs.msg import DobotAlarmCodes
from dobot_msgs.srv import GripperControl
from geometry_msgs.msg import PoseStamped
from mani_core.models import Pose, TaskAxes
from mani_core.safety import SafetyEnvelope, SafetyViolation
from mani_interfaces.action import MoveToPose
from mani_interfaces.msg import (
    EndEffectorCapability,
    ManipulatorCapability,
    RobotCapabilities,
)
from mani_interfaces.srv import CommandEndEffector
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from .kinematics import RAW_JOINT_LOWER, RAW_JOINT_UPPER


SUPPORTED_AXES = TaskAxes(True, True, True, False, False, True)


def _rpy_from_xyzw(quaternion: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = (float(value) for value in quaternion)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class DobotPortableBridge(Node):
    """Expose generic manipulation interfaces without changing magician_ros2."""

    def __init__(self) -> None:
        super().__init__("mani_dobot_bridge")
        self._group = ReentrantCallbackGroup()
        self.declare_parameter("group_name", "arm")
        self.declare_parameter("end_effector_name", "gripper")
        self.declare_parameter("base_frame", "magician_base_link")
        self.declare_parameter("tip_frame", "TCP")
        self.declare_parameter("source_move_action", "PTP_action")
        self.declare_parameter("source_gripper_service", "dobot_gripper_service")
        self.declare_parameter("source_tcp_topic", "dobot_TCP")
        self.declare_parameter("source_alarm_topic", "dobot_alarms")
        self.declare_parameter("upstream_wait_timeout_s", 2.0)

        self.group_name = str(self.get_parameter("group_name").value)
        self.end_effector_name = str(self.get_parameter("end_effector_name").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tip_frame = str(self.get_parameter("tip_frame").value)
        self._current_pose: PoseStamped | None = None
        self._current_pose_received_at: float | None = None
        self._has_active_alarm = False

        self._safety = SafetyEnvelope(
            joint_lower=RAW_JOINT_LOWER,
            joint_upper=RAW_JOINT_UPPER,
            workspace_lower=np.array([-0.35, -0.35, 0.015]),
            workspace_upper=np.array([0.35, 0.35, 0.38]),
            supported_axes=SUPPORTED_AXES,
            max_state_age_s=0.25,
        )

        capability_qos = QoSProfile(depth=1)
        capability_qos.reliability = ReliabilityPolicy.RELIABLE
        capability_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._capability_publisher = self.create_publisher(
            RobotCapabilities, "/mani/capabilities", capability_qos
        )
        self._pose_subscription = self.create_subscription(
            PoseStamped,
            str(self.get_parameter("source_tcp_topic").value),
            self._pose_callback,
            10,
            callback_group=self._group,
        )
        self._alarm_subscription = self.create_subscription(
            DobotAlarmCodes,
            str(self.get_parameter("source_alarm_topic").value),
            self._alarm_callback,
            10,
            callback_group=self._group,
        )
        self._ptp_client = ActionClient(
            self,
            PointToPoint,
            str(self.get_parameter("source_move_action").value),
            callback_group=self._group,
        )
        self._gripper_client = self.create_client(
            GripperControl,
            str(self.get_parameter("source_gripper_service").value),
            callback_group=self._group,
        )
        self._move_server = ActionServer(
            self,
            MoveToPose,
            "/mani/move_to_pose",
            execute_callback=self._execute_move,
            goal_callback=self._move_goal_callback,
            cancel_callback=self._move_cancel_callback,
            callback_group=self._group,
        )
        self._end_effector_service = self.create_service(
            CommandEndEffector,
            "/mani/command_end_effector",
            self._command_end_effector,
            callback_group=self._group,
        )
        self._capability_timer = self.create_timer(1.0, self._publish_capabilities)
        self._publish_capabilities()

    def _publish_capabilities(self) -> None:
        manipulator = ManipulatorCapability()
        manipulator.name = self.group_name
        manipulator.root_frame = self.base_frame
        manipulator.tip_frame = self.tip_frame
        manipulator.joint_names = [
            "magician_joint_1",
            "magician_joint_2",
            "magician_joint_3",
            "magician_joint_4",
        ]
        manipulator.task_axes = list(SUPPORTED_AXES.as_tuple())
        manipulator.control_modes = ["task_pose"]
        manipulator.end_effector = self.end_effector_name
        manipulator.command_rate_hz = 20.0

        end_effector = EndEffectorCapability()
        end_effector.name = self.end_effector_name
        end_effector.kind = "parallel_gripper"
        end_effector.command_modes = ["open", "close"]
        end_effector.max_opening_m = 0.027

        message = RobotCapabilities()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.base_frame
        message.embodiment_name = "dobot_magician"
        message.manipulators = [manipulator]
        message.end_effectors = [end_effector]
        message.max_payload_kg = 0.5
        self._capability_publisher.publish(message)

    def _pose_callback(self, message: PoseStamped) -> None:
        self._current_pose = message
        self._current_pose_received_at = monotonic()

    def _alarm_callback(self, message: DobotAlarmCodes) -> None:
        self._has_active_alarm = bool(message.alarms_list)

    def _goal_pose(self, goal: MoveToPose.Goal) -> tuple[Pose, tuple[float, float, float]]:
        target = goal.target
        frame_id = target.header.frame_id or self.base_frame
        if frame_id != self.base_frame:
            raise SafetyViolation(
                f"target frame {frame_id!r} is not {self.base_frame!r}; TF conversion is not enabled"
            )
        quaternion = target.pose.orientation
        roll, pitch, yaw = _rpy_from_xyzw(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        )
        pose = Pose.from_xyz_yaw(
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z,
            yaw,
            frame_id=frame_id,
        )
        return pose, (roll, pitch, yaw)

    def _move_goal_callback(self, request: MoveToPose.Goal) -> GoalResponse:
        try:
            if request.group_name != self.group_name:
                raise SafetyViolation(f"unknown group {request.group_name!r}")
            if self._has_active_alarm:
                raise SafetyViolation("Dobot reports active alarms")
            if self._current_pose is None:
                raise SafetyViolation("no Dobot TCP state received yet")
            state_age = monotonic() - float(self._current_pose_received_at)
            if state_age > self._safety.max_state_age_s:
                raise SafetyViolation(
                    f"Dobot TCP state is stale ({state_age:.3f}s > "
                    f"{self._safety.max_state_age_s:.3f}s)"
                )
            required = TaskAxes.from_iterable(request.controlled_axes)
            pose, (roll, pitch, _) = self._goal_pose(request)
            self._safety.validate_pose(pose, required)
            if required.roll or required.pitch or abs(roll) > 1e-3 or abs(pitch) > 1e-3:
                raise SafetyViolation("Dobot supports xyz+yaw only; roll and pitch must be zero")
            if not (0.0 < request.velocity_scale <= 1.0):
                raise SafetyViolation("velocity_scale must be in (0, 1]")
            if not (0.0 < request.acceleration_scale <= 1.0):
                raise SafetyViolation("acceleration_scale must be in (0, 1]")
        except (ValueError, SafetyViolation) as exc:
            self.get_logger().warning(f"Rejecting move goal: {exc}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _move_cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _copy_current_pose(self) -> PoseStamped:
        message = PoseStamped()
        if self._current_pose is not None:
            message.header = self._current_pose.header
            message.pose = self._current_pose.pose
        else:
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self.base_frame
            message.pose.orientation.w = 1.0
        return message

    def _upstream_feedback(self, portable_goal_handle, target_xyz: np.ndarray, upstream_feedback) -> None:
        feedback = MoveToPose.Feedback()
        feedback.current_pose = self._copy_current_pose()
        if self._current_pose is None:
            feedback.progress = 0.0
        else:
            current_xyz = np.array(
                [
                    self._current_pose.pose.position.x,
                    self._current_pose.pose.position.y,
                    self._current_pose.pose.position.z,
                ]
            )
            upstream_current = np.asarray(upstream_feedback.feedback.current_pose[:3]) / 1000.0
            remaining = min(
                float(np.linalg.norm(current_xyz - target_xyz)),
                float(np.linalg.norm(upstream_current - target_xyz)),
            )
            feedback.progress = float(np.clip(1.0 - remaining / 0.35, 0.0, 1.0))
        portable_goal_handle.publish_feedback(feedback)

    async def _execute_move(self, goal_handle) -> MoveToPose.Result:
        result = MoveToPose.Result()
        request = goal_handle.request
        target, (_, _, yaw) = self._goal_pose(request)
        timeout = float(self.get_parameter("upstream_wait_timeout_s").value)
        if not self._ptp_client.wait_for_server(timeout_sec=timeout):
            result.success = False
            result.message = "upstream PTP_action server is unavailable"
            result.achieved_pose = self._copy_current_pose()
            goal_handle.abort()
            return result

        upstream_goal = PointToPoint.Goal()
        upstream_goal.motion_type = PointToPoint.Goal.MOTION_TYPE_MOVJ_XYZ
        upstream_goal.target_pose = [
            1000.0 * target.position[0],
            1000.0 * target.position[1],
            1000.0 * target.position[2],
            math.degrees(yaw),
        ]
        upstream_goal.velocity_ratio = float(request.velocity_scale)
        upstream_goal.acceleration_ratio = float(request.acceleration_scale)

        upstream_handle = await self._ptp_client.send_goal_async(
            upstream_goal,
            feedback_callback=lambda feedback: self._upstream_feedback(
                goal_handle, target.position, feedback
            ),
        )
        if not upstream_handle.accepted:
            result.success = False
            result.message = "upstream Dobot driver rejected the motion"
            result.achieved_pose = self._copy_current_pose()
            goal_handle.abort()
            return result

        upstream_result_future = upstream_handle.get_result_async()
        while not upstream_result_future.done():
            if goal_handle.is_cancel_requested:
                await upstream_handle.cancel_goal_async()
                goal_handle.canceled()
                result.success = False
                result.message = "motion canceled"
                result.achieved_pose = self._copy_current_pose()
                return result
            await asyncio.sleep(0.02)

        wrapped_result = upstream_result_future.result()
        # The upstream action can complete just before the 20 Hz TCP publisher
        # emits its next sample.  Give it one publication interval before
        # checking the portable contract's tolerance.
        await asyncio.sleep(0.06)
        result.achieved_pose = self._copy_current_pose()
        achieved = result.achieved_pose.pose
        position_error = float(
            np.linalg.norm(
                np.array([achieved.position.x, achieved.position.y, achieved.position.z])
                - target.position
            )
        )
        _, _, achieved_yaw = _rpy_from_xyzw(
            [
                achieved.orientation.x,
                achieved.orientation.y,
                achieved.orientation.z,
                achieved.orientation.w,
            ]
        )
        yaw_error = abs((achieved_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi)
        position_tolerance = (
            float(request.position_tolerance_m)
            if request.position_tolerance_m > 0.0
            else 0.005
        )
        orientation_tolerance = (
            float(request.orientation_tolerance_rad)
            if request.orientation_tolerance_rad > 0.0
            else math.radians(1.0)
        )
        upstream_succeeded = wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
        result.success = bool(
            upstream_succeeded
            and position_error <= position_tolerance
            and yaw_error <= orientation_tolerance
        )
        result.message = (
            f"position error {position_error * 1000.0:.2f} mm, "
            f"yaw error {math.degrees(yaw_error):.2f} deg"
            if upstream_succeeded
            else "upstream motion failed"
        )
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    async def _command_end_effector(self, request, response):
        if request.end_effector_name != self.end_effector_name:
            response.success = False
            response.message = f"unknown end effector {request.end_effector_name!r}"
            return response
        command_map = {
            CommandEndEffector.Request.OPEN: "open",
            CommandEndEffector.Request.CLOSE: "close",
        }
        if request.command not in command_map:
            response.success = False
            response.message = "Dobot bridge currently supports OPEN and CLOSE only"
            return response
        timeout = float(self.get_parameter("upstream_wait_timeout_s").value)
        if not self._gripper_client.wait_for_service(timeout_sec=timeout):
            response.success = False
            response.message = "upstream gripper service is unavailable"
            return response

        upstream_request = GripperControl.Request()
        upstream_request.gripper_state = command_map[request.command]
        upstream_request.keep_compressor_running = bool(request.hold)
        upstream_response = await self._gripper_client.call_async(upstream_request)
        response.success = bool(upstream_response.success)
        response.message = str(upstream_response.message)
        response.achieved_position_m = (
            0.027 if request.command == CommandEndEffector.Request.OPEN else 0.0
        )
        return response

    def destroy_node(self) -> bool:
        self._move_server.destroy()
        self._ptp_client.destroy()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DobotPortableBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
