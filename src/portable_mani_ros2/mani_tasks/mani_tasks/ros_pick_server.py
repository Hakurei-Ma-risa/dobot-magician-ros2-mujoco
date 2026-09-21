"""ROS 2 action server implementing the portable top-down pick state machine."""

from __future__ import annotations

import numpy as np
import rclpy
import time
from mani_core import Pose, SceneObjectState
from mani_interfaces.action import MoveToPose, PickObject
from mani_interfaces.msg import SceneObjectArray
from mani_interfaces.srv import CommandEndEffector
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .pick import PickConfig, PickStage, plan_top_down_pick


class PickServer(Node):
    def __init__(self) -> None:
        super().__init__("mani_pick_server")
        self._callbacks = ReentrantCallbackGroup()
        self._objects: dict[str, SceneObjectState] = {}
        self._move = ActionClient(
            self,
            MoveToPose,
            "/mani/move_to_pose",
            callback_group=self._callbacks,
        )
        self._gripper = self.create_client(
            CommandEndEffector,
            "/mani/command_end_effector",
            callback_group=self._callbacks,
        )
        self._subscription = self.create_subscription(
            SceneObjectArray,
            "/mani/scene_objects",
            self._objects_callback,
            10,
            callback_group=self._callbacks,
        )
        self._server = ActionServer(
            self,
            PickObject,
            "/mani/pick_object",
            execute_callback=self._execute,
            goal_callback=self._goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=self._callbacks,
        )
        self.get_logger().info("Portable top-down pick server is ready")

    @staticmethod
    def _from_message(message) -> SceneObjectState:
        pose = message.pose.pose
        return SceneObjectState(
            uuid=message.uuid,
            class_id=message.class_id,
            pose=Pose(
                position=np.array([pose.position.x, pose.position.y, pose.position.z]),
                quaternion_xyzw=np.array(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                ),
                frame_id=message.header.frame_id or "magician_base_link",
            ),
            size=np.array([message.size.x, message.size.y, message.size.z]),
            confidence=message.confidence,
            source=message.source,
        )

    def _objects_callback(self, message: SceneObjectArray) -> None:
        self._objects = {
            item.uuid: self._from_message(item) for item in message.objects
        }

    def _goal(self, request: PickObject.Goal) -> GoalResponse:
        if request.object_uuid not in self._objects:
            self.get_logger().warning(
                f"Rejecting pick: object {request.object_uuid!r} is unavailable"
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    @staticmethod
    def _feedback(goal_handle, stage: PickStage, progress: float) -> None:
        feedback = PickObject.Feedback()
        feedback.stage = stage.value
        feedback.progress = float(progress)
        goal_handle.publish_feedback(feedback)

    async def _gripper_command(self, name: str, command: int) -> tuple[bool, str]:
        if not self._gripper.wait_for_service(timeout_sec=2.0):
            return False, "end-effector service unavailable"
        request = CommandEndEffector.Request()
        request.end_effector_name = name
        request.command = command
        request.hold = True
        response = await self._gripper.call_async(request)
        return bool(response.success), str(response.message)

    async def _move_to(
        self, goal_request: PickObject.Goal, pose: Pose
    ) -> tuple[bool, str]:
        if not self._move.wait_for_server(timeout_sec=2.0):
            return False, "move action unavailable"
        goal = MoveToPose.Goal()
        goal.group_name = goal_request.group_name or "arm"
        goal.target.header.frame_id = pose.frame_id
        goal.target.pose.position.x, goal.target.pose.position.y, goal.target.pose.position.z = (
            float(value) for value in pose.position
        )
        (
            goal.target.pose.orientation.x,
            goal.target.pose.orientation.y,
            goal.target.pose.orientation.z,
            goal.target.pose.orientation.w,
        ) = (float(value) for value in pose.quaternion_xyzw)
        goal.controlled_axes = [True, True, True, False, False, True]
        goal.position_tolerance_m = 0.005
        goal.orientation_tolerance_rad = float(np.deg2rad(1.0))
        goal.velocity_scale = min(max(float(goal_request.velocity_scale), 0.2), 1.0)
        goal.acceleration_scale = min(
            max(float(goal_request.acceleration_scale), 0.2), 1.0
        )
        handle = await self._move.send_goal_async(goal)
        if not handle.accepted:
            return False, "move goal rejected"
        wrapped = await handle.get_result_async()
        return bool(wrapped.result.success), str(wrapped.result.message)

    async def _execute(self, goal_handle) -> PickObject.Result:
        request = goal_handle.request
        result = PickObject.Result()
        target = self._objects.get(request.object_uuid)
        if target is None:
            result.failure_stage = "lookup"
            result.message = "object disappeared before execution"
            goal_handle.abort()
            return result

        config = PickConfig(
            group_name=request.group_name or "arm",
            end_effector_name=request.end_effector_name or "gripper",
            approach_distance_m=(
                request.approach_distance_m
                if request.approach_distance_m > 0.0
                else 0.080
            ),
            lift_distance_m=(
                request.lift_distance_m if request.lift_distance_m > 0.0 else 0.090
            ),
        )
        plan = plan_top_down_pick(target, config)
        initial_z = float(target.pose.position[2])

        stages = [
            (PickStage.OPEN, None),
            (PickStage.PREGRASP, plan.pregrasp_pose),
            (PickStage.DESCEND, plan.grasp_pose),
            (PickStage.CLOSE, None),
            (PickStage.LIFT, plan.lift_pose),
        ]
        for index, (stage, pose) in enumerate(stages):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.failure_stage = stage.value
                result.message = "pick canceled"
                return result
            self._feedback(goal_handle, stage, index / len(stages))
            if stage is PickStage.OPEN:
                success, message = await self._gripper_command(
                    config.end_effector_name, CommandEndEffector.Request.OPEN
                )
            elif stage is PickStage.CLOSE:
                success, message = await self._gripper_command(
                    config.end_effector_name, CommandEndEffector.Request.CLOSE
                )
            else:
                success, message = await self._move_to(request, pose)
            if not success:
                result.failure_stage = stage.value
                result.message = message
                goal_handle.abort()
                return result

        self._feedback(goal_handle, PickStage.VERIFY, 0.95)
        # Allow the RGB-D/localizer pipeline to observe the post-lift state.
        # This is also required on hardware, where perception is asynchronous.
        time.sleep(0.5)
        final = self._objects.get(request.object_uuid)
        object_lift = 0.0 if final is None else float(final.pose.position[2]) - initial_z
        result.object_lift_m = object_lift
        result.success = object_lift >= config.required_object_lift_m
        result.failure_stage = "" if result.success else PickStage.VERIFY.value
        result.message = f"object lifted {object_lift * 1000.0:.1f} mm"
        if result.success:
            self._feedback(goal_handle, PickStage.COMPLETE, 1.0)
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def destroy_node(self) -> bool:
        self._server.destroy()
        self._move.destroy()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickServer()
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
