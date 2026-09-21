"""Batch evaluation of the ROS RGB-D-to-physical-pick simulation loop."""

from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from mani_interfaces.action import PickObject
from mani_interfaces.msg import SceneObjectArray
from mani_interfaces.srv import ResetScene
from rclpy.action import ActionClient
from rclpy.node import Node


class ClosedLoopEvaluator(Node):
    def __init__(self) -> None:
        super().__init__("mani_closed_loop_evaluator")
        self.declare_parameter("trials", 10)
        self.declare_parameter("seed", 20260918)
        self.declare_parameter("settle_sec", 0.6)
        self.declare_parameter("max_localization_error_m", 0.006)
        self.declare_parameter("min_success_rate", 0.90)
        self._localized = {}
        self._scene_sub = self.create_subscription(
            SceneObjectArray,
            "/mani/scene_objects",
            self._on_scene,
            10,
        )
        self._reset = self.create_client(ResetScene, "/mani/sim/reset_scene")
        self._pick = ActionClient(self, PickObject, "/mani/pick_object")

    def _on_scene(self, message: SceneObjectArray) -> None:
        self._localized = {item.uuid: item for item in message.objects}

    def _spin_for(self, duration_sec: float) -> None:
        deadline = time.monotonic() + duration_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, deadline - time.monotonic()))

    def _wait_for_localized(self, uuid: str, timeout_sec: float = 2.0):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            item = self._localized.get(uuid)
            if item is not None:
                return item
            rclpy.spin_once(self, timeout_sec=0.05)
        return None

    def evaluate(self) -> dict:
        trials = int(self.get_parameter("trials").value)
        seed = int(self.get_parameter("seed").value)
        settle_sec = float(self.get_parameter("settle_sec").value)
        max_error = float(self.get_parameter("max_localization_error_m").value)
        min_success_rate = float(self.get_parameter("min_success_rate").value)
        if trials <= 0:
            raise ValueError("trials must be positive")
        if not self._reset.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("reset service unavailable")
        if not self._pick.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("pick action unavailable")

        records = []
        pick_successes = 0
        localization_passes = 0
        for trial in range(trials):
            self._localized = {}
            reset = ResetScene.Request()
            reset.randomize = True
            reset.seed = seed + trial
            reset_future = self._reset.call_async(reset)
            rclpy.spin_until_future_complete(self, reset_future, timeout_sec=5.0)
            response = reset_future.result()
            if response is None or not response.success:
                records.append(
                    {"trial": trial, "error": "scene reset failed", "pick": False}
                )
                continue

            self._spin_for(settle_sec)
            localized = self._wait_for_localized(response.object.uuid)
            if localized is None:
                records.append(
                    {
                        "trial": trial,
                        "error": "localized object unavailable",
                        "pick": False,
                    }
                )
                continue
            truth_position = response.object.pose.pose.position
            estimate_position = localized.pose.pose.position
            truth = np.array(
                [truth_position.x, truth_position.y, truth_position.z],
                dtype=np.float64,
            )
            estimate = np.array(
                [estimate_position.x, estimate_position.y, estimate_position.z],
                dtype=np.float64,
            )
            error = float(np.linalg.norm(estimate - truth))
            if error <= max_error:
                localization_passes += 1

            goal = PickObject.Goal()
            goal.object_uuid = response.object.uuid
            goal.group_name = "arm"
            goal.end_effector_name = "gripper"
            goal.approach_distance_m = 0.08
            goal.lift_distance_m = 0.09
            goal.velocity_scale = 0.5
            goal.acceleration_scale = 0.5
            send_future = self._pick.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
            handle = send_future.result()
            if handle is None or not handle.accepted:
                records.append(
                    {
                        "trial": trial,
                        "localization_error_mm": round(error * 1000.0, 3),
                        "error": "pick goal rejected",
                        "pick": False,
                    }
                )
                continue
            result_future = handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)
            wrapped = result_future.result()
            success = bool(wrapped is not None and wrapped.result.success)
            lift = 0.0 if wrapped is None else float(wrapped.result.object_lift_m)
            if success:
                pick_successes += 1
            records.append(
                {
                    "trial": trial,
                    "seed": seed + trial,
                    "truth_xyz_m": np.round(truth, 5).tolist(),
                    "estimate_xyz_m": np.round(estimate, 5).tolist(),
                    "localization_error_mm": round(error * 1000.0, 3),
                    "pick": success,
                    "visual_lift_mm": round(lift * 1000.0, 2),
                }
            )

        success_rate = pick_successes / trials
        localization_rate = localization_passes / trials
        passed = success_rate >= min_success_rate and localization_rate == 1.0
        return {
            "passed": passed,
            "trials": trials,
            "pick_successes": pick_successes,
            "pick_success_rate": success_rate,
            "localization_within_threshold": localization_passes,
            "localization_pass_rate": localization_rate,
            "max_localization_error_mm": max_error * 1000.0,
            "required_pick_success_rate": min_success_rate,
            "records": records,
        }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ClosedLoopEvaluator()
    exit_code = 1
    try:
        report = node.evaluate()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        exit_code = 0 if report["passed"] else 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
