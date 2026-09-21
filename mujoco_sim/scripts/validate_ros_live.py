#!/usr/bin/env python3
"""Read-only live validation of magician_ros2 topics against model conventions.

Run this with the ROS 2/system Python environment, not the MuJoCo Conda
environment.  The node never publishes and never sends a motion command.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

REAR_ARM_LENGTH = 0.135
FOREARM_LENGTH = 0.147


class ReadOnlyProbe(Node):
    def __init__(self) -> None:
        super().__init__("dobot_read_only_validation_probe")
        self.raw_joint: JointState | None = None
        self.urdf_joint: JointState | None = None
        self.pose: Float64MultiArray | None = None
        self.raw_by_stamp: dict[tuple[int, int], JointState] = {}
        self.urdf_by_stamp: dict[tuple[int, int], JointState] = {}
        self.raw_arrivals: list[float] = []
        self.create_subscription(JointState, "/dobot_joint_states", self._on_raw, 20)
        self.create_subscription(JointState, "/joint_states", self._on_urdf, 20)
        self.create_subscription(Float64MultiArray, "/dobot_pose_raw", self._on_pose, 20)

    def _on_raw(self, message: JointState) -> None:
        self.raw_joint = message
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        self.raw_by_stamp[stamp] = message
        self.raw_arrivals.append(time.monotonic())

    def _on_urdf(self, message: JointState) -> None:
        self.urdf_joint = message
        stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
        self.urdf_by_stamp[stamp] = message

    def _on_pose(self, message: Float64MultiArray) -> None:
        self.pose = message


def forward_kinematics(raw: list[float]) -> list[float]:
    theta1, theta2, theta3 = raw[:3]
    radius = REAR_ARM_LENGTH * math.sin(theta2) + FOREARM_LENGTH * math.cos(theta3)
    return [
        radius * math.cos(theta1),
        radius * math.sin(theta1),
        REAR_ARM_LENGTH * math.cos(theta2) - FOREARM_LENGTH * math.sin(theta3),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fk-tolerance-mm", type=float, default=5.0)
    args = parser.parse_args()

    rclpy.init()
    node = ReadOnlyProbe()
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        common_stamps = node.raw_by_stamp.keys() & node.urdf_by_stamp.keys()
        if common_stamps:
            latest_common_stamp = max(common_stamps)
            raw_message = node.raw_by_stamp[latest_common_stamp]
            urdf_message = node.urdf_by_stamp[latest_common_stamp]
        else:
            raw_message = node.raw_joint
            urdf_message = node.urdf_joint
        pose_message = node.pose
        arrivals = node.raw_arrivals.copy()
        node.destroy_node()
        rclpy.shutdown()

    missing = []
    if raw_message is None:
        missing.append("/dobot_joint_states")
    if urdf_message is None:
        missing.append("/joint_states")
    if pose_message is None:
        missing.append("/dobot_pose_raw")
    if missing:
        print(json.dumps({"status": "FAIL", "missing_topics": missing}, indent=2))
        raise SystemExit(1)

    assert raw_message is not None
    assert urdf_message is not None
    assert pose_message is not None
    raw = list(raw_message.position[:4])
    urdf = list(urdf_message.position[:4])
    if len(raw) != 4 or len(urdf) != 4 or len(pose_message.data) < 3:
        print(json.dumps({"status": "FAIL", "reason": "Unexpected message dimensions"}, indent=2))
        raise SystemExit(1)

    expected_urdf = [raw[0], raw[1], raw[2] - raw[1], raw[3]]
    mapping_error = max(abs(actual - expected) for actual, expected in zip(urdf, expected_urdf))
    expected_xyz = forward_kinematics(raw)
    measured_xyz = list(pose_message.data[:3])
    fk_error_m = math.dist(expected_xyz, measured_xyz)

    intervals = [later - earlier for earlier, later in zip(arrivals, arrivals[1:])]
    rate_hz = 1.0 / statistics.mean(intervals) if intervals else 0.0
    checks = {
        "joint_mapping": mapping_error <= 1e-5,
        "forward_kinematics": fk_error_m <= args.fk_tolerance_mm / 1000.0,
        "state_rate": 15.0 <= rate_hz <= 25.0,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "samples": len(arrivals),
        "state_rate_hz": rate_hz,
        "joint_mapping_max_error_rad": mapping_error,
        "fk_error_mm": fk_error_m * 1000.0,
        "raw_joint_deg": [math.degrees(value) for value in raw],
        "ros_joint_deg": [math.degrees(value) for value in urdf],
        "expected_wrist_m": expected_xyz,
        "measured_dobot_pose_m": measured_xyz,
        "note": "Read-only probe: no commands were published.",
    }
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
