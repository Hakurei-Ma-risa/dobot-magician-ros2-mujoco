"""One command for the whole chain:  ros2 run mani_vision pick "the red can"

Steps (each one is a plain ROS call you can also type by hand):
  1. park the arm out of the camera view      (action /mani/move_to_pose)
  2. send the instruction                     (topic  /mani/vision/instruction)
  3. wait for the choice                      (topic  /mani/vision/choice)
  4. move back to the ready pose, wait until the target has a 3-D position
                                              (topic  /mani/scene_objects)
  5. pick it                                  (action /mani/pick_object)

The default park / ready paths fit the MuJoCo scene (camera on the right of
the robot). Moves are straight lines, so a path goes around the robot base
in short steps instead of cutting across it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from mani_interfaces.action import MoveToPose, PickObject
from mani_interfaces.msg import SceneObjectArray
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String


class PickByWords(Node):
    def __init__(self) -> None:
        super().__init__("mani_pick_by_words")
        self.choice: dict | None = None
        self.target_seen_at: float | None = None
        self.target_z: float | None = None
        self._instruction = self.create_publisher(String, "/mani/vision/instruction", 10)
        self.create_subscription(String, "/mani/vision/choice", self._on_choice, 10)
        self.create_subscription(SceneObjectArray, "/mani/scene_objects", self._on_scene, 10)
        self.move = ActionClient(self, MoveToPose, "/mani/move_to_pose")
        self.pick = ActionClient(self, PickObject, "/mani/pick_object")

    def _on_choice(self, message: String) -> None:
        self.choice = json.loads(message.data)

    def _on_scene(self, message: SceneObjectArray) -> None:
        for obj in message.objects:
            if obj.uuid == "target":
                self.target_seen_at = time.time()
                self.target_z = obj.pose.pose.position.z

    def wait_for(self, condition, timeout_s: float) -> bool:
        end = time.time() + timeout_s
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if condition():
                return True
        return False

    def follow_path(self, path: str, speed: float, title: str) -> bool:
        if path.lower() == "none":
            return True
        points = [tuple(float(v) for v in step.split(",")) for step in path.split(";") if step.strip()]
        print(title + ": " + " -> ".join(f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in points), flush=True)
        for x, y, z in points:
            goal = MoveToPose.Goal()
            goal.group_name = "arm"
            goal.target = PoseStamped()
            goal.target.header.frame_id = "magician_base_link"
            goal.target.pose.position.x, goal.target.pose.position.y, goal.target.pose.position.z = x, y, z
            goal.target.pose.orientation.w = 1.0
            goal.controlled_axes = [True, True, True, False, False, False]
            goal.position_tolerance_m = 0.01
            goal.orientation_tolerance_rad = 3.2
            goal.velocity_scale = goal.acceleration_scale = speed
            result = self.run_action(self.move, goal, 30.0)
            if not result.success:
                print("      move failed: " + result.message)
                return False
        return True

    def run_action(self, client: ActionClient, goal, timeout_s: float):
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"action server {client._action_name} is not running")
        sent = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=10.0)
        handle = sent.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"{client._action_name} rejected the goal")
        done = handle.get_result_async()
        rclpy.spin_until_future_complete(self, done, timeout_sec=timeout_s)
        if done.result() is None:
            raise RuntimeError(f"{client._action_name} timed out")
        return done.result().result


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick the object described in words.")
    parser.add_argument("instruction", help='e.g. "the red can" or "the block on the left"')
    parser.add_argument("--park", default="0.26,0.0,0.14;0.20,0.12,0.12;0.10,0.18,0.12",
                        help="path x,y,z;x,y,z (metres, base frame) out of the camera view; 'none' to skip")
    parser.add_argument("--ready", default="0.20,0.12,0.12;0.24,0.0,0.12",
                        help="path back to a pose from which picks start; 'none' to skip")
    parser.add_argument("--speed", type=float, default=0.5, help="velocity/acceleration scale 0..1")
    parser.add_argument("--approach", type=float, default=0.05,
                        help="metres above the grasp height to start the descent")
    parser.add_argument("--lift", type=float, default=0.05,
                        help="metres to lift; near the robot (r < 0.21 m) the Dobot cannot go much higher")
    parser.add_argument("--choice-timeout", type=float, default=40.0)
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = PickByWords()
    try:
        if not node.follow_path(args.park, args.speed, "[1/5] park the arm out of the camera view"):
            return 2
        print(f"[2/5] instruction: {args.instruction!r}", flush=True)
        time.sleep(0.3)  # let the camera see the parked arm
        node.choice = None
        node._instruction.publish(String(data=args.instruction))
        if not node.wait_for(lambda: node.choice is not None, args.choice_timeout):
            print("      no answer from choose_target (is it running?)")
            return 3
        c = node.choice
        print(f"[3/5] chose box {c['index']} of {c['candidates']} ({c['selector']}, {c['seconds']} s): {c['reason']}", flush=True)
        if c["index"] < 0:
            return 4
        if not node.follow_path(args.ready, args.speed, "      back to the ready pose"):
            return 2
        node.target_seen_at = None
        if not node.wait_for(lambda: node.target_seen_at is not None, 10.0):
            print("      the localizer did not report the target (is rgbd_localizer running?)")
            return 5
        print(f"[4/5] target located, centre height {node.target_z:.3f} m", flush=True)
        goal = PickObject.Goal()
        goal.object_uuid = "target"
        goal.group_name, goal.end_effector_name = "arm", "gripper"
        goal.approach_distance_m, goal.lift_distance_m = args.approach, args.lift
        goal.velocity_scale = goal.acceleration_scale = args.speed
        result = node.run_action(node.pick, goal, 90.0)
        status = "SUCCEEDED" if result.success else f"FAILED at {result.failure_stage}"
        print(f"[5/5] pick {status}: {result.message}", flush=True)
        if not result.success and result.failure_stage == "verify":
            print("      The camera could not confirm the lift. Look at the simulator: if the object\n"
                  "      is in the gripper, the pick worked. With the arm in view, the depth image can\n"
                  "      join the object to its neighbours and the follower loses it (README, Limits).", flush=True)
        return 0 if result.success else 1
    except RuntimeError as exc:
        print("error:", exc)
        return 6
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
