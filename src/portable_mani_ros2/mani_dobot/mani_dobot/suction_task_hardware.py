"""Bounded real-Dobot commands with independent, timestamped ROS feedback.

Importing this module does not import ROS or open a serial port.  The task CLI
must only construct ``RosDobotHardware`` after an explicit execution request.
Suction state means the controller output, never a measured vacuum or grasp.
"""

from __future__ import annotations

from collections import deque
import fcntl
import hashlib
import math
import os
import select
import struct
import sys
import time


RAW_JOINT_LIMITS_DEG = ((-125., 125.), (-5., 90.), (-15., 70.), (-150., 150.))
STATE_TOPICS = ("/dobot_TCP", "/dobot_joint_states", "/dobot_alarms")


class _DiscoveryPending(RuntimeError):
    """Only the read-only initial gate may wait for a missing graph entry."""


def _finite_vector(values, length, label):
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain {length} finite numbers") from exc
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain {length} finite numbers")
    return result


def _angle_delta(first, second):
    return (second - first + 180.) % 360. - 180.


def _distance(first, second):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _open_driver(robot_port):
    # Do not import dobot_handle: it creates a global, second serial client.
    from dobot_driver.interface import Interface
    return Interface(robot_port)


class RosDobotHardware:
    """One command client; pose/joints/alarms come only from real ROS state.

    Construction opens the selected port and reads state/parameters. It never
    clears alarms, changes a tool offset, switches suction, or sends motion.
    ``arm()`` is the explicit boundary for configuring and enabling commands.
    """

    def __init__(self, robot_port, expected_bias_mm, velocity_mm_s=10., acceleration_mm_s2=20.):
        self.robot_port = os.fspath(robot_port)
        if not self.robot_port.startswith("/dev/serial/by-id/"):
            raise ValueError("Use the taught robot's persistent /dev/serial/by-id/ port")
        if not os.path.exists(self.robot_port):
            raise ValueError(f"Robot port does not exist: {self.robot_port}")
        self.expected_bias_mm = _finite_vector(expected_bias_mm, 3, "expected tool bias")
        self.velocity_mm_s = float(velocity_mm_s)
        self.acceleration_mm_s2 = float(acceleration_mm_s2)
        if not math.isfinite(self.velocity_mm_s) or not 5. <= self.velocity_mm_s <= 50.:
            raise ValueError("Coordinate velocity must be between 5 and 50 mm/s")
        if not math.isfinite(self.acceleration_mm_s2) or not 5. <= self.acceleration_mm_s2 <= 30.:
            raise ValueError("Coordinate acceleration must be between 5 and 30 mm/s²")
        self._armed = False
        self._commands_started = False
        self._closed = False
        self._bot = None
        self._node = None
        self._context = None
        self._executor = None
        self._command_lock_fd = None
        self._pose = self._joints = self._alarms = None
        self._received = {}
        self._stamps = {}
        self._fault = None
        self._pose_history = deque(maxlen=50)
        self._joint_history = deque(maxlen=50)
        self._suction_on = False
        try:
            lock_id = hashlib.sha256(os.path.realpath(self.robot_port).encode()).hexdigest()[:16]
            self._command_lock_fd = os.open(
                f"/tmp/dobot-suction-task-{lock_id}.lock", os.O_CREAT | os.O_RDWR, 0o600
            )
            try:
                fcntl.flock(self._command_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another suction task command client owns this robot") from exc
            self._start_ros()
            self._wait_initial_feedback()
            self._bot = _open_driver(self.robot_port)
            self._verify_tool_bias()
            self._read_suction()
            self._spin_once(0.)
            self._feedback()
        except BaseException:
            # A failed read-only constructor must not change the controller.
            self._release_resources()
            raise

    def _start_ros(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from dobot_msgs.msg import DobotAlarmCodes
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState

        self._rclpy = rclpy
        self._context = Context()
        rclpy.init(args=[], context=self._context)
        self._node = Node("dobot_suction_task_cycle", context=self._context, use_global_arguments=False)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._subscriptions = [
            self._node.create_subscription(PoseStamped, STATE_TOPICS[0], self._on_pose, 10),
            self._node.create_subscription(JointState, STATE_TOPICS[1], self._on_joints, 10),
            self._node.create_subscription(DobotAlarmCodes, STATE_TOPICS[2], self._on_alarms, 10),
        ]

    def _spin_once(self, timeout=0.02):
        if self._closed or not self._rclpy.ok(context=self._context):
            raise RuntimeError("ROS feedback context is closed")
        self._executor.spin_once(timeout_sec=timeout)
        # A single executor spin handles only one callback. The three feedback
        # subscriptions produce about 50 callbacks/s, exceeding the terminal
        # confirmation loop's cadence. Drain ready work without waiting so a
        # fresh receipt cannot conceal an old header in a growing local queue.
        # Bound the work; genuine source loss still fails the unchanged ages.
        for _ in range(24):
            self._executor.spin_once(timeout_sec=0.)

    def _ros_now(self):
        return self._node.get_clock().now().nanoseconds * 1e-9

    def _record_stamp(self, kind, message):
        stamp = message.header.stamp
        self._stamps[kind] = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        self._received[kind] = time.monotonic()

    def _on_pose(self, message):
        try:
            frame = message.header.frame_id
            if frame not in ("", "magician_base_link"):
                raise ValueError(f"Unexpected real TCP frame: {frame!r}")
            q = message.pose.orientation
            x, y, z, w = _finite_vector((q.x, q.y, q.z, q.w), 4, "TCP quaternion")
            norm = math.sqrt(x*x + y*y + z*z + w*w)
            if abs(norm - 1.) > 0.01:
                raise ValueError("TCP quaternion is not normalized")
            roll = math.atan2(2.*(w*x+y*z), 1.-2.*(x*x+y*y))
            pitch = math.asin(max(-1., min(1., 2.*(w*y-z*x))))
            if max(abs(roll), abs(pitch)) > math.radians(1.):
                raise ValueError("Unexpected roll/pitch in the real Dobot TCP")
            yaw = math.degrees(math.atan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z)))
            p = message.pose.position
            self._pose = _finite_vector((p.x*1000., p.y*1000., p.z*1000., yaw), 4, "TCP")
            self._record_stamp("pose", message)
            self._pose_history.append((self._received["pose"], self._pose))
        except (TypeError, ValueError, AttributeError) as exc:
            self._fault = str(exc)

    def _on_joints(self, message):
        try:
            expected = [f"magician_joint_{index}" for index in range(1, 5)]
            if any(message.name.count(name) != 1 for name in expected):
                raise ValueError("Real raw joint names are missing or duplicated")
            raw = [math.degrees(message.position[message.name.index(name)]) for name in expected]
            self._joints = _finite_vector(raw, 4, "raw vendor joints")
            self._record_stamp("joints", message)
            self._joint_history.append((self._received["joints"], self._joints))
        except (TypeError, ValueError, IndexError, AttributeError) as exc:
            self._fault = str(exc)

    def _on_alarms(self, message):
        try:
            alarms = list(message.alarms_list)
            if not all(isinstance(code, int) and 0 <= code <= 255 for code in alarms):
                raise ValueError("Malformed real alarm list")
            self._alarms = alarms
            self._record_stamp("alarms", message)
        except (TypeError, ValueError, AttributeError) as exc:
            self._fault = str(exc)

    def _assert_graph(self, *, allow_undiscovered=False):
        missing_topics = []
        for topic in STATE_TOPICS:
            publishers = self._node.get_publishers_info_by_topic(topic)
            if not publishers and allow_undiscovered:
                missing_topics.append(topic)
                continue
            if (allow_undiscovered and len(publishers) == 1
                    and publishers[0].node_name == "_NODE_NAME_UNKNOWN_"
                    and publishers[0].node_namespace == "_NODE_NAMESPACE_UNKNOWN_"):
                # DDS may expose the endpoint before participant name metadata.
                # Only this exact, single unresolved identity is retryable in
                # the initial gate. Runtime and all known wrong names fail.
                missing_topics.append(f"{topic} (participant identity pending)")
                continue
            if len(publishers) != 1 or publishers[0].node_name != "dobot_state_publisher":
                details = [(item.node_name, item.node_namespace) for item in publishers]
                raise RuntimeError(f"{topic} must have exactly one real dobot_state_publisher; observed {details}")
            if publishers[0].node_namespace not in ("", "/"):
                raise RuntimeError(f"Unexpected feedback namespace on {topic}")
        forbidden = ("ptp", "homing", "gripper", "suction_cup", "control_panel", "sliding_rail", "jog")
        own_name = self._node.get_name()
        for name, _namespace in self._node.get_node_names_and_namespaces():
            lower = name.lower()
            if name == own_name:
                continue
            if (any(part in lower for part in forbidden)
                    or lower in ("mani_dobot_bridge", "dobot_bridge")
                    or lower.startswith("mani_mujoco")):
                raise RuntimeError(f"Competing motion/simulation node is active: {name}")
        if missing_topics:
            raise _DiscoveryPending(f"Waiting for ROS graph discovery of {missing_topics}")

    def _feedback(self, *, allow_undiscovered=False):
        self._assert_graph(allow_undiscovered=allow_undiscovered)
        if self._fault:
            raise RuntimeError(f"Invalid feedback: {self._fault}")
        now, ros_now = time.monotonic(), self._ros_now()
        for kind in ("pose", "joints", "alarms"):
            age = now - self._received.get(kind, -math.inf)
            header_age = ros_now - self._stamps.get(kind, -math.inf)
            if not 0. <= age <= 0.4:
                raise RuntimeError(f"Missing/stale {kind} receipt ({age:.3f} s)")
            if not -0.05 <= header_age <= 0.5 or self._stamps[kind] <= 0.:
                raise RuntimeError(f"Missing/stale/future {kind} header ({header_age:.3f} s)")
        if abs(self._stamps["pose"] - self._stamps["joints"]) > 0.15:
            raise RuntimeError("TCP and raw-joint feedback timestamps disagree")
        if self._alarms:
            raise RuntimeError(f"Dobot alarms: {self._alarms}")
        for value, (lower, upper) in zip(self._joints, RAW_JOINT_LIMITS_DEG):
            if not lower - 0.05 <= value <= upper + 0.05:
                raise RuntimeError(f"Raw vendor joint outside limits: {self._joints}")
        return {"pose": list(self._pose), "joints": list(self._joints),
                "alarms": list(self._alarms), "suction_on": self._suction_on}

    def _is_stable(self, since=-math.inf):
        now = time.monotonic()
        history = [(stamp, pose) for stamp, pose in self._pose_history
                   if stamp >= since and now - stamp <= 0.4]
        if len(history) < 5 or history[-1][0] - history[0][0] < 0.25:
            return False
        reference = history[-1][1]
        joint_history = [(stamp, joints) for stamp, joints in self._joint_history
                         if stamp >= since and now - stamp <= 0.4]
        if len(joint_history) < 5 or joint_history[-1][0] - joint_history[0][0] < 0.25:
            return False
        joint_reference = joint_history[-1][1]
        return (all(_distance(pose[:3], reference[:3]) <= 0.15
                    and abs(_angle_delta(pose[3], reference[3])) <= 0.15
                    for _, pose in history)
                and all(max(abs(a-b) for a, b in zip(joints, joint_reference)) <= 0.2
                        for _, joints in joint_history))

    def _wait_initial_feedback(self):
        deadline = time.monotonic() + 3.
        graph_ready_since = None
        discovery_reason = None
        while time.monotonic() < deadline:
            self._spin_once()
            try:
                # Discoveries may lag data delivery. Only zero endpoints are
                # retryable here; wrong/duplicate endpoints and competing
                # writers are rejected even before all messages have arrived.
                self._assert_graph(allow_undiscovered=True)
                if len(self._received) != 3:
                    continue
                self._feedback(allow_undiscovered=True)
            except _DiscoveryPending as exc:
                graph_ready_since = None
                discovery_reason = str(exc)
                continue
            if graph_ready_since is None:
                graph_ready_since = time.monotonic()
            if self._is_stable(since=graph_ready_since):
                return
        suffix = f"; {discovery_reason}" if discovery_reason else ""
        raise RuntimeError("Timed out waiting for fresh, stable real Dobot feedback" + suffix)

    def _query(self, command_id, expected_size):
        """Read one checked getter packet under the driver's process lock.

        The upstream Message parser only admits pose/alarm IDs and has broken
        parameter/suction getters, so validate these read-only packets here.
        No pose getter is used by this command client.
        """
        if command_id not in (60, 62, 80, 81, 83):
            raise ValueError("Only explicitly allowed parameter getters are supported")
        with self._bot._transaction():
            serial = self._bot.serial
            serial.reset_input_buffer()
            serial.write(bytes((0xAA, 0xAA, 2, command_id, 0, (-command_id) & 0xFF)))
            serial.flush()
            header = serial.read(3)
            if len(header) != 3 or header[:2] != b"\xaa\xaa" or header[2] != expected_size + 2:
                raise RuntimeError(f"Invalid/missing getter {command_id} response header")
            payload = serial.read(header[2] + 1)
            if (len(payload) != header[2] + 1 or payload[0] != command_id
                    or payload[1] != 0 or sum(payload) % 256 != 0):
                raise RuntimeError(f"Invalid/missing getter {command_id} response/checksum")
            return bytes(payload[2:-1])

    def _query_floats(self, command_id, count):
        return _finite_vector(struct.unpack("<" + "f"*count, self._query(command_id, count*4)),
                              count, f"getter {command_id}")

    def _read_suction(self):
        values = self._query(62, 2)
        if values not in (b"\0\0", b"\1\1"):
            raise RuntimeError(f"Inconsistent suction output state: {list(values)}")
        self._suction_on = values == b"\1\1"
        return self._suction_on

    def _verify_tool_bias(self):
        actual_bias = self._query_floats(60, 3)
        if any(abs(a-b) > 0.05 for a, b in zip(actual_bias, self.expected_bias_mm)):
            raise RuntimeError(f"Current tool bias {actual_bias} differs from taught bias {self.expected_bias_mm}")

    def state(self):
        try:
            self._spin_once(0.)
            self._feedback()
            self._read_suction()
            self._spin_once(0.)
            return self._feedback()
        except BaseException:
            self._emergency_stop()
            raise

    def spin_for(self, seconds):
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0.:
            raise ValueError("Spin duration must be finite and nonnegative")
        try:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                self._spin_once(min(0.02, max(0., deadline-time.monotonic())))
                self._feedback()
        except BaseException:
            self._emergency_stop()
            raise

    def _configure_speed(self, velocity, *, configure_common):
        acceleration = self.acceleration_mm_s2
        self._bot.set_point_to_point_joint_params([velocity]*4, [acceleration]*4, queue=False)
        self._bot.set_point_to_point_coordinate_params(velocity, velocity, acceleration, acceleration, queue=False)
        if configure_common:
            self._bot.set_point_to_point_common_params(100., 100., queue=False)
        expected = ((80, [velocity]*4+[acceleration]*4),
                    (81, [velocity, velocity, acceleration, acceleration]), (83, [100., 100.]))
        for command_id, values in expected:
            observed = self._query_floats(command_id, len(values))
            if any(abs(a-b) > 0.01 for a, b in zip(observed, values)):
                raise RuntimeError(f"PTP parameter readback mismatch for {command_id}: {observed}")
            self._spin_once(0.)
            self._feedback()

    def set_speed(self, velocity_mm_s):
        """Change a stationary armed robot's speed, preserving acceleration/pump.

        Returns the verified new limit in mm/s. No motion or queue start/stop
        is sent on success; a failure stops/clears motion and retains suction.
        """
        try:
            if not self._armed:
                raise RuntimeError("Call arm() before changing speed")
            velocity = float(velocity_mm_s)
            if not math.isfinite(velocity) or not 5. <= velocity <= 50.:
                raise ValueError("Coordinate velocity must be between 5 and 50 mm/s")
            state = self.state()
            if not self._is_stable():
                raise RuntimeError("Speed changes require a stationary robot")
            self._configure_speed(velocity, configure_common=False)
            if self._read_suction() is not state["suction_on"]:
                raise RuntimeError("Suction output changed while updating speed")
            self._spin_once(0.)
            self._feedback()
            self.velocity_mm_s = velocity
            return velocity
        except BaseException:
            self._emergency_stop()
            raise

    def arm(self, *, holding_confirmed=False):
        try:
            if type(holding_confirmed) is not bool:
                raise ValueError("holding_confirmed must be an explicit boolean")
            state = self.state()
            if state["suction_on"] and not holding_confirmed:
                raise RuntimeError("Cannot arm with suction already on; resolve the current held object")
            if holding_confirmed and not state["suction_on"]:
                raise RuntimeError("Holding-confirmed arming requires suction already on")
            if not self._is_stable():
                raise RuntimeError("Cannot arm while the robot is moving")
            self._verify_tool_bias()
            # Read before changing: malformed replies are a preflight failure.
            for command_id, count in ((80, 8), (81, 4), (83, 2)):
                if any(value <= 0. for value in self._query_floats(command_id, count)):
                    raise RuntimeError(f"Invalid PTP parameters in getter {command_id}")
                self._spin_once(0.)
                self._feedback()
            self._commands_started = True
            self.stop()
            self._configure_speed(self.velocity_mm_s, configure_common=True)
            if self._read_suction() is not state["suction_on"]:
                raise RuntimeError("Suction changed during preflight")
            self._spin_once(0.)
            self._feedback()
            # arm() already force-stopped and cleared the queue above. Start
            # only this empty queue; fault paths never start or resume it.
            self._bot.start_queue()
            self._armed = True
        except BaseException:
            self._emergency_stop()
            raise

    @staticmethod
    def _segment_geometry(start, goal, pose):
        vector = [b-a for a, b in zip(start[:3], goal[:3])]
        length_squared = sum(value*value for value in vector)
        if length_squared < 1e-12:
            if _distance(start[:3], pose[:3]) > 1.:
                raise RuntimeError("TCP deviated from pure-yaw segment")
            low, high = sorted((0., _angle_delta(start[3], goal[3])))
            actual = _angle_delta(start[3], pose[3])
            if actual < low-1. or actual > high+1.:
                raise RuntimeError("Yaw deviated from commanded segment")
            return
        fraction = sum((p-a)*v for p, a, v in zip(pose[:3], start[:3], vector)) / length_squared
        closest = [a + min(1., max(0., fraction))*v for a, v in zip(start[:3], vector)]
        if _distance(pose[:3], closest) > 1.:
            raise RuntimeError("TCP departed from MOVL line by more than 1 mm")
        expected_yaw = start[3] + _angle_delta(start[3], goal[3])*min(1., max(0., fraction))
        if abs(_angle_delta(expected_yaw, pose[3])) > 1.:
            raise RuntimeError("Yaw departed from MOVL interpolation by more than 1 degree")

    def move_segment(self, goal4):
        """Execute one small XYZ/yaw step with the original contact limits."""
        return self._move_checked(goal4, transfer=False)

    def move_transfer(self, goal4):
        """Execute one continuous, fixed-heading XY move on the high plane.

        Only this high-plane operation can exceed the contact step size. The
        ideal firmware path is checked densely before sending its one endpoint;
        all live-feedback, alarm and line-deviation interlocks remain active.
        """
        return self._move_checked(goal4, transfer=True)

    def _move_checked(self, goal4, *, transfer):
        try:
            if not self._armed:
                raise RuntimeError("Call arm() before motion")
            goal = _finite_vector(goal4, 4, "segment goal XYZ mm/yaw degrees")
            initial_state = self.state()
            start = initial_state["pose"]
            length = _distance(start[:3], goal[:3])
            yaw_change = abs(_angle_delta(start[3], goal[3]))
            if transfer:
                self._validate_transfer(start, goal, length, yaw_change)
            elif length > 5.01 or yaw_change > 2.0001:
                raise ValueError("MOVL segment exceeds 5.01 mm or 2 degrees")
            if not self._is_stable():
                raise RuntimeError("Robot has not settled before the next MOVL segment")
            if transfer and length < 1e-9:
                return initial_state
            started = time.monotonic()
            self._bot.set_point_to_point_command(2, *goal, queue=False)
            timeout = max(4., 4.*length/self.velocity_mm_s + 4.*yaw_change/self.velocity_mm_s + 2.)
            if transfer:
                timeout = max(timeout, 4.*math.sqrt(length/self.acceleration_mm_s2) + 2.)
            while time.monotonic() - started <= timeout:
                self._spin_once()
                state = self._feedback()
                pose = state["pose"]
                self._segment_geometry(start, goal, pose)
                if (_distance(pose[:3], goal[:3]) <= 0.25
                        and abs(_angle_delta(pose[3], goal[3])) <= 0.3
                        and self._is_stable(since=started)):
                    return state
            raise RuntimeError(f"MOVL segment timed out after {timeout:.1f} s")
        except BaseException:
            self._emergency_stop()
            raise

    def _validate_transfer(self, start, goal, length, yaw_change):
        if length > 250.000001:
            raise ValueError("High-plane MOVL transfer exceeds 250 mm")
        if abs(goal[2]-start[2]) > 0.25 or min(start[2], goal[2]) < -10.1:
            raise ValueError("Long transfer requires the same high plane at Z >= -10.1 mm")
        if yaw_change > 0.300001:
            raise ValueError("Long transfer must retain fixed yaw within 0.3 degree")
        from .suction_task_plan import firmware_ik

        # Use this installation's verified bias, never the MuJoCo tool offset.
        # At most 251 samples because length is bounded above by 250 mm.
        count = max(1, math.ceil(length))
        yaw_delta = _angle_delta(start[3], goal[3])
        for index in range(count+1):
            fraction = index/count
            pose = tuple(a+(b-a)*fraction for a, b in zip(start[:3], goal[:3])) + (
                start[3]+yaw_delta*fraction,
            )
            try:
                firmware_ik(pose, self.expected_bias_mm)
            except ValueError as exc:
                raise ValueError(f"High-plane transfer is invalid at fraction {fraction:.6f}: {exc}") from exc

    def suction(self, on):
        try:
            if not self._armed:
                raise RuntimeError("Call arm() before suction commands")
            if type(on) is not bool:
                raise ValueError("Suction command must be boolean")
            self._spin_once(0.)
            self._feedback()
            if not self._is_stable():
                raise RuntimeError("Suction switching requires a stationary robot")
            self._bot.set_end_effector_suction_cup(on, on, queue=False)
            observed = self._read_suction()
            if observed is not on:
                raise RuntimeError("Suction output readback does not match the command")
            self._spin_once(0.)
            self._feedback()
            # This is output-settling time, not evidence of vacuum or release.
            # Continue monitoring feedback while the pneumatic system reacts.
            self.spin_for(0.6)
            if self._read_suction() is not on:
                raise RuntimeError("Suction output changed during pneumatic settling")
            self._spin_once(0.)
            self._feedback()
            return observed
        except BaseException:
            self._emergency_stop()
            raise

    def confirm(self, prompt, exact):
        """Read one exact onsite confirmation while continuously checking ROS."""
        try:
            if not isinstance(exact, str) or not exact.strip() or "\n" in exact:
                raise ValueError("Confirmation requires a nonempty exact single-line phrase")
            if not sys.stdin.isatty():
                raise RuntimeError("Execution confirmation requires an interactive terminal")
            print(f"{prompt}\nType exactly: {exact}", flush=True)
            while True:
                self._spin_once(0.01)
                self._feedback()
                ready, _, _ = select.select([sys.stdin], [], [], 0.04)
                if not ready:
                    continue
                line = sys.stdin.readline()
                if line == "":
                    raise RuntimeError("Confirmation input reached EOF; execution stopped")
                if line.strip().casefold() in ("stop", "cancel", "停止", "取消"):
                    raise RuntimeError("Operator canceled execution; motion stopped, suction retained")
                if line.rstrip("\r\n") == exact:
                    return True
                print(f"Not confirmed. Type exactly: {exact}", flush=True)
        except BaseException:
            self._emergency_stop()
            raise

    def stop(self):
        """Stop and clear motion only. Never disable suction or resume a queue."""
        self._armed = False
        if self._closed or self._bot is None:
            return
        errors = []
        for operation in (lambda: self._bot.stop_queue(force=True), self._bot.clear_queue):
            try:
                operation()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("Motion stop/clear failed: " + "; ".join(errors))

    def _emergency_stop(self):
        if not self._commands_started:
            return
        try:
            self.stop()
        except Exception as exc:
            print(f"URGENT: {exc}; use the onsite stop control", file=sys.stderr, flush=True)

    def _release_resources(self):
        self._armed = False
        self._closed = True
        operations = []
        if self._bot is not None:
            operations.append(self._bot.serial.close)
            descriptor = getattr(self._bot, "_process_lock_fd", None)
            if descriptor is not None:
                operations.append(lambda descriptor=descriptor: os.close(descriptor))
        if self._executor is not None:
            if self._node is not None:
                operations.append(lambda: self._executor.remove_node(self._node))
            operations.append(self._executor.shutdown)
        if self._node is not None:
            operations.append(self._node.destroy_node)
        if self._context is not None and self._rclpy.ok(context=self._context):
            operations.append(lambda: self._rclpy.shutdown(context=self._context))
        if self._command_lock_fd is not None:
            descriptor = self._command_lock_fd
            self._command_lock_fd = None
            operations.append(lambda descriptor=descriptor: os.close(descriptor))
        # A failed serial close must not retain the exclusive task lock or ROS
        # resources, and must not mask the original preflight/motion failure.
        for operation in operations:
            try:
                operation()
            except Exception as exc:
                print(f"Resource cleanup failed: {exc}", file=sys.stderr, flush=True)

    def close(self):
        if self._closed:
            return
        try:
            if self._commands_started:
                self.stop()
        finally:
            self._release_resources()
