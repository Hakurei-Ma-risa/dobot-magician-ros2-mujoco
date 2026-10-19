"""Real command interlocks tested with in-memory ROS/serial substitutes only."""

from collections import deque
from contextlib import nullcontext
import io
import math
import struct
from types import SimpleNamespace as NS

import pytest

from mani_dobot import suction_task_hardware as hardware


class Clock:
    value = 10.

    def now(self):
        return self.value


class FakeSerial:
    def __init__(self, driver):
        self.driver = driver
        self.pending = b""
        self.queries = []
        self.closed = False
        self.corrupt = None

    def reset_input_buffer(self):
        self.pending = b""

    def write(self, packet):
        assert packet[:3] == b"\xaa\xaa\x02"
        command_id = packet[3]
        assert packet[4] == 0 and sum(packet[3:]) % 256 == 0
        self.queries.append(command_id)
        values = self.driver.parameters[command_id]
        parameters = bytes(values) if command_id == 62 else struct.pack("<"+"f"*len(values), *values)
        payload = bytes((command_id, 0)) + parameters
        response = b"\xaa\xaa" + bytes((len(payload),)) + payload + bytes(((-sum(payload)) & 255,))
        self.pending = self.corrupt(response) if self.corrupt else response

    def flush(self):
        pass

    def read(self, count):
        value, self.pending = self.pending[:count], self.pending[count:]
        return value

    def close(self):
        self.closed = True


class FakeDriver:
    def __init__(self):
        self.parameters = {60: [59.7, 0., 0.], 62: [0, 0], 80: [50.]*8,
                           81: [50.]*4, 83: [100., 100.]}
        self.serial = FakeSerial(self)
        self.actions = []
        self.target = None

    def _transaction(self):
        return nullcontext()

    def stop_queue(self, *, force):
        self.actions.append(("stop", force))

    def clear_queue(self):
        self.actions.append(("clear",))

    def start_queue(self):
        self.actions.append(("start",))

    def set_point_to_point_joint_params(self, velocity, acceleration, *, queue):
        assert queue is False
        self.actions.append(("joint_params",))
        self.parameters[80] = velocity + acceleration

    def set_point_to_point_coordinate_params(self, *values, queue):
        assert queue is False
        self.actions.append(("coordinate_params",))
        self.parameters[81] = list(values)

    def set_point_to_point_common_params(self, *values, queue):
        assert queue is False
        self.actions.append(("common_params",))
        self.parameters[83] = list(values)

    def set_point_to_point_command(self, mode, *goal, queue):
        assert mode == 2 and queue is False
        self.actions.append(("move", *goal))
        self.target = list(goal)

    def set_end_effector_suction_cup(self, control, suction, *, queue):
        assert queue is False
        self.actions.append(("suction", control, suction))
        self.parameters[62] = [int(control), int(suction)]

    def get_pose(self):
        raise AssertionError("The command client must never poll pose over serial")


class FakeNode:
    def __init__(self, clock):
        self.clock = clock
        self.publishers = {topic: [NS(node_name="dobot_state_publisher", node_namespace="/")]
                           for topic in hardware.STATE_TOPICS}
        self.names = [("dobot_state_publisher", "/"), ("dobot_suction_task_cycle", "/")]
        self.destroyed = False

    def get_name(self):
        return "dobot_suction_task_cycle"

    def get_publishers_info_by_topic(self, topic):
        return self.publishers[topic]

    def get_node_names_and_namespaces(self):
        return self.names

    def get_clock(self):
        return NS(now=lambda: NS(nanoseconds=int((100.+self.clock.value)*1e9)))

    def destroy_node(self):
        self.destroyed = True


def stamp(clock):
    nanoseconds = round((100.+clock.value)*1e9)
    return NS(sec=nanoseconds//1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


def feed(node, clock, pose=(210., 0., -45., 18.), joints=(18., 48., 67., 2.77), alarms=()):
    yaw = math.radians(pose[3])
    header = NS(stamp=stamp(clock), frame_id="")
    node._on_pose(NS(header=header, pose=NS(position=NS(x=pose[0]/1000., y=pose[1]/1000., z=pose[2]/1000.),
        orientation=NS(x=0., y=0., z=math.sin(yaw/2.), w=math.cos(yaw/2.)))))
    node._on_joints(NS(header=header, name=[f"magician_joint_{index}" for index in range(1, 5)],
                        position=[math.radians(value) for value in joints]))
    node._on_alarms(NS(header=header, alarms_list=list(alarms)))


@pytest.fixture
def rig(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(hardware, "time", NS(monotonic=clock.now))
    node = hardware.RosDobotHardware.__new__(hardware.RosDobotHardware)
    node.robot_port = "/dev/serial/by-id/test-only"
    node.expected_bias_mm = (59.7, 0., 0.)
    node.velocity_mm_s, node.acceleration_mm_s2 = 10., 20.
    node._armed = node._commands_started = node._closed = False
    node._bot, node._node = FakeDriver(), FakeNode(clock)
    node._context = node._command_lock_fd = None
    node._executor = None
    node._pose = node._joints = node._alarms = None
    node._received, node._stamps = {}, {}
    node._fault = None
    node._pose_history, node._joint_history = deque(maxlen=50), deque(maxlen=50)
    node._suction_on = False
    node._rclpy = NS(ok=lambda **kwargs: True)
    for moment in (9.65, 9.70, 9.75, 9.80, 9.85, 9.90, 9.95, 10.):
        clock.value = moment
        feed(node, clock)

    def spin(timeout=0.02):
        if timeout:
            clock.value += max(0.05, timeout)
        pose = list(node._pose)
        if node._bot.target is not None:
            target = node._bot.target
            distance = hardware._distance(pose[:3], target[:3])
            fraction = min(1., 0.5/distance) if distance else 1.
            pose = [a+(b-a)*fraction for a, b in zip(pose[:3], target[:3])] + [
                pose[3] + hardware._angle_delta(pose[3], target[3])*fraction]
        feed(node, clock, pose)

    node._spin_once = spin
    return node, clock


def test_unarmed_read_only_check_and_close_never_write(rig):
    node, _ = rig
    state = node.state()
    assert set(state) == {"pose", "joints", "alarms", "suction_on"}
    assert state["pose"] == pytest.approx([210., 0., -45., 18.])
    assert state["joints"][-1] == pytest.approx(2.77)
    node.close()
    assert node._bot.actions == []
    assert node._bot.serial.queries == [62]
    assert node._bot.serial.closed and node._node.destroyed


def test_failed_constructor_remains_read_only(monkeypatch):
    driver = FakeDriver()
    driver.serial.corrupt = lambda response: response[:-1]+bytes(((response[-1]+1) % 256,))
    monkeypatch.setattr(hardware.os.path, "exists", lambda path: True)
    monkeypatch.setattr(hardware, "_open_driver", lambda path: driver)
    monkeypatch.setattr(hardware.RosDobotHardware, "_start_ros", lambda node: None)
    monkeypatch.setattr(hardware.RosDobotHardware, "_wait_initial_feedback", lambda node: None)
    with pytest.raises(RuntimeError, match="checksum"):
        hardware.RosDobotHardware("/dev/serial/by-id/test-only", [59.7, 0., 0.])
    assert driver.actions == [] and driver.serial.closed


def test_constructor_refuses_changed_tool_bias_without_writing(monkeypatch):
    driver = FakeDriver()
    driver.parameters[60] = [70., 0., 0.]
    monkeypatch.setattr(hardware.os.path, "exists", lambda path: True)
    monkeypatch.setattr(hardware, "_open_driver", lambda path: driver)
    monkeypatch.setattr(hardware.RosDobotHardware, "_start_ros", lambda node: None)
    monkeypatch.setattr(hardware.RosDobotHardware, "_wait_initial_feedback", lambda node: None)
    with pytest.raises(RuntimeError, match="differs from taught bias"):
        hardware.RosDobotHardware("/dev/serial/by-id/test-only", [59.7, 0., 0.])
    assert driver.actions == [] and driver.serial.closed
    assert driver.serial.queries == [60]


@pytest.mark.parametrize("velocity,acceleration", [(4.9, 20.), (50.1, 20.), (float("nan"), 20.),
                                                   (10., 0.), (10., float("inf"))])
def test_invalid_motion_parameters_rejected_before_ros_or_serial(monkeypatch, velocity, acceleration):
    monkeypatch.setattr(hardware.os.path, "exists", lambda path: True)
    monkeypatch.setattr(hardware, "_open_driver", lambda path: pytest.fail("Must not open the serial port"))
    with pytest.raises(ValueError):
        hardware.RosDobotHardware("/dev/serial/by-id/test-only", [59.7, 0., 0.], velocity, acceleration)


@pytest.mark.parametrize("fault", ["duplicate", "simulation", "stale", "future", "alarms", "joint_limit"])
def test_faulty_feedback_is_rejected(rig, fault):
    node, clock = rig
    if fault == "duplicate":
        node._node.publishers[hardware.STATE_TOPICS[0]] *= 2
    elif fault == "simulation":
        node._node.publishers[hardware.STATE_TOPICS[0]][0].node_name = "mani_mujoco_server"
    elif fault == "stale":
        clock.value += 0.41
    elif fault == "future":
        node._stamps["pose"] += 1.
    elif fault == "alarms":
        node._alarms = [0x49]
    else:
        node._joints = (0., 48., 71., 2.77)
    with pytest.raises(RuntimeError):
        node._feedback()
    assert node._bot.actions == []


@pytest.mark.parametrize("writer", ["dobot_PTP_server", "dobot_suction_cup_srv", "dobot_homing_srv",
                                     "mani_dobot_bridge", "mani_mujoco_server"])
def test_competing_writers_and_simulation_nodes_are_rejected(rig, writer):
    node, _ = rig
    node._node.names.append((writer, "/"))
    with pytest.raises(RuntimeError, match="Competing"):
        node._feedback()


@pytest.mark.parametrize("corrupt", [lambda p: p[:-1]+bytes(((p[-1]+1) % 256,)),
                                      lambda p: p[:3]+b"\x3d"+p[4:],
                                      lambda p: p[:4]+b"\x01"+p[5:], lambda p: p[:-1],
                                      lambda p: b"\x00"+p[1:], lambda p: p[:2]+b"\x01"+p[3:]])
def test_raw_parameter_queries_validate_packet_integrity(rig, corrupt):
    node, _ = rig
    node._bot.serial.corrupt = corrupt
    with pytest.raises(RuntimeError):
        node._query_floats(60, 3)


def test_pose_getter_cannot_be_used_through_raw_query(rig):
    node, _ = rig
    with pytest.raises(ValueError, match="allowed"):
        node._query(10, 32)


def test_arm_reads_bias_and_verifies_low_speed_without_writing_tool_offset(rig):
    node, _ = rig
    node.arm()
    assert node._armed
    assert node._bot.actions == [("stop", True), ("clear",), ("joint_params",),
                                 ("coordinate_params",), ("common_params",), ("start",)]
    assert node._bot.parameters[60] == [59.7, 0., 0.]
    assert node._bot.parameters[81] == [10., 10., 20., 20.]
    assert set(node._bot.serial.queries) == {60, 62, 80, 81, 83}


@pytest.mark.parametrize("velocity", [25., 50.])
def test_constructor_accepts_requested_velocity_without_changing_acceleration(monkeypatch, velocity):
    driver = FakeDriver()
    monkeypatch.setattr(hardware.os.path, "exists", lambda path: True)
    monkeypatch.setattr(hardware, "_open_driver", lambda path: driver)
    monkeypatch.setattr(hardware.RosDobotHardware, "_start_ros", lambda node: None)
    monkeypatch.setattr(hardware.RosDobotHardware, "_wait_initial_feedback", lambda node: None)
    monkeypatch.setattr(hardware.RosDobotHardware, "_spin_once", lambda node, timeout=0.: None)
    monkeypatch.setattr(hardware.RosDobotHardware, "_feedback", lambda node: {})
    node = hardware.RosDobotHardware("/dev/serial/by-id/test-only", [59.7, 0., 0.],
                                     velocity_mm_s=velocity, acceleration_mm_s2=10.)
    assert node.velocity_mm_s == velocity
    assert node.acceleration_mm_s2 == 10.
    node.close()
    assert driver.actions == []


def test_arm_50_mm_s_keeps_acceleration_at_10_and_checks_readback(rig):
    node, _ = rig
    node.velocity_mm_s, node.acceleration_mm_s2 = 50., 10.
    node.arm()
    assert node._bot.parameters[80] == [50.]*4+[10.]*4
    assert node._bot.parameters[81] == [50., 50., 10., 10.]
    assert node._bot.parameters[83] == [100., 100.]
    for command_id in (80, 81, 83):
        assert node._bot.serial.queries.count(command_id) == 2
    assert node._armed


@pytest.mark.parametrize("cause", ["bias", "pump", "readback"])
def test_arm_refuses_wrong_tool_held_object_or_bad_parameter_readback(rig, cause):
    node, _ = rig
    if cause == "bias":
        node._bot.parameters[60][0] = 70.
    elif cause == "pump":
        node._bot.parameters[62] = [1, 1]
    else:
        node._bot.set_point_to_point_common_params = lambda *args, **kwargs: None
        node._bot.parameters[83] = [50., 50.]
    with pytest.raises(RuntimeError):
        node.arm()
    assert not node._armed
    assert all(action[0] != "move" and action[0] != "suction" for action in node._bot.actions)
    if cause in ("bias", "pump"):
        assert node._bot.actions == []
    else:
        assert node._bot.actions[-2:] == [("stop", True), ("clear",)]


def test_five_mm_linear_move_is_monitored_and_allows_nonzero_wrist(rig):
    node, clock = rig
    node.arm()
    before = clock.value
    state = node.move_segment([215., 0., -45., 19.])
    assert state["pose"] == pytest.approx([215., 0., -45., 19.])
    assert state["joints"][-1] == pytest.approx(2.77)
    assert clock.value-before >= 0.75
    assert node._bot.serial.queries.count(10) == 0


@pytest.mark.parametrize("goal", [[216., 0., -45., 18.], [210., 0., -45., 20.1],
                                  [float("nan"), 0., -45., 18.]])
def test_oversized_or_nonfinite_segments_stop_without_sending_motion(rig, goal):
    node, _ = rig
    node.arm()
    with pytest.raises((ValueError, RuntimeError)):
        node.move_segment(goal)
    assert all(action[0] != "move" for action in node._bot.actions)
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]


def test_stalled_motion_times_out_and_retains_suction(rig):
    node, clock = rig
    node.arm()
    node.suction(True)

    def stalled(timeout=0.02):
        clock.value += 0.05
        feed(node, clock)

    node._spin_once = stalled
    with pytest.raises(RuntimeError, match="timed out"):
        node.move_segment([215., 0., -45., 18.])
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]
    assert node._bot.parameters[62] == [1, 1]
    assert not node._armed


def test_feedback_loss_mid_segment_stops_and_preserves_pump(rig):
    node, clock = rig
    node.arm()
    node.suction(True)
    original = node._spin_once

    def lose_feedback(timeout=0.02):
        if timeout:
            clock.value += 0.5
        else:
            original(timeout)

    node._spin_once = lose_feedback
    with pytest.raises(RuntimeError, match="stale"):
        node.move_segment([215., 0., -45., 18.])
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]
    assert node._bot.parameters[62] == [1, 1]


@pytest.mark.parametrize("pose", [[212., 1.01, -45., 18.], [212., 0., -45., 20.]])
def test_motion_monitor_rejects_off_line_or_bad_yaw(pose):
    with pytest.raises(RuntimeError):
        hardware.RosDobotHardware._segment_geometry([210., 0., -45., 18.], [215., 0., -45., 18.], pose)


def test_suction_off_disables_control_and_pump_without_restarting_queue(rig):
    node, _ = rig
    node.arm()
    assert node.suction(True) is True
    assert node.suction(False) is False
    assert node._bot.actions[-1] == ("suction", False, False)
    assert node._bot.parameters[62] == [0, 0]
    assert node._bot.actions.count(("start",)) == 1


def test_suction_switching_monitors_feedback_during_pneumatic_settling(rig):
    node, clock = rig
    node.arm()
    started = clock.value
    node.suction(True)
    assert clock.value - started >= 0.6
    assert node._received["pose"] == clock.value
    assert node._bot.parameters[62] == [1, 1]


def test_read_only_close_does_not_hide_bad_suction_state(rig):
    node, _ = rig
    node._bot.parameters[62] = [1, 0]
    with pytest.raises(RuntimeError, match="Inconsistent"):
        node.state()
    node.close()
    assert node._bot.actions == []


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_confirmation_requires_exact_phrase_and_keeps_feedback_spinning(rig, monkeypatch):
    node, clock = rig
    node.arm()
    monkeypatch.setattr(hardware.sys, "stdin", Terminal("yes\n READY A\nREADY A\n"))
    monkeypatch.setattr(hardware.select, "select", lambda *args: ([hardware.sys.stdin], [], []))
    before = clock.value
    assert node.confirm("Refill A", "READY A") is True
    assert clock.value > before


@pytest.mark.parametrize("failure", ["EOF", "interrupt", "noninteractive"])
def test_confirmation_failure_stops_and_keeps_suction(rig, monkeypatch, failure):
    node, _ = rig
    node.arm()
    node.suction(True)
    monkeypatch.setattr(hardware.sys, "stdin", Terminal(""))
    if failure == "interrupt":
        def interrupt(*args):
            raise KeyboardInterrupt
        monkeypatch.setattr(hardware.select, "select", interrupt)
        expected = KeyboardInterrupt
    elif failure == "noninteractive":
        monkeypatch.setattr(hardware.sys, "stdin", io.StringIO("READY A\n"))
        expected = RuntimeError
    else:
        monkeypatch.setattr(hardware.select, "select", lambda *args: ([hardware.sys.stdin], [], []))
        expected = RuntimeError
    with pytest.raises(expected):
        node.confirm("Refill A", "READY A")
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]
    assert node._bot.parameters[62] == [1, 1]


def test_close_after_arming_stops_motion_and_never_releases_held_object(rig):
    node, _ = rig
    node.arm()
    node.suction(True)
    node.close()
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]
    assert node._bot.parameters[62] == [1, 1]
    assert node._bot.serial.closed
    action_count = len(node._bot.actions)
    node.close()
    assert len(node._bot.actions) == action_count


def test_old_header_is_rejected_even_with_a_fresh_callback_receipt(rig):
    node, _ = rig
    node._stamps["alarms"] -= 0.51
    with pytest.raises(RuntimeError, match="header"):
        node._feedback()


def test_joint_motion_with_stationary_tcp_does_not_pass_settle_gate(rig):
    node, _ = rig
    stamp, joints = node._joint_history[0]
    node._joint_history[0] = (stamp, (*joints[:3], joints[3]+1.))
    assert not node._is_stable()


def test_alarm_during_pneumatic_settling_stops_and_keeps_output(rig):
    node, clock = rig
    node.arm()
    original = node._spin_once

    def alarm(timeout=0.02):
        original(timeout)
        if timeout:
            feed(node, clock, alarms=(0x43,))

    node._spin_once = alarm
    with pytest.raises(RuntimeError, match="alarms"):
        node.suction(True)
    assert node._bot.parameters[62] == [1, 1]
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]
    assert node._bot.actions.count(("start",)) == 1


def test_stop_still_clears_queue_if_force_stop_write_fails(rig):
    node, _ = rig
    node.arm()

    def failed_stop(**kwargs):
        raise OSError("USB disconnected")

    node._bot.stop_queue = failed_stop
    with pytest.raises(RuntimeError, match="USB disconnected"):
        node.stop()
    assert node._bot.actions[-1] == ("clear",)
    assert not node._armed


def test_read_only_cleanup_continues_if_serial_close_fails(rig):
    node, _ = rig

    def failed_close():
        raise OSError("close failed")

    node._bot.serial.close = failed_close
    node.close()
    assert node._node.destroyed
    assert node._bot.actions == []


@pytest.mark.parametrize("text", ["STOP\n", "stop\n", "CANCEL\n", "停止\n", "取消\n"])
def test_operator_cancel_text_stops_without_releasing_object(rig, monkeypatch, text):
    node, _ = rig
    node.arm()
    node.suction(True)
    monkeypatch.setattr(hardware.sys, "stdin", Terminal(text))
    monkeypatch.setattr(hardware.select, "select", lambda *args: ([hardware.sys.stdin], [], []))
    with pytest.raises(RuntimeError, match="Operator canceled"):
        node.confirm("Refill A", "READY A")
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]
    assert node._bot.parameters[62] == [1, 1]


def test_custom_ros_context_uses_its_own_executor_and_shutdown_order(monkeypatch):
    events = []

    class Context:
        pass

    class Executor:
        def __init__(self, *, context):
            self.context = context

        def add_node(self, node):
            assert node.context is self.context
            events.append("add")

        def spin_once(self, *, timeout_sec):
            events.append(("spin", timeout_sec))

        def remove_node(self, node):
            assert node.context is self.context
            events.append("remove")

        def shutdown(self):
            events.append("executor_shutdown")

    class Node:
        def __init__(self, name, *, context, use_global_arguments):
            self.context = context
            assert use_global_arguments is False

        def create_subscription(self, *args):
            return object()

        def destroy_node(self):
            events.append("destroy_node")

    def global_spin(*args, **kwargs):
        pytest.fail("A custom context must not use the uninitialized global executor")

    fake_ros = NS(init=lambda **kwargs: events.append("init"), ok=lambda **kwargs: True,
                  spin_once=global_spin, shutdown=lambda **kwargs: events.append("context_shutdown"))
    modules = {"rclpy": fake_ros, "rclpy.context": NS(Context=Context),
               "rclpy.executors": NS(SingleThreadedExecutor=Executor), "rclpy.node": NS(Node=Node),
               "dobot_msgs.msg": NS(DobotAlarmCodes=object),
               "geometry_msgs.msg": NS(PoseStamped=object), "sensor_msgs.msg": NS(JointState=object)}
    for name, module in modules.items():
        monkeypatch.setitem(hardware.sys.modules, name, module)
    node = hardware.RosDobotHardware.__new__(hardware.RosDobotHardware)
    node._bot = node._command_lock_fd = None
    node._closed = False
    node._start_ros()
    node._spin_once(0.03)
    node._release_resources()
    assert events == ["init", "add", ("spin", 0.03)] + [("spin", 0.)]*24 + [
        "remove", "executor_shutdown", "destroy_node", "context_shutdown"]


def test_confirmation_drains_multitopic_backlog_without_weakening_header_age(rig, monkeypatch):
    node, clock = rig
    node.arm()
    started = clock.value
    callbacks = deque()
    next_emit = [clock.value + 0.05]

    def enqueue_feedback():
        while next_emit[0] <= clock.value + 1e-9:
            timestamp = round((100.+next_emit[0])*1e9)
            header = NS(stamp=NS(sec=timestamp//1_000_000_000, nanosec=timestamp % 1_000_000_000), frame_id="")
            yaw = math.radians(18.)
            pose = NS(header=header, pose=NS(position=NS(x=.210, y=0., z=-.045),
                      orientation=NS(x=0., y=0., z=math.sin(yaw/2.), w=math.cos(yaw/2.))))
            joint = NS(header=header, name=[f"magician_joint_{index}" for index in range(1, 5)],
                       position=[math.radians(value) for value in (18., 48., 67., 2.77)])
            alarms = NS(header=header, alarms_list=[])
            callbacks.extend((lambda message=pose: node._on_pose(message),
                              lambda message=joint: node._on_joints(message)))
            # 20 Hz pose + 20 Hz joints + 10 Hz alarms.
            if round((next_emit[0]-started)/.05) % 2 == 0:
                callbacks.append(lambda message=alarms: node._on_alarms(message))
            next_emit[0] += .05

    class Executor:
        def spin_once(self, *, timeout_sec):
            clock.value += timeout_sec
            enqueue_feedback()
            if callbacks:
                callbacks.popleft()()

    node._executor = Executor()
    node._spin_once = hardware.RosDobotHardware._spin_once.__get__(node)
    monkeypatch.setattr(hardware.sys, "stdin", Terminal("READY A\n"))

    def select_after_ten_seconds(*args):
        clock.value += .04
        if clock.value-started >= 10.:
            return [hardware.sys.stdin], [], []
        return [], [], []

    monkeypatch.setattr(hardware.select, "select", select_after_ten_seconds)
    assert node.confirm("Refill A", "READY A") is True
    assert clock.value-started >= 10.
    assert len(callbacks) <= 3
    assert node._ros_now()-node._stamps["joints"] < .11
    assert node._armed


def test_draining_empty_executor_still_rejects_actual_feedback_loss(rig):
    node, clock = rig
    node.arm()
    node._executor = NS(spin_once=lambda **kwargs: None)
    node._spin_once = hardware.RosDobotHardware._spin_once.__get__(node)
    clock.value += .51
    with pytest.raises(RuntimeError, match="stale"):
        node.state()
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]


def prime_high_plane(node, clock, pose=(210., 69., -10., 18.)):
    node._pose_history.clear()
    node._joint_history.clear()
    latest = clock.value
    for offset in (.35, .30, .25, .20, .15, .10, .05, 0.):
        clock.value = latest-offset
        feed(node, clock, pose)


def test_long_high_xy_transfer_sends_one_movl_with_50_velocity_10_acceleration(rig):
    node, clock = rig
    prime_high_plane(node, clock)
    node.velocity_mm_s, node.acceleration_mm_s2 = 50., 10.
    node.arm()
    node.suction(True)
    goal = [210., -92., -10., 18.]
    result = node.move_transfer(goal)
    moves = [action for action in node._bot.actions if action[0] == "move"]
    assert moves == [("move", *goal)]
    assert result["pose"] == pytest.approx(goal)
    assert node._bot.parameters[81] == [50., 50., 10., 10.]
    assert node._bot.parameters[62] == [1, 1]
    assert node._armed


def test_high_plane_zero_move_returns_state_without_motion_command(rig):
    node, clock = rig
    prime_high_plane(node, clock)
    node.arm()
    result = node.move_transfer(list(node._pose))
    assert result["pose"] == pytest.approx([210., 69., -10., 18.])
    assert all(action[0] != "move" for action in node._bot.actions)


@pytest.mark.parametrize("start,goal", [
    ([210., 69., -45., 18.], [210., -92., -45., 18.]),
    ([210., 69., -10., 18.], [210., -92., -9.7, 18.]),
    ([210., 69., -10., 18.], [210., -92., -10., 18.31]),
    ([210., 69., -10., 18.], [210., -181.1, -10., 18.]),
    ([210., 69., -10.2, 18.], [210., -92., -10.2, 18.]),
])
def test_long_transfer_rejects_low_plane_mixed_z_rotation_and_excess_distance(rig, start, goal):
    node, clock = rig
    prime_high_plane(node, clock, start)
    node.arm()
    node.suction(True)
    with pytest.raises(ValueError):
        node.move_transfer(goal)
    assert all(action[0] != "move" for action in node._bot.actions)
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]
    assert node._bot.parameters[62] == [1, 1]


def test_long_transfer_dense_ik_checks_interior_before_any_command(rig, monkeypatch):
    from mani_dobot import suction_task_plan
    node, clock = rig
    prime_high_plane(node, clock)
    node.arm()
    checked = []
    original = suction_task_plan.firmware_ik

    def fail_inside(pose, bias):
        checked.append(pose)
        assert bias == node.expected_bias_mm
        if -1. < pose[1] < 1.:
            raise ValueError("Interior fails a joint limit")
        return original(pose, bias)

    monkeypatch.setattr(suction_task_plan, "firmware_ik", fail_inside)
    with pytest.raises(ValueError, match="fraction"):
        node.move_transfer([210., -92., -10., 18.])
    assert len(checked) >= 69
    assert all(action[0] != "move" for action in node._bot.actions)


@pytest.mark.parametrize("fault", ["stale", "line", "timeout"])
def test_long_transfer_failure_stops_and_retains_suction_without_retry(rig, fault):
    node, clock = rig
    prime_high_plane(node, clock)
    node.arm()
    node.suction(True)
    original_spin = node._spin_once
    start = list(node._pose)

    def fail_motion(timeout=0.02):
        if not timeout:
            original_spin(timeout)
            return
        if fault == "stale":
            clock.value += .51
        elif fault == "line":
            clock.value += .05
            feed(node, clock, [start[0]+1.1, start[1]-1., start[2], start[3]])
        else:
            clock.value += .05
            feed(node, clock, start)

    node._spin_once = fail_motion
    with pytest.raises(RuntimeError):
        node.move_transfer([210., -92., -10., 18.])
    assert len([action for action in node._bot.actions if action[0] == "move"]) == 1
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]
    assert node._bot.parameters[62] == [1, 1]
    assert not node._armed


def test_stationary_set_speed_preserves_acceleration_suction_and_queue(rig):
    node, _ = rig
    node.acceleration_mm_s2 = 10.
    node.arm()
    node.suction(True)
    node._bot.actions.clear()
    assert node.set_speed(50.) == 50.
    assert node.velocity_mm_s == 50.
    assert node.acceleration_mm_s2 == 10.
    assert node._bot.parameters[80] == [50.]*4+[10.]*4
    assert node._bot.parameters[81] == [50., 50., 10., 10.]
    assert node._bot.actions == [("joint_params",), ("coordinate_params",)]
    assert node._bot.parameters[62] == [1, 1]
    assert node._armed


@pytest.mark.parametrize("velocity", [4.9, 50.1, float("nan"), float("inf")])
def test_invalid_speed_change_stops_motion_and_keeps_old_limit_and_pump(rig, velocity):
    node, _ = rig
    node.arm()
    node.suction(True)
    before = node.velocity_mm_s
    with pytest.raises(ValueError):
        node.set_speed(velocity)
    assert node.velocity_mm_s == before
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]
    assert node._bot.parameters[62] == [1, 1]


def test_unarmed_speed_change_is_refused_without_any_writes(rig):
    node, _ = rig
    with pytest.raises(RuntimeError, match="arm"):
        node.set_speed(50.)
    assert node._bot.actions == []


@pytest.mark.parametrize("fault", ["moving", "stale", "readback"])
def test_speed_change_fault_stops_and_preserves_pump(rig, fault):
    node, clock = rig
    node.arm()
    node.suction(True)
    if fault == "moving":
        node._is_stable = lambda **kwargs: False
    elif fault == "stale":
        node._spin_once = lambda timeout=0.02: None
        clock.value += .51
    else:
        node._bot.set_point_to_point_coordinate_params = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError):
        node.set_speed(50.)
    assert node.velocity_mm_s == 10.
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]
    assert node._bot.parameters[62] == [1, 1]


def test_explicit_holding_confirmed_arm_retains_on_output_and_uses_full_preflight(rig):
    node, _ = rig
    node._bot.parameters[62] = [1, 1]
    node.velocity_mm_s, node.acceleration_mm_s2 = 50., 10.
    node.arm(holding_confirmed=True)
    assert node._armed
    assert node._bot.parameters[62] == [1, 1]
    assert node._bot.parameters[81] == [50., 50., 10., 10.]
    assert all(action[0] not in ("move", "suction") for action in node._bot.actions)
    assert node._bot.actions[0:2] == [("stop", True), ("clear",)]
    assert node._bot.actions[-1] == ("start",)
    assert set(node._bot.serial.queries) == {60, 62, 80, 81, 83}


@pytest.mark.parametrize("confirmation", [1, "yes", None])
def test_holding_confirmation_requires_strict_boolean(rig, confirmation):
    node, _ = rig
    node._bot.parameters[62] = [1, 1]
    with pytest.raises(ValueError, match="explicit boolean"):
        node.arm(holding_confirmed=confirmation)
    assert node._bot.actions == []


def test_holding_confirmed_arm_requires_output_already_on(rig):
    node, _ = rig
    with pytest.raises(RuntimeError, match="requires suction already on"):
        node.arm(holding_confirmed=True)
    assert node._bot.actions == []


def test_holding_confirmed_arm_rejects_bad_bias_without_releasing(rig):
    node, _ = rig
    node._bot.parameters[62] = [1, 1]
    node._bot.parameters[60] = [70., 0., 0.]
    with pytest.raises(RuntimeError, match="differs from taught bias"):
        node.arm(holding_confirmed=True)
    assert node._bot.actions == []
    assert node._bot.parameters[62] == [1, 1]


def test_initial_feedback_waits_for_zero_endpoint_discovery_then_new_stable_samples(rig):
    node, clock = rig
    initial = clock.value
    original = node._spin_once

    def discovery(timeout=0.02):
        original(timeout)
        for topic in hardware.STATE_TOPICS:
            node._node.publishers[topic] = ([] if clock.value-initial < .25 else [
                NS(node_name="dobot_state_publisher", node_namespace="/")])

    node._spin_once = discovery
    node._wait_initial_feedback()
    assert clock.value-initial >= .5
    assert node._bot.actions == []
    assert node._bot.serial.queries == []


@pytest.mark.parametrize("fault", ["wrong", "duplicate"])
def test_initial_feedback_rejects_wrong_or_duplicate_publishers_immediately(rig, fault):
    node, clock = rig
    if fault == "wrong":
        node._node.publishers[hardware.STATE_TOPICS[0]][0].node_name = "mani_mujoco_server"
    else:
        node._node.publishers[hardware.STATE_TOPICS[0]] *= 2
    initial = clock.value
    with pytest.raises(RuntimeError, match="observed"):
        node._wait_initial_feedback()
    assert clock.value-initial < .1
    assert node._bot.actions == []


def test_initial_graph_discovery_has_finite_deadline_and_never_sends_commands(rig):
    node, clock = rig
    for topic in hardware.STATE_TOPICS:
        node._node.publishers[topic] = []
    initial = clock.value
    with pytest.raises(RuntimeError, match="Timed out.*discovery"):
        node._wait_initial_feedback()
    assert 3. <= clock.value-initial < 3.1
    assert node._bot.actions == []


def test_runtime_zero_endpoint_still_immediately_stops(rig):
    node, _ = rig
    node.arm()
    node.suction(True)
    node._node.publishers[hardware.STATE_TOPICS[0]] = []
    with pytest.raises(RuntimeError, match="observed"):
        node.state()
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]
    assert node._bot.parameters[62] == [1, 1]


def test_initial_single_unresolved_dds_identity_waits_until_resolved_and_stable(rig):
    node, clock = rig
    initial = clock.value
    original = node._spin_once

    def discovery(timeout=0.02):
        original(timeout)
        unresolved = clock.value-initial < .25
        node._node.publishers[hardware.STATE_TOPICS[0]] = [NS(
            node_name="_NODE_NAME_UNKNOWN_" if unresolved else "dobot_state_publisher",
            node_namespace="_NODE_NAMESPACE_UNKNOWN_" if unresolved else "/")]

    node._spin_once = discovery
    node._wait_initial_feedback()
    assert clock.value-initial >= .5
    assert node._bot.actions == [] and node._bot.serial.queries == []


def test_initial_unresolved_identity_never_authorizes_hardware_and_times_out(rig):
    node, clock = rig
    node._node.publishers[hardware.STATE_TOPICS[0]] = [NS(
        node_name="_NODE_NAME_UNKNOWN_", node_namespace="_NODE_NAMESPACE_UNKNOWN_")]
    initial = clock.value
    with pytest.raises(RuntimeError, match="Timed out.*identity pending"):
        node._wait_initial_feedback()
    assert 3. <= clock.value-initial < 3.1
    assert node._bot.actions == [] and node._bot.serial.queries == []


@pytest.mark.parametrize("name,namespace", [("other_publisher", "/"),
    ("_NODE_NAME_UNKNOWN_", "/"), ("dobot_state_publisher", "_NODE_NAMESPACE_UNKNOWN_")])
def test_initial_wrong_or_partially_unknown_identity_is_not_retryable(rig, name, namespace):
    node, clock = rig
    node._node.publishers[hardware.STATE_TOPICS[0]] = [NS(node_name=name, node_namespace=namespace)]
    initial = clock.value
    with pytest.raises(RuntimeError):
        node._wait_initial_feedback()
    assert clock.value-initial < .1
    assert node._bot.actions == []


def test_initial_multiple_unknown_endpoints_are_rejected_immediately(rig):
    node, clock = rig
    node._node.publishers[hardware.STATE_TOPICS[0]] = [NS(
        node_name="_NODE_NAME_UNKNOWN_", node_namespace="_NODE_NAMESPACE_UNKNOWN_")]*2
    initial = clock.value
    with pytest.raises(RuntimeError, match="observed"):
        node._wait_initial_feedback()
    assert clock.value-initial < .1


def test_runtime_unknown_dds_identity_still_stops_immediately(rig):
    node, _ = rig
    node.arm()
    node.suction(True)
    node._node.publishers[hardware.STATE_TOPICS[0]] = [NS(
        node_name="_NODE_NAME_UNKNOWN_", node_namespace="_NODE_NAMESPACE_UNKNOWN_")]
    with pytest.raises(RuntimeError, match="observed"):
        node.state()
    assert node._bot.actions[-2:] == [("stop", True), ("clear",)]
    assert node._bot.parameters[62] == [1, 1]
