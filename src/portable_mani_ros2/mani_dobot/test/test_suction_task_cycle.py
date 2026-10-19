"""Operator gates and fault handling, using no ROS or serial hardware."""

import builtins
from dataclasses import replace
import json
import math
import sys
from types import ModuleType

import pytest

from mani_dobot import suction_task_cycle as cycle
from mani_dobot.suction_task_plan import (
    COORDINATE_CONVENTION,
    EXPECTED_ROBOT_PORT,
    firmware_fk,
    fixed_yaw_plan,
    load_plan,
)


# Snapshot of the four taught contacts; tests never read installation reports.
CONTACTS = {
    "pick_A": ([204.83059692382812, 69.3899154663086, -45.65318298339844],
               18.71470069885254, [18.71470069885254, 48.07060623168945, 67.55284881591797, 0.0]),
    "place_B": ([206.99929809570312, -63.36740112304687, -45.26304626464844],
                -17.020595550537113, [-17.020593643188477, 47.94572067260742, 67.37850189208984, -0.0000019073486328125]),
    "place_C": ([209.57937622070312, -92.19148254394531, -47.31372833251953],
                -23.74412536621094, [-23.744123458862305, 51.040321350097656, 64.06710815429688, -0.0000019073486328125]),
    "place_D": ([210.87490844726562, -147.76919555664062, -46.8529052734375],
                -35.020599365234375, [-35.02059555053711, 57.050655364990234, 54.90747833251953, -0.0000019073486328125]),
}


@pytest.fixture
def waypoint_path(tmp_path):
    data = {
        "schema_version": 1,
        "coordinate_convention": COORDINATE_CONVENTION,
        "robot_port": EXPECTED_ROBOT_PORT,
        "controller_tool_bias_mm": [59.70000076293945, 0.0, 0.0],
        "tool": "suction_cup",
        "all_contact_waypoints_recorded": True,
        "destinations_in_order": ["place_B", "place_C", "place_D"],
    }
    for name, (xyz, yaw, joints) in CONTACTS.items():
        data[name] = {
            "status": "manually_taught_contact_recorded",
            "controller_xyz_mm": xyz,
            "controller_yaw_deg": yaw,
            "vendor_joints_deg": joints,
            "alarm_codes": [],
        }
    path = tmp_path / "task_waypoints.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def plan(waypoint_path):
    return load_plan(waypoint_path)


class FakeHardware:
    def __init__(self, pose, *, responses=None, reject_gate=None, fail_loaded_xy=False,
                 gate_deltas=None):
        self.pose = tuple(pose)
        self.suction_on = False
        self.alarms = []
        self.events = []
        self.responses = iter(responses) if responses is not None else None
        self.reject_gate = reject_gate
        self.gate_count = 0
        self.fail_loaded_xy = fail_loaded_xy
        self.failed_goals = []
        self.gate_deltas = gate_deltas or {}

    def state(self):
        return {"pose": self.pose, "joints": [0.0] * 4,
                "alarms": self.alarms, "suction_on": self.suction_on}

    def confirm(self, prompt, exact):
        self.gate_count += 1
        self.events.append(("confirm", exact, self.pose, self.suction_on))
        response = next(self.responses) if self.responses is not None else exact
        if self.gate_count == self.reject_gate or response != exact:
            raise RuntimeError("Operator cancelled")
        if self.gate_count in self.gate_deltas:
            delta = self.gate_deltas[self.gate_count]
            self.pose = tuple(value + change for value, change in zip(self.pose, delta))
            self.events.append(("manual_pose_change", self.pose))

    def arm(self, *, holding_confirmed=False):
        self.events.append(("arm_held",) if holding_confirmed else ("arm",))

    def set_speed(self, speed):
        self.events.append(("speed", speed))

    def move_segment(self, goal):
        goal = tuple(goal)
        self.events.append(("move", goal, self.suction_on))
        if self.fail_loaded_xy and self.suction_on and goal[:2] != self.pose[:2]:
            self.failed_goals.append(goal)
            raise RuntimeError("Injected motion failure")
        self.pose = goal

    def move_transfer(self, goal):
        goal = tuple(goal)
        self.events.append(("transfer", goal, self.suction_on))
        if self.fail_loaded_xy and self.suction_on and goal[:2] != self.pose[:2]:
            self.failed_goals.append(goal)
            raise RuntimeError("Injected motion failure")
        self.pose = goal

    def suction(self, enabled):
        self.events.append(("suction", enabled, self.pose))
        self.suction_on = enabled

    def stop(self):
        self.events.append(("stop",))

    def close(self):
        self.events.append(("close",))


def high(pose, z):
    return (pose[0], pose[1], z, pose[3])


def lifted(pose):
    return (pose[0], pose[1], pose[2] + 10.0, pose[3])


@pytest.mark.parametrize("mode,sources", [
    ("refill", ["pick_A", "pick_A", "pick_A"]),
    ("chain", ["pick_A", "place_B", "place_C"]),
])
def test_complete_routes_obey_contact_trial_lift_and_release_gates(waypoint_path, mode, sources):
    task = load_plan(waypoint_path, mode=mode)
    hardware = FakeHardware(task.waypoints["pick_A"].pose)
    result = cycle.run_task(task, hardware)
    destinations = ["place_B", "place_C", "place_D"]
    assert result["status"] == "complete"
    assert result["completed_destinations"] == destinations
    assert result["routes"] == [list(route) for route in zip(sources, destinations)]
    assert hardware.events.count(("arm",)) == 1
    assert hardware.events.count(("stop",)) == 1
    gates = [(i, event) for i, event in enumerate(hardware.events) if event[0] == "confirm"]
    assert [event[1] for _, event in gates] == [
        token for source in sources
        for token in ("READY " + source.split("_")[1], "HELD", "RELEASED")
    ]
    pump_events = [(i, event) for i, event in enumerate(hardware.events) if event[0] == "suction"]
    assert [event[1] for _, event in pump_events] == [True, False] * 3

    previous_release = 0
    for route_index, (source_name, target_name) in enumerate(zip(sources, destinations)):
        source = task.waypoints[source_name].pose
        target = task.waypoints[target_name].pose
        ready_i, ready = gates[route_index * 3]
        held_i, held = gates[route_index * 3 + 1]
        released_i, released = gates[route_index * 3 + 2]
        on_i, on = pump_events[route_index * 2]
        off_i, off = pump_events[route_index * 2 + 1]
        assert previous_release <= ready_i < on_i < held_i < off_i < released_i
        assert ready[3] is False
        assert on[2] == pytest.approx(source)
        assert held[2] == pytest.approx(lifted(source))
        assert held[3] is True
        # Prior to HELD, loaded motion is only the local 10 mm test lift.
        trial = [event for event in hardware.events[on_i + 1:held_i] if event[0] == "move"]
        assert trial
        assert all(event[1][:2] == pytest.approx(source[:2]) for event in trial)
        assert all(source[2] <= event[1][2] <= source[2] + 10.0 for event in trial)
        assert trial[-1][1] == pytest.approx(lifted(source))
        loaded = [event for event in hardware.events[held_i + 1:off_i] if event[0] == "move"]
        previous = held[2]
        lateral = []
        for event in loaded:
            if math.dist(previous[:2], event[1][:2]) > 1e-8:
                assert previous[2] == pytest.approx(task.safe_z_mm)
                lateral.append(event)
            previous = event[1]
        assert lateral
        assert all(event[1][2] == pytest.approx(task.safe_z_mm) for event in lateral)
        assert off[2] == pytest.approx(target)
        assert released[2] == pytest.approx(lifted(target))
        assert released[3] is False
        retract = [event for event in hardware.events[off_i + 1:released_i] if event[0] == "move"]
        assert retract
        assert all(event[1][:2] == pytest.approx(target[:2]) for event in retract)
        assert all(target[2] <= event[1][2] <= target[2] + 10.0 for event in retract)
        previous_release = released_i
    assert hardware.pose == pytest.approx(high(task.waypoints["place_D"].pose, task.safe_z_mm))


@pytest.mark.parametrize("rejected_gate", range(1, 10))
def test_rejecting_any_operator_gate_stops_without_skipping_retry_or_pump_changes(plan, rejected_gate):
    hardware = FakeHardware(plan.waypoints["pick_A"].pose, reject_gate=rejected_gate)
    journal = cycle.TaskJournal(plan)
    with pytest.raises(RuntimeError, match="Operator cancelled"):
        cycle.run_task(plan, hardware, journal)
    assert hardware.gate_count == rejected_gate
    last_gate = max(i for i, event in enumerate(hardware.events) if event[0] == "confirm")
    # A declined initial READY has never armed; all later failures stop once.
    assert hardware.events[last_gate + 1:] == ([] if rejected_gate == 1 else [("stop",)])
    assert journal.data["status"] == "stopped"
    assert journal.data["completed_destinations"] == ["place_B", "place_C", "place_D"][:(rejected_gate - 1) // 3]
    assert hardware.suction_on is (rejected_gate % 3 == 2)
    assert len([event for event in hardware.events if event[0] == "suction"]) == (
        ((rejected_gate - 1) // 3) * 2 + (0 if rejected_gate % 3 == 1 else 1 if rejected_gate % 3 == 2 else 2)
    )


@pytest.mark.parametrize("response", ["", "ready A", "READY", "READY B", "yes"])
def test_source_gate_requires_exact_ready_token(plan, response):
    hardware = FakeHardware(plan.waypoints["pick_A"].pose, responses=[response])
    with pytest.raises(RuntimeError, match="Operator cancelled"):
        cycle.run_task(plan, hardware)
    assert hardware.events == [("confirm", "READY A", hardware.pose, False)]


def test_loaded_motion_failure_stops_with_suction_retained_and_no_automatic_retry(plan):
    hardware = FakeHardware(plan.waypoints["pick_A"].pose, fail_loaded_xy=True)
    journal = cycle.TaskJournal(plan)
    with pytest.raises(RuntimeError, match="Injected motion failure"):
        cycle.run_task(plan, hardware, journal)
    assert len(hardware.failed_goals) == 1
    assert hardware.events[-1] == ("stop",)
    assert hardware.suction_on is True
    assert [event[1] for event in hardware.events if event[0] == "suction"] == [True]
    assert [event[1] for event in hardware.events if event[0] == "confirm"] == ["READY A", "HELD"]
    assert journal.data["completed_destinations"] == []
    assert journal.data["holding_assumption"] == "operator_confirmed_held"
    assert journal.data["status"] == "stopped"


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_pose_change_during_held_gate_stops_with_pump_on_before_any_transfer(plan, axis):
    delta = [0.0] * 4
    delta[axis] = 2.0
    source = plan.waypoints["pick_A"].pose
    hardware = FakeHardware(source, gate_deltas={2: delta})
    journal = cycle.TaskJournal(plan)
    with pytest.raises(RuntimeError, match="moved while awaiting grasp confirmation"):
        cycle.run_task(plan, hardware, journal)
    assert hardware.gate_count == 2
    assert hardware.events[-2][0] == "manual_pose_change"
    assert hardware.events[-1] == ("stop",)
    assert hardware.suction_on is True
    assert [event[1] for event in hardware.events if event[0] == "suction"] == [True]
    assert all(event[1][:2] == pytest.approx(source[:2])
               for event in hardware.events if event[0] == "move")
    assert journal.data["status"] == "stopped"
    assert journal.data["completed_destinations"] == []


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_pose_change_during_released_gate_stops_without_next_target(plan, axis):
    delta = [0.0] * 4
    delta[axis] = 2.0
    hardware = FakeHardware(plan.waypoints["pick_A"].pose, gate_deltas={3: delta})
    journal = cycle.TaskJournal(plan)
    with pytest.raises(RuntimeError, match="moved while awaiting release confirmation"):
        cycle.run_task(plan, hardware, journal)
    assert hardware.gate_count == 3
    assert hardware.events[-2][0] == "manual_pose_change"
    assert hardware.events[-1] == ("stop",)
    assert hardware.suction_on is False
    assert [event[1] for event in hardware.events if event[0] == "suction"] == [True, False]
    assert journal.data["status"] == "stopped"
    assert journal.data["completed_destinations"] == []


def test_pose_changed_to_unknown_location_at_second_ready_is_rejected_before_motion(plan):
    hardware = FakeHardware(plan.waypoints["pick_A"].pose,
                            gate_deltas={4: [15.0, 15.0, 0.0, 0.0]})
    journal = cycle.TaskJournal(plan)
    with pytest.raises(RuntimeError, match="not a taught contact/high point"):
        cycle.run_task(plan, hardware, journal)
    assert hardware.gate_count == 4
    assert hardware.events[-2][0] == "manual_pose_change"
    assert hardware.events[-1] == ("stop",)
    assert hardware.events.count(("arm",)) == 1
    assert hardware.suction_on is False
    assert journal.data["completed_destinations"] == ["place_B"]
    assert journal.data["status"] == "stopped"


@pytest.mark.parametrize("stage", [
    "raise_empty_at_current", "descend_to_source", "trial_lift_10mm",
    "lift_loaded", "descend_to_target", "release_check_retract_10mm", "retract_empty",
])
@pytest.mark.parametrize("axis,change", [(0, 2.0), (1, 2.0), (3, 2.0)])
def test_vertical_stage_rejects_xy_or_orientation_change_before_any_command(plan, stage, axis, change):
    source = plan.waypoints["pick_A"].pose
    hardware = FakeHardware(source)
    destination = list(lifted(source))
    destination[axis] += change
    with pytest.raises(RuntimeError, match="must be vertical with unchanged orientation"):
        cycle._move(hardware, destination, cycle.TaskJournal(plan), stage)
    assert hardware.events == []


@pytest.mark.parametrize("stage", ["above_source", "transfer_at_safe_height"])
@pytest.mark.parametrize("height_change", [-2.0, 2.0])
def test_horizontal_stage_rejects_changed_height_before_any_command(plan, stage, height_change):
    hardware = FakeHardware(high(plan.waypoints["pick_A"].pose, plan.safe_z_mm))
    destination = high(plan.waypoints["place_B"].pose, plan.safe_z_mm + height_change)
    with pytest.raises(RuntimeError, match="same high plane"):
        cycle._move(hardware, destination, cycle.TaskJournal(plan), stage)
    assert hardware.events == []


@pytest.mark.parametrize("mode", ["refill", "chain"])
def test_actual_static_command_steps_leave_hardware_settling_margin(waypoint_path, mode):
    task = load_plan(waypoint_path, mode=mode)
    previous = task.waypoints["pick_A"].pose
    hardware = FakeHardware(previous)
    cycle.run_task(task, hardware)
    moves = [event[1] for event in hardware.events if event[0] == "move"]
    assert moves
    for goal in moves:
        assert math.dist(previous[:3], goal[:3]) <= 4.5 + 1e-8
        yaw_delta = (goal[3] - previous[3] + 180.0) % 360.0 - 180.0
        assert abs(yaw_delta) <= 1.5 + 1e-8
        previous = goal


@pytest.mark.parametrize("unsafe_state", ["pump_on", "unknown_pump", "unknown_pose", "alarm", "nan_pose"])
def test_initial_unknown_or_unsafe_state_is_rejected_before_arming(plan, unsafe_state):
    hardware = FakeHardware(plan.waypoints["pick_A"].pose)
    if unsafe_state == "pump_on":
        hardware.suction_on = True
    elif unsafe_state == "unknown_pump":
        hardware.suction_on = None
    elif unsafe_state == "unknown_pose":
        hardware.pose = (210.0, 0.0, 65.0, 0.0)
    elif unsafe_state == "alarm":
        hardware.alarms = [0x43]
    else:
        hardware.pose = (float("nan"), 0.0, 0.0, 0.0)
    with pytest.raises((RuntimeError, ValueError)):
        cycle.run_task(plan, hardware)
    assert [event[0] for event in hardware.events] == ["confirm"]


def test_existing_log_is_never_overwritten_or_resumed(plan, tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("keep existing evidence\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        cycle.TaskJournal(plan, path)
    assert path.read_text(encoding="utf-8") == "keep existing evidence\n"


def test_successful_run_persists_finite_route_and_no_resume_permission(plan, tmp_path):
    path = tmp_path / "new-run.json"
    journal = cycle.TaskJournal(plan, path)
    cycle.run_task(plan, FakeHardware(plan.waypoints["pick_A"].pose), journal)
    log = json.loads(path.read_text(encoding="utf-8"))
    assert log["status"] == "complete"
    assert log["completed_destinations"] == ["place_B", "place_C", "place_D"]
    assert log["automatic_resume_allowed"] is False
    assert log["config_sha256"] == plan.config_sha256
    assert [event["stage"] for event in log["events"]].count("trial_lift_10mm") == 3


def block_hardware_import(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.endswith("suction_task_hardware") or name in {"rclpy", "serial"}:
            pytest.fail("Offline CLI attempted to import hardware")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_cli_default_is_offline_and_never_imports_hardware(waypoint_path, monkeypatch, capsys):
    block_hardware_import(monkeypatch)
    assert cycle.main(["--waypoints", str(waypoint_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "model_geometry_pass_only"


def test_cli_execute_rejects_noninteractive_input_before_hardware_import(waypoint_path, monkeypatch):
    monkeypatch.setattr(cycle.sys.stdin, "isatty", lambda: False)
    block_hardware_import(monkeypatch)
    with pytest.raises(ValueError, match="interactive terminal"):
        cycle.main(["--waypoints", str(waypoint_path), "--execute"])


def test_cli_check_reads_and_closes_without_arming_or_moving(waypoint_path, plan, monkeypatch, capsys):
    hardware = FakeHardware(plan.waypoints["pick_A"].pose)
    module = ModuleType("mani_dobot.suction_task_hardware")
    constructions = []

    def construct(*args, **kwargs):
        constructions.append((args, kwargs))
        return hardware

    module.RosDobotHardware = construct
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert cycle.main(["--waypoints", str(waypoint_path), "--check"]) == 0
    assert len(constructions) == 1
    assert hardware.events == [("close",)]
    assert '"motion_started": false' in capsys.readouterr().out


class GraspYawReadbackHardware(FakeHardware):
    """Model small real yaw settling that is visible immediately after pump-on."""

    def __init__(self, pose, drifts):
        super().__init__(pose)
        self.drifts = iter(drifts)

    def suction(self, enabled):
        super().suction(enabled)
        if enabled:
            self.pose = self.pose[:3] + (self.pose[3] + next(self.drifts),)
            self.events.append(("grasp_readback", self.pose))


@pytest.mark.parametrize("start_at_trial_lift", [False, True])
@pytest.mark.parametrize("drifts", [(0.0, 0.0), (0.17, -0.21)])
@pytest.mark.parametrize("continuous", [False, True])
def test_hold_yaw_uses_each_actual_grasp_readback_through_release_in_two_rounds(
        waypoint_path, start_at_trial_lift, drifts, continuous):
    original_bytes = waypoint_path.read_bytes()
    taught = load_plan(waypoint_path)
    original_points = dict(taught.waypoints)
    assert taught.yaw_policy == "taught"
    task = fixed_yaw_plan(replace(taught, order=("place_B", "place_C"),
                                  routes=(("pick_A", "place_B"), ("pick_A", "place_C"))))
    source = task.waypoints["pick_A"].pose
    hardware = GraspYawReadbackHardware(lifted(source) if start_at_trial_lift else source, drifts)
    journal = cycle.TaskJournal(task)
    result = cycle.run_task(task, hardware, journal, continuous_transfers=continuous)
    assert result["status"] == "complete"
    assert result["completed_destinations"] == ["place_B", "place_C"]
    assert result["yaw_policy"] == "hold"
    locks = [event["locked_grasp_yaw_deg"] for event in result["events"]
             if event["stage"] == "grasp_yaw_locked"]
    assert locks == pytest.approx([source[3] + drift for drift in drifts])
    assert result["locked_grasp_yaw_deg"] == pytest.approx(locks[-1])
    on_indices = [i for i, event in enumerate(hardware.events)
                  if event[0] == "suction" and event[1] is True]
    off_indices = [i for i, event in enumerate(hardware.events)
                   if event[0] == "suction" and event[1] is False]
    for round_index, (on_i, off_i, locked) in enumerate(zip(on_indices, off_indices, locks)):
        assert hardware.events[on_i + 1] == ("grasp_readback", source[:3] + (locked,))
        loaded = [event for event in hardware.events[on_i + 1:off_i] if event[0] in ("move", "transfer")]
        assert loaded
        assert all(event[1][3] == pytest.approx(locked, abs=1e-10) for event in loaded)
        assert hardware.events[off_i][2][3] == pytest.approx(locked, abs=1e-10)
        end = on_indices[round_index + 1] if round_index + 1 < len(on_indices) else len(hardware.events)
        next_ready = next((i for i in range(off_i + 1, end)
                           if hardware.events[i][0:2] == ("confirm", "READY A")), end)
        # Both the local 10 mm release check and full empty retract retain R.
        release_moves = [event for event in hardware.events[off_i + 1:next_ready] if event[0] == "move"]
        assert release_moves
        assert all(event[1][3] == pytest.approx(locked, abs=1e-10) for event in release_moves)
    assert taught.waypoints == original_points
    assert waypoint_path.read_bytes() == original_bytes


@pytest.mark.parametrize("speed", [25.0, 50.0])
def test_cli_hold_continuous_defaults_select_only_b_c_and_accept_speed_up_to_50(
        waypoint_path, plan, tmp_path, monkeypatch, speed):
    hardware = GraspYawReadbackHardware(lifted(plan.waypoints["pick_A"].pose), [0.17, -0.21])
    module = ModuleType("mani_dobot.suction_task_hardware")
    constructions = []

    def construct(*args, **kwargs):
        constructions.append((args, kwargs))
        return hardware

    module.RosDobotHardware = construct
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cycle.sys.stdin, "isatty", lambda: True)
    log_path = tmp_path / "hold-b-c.json"
    assert cycle.main(["--waypoints", str(waypoint_path), "--execute", "--targets", "B", "C",
                       "--speed-mm-s", str(speed), "--log", str(log_path)]) == 0
    assert len(constructions) == 1
    assert constructions[0][1]["velocity_mm_s"] == speed
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["yaw_policy"] == "hold"
    assert log["routes"] == [["pick_A", "place_B"], ["pick_A", "place_C"]]
    assert log["completed_destinations"] == ["place_B", "place_C"]
    assert all(event.get("target") != "place_D" for event in log["events"])
    assert hardware.gate_count == 6
    assert any(event[0] == "transfer" for event in hardware.events)
    assert log["continuous_transfers"] is True
    assert hardware.events[-1] == ("close",)


@pytest.mark.parametrize("speed", ["50.001", "51", "nan", "inf"])
def test_cli_rejects_speed_above_50_or_nonfinite_before_hardware_import(waypoint_path, monkeypatch, speed):
    block_hardware_import(monkeypatch)
    with pytest.raises(ValueError, match="between 5 and 50"):
        cycle.main(["--waypoints", str(waypoint_path), "--speed-mm-s", speed, "--execute"])


def test_cli_explicit_taught_policy_preserves_old_endpoint_yaws(waypoint_path, monkeypatch, capsys):
    block_hardware_import(monkeypatch)
    assert cycle.main(["--waypoints", str(waypoint_path), "--yaw-policy", "taught",
                       "--transfer-mode", "segmented"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["yaw_policy"] == "taught"
    assert summary["order"] == ["place_B", "place_C", "place_D"]
    for name, (_, yaw, _) in CONTACTS.items():
        assert summary["waypoints"][name]["pose_mm_deg"][3] == pytest.approx(yaw)


def test_continuous_two_route_run_uses_long_moves_only_on_high_plane_and_small_contact_steps(plan):
    task = fixed_yaw_plan(replace(plan, order=("place_B", "place_C"),
                                  routes=(("pick_A", "place_B"), ("pick_A", "place_C"))))
    previous = lifted(task.waypoints["pick_A"].pose)
    hardware = GraspYawReadbackHardware(previous, [0.17, -0.21])
    journal = cycle.TaskJournal(task)
    cycle.run_task(task, hardware, journal, continuous_transfers=True)
    locked = None
    transfers = []
    for event in hardware.events:
        if event[0] == "grasp_readback":
            previous = event[1]
            locked = previous[3]
        elif event[0] in ("move", "transfer"):
            goal = event[1]
            yaw_delta = (goal[3] - previous[3] + 180.0) % 360.0 - 180.0
            if event[0] == "transfer":
                transfers.append(event)
                assert previous[2] == pytest.approx(task.safe_z_mm)
                assert goal[2] == pytest.approx(task.safe_z_mm)
                assert math.dist(previous[:2], goal[:2]) > 0.25
                assert abs(yaw_delta) <= 0.3 + 1e-8
            else:
                assert math.dist(previous[:3], goal[:3]) <= 4.5 + 1e-8
                assert abs(yaw_delta) <= 1.5 + 1e-8
            if event[2]:
                assert goal[3] == pytest.approx(locked, abs=1e-10)
            previous = goal
    assert len(transfers) == 3
    assert [event[2] for event in transfers] == [True, False, True]
    continuous_stages = [event for event in journal.data["events"]
                         if event.get("motion_profile") == "continuous"]
    assert [event["stage"] for event in continuous_stages] == [
        "transfer_at_safe_height", "above_source", "transfer_at_safe_height",
    ]
    assert all(event["segments"] == 1 and event["validation_segment_count"] > 1
               for event in continuous_stages)


@pytest.mark.parametrize("stage", [
    "raise_empty_at_current", "descend_to_source", "trial_lift_10mm",
    "lift_loaded", "descend_to_target", "release_check_retract_10mm", "retract_empty",
])
def test_continuous_flag_never_turns_vertical_contact_stages_into_long_commands(plan, stage):
    start = plan.waypoints["pick_A"].pose
    hardware = FakeHardware(start)
    cycle._move(hardware, lifted(start), cycle.TaskJournal(plan), stage, continuous=True)
    assert hardware.events
    previous = start
    for event in hardware.events:
        assert event[0] == "move"
        assert event[1][:2] == pytest.approx(start[:2])
        assert math.dist(previous[:3], event[1][:3]) <= 4.5 + 1e-8
        previous = event[1]


def test_continuous_transfer_validates_whole_path_before_sending_long_command(plan):
    # Valid endpoints, but the high-plane chord crosses the central dead zone.
    start = firmware_fk((-90.0, 35.0, 40.0, 90.0))
    end = firmware_fk((90.0, 35.0, 40.0, -90.0))
    hardware = FakeHardware(start)
    with pytest.raises(ValueError, match="Path is invalid"):
        cycle._move(hardware, end, cycle.TaskJournal(plan), "transfer_at_safe_height", continuous=True)
    assert hardware.events == []


def test_continuous_loaded_transfer_failure_stops_and_retains_suction_without_retry(plan):
    task = fixed_yaw_plan(plan)
    hardware = FakeHardware(task.waypoints["pick_A"].pose, fail_loaded_xy=True)
    journal = cycle.TaskJournal(task)
    with pytest.raises(RuntimeError, match="Injected motion failure"):
        cycle.run_task(task, hardware, journal, continuous_transfers=True)
    assert len(hardware.failed_goals) == 1
    assert len([event for event in hardware.events if event[0] == "transfer"]) == 1
    assert hardware.events[-1] == ("stop",)
    assert hardware.suction_on is True
    assert [event[1] for event in hardware.events if event[0] == "suction"] == [True]
    assert journal.data["completed_destinations"] == []
    assert journal.data["status"] == "stopped"


def test_continuous_runtime_requires_hold_policy_before_any_hardware_operation(plan):
    hardware = FakeHardware(plan.waypoints["pick_A"].pose)
    with pytest.raises(ValueError, match="fixed grasp-yaw policy"):
        cycle.run_task(plan, hardware, continuous_transfers=True)
    assert hardware.events == []


def test_cli_taught_continuous_combination_is_rejected_before_hardware_import(waypoint_path, monkeypatch):
    block_hardware_import(monkeypatch)
    with pytest.raises(ValueError, match="requires --yaw-policy hold"):
        cycle.main(["--waypoints", str(waypoint_path), "--yaw-policy", "taught"])


def assert_split_speed_profile(hardware, initial_pose, *, vertical=30.0, transfer=50.0):
    previous, speed = initial_pose, None
    vertical_count = transfer_count = 0
    for event in hardware.events:
        if event[0] == "speed":
            speed = event[1]
        elif event[0] == "grasp_readback":
            previous = event[1]
        elif event[0] in ("move", "transfer"):
            goal = event[1]
            if event[0] == "transfer" or math.dist(previous[:2], goal[:2]) > 1e-8:
                assert speed == transfer
                transfer_count += 1
            elif abs(goal[2] - previous[2]) > 1e-8:
                assert speed == vertical
                vertical_count += 1
            previous = goal
    assert vertical_count > 0
    assert transfer_count > 0


@pytest.mark.parametrize("continuous", [False, True])
def test_start_only_retains_trial_lift_and_release_retract_with_split_speeds(plan, continuous):
    task = fixed_yaw_plan(replace(plan, order=("place_B", "place_C"),
                                  routes=(("pick_A", "place_B"), ("pick_A", "place_C"))))
    initial = lifted(task.waypoints["pick_A"].pose)
    hardware = GraspYawReadbackHardware(initial, [0.17, -0.21])
    journal = cycle.TaskJournal(task)
    result = cycle.run_task(task, hardware, journal, continuous_transfers=continuous,
                            vertical_speed_mm_s=30.0, transfer_speed_mm_s=50.0,
                            confirm_each_action=False)
    assert result["completed_destinations"] == ["place_B", "place_C"]
    assert [event[1] for event in hardware.events if event[0] == "confirm"] == ["READY A", "READY A"]
    assert [event[1] for event in hardware.events if event[0] == "suction"] == [True, False, True, False]
    assert [event["stage"] for event in result["events"]].count("trial_lift_10mm") == 2
    assert [event["stage"] for event in result["events"]].count("release_check_retract_10mm") == 2
    assert not any("operator_confirmed" in str(event.get("holding_assumption", ""))
                   for event in result["events"])
    assert_split_speed_profile(hardware, initial)


@pytest.fixture
def held_task(plan):
    return fixed_yaw_plan(replace(plan, order=("place_C",), routes=(("pick_A", "place_C"),)))


@pytest.mark.parametrize("confirm_each_action,expected_gates", [
    (True, ["CONTINUE HELD", "RELEASED"]),
    (False, ["CONTINUE HELD"]),
])
def test_finish_held_route_requires_explicit_continuation_then_releases_only_target(
        held_task, confirm_each_action, expected_gates):
    source = held_task.waypoints["pick_A"].pose
    initial = lifted(source)[:3] + (source[3] + 0.17,)
    hardware = FakeHardware(initial)
    hardware.suction_on = True
    journal = cycle.TaskJournal(held_task)
    result = cycle.finish_held_route(
        held_task, hardware, journal, target_name="place_C",
        vertical_speed_mm_s=30.0, transfer_speed_mm_s=50.0,
        continuous_transfers=True, confirm_each_action=confirm_each_action,
    )
    assert result["status"] == "complete"
    assert result["completed_destinations"] == ["place_C"]
    assert [event[1] for event in hardware.events if event[0] == "confirm"] == expected_gates
    assert hardware.events.count(("arm_held",)) == 1
    assert ("arm",) not in hardware.events
    assert [event[1] for event in hardware.events if event[0] == "suction"] == [False]
    assert hardware.suction_on is False
    first_move = next(event for event in hardware.events if event[0] in ("move", "transfer"))
    assert first_move[1][:2] == pytest.approx(source[:2])
    assert first_move[1][2] > initial[2]
    # The continuation must not descend to A or toggle its already-active pump.
    assert not any(event.get("stage") == "descend_to_source" for event in result["events"])
    for event in hardware.events:
        if event[0] in ("move", "transfer"):
            assert event[1][3] == pytest.approx(initial[3], abs=1e-10)
    assert_split_speed_profile(hardware, initial)


@pytest.mark.parametrize("bad_start", ["pump_off", "unknown_pump", "xy", "z", "yaw", "alarm"])
def test_finish_held_route_rejects_wrong_start_without_motion_or_pump_commands(held_task, bad_start):
    initial = list(lifted(held_task.waypoints["pick_A"].pose))
    hardware = FakeHardware(initial)
    hardware.suction_on = True
    if bad_start == "pump_off":
        hardware.suction_on = False
    elif bad_start == "unknown_pump":
        hardware.suction_on = None
    elif bad_start in ("xy", "z", "yaw"):
        axis = {"xy": 0, "z": 2, "yaw": 3}[bad_start]
        initial[axis] += 2.0
        hardware.pose = tuple(initial)
    else:
        hardware.alarms = [0x43]
    with pytest.raises((RuntimeError, ValueError)):
        cycle.finish_held_route(held_task, hardware, cycle.TaskJournal(held_task), target_name="place_C")
    assert not any(event[0] in ("arm", "arm_held", "move", "transfer", "suction")
                   for event in hardware.events)


@pytest.mark.parametrize("bad_plan", ["taught", "chain", "target_not_in_routes"])
def test_finish_held_route_rejects_nonhold_nonrefill_or_unselected_target(plan, held_task, bad_plan):
    task, target = held_task, "place_C"
    if bad_plan == "taught":
        task = plan
    elif bad_plan == "chain":
        task = replace(held_task, mode="chain")
    else:
        target = "place_D"
    hardware = FakeHardware(lifted(held_task.waypoints["pick_A"].pose))
    hardware.suction_on = True
    with pytest.raises((RuntimeError, ValueError)):
        cycle.finish_held_route(task, hardware, cycle.TaskJournal(task), target_name=target)
    assert not any(event[0] in ("arm", "arm_held", "move", "transfer", "suction")
                   for event in hardware.events)


def test_finish_held_transfer_failure_stops_without_disabling_suction_or_retrying(held_task):
    hardware = FakeHardware(lifted(held_task.waypoints["pick_A"].pose), fail_loaded_xy=True)
    hardware.suction_on = True
    journal = cycle.TaskJournal(held_task)
    with pytest.raises(RuntimeError, match="Injected motion failure"):
        cycle.finish_held_route(held_task, hardware, journal, target_name="place_C",
                                confirm_each_action=False)
    assert hardware.suction_on is True
    assert len(hardware.failed_goals) == 1
    assert [event[1] for event in hardware.events if event[0] == "confirm"] == ["CONTINUE HELD"]
    assert hardware.events[-1] == ("stop",)
    assert not any(event[0] == "suction" for event in hardware.events)
    assert journal.data["status"] == "stopped"
    assert journal.data["completed_destinations"] == []


@pytest.mark.parametrize("changed", ["pose", "pump"])
def test_finish_held_rechecks_pose_and_pump_after_arming_before_first_move(held_task, changed):
    class ChangedOnArmHardware(FakeHardware):
        def arm(self, *, holding_confirmed=False):
            super().arm(holding_confirmed=holding_confirmed)
            if changed == "pose":
                self.pose = (self.pose[0] + 2.0,) + self.pose[1:]
            else:
                self.suction_on = False

    hardware = ChangedOnArmHardware(lifted(held_task.waypoints["pick_A"].pose))
    hardware.suction_on = True
    with pytest.raises((RuntimeError, ValueError)):
        cycle.finish_held_route(held_task, hardware, cycle.TaskJournal(held_task), target_name="place_C")
    assert ("arm_held",) in hardware.events
    assert hardware.events[-1] == ("stop",)
    assert not any(event[0] in ("move", "transfer", "suction") for event in hardware.events)


@pytest.mark.parametrize("args", [
    ["--continue-held", "--targets", "C"],
    ["--continue-held", "--execute", "--targets", "B", "C"],
    ["--continue-held", "--execute", "--targets", "C", "--mode", "chain"],
    ["--continue-held", "--execute", "--targets", "C", "--yaw-policy", "taught",
     "--transfer-mode", "segmented"],
])
def test_cli_rejects_invalid_continue_held_modes_before_hardware_import(waypoint_path, monkeypatch, args):
    block_hardware_import(monkeypatch)
    with pytest.raises(ValueError):
        cycle.main(["--waypoints", str(waypoint_path)] + args)


def test_cli_continue_held_start_only_uses_30_vertical_50_transfer_and_single_target(
        waypoint_path, plan, tmp_path, monkeypatch):
    initial = lifted(plan.waypoints["pick_A"].pose)
    hardware = FakeHardware(initial)
    hardware.suction_on = True
    module = ModuleType("mani_dobot.suction_task_hardware")
    module.RosDobotHardware = lambda *args, **kwargs: hardware
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cycle.sys.stdin, "isatty", lambda: True)
    path = tmp_path / "continue-c.json"
    assert cycle.main([
        "--waypoints", str(waypoint_path), "--execute", "--continue-held", "--targets", "C",
        "--speed-mm-s", "50", "--confirmation-policy", "start-only", "--log", str(path),
    ]) == 0
    log = json.loads(path.read_text(encoding="utf-8"))
    assert log["completed_destinations"] == ["place_C"]
    assert [event[1] for event in hardware.events if event[0] == "confirm"] == ["CONTINUE HELD"]
    assert [event[1] for event in hardware.events if event[0] == "suction"] == [False]
    assert hardware.events[-1] == ("close",)
    assert_split_speed_profile(hardware, initial)


@pytest.mark.parametrize("failed_stage", ["trial_pump_lost", "release_pump_stuck"])
def test_start_only_still_checks_pump_state_before_transfer_or_next_target(plan, failed_stage):
    task = fixed_yaw_plan(plan)
    trial_pose = lifted(task.waypoints["pick_A"].pose)

    class BadPumpStateHardware(FakeHardware):
        def move_segment(self, goal):
            super().move_segment(goal)
            if (failed_stage == "trial_pump_lost" and self.suction_on
                    and self.pose == pytest.approx(trial_pose)):
                self.suction_on = False

        def suction(self, enabled):
            super().suction(enabled)
            if not enabled and failed_stage == "release_pump_stuck":
                self.suction_on = True

    hardware = BadPumpStateHardware(task.waypoints["pick_A"].pose)
    journal = cycle.TaskJournal(task)
    with pytest.raises(RuntimeError):
        cycle.run_task(task, hardware, journal, continuous_transfers=True,
                        vertical_speed_mm_s=30.0, transfer_speed_mm_s=50.0,
                        confirm_each_action=False)
    assert hardware.events[-1] == ("stop",)
    assert [event[1] for event in hardware.events if event[0] == "confirm"] == ["READY A"]
    assert journal.data["completed_destinations"] == []
    assert journal.data["status"] == "stopped"
    pump_commands = [event[1] for event in hardware.events if event[0] == "suction"]
    assert pump_commands == ([True] if failed_stage == "trial_pump_lost" else [True, False])
    transfers = [event for event in hardware.events if event[0] == "transfer"]
    assert len(transfers) == (0 if failed_stage == "trial_pump_lost" else 1)


def test_finish_held_continuation_cannot_bypass_explicit_gate_under_start_only(held_task):
    hardware = FakeHardware(lifted(held_task.waypoints["pick_A"].pose), reject_gate=1)
    hardware.suction_on = True
    with pytest.raises(RuntimeError, match="Operator cancelled"):
        cycle.finish_held_route(held_task, hardware, target_name="place_C", confirm_each_action=False)
    assert hardware.events == [("confirm", "CONTINUE HELD", hardware.pose, True)]
    assert hardware.suction_on is True


def test_finish_held_rechecks_manual_pose_change_during_continuation_gate(held_task):
    hardware = FakeHardware(lifted(held_task.waypoints["pick_A"].pose),
                            gate_deltas={1: (2.0, 0.0, 0.0, 0.0)})
    hardware.suction_on = True
    with pytest.raises(RuntimeError, match="changed during continuation confirmation"):
        cycle.finish_held_route(held_task, hardware, target_name="place_C", confirm_each_action=False)
    assert hardware.suction_on is True
    assert not any(event[0] in ("arm", "arm_held", "move", "transfer", "suction")
                   for event in hardware.events)
