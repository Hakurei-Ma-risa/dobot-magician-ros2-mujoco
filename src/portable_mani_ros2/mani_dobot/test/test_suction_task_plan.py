"""Offline model properties of the actual four taught task points."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math

import pytest

from mani_dobot.suction_task_plan import (
    COORDINATE_CONVENTION,
    EXPECTED_ROBOT_PORT,
    firmware_fk,
    firmware_ik,
    fixed_yaw_plan,
    interpolate_segments,
    load_plan,
    plan_summary,
)


POINTS = {
    "pick_A": ([204.83059692382812, 69.3899154663086, -45.65318298339844],
               18.71470069885254, [18.71470069885254, 48.07060623168945, 67.55284881591797, 0.]),
    "place_B": ([206.99929809570312, -63.36740112304687, -45.26304626464844],
                -17.020595550537113, [-17.020593643188477, 47.94572067260742, 67.37850189208984, -0.0000019073486328125]),
    "place_C": ([209.57937622070312, -92.19148254394531, -47.31372833251953],
                -23.74412536621094, [-23.744123458862305, 51.040321350097656, 64.06710815429688, -0.0000019073486328125]),
    "place_D": ([210.87490844726562, -147.76919555664062, -46.8529052734375],
                -35.020599365234375, [-35.02059555053711, 57.050655364990234, 54.90747833251953, -0.0000019073486328125]),
}


def config():
    data = {"schema_version": 1, "coordinate_convention": COORDINATE_CONVENTION,
            "robot_port": EXPECTED_ROBOT_PORT, "controller_tool_bias_mm": [59.70000076293945, 0., 0.],
            "tool": "suction_cup", "all_contact_waypoints_recorded": True,
            "destinations_in_order": ["place_B", "place_C", "place_D"],
            "automatic_pick_and_place_ready": False, "tcp_reference_verified": False}
    for name, (xyz, yaw, joints) in POINTS.items():
        data[name] = {"status": "manually_taught_contact_recorded", "controller_xyz_mm": list(xyz),
                      "controller_yaw_deg": yaw, "vendor_joints_deg": list(joints)}
    return data


def write_config(tmp_path, data=None):
    path = tmp_path / "waypoints.json"
    path.write_text(json.dumps(config() if data is None else data))
    return path


@pytest.mark.parametrize("name", list(POINTS))
def test_actual_taught_fk_and_ik_match_each_other(name):
    xyz, yaw, raw = POINTS[name]
    predicted = firmware_fk(raw)
    assert math.dist(predicted[:3], xyz) < 0.0001
    assert predicted[3] == pytest.approx(yaw, abs=0.00001)
    assert firmware_ik(tuple(xyz)+(yaw,)) == pytest.approx(raw, abs=0.00001)


def test_default_plan_preserves_taught_points_and_does_not_claim_hardware_pass(tmp_path):
    path = write_config(tmp_path)
    plan = load_plan(path)
    assert plan.mode == "refill"
    assert plan.safe_z_mm == -10
    assert plan.routes == (("pick_A", "place_B"), ("pick_A", "place_C"), ("pick_A", "place_D"))
    assert plan.order == ("place_B", "place_C", "place_D")
    assert plan.config_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert plan.expected_bias_mm == (59.70000076293945, 0., 0.)
    assert plan.waypoints["pick_A"].pose[:3] == tuple(POINTS["pick_A"][0])
    with pytest.raises(FrozenInstanceError):
        plan.mode = "chain"
    summary = plan_summary(plan)
    assert summary["model_geometry_pass"]
    assert not summary["physical_transfer_validated"]
    assert not summary["automatic_pick_and_place_ready"]
    assert summary["requires_refill_before_each_pick_A"]
    assert summary["raw_joint_max_deg"][3] > 2.7
    assert summary["minimum_joint_limit_margin_deg"] == pytest.approx(2.447152, abs=0.00001)
    json.dumps(summary, allow_nan=False)


def test_chain_mode_reuses_each_placed_contact(tmp_path):
    plan = load_plan(write_config(tmp_path), mode="chain")
    assert plan.routes == (("pick_A", "place_B"), ("place_B", "place_C"), ("place_C", "place_D"))
    assert not plan_summary(plan)["requires_refill_before_each_pick_A"]


@pytest.mark.parametrize("key,value", [
    ("schema_version", 2), ("schema_version", True), ("tool", "gripper"),
    ("all_contact_waypoints_recorded", False), ("coordinate_convention", "metres"),
    ("units", {"position": "m", "angles": "radians"}),
    ("robot_port", "/dev/ttyACM0"), ("robot_port", "/dev/serial/by-id/another-robot"),
    ("controller_tool_bias_mm", [0., 0., 0.]), ("controller_tool_bias_mm", [59.7, 1., 0.]),
    ("controller_tool_bias_mm", [59.7, 0., 1.]), ("controller_tool_bias_mm", [float("nan"), 0., 0.]),
    ("destinations_in_order", ["place_C", "place_B", "place_D"]),
])
def test_changed_robot_geometry_units_or_schema_are_rejected(tmp_path, key, value):
    data = config()
    data[key] = value
    with pytest.raises(ValueError):
        load_plan(write_config(tmp_path, data))


@pytest.mark.parametrize("name", ["pick_A", "place_B", "place_C", "place_D"])
def test_each_missing_or_untaught_contact_blocks_plan(tmp_path, name):
    data = config()
    data[name]["status"] = "pending_manual_teaching"
    with pytest.raises(ValueError, match="manually taught"):
        load_plan(write_config(tmp_path, data))
    del data[name]
    with pytest.raises(ValueError):
        load_plan(write_config(tmp_path, data))


def test_duplicate_contacts_are_not_a_new_destination(tmp_path):
    data = config()
    data["place_C"] = dict(data["place_B"])
    with pytest.raises(ValueError, match="Duplicate"):
        load_plan(write_config(tmp_path, data))


@pytest.mark.parametrize("field,value", [
    ("controller_xyz_mm", [float("inf"), 0., 0.]),
    ("controller_xyz_mm", [204., 0.]), ("controller_xyz_mm", [True, 0., 0.]),
    ("controller_yaw_deg", float("nan")), ("controller_yaw_deg", "18.7"),
    ("vendor_joints_deg", [0., 0., 0.]), ("vendor_joints_deg", [0., 0., float("nan"), 0.]),
    ("alarm_codes", [0x43]),
])
def test_malformed_or_alarmed_contact_blocks_plan(tmp_path, field, value):
    data = config()
    data["pick_A"][field] = value
    with pytest.raises(ValueError):
        load_plan(write_config(tmp_path, data))


def test_inconsistent_pose_and_raw_joints_blocks_wrong_geometry(tmp_path):
    data = config()
    data["pick_A"]["controller_xyz_mm"][0] += 1.0
    with pytest.raises(ValueError, match="disagree"):
        load_plan(write_config(tmp_path, data))
    data = config()
    data["pick_A"]["controller_yaw_deg"] += 2.0
    with pytest.raises(ValueError, match="yaw"):
        load_plan(write_config(tmp_path, data))


@pytest.mark.parametrize("height", [-15., -50., 1000., float("nan")])
def test_transfer_height_requires_35mm_clearance_and_model_reachability(tmp_path, height):
    with pytest.raises(ValueError):
        load_plan(write_config(tmp_path), safe_z_mm=height)


def test_exact_clearance_threshold_and_higher_reachable_plane(tmp_path):
    path = write_config(tmp_path)
    minimum = max(point[0][2] for point in POINTS.values()) + 35.
    assert load_plan(path, safe_z_mm=minimum).safe_z_mm == minimum
    assert load_plan(path, safe_z_mm=0.).safe_z_mm == 0.


def test_segment_steps_are_bounded_and_shortest_yaw_wrap_is_used():
    # theta1=100 deg, yaw near 180 leaves raw theta4 near 80, inside limits.
    start = firmware_fk((100., 35., 40., 79.))
    end = (start[0]+1., start[1]+1., start[2]+1., -179.)
    goals = interpolate_segments(start, end, max_step_mm=0.5, max_yaw_step_deg=0.4)
    assert goals
    previous = start
    yaw_travel = 0.
    for goal in goals:
        assert math.dist(previous[:3], goal[:3]) <= 0.5+1e-10
        yaw_step = (goal[3]-previous[3]+180.)%360.-180.
        assert abs(yaw_step) <= 0.4+1e-10
        yaw_travel += yaw_step
        firmware_ik(goal)
        previous = goal
    assert yaw_travel == pytest.approx(2.)
    assert goals[-1] == pytest.approx(end)
    assert goals[0] != start


def test_zero_motion_returns_no_goals_but_still_checks_start():
    pose = tuple(POINTS["pick_A"][0])+(POINTS["pick_A"][1],)
    assert interpolate_segments(pose, pose) == []
    with pytest.raises(ValueError):
        interpolate_segments((1000., 0., 0., 0.), (1000., 0., 0., 0.))


def test_path_interior_is_checked_even_with_coarse_steps():
    # Both endpoints have valid angles, but the straight chord passes through
    # the unsupported central workspace at x~0 and must never be accepted.
    start = firmware_fk((-90., 35., 40., 0.))
    end = firmware_fk((90., 35., 40., 0.))
    firmware_ik(start)
    firmware_ik(end)
    with pytest.raises(ValueError, match="Path is invalid"):
        interpolate_segments(start, end, max_step_mm=1000., max_yaw_step_deg=360.)


@pytest.mark.parametrize("pose", [(1000., 0., 0., 0.), (20., 0., 0., 0.),
                                  (210., 0., -300., 0.), (210., 0., 0., 170.),
                                  (-210., 0., 0., 0.), (210., float("nan"), 0., 0.)])
def test_ik_rejects_unreachable_limits_and_nonfinite_inputs(pose):
    with pytest.raises(ValueError):
        firmware_ik(pose)


def test_firmware_bias_is_radial_not_the_absolute_wrist_yaw():
    # Changing only the wrist rotates R, but cannot change XYZ in this model.
    first = firmware_fk((18., 35., 40., 0.))
    second = firmware_fk((18., 35., 40., 50.))
    assert second[:3] == first[:3]
    assert second[3]-first[3] == pytest.approx(50.)


def test_invalid_mode_and_bad_json_are_rejected(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(ValueError, match="mode"):
        load_plan(path, mode="unknown")
    path.write_text("not json")
    with pytest.raises(ValueError, match="JSON"):
        load_plan(path)


@pytest.mark.parametrize("step", [0., -1., float("nan"), float("inf"), 1e-310])
def test_bad_command_step_sizes_are_rejected_without_unbounded_allocation(step):
    start = tuple(POINTS["pick_A"][0])+(POINTS["pick_A"][1],)
    end = (start[0], start[1], start[2]+5., start[3])
    with pytest.raises(ValueError):
        interpolate_segments(start, end, max_step_mm=step)


def test_absurd_finite_pose_is_rejected_without_overflow():
    with pytest.raises(ValueError):
        interpolate_segments((1e308, 0., 0., 0.), (210., 0., 0., 0.))


def test_fixed_yaw_copies_without_mutating_taught_points_or_source_file(tmp_path):
    path = write_config(tmp_path)
    content = path.read_bytes()
    original = load_plan(path)
    taught_points = dict(original.waypoints)
    held = fixed_yaw_plan(original)
    reference = original.waypoints["pick_A"].pose[3]
    assert held is not original and held.waypoints is not original.waypoints
    assert original.yaw_policy == "taught" and original.yaw_reference_deg is None
    assert held.yaw_policy == "hold" and held.yaw_reference_deg == reference
    assert held.config_sha256 == original.config_sha256
    assert path.read_bytes() == content
    for name, source in taught_points.items():
        assert original.waypoints[name] is source
        assert held.waypoints[name] is not source
        assert held.waypoints[name].pose[:3] == source.pose[:3]
        assert held.waypoints[name].pose[3] == reference
        assert held.waypoints[name].joints[:3] == pytest.approx(source.joints[:3], abs=0.00001)
    assert held.waypoints["place_B"].joints[3] == pytest.approx(35.7352944, abs=0.00001)
    assert held.waypoints["place_C"].joints[3] == pytest.approx(42.4588255, abs=0.00001)
    summary = plan_summary(held)
    assert summary["yaw_policy"] == "hold"
    assert summary["yaw_reference_deg"] == reference
    assert not summary["physical_transfer_validated"]


@pytest.mark.parametrize("destination", ["place_B", "place_C", "place_D"])
def test_dense_held_yaw_routes_counterrotate_wrist_without_xyz_changes(tmp_path, destination):
    taught = load_plan(write_config(tmp_path))
    held = fixed_yaw_plan(taught)
    reference = held.yaw_reference_deg
    source, target = held.waypoints["pick_A"].pose, held.waypoints[destination].pose
    source_high = source[:2]+(held.safe_z_mm, reference)
    target_high = target[:2]+(held.safe_z_mm, reference)
    for start, end in [(source, source_high), (source_high, target_high), (target_high, target)]:
        goals = interpolate_segments(start, end, max_step_mm=0.5)
        for goal in [start]+goals:
            assert goal[3] == reference
            raw = firmware_ik(goal, held.expected_bias_mm)
            assert raw[0]+raw[3] == pytest.approx(reference, abs=1e-10)
            assert firmware_fk(raw, held.expected_bias_mm)[:3] == pytest.approx(goal[:3], abs=1e-10)
    # Base heading decreases towards B/C/D, while wrist angle grows to hold R.
    assert held.waypoints[destination].joints[0] < held.waypoints["pick_A"].joints[0]
    assert held.waypoints[destination].joints[3] > held.waypoints["pick_A"].joints[3]
    assert taught.waypoints[destination].joints[3] == pytest.approx(0., abs=0.00001)


def test_live_yaw_reference_can_override_pick_pose_and_is_validated(tmp_path):
    plan = load_plan(write_config(tmp_path))
    held = fixed_yaw_plan(plan, yaw_deg=18.9)
    assert held.yaw_reference_deg == 18.9
    assert all(point.pose[3] == 18.9 for point in held.waypoints.values())
    source = held.waypoints["pick_A"].pose
    target = held.waypoints["place_B"].pose
    source_high = source[:2]+(held.safe_z_mm, 18.9)
    target_high = target[:2]+(held.safe_z_mm, 18.9)
    assert all(goal[3] == 18.9 for goal in interpolate_segments(source_high, target_high))
    with pytest.raises(ValueError, match="joint 4"):
        fixed_yaw_plan(plan, yaw_deg=170.)
    with pytest.raises(ValueError, match="finite"):
        fixed_yaw_plan(plan, yaw_deg=float("nan"))


def test_hold_summary_rejects_inconsistent_yaw_even_if_each_pose_is_reachable(tmp_path):
    plan = load_plan(write_config(tmp_path))
    with pytest.raises(ValueError, match="same absolute R"):
        plan_summary(replace(plan, yaw_policy="hold", yaw_reference_deg=plan.waypoints["pick_A"].pose[3]))
