"""Offline geometry checks for this robot's manually taught suction task.

All coordinates are Dobot firmware XYZ in millimetres and absolute R in degrees.
No camera transform, hardware access, collision checker, or motion execution is
provided. A model pass never certifies the physical transfer or suction hold.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Sequence


EXPECTED_ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_58EA056287-if00"
EXPECTED_BIAS_MM = (59.7, 0.0, 0.0)
COORDINATE_CONVENTION = (
    "Dobot controller XYZ in millimetres and R in degrees; "
    "not a calibrated camera-to-base transform."
)
WAYPOINT_NAMES = ("pick_A", "place_B", "place_C", "place_D")
DESTINATION_NAMES = ("place_B", "place_C", "place_D")
RAW_JOINT_LOWER_DEG = (-125.0, -5.0, -15.0, -150.0)
RAW_JOINT_UPPER_DEG = (125.0, 90.0, 70.0, 150.0)
_TAUGHT_STATUSES = {
    "manually_taught_contact_recorded",
    "manually_taught_contact_and_vertical_lift_verified",
}


@dataclass(frozen=True)
class Waypoint:
    name: str
    pose: tuple[float, float, float, float]
    joints: tuple[float, ...]


@dataclass(frozen=True)
class TaskPlan:
    mode: str
    order: tuple[str, ...]
    routes: tuple[tuple[str, str], ...]
    safe_z_mm: float
    waypoints: dict[str, Waypoint]
    config_sha256: str
    robot_port: str
    expected_bias_mm: tuple[float, float, float]
    yaw_policy: str = "taught"
    yaw_reference_deg: float | None = None


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _vector(value, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} finite numbers")
    return tuple(_finite_number(item, name) for item in value)


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _bias(value: Sequence[float]) -> tuple[float, float, float]:
    bias = _vector(value, 3, "bias_mm")
    if bias[0] < 0.0 or abs(bias[1]) > 1e-8 or abs(bias[2]) > 1e-8:
        raise ValueError("This verified firmware model supports positive radial X bias and zero Y/Z bias only")
    return bias


def _validate_joints(raw: Sequence[float]) -> tuple[float, float, float, float]:
    joints = _vector(raw, 4, "raw joints")
    for index, (angle, lower, upper) in enumerate(zip(joints, RAW_JOINT_LOWER_DEG, RAW_JOINT_UPPER_DEG)):
        if not lower-1e-7 <= angle <= upper+1e-7:
            raise ValueError(f"joint {index+1}={angle:.6f} deg outside [{lower}, {upper}]")
    lower_third = -15.0 + max(joints[1] - 40.0, 0.0)
    if joints[2] < lower_third-1e-7:
        raise ValueError("Parallelogram coupled lower joint limit is violated")
    return joints


def firmware_fk(raw_joints_deg: Sequence[float], bias_mm=EXPECTED_BIAS_MM) -> tuple[float, float, float, float]:
    """Ideal firmware FK; the radial bias follows axis 1, not the wrist yaw."""
    joints = _validate_joints(raw_joints_deg)
    bias = _bias(bias_mm)
    first, rear, front, _ = map(math.radians, joints)
    radial = 135.0*math.sin(rear) + 147.0*math.cos(front) + bias[0]
    return (
        radial*math.cos(first), radial*math.sin(first),
        135.0*math.cos(rear) - 147.0*math.sin(front),
        _wrap_degrees(joints[0] + joints[3]),
    )


def firmware_ik(pose4: Sequence[float], bias_mm=EXPECTED_BIAS_MM) -> tuple[float, float, float, float]:
    """Invert the elbow branch verified by this robot's four taught poses."""
    x, y, z, yaw = _vector(pose4, 4, "firmware pose")
    bias = _bias(bias_mm)
    radial = math.hypot(x, y) - bias[0]
    if radial <= 0.0:
        raise ValueError("Pose lies inside the supported positive-radial firmware branch")
    distance = math.hypot(radial, z)
    first = math.degrees(math.atan2(y, x))
    c_shoulder = (distance*distance + 135.0**2 - 147.0**2) / (270.0*distance)
    c_elbow = (distance*distance - 135.0**2 - 147.0**2) / (2.0*135.0*147.0)
    if abs(c_shoulder) > 1.0+1e-12 or abs(c_elbow) > 1.0+1e-12:
        raise ValueError("Firmware pose is geometrically unreachable")
    rear = 90.0 - math.degrees(
        math.atan2(z, radial) + math.acos(max(-1.0, min(1.0, c_shoulder)))
    )
    front = math.degrees(math.acos(max(-1.0, min(1.0, c_elbow)))) - 90.0 + rear
    wrist = _wrap_degrees(yaw - first)
    return _validate_joints((first, rear, front, wrist))


def _interpolate(start4, end4, max_step_mm, max_yaw_step_deg, bias_mm) -> list[tuple[float, float, float, float]]:
    start, end = _vector(start4, 4, "start pose"), _vector(end4, 4, "end pose")
    step = _finite_number(max_step_mm, "max_step_mm")
    yaw_step = _finite_number(max_yaw_step_deg, "max_yaw_step_deg")
    if step <= 0 or yaw_step <= 0:
        raise ValueError("Interpolation step limits must be positive")
    firmware_ik(start, bias_mm)
    firmware_ik(end, bias_mm)
    distance = math.dist(start[:3], end[:3])
    yaw_delta = _wrap_degrees(end[3] - start[3])
    translation_parts, yaw_parts = distance/step, abs(yaw_delta)/yaw_step
    if max(translation_parts, yaw_parts) > 100000:
        raise ValueError("Interpolation is unreasonably large; check units and step limits")
    count = max(math.ceil(translation_parts), math.ceil(yaw_parts))
    dense_count = max(count, math.ceil(distance), math.ceil(abs(yaw_delta)/0.5), 1)
    if count > 100000 or dense_count > 500000:
        raise ValueError("Interpolation is unreasonably large; check units and step limits")

    def at(fraction):
        return tuple(start[i] + (end[i]-start[i])*fraction for i in range(3)) + (
            start[3] if yaw_delta == 0.0 else _wrap_degrees(start[3] + yaw_delta*fraction),
        )

    # The denser check catches an unreachable interior even when both endpoints
    # or the caller's coarse command segments are individually reachable.
    for index in range(dense_count+1):
        try:
            firmware_ik(at(index/dense_count), bias_mm)
        except ValueError as exc:
            raise ValueError(f"Path is invalid at fraction {index/dense_count:.6f}: {exc}") from exc
    goals = [at(index/count) for index in range(1, count+1)] if count else []
    for goal in goals:
        firmware_ik(goal, bias_mm)
    return goals


def interpolate_segments(start4, end4, max_step_mm=5.0, max_yaw_step_deg=2.0) -> list[tuple[float, float, float, float]]:
    """Return MOVL segment endpoints, excluding start, after dense IK checks.

    Yaw uses the shortest wrapped angular difference. A zero-motion request
    returns no goals but still validates the starting pose. This pure planner
    cannot guarantee a controller's rotation convention at the +/-180 boundary.
    """
    return _interpolate(start4, end4, max_step_mm, max_yaw_step_deg, EXPECTED_BIAS_MM)


def _high(waypoint: Waypoint, safe_z_mm: float) -> tuple[float, float, float, float]:
    x, y, _, yaw = waypoint.pose
    return (x, y, safe_z_mm, yaw)


def _check_waypoints(waypoints: dict[str, Waypoint], safe_z_mm: float, bias) -> None:
    height = _finite_number(safe_z_mm, "safe_z_mm")
    if set(waypoints) != set(WAYPOINT_NAMES):
        raise ValueError("Exactly pick_A/place_B/place_C/place_D must be provided")
    for index, name in enumerate(WAYPOINT_NAMES):
        waypoint = waypoints[name]
        if waypoint.name != name:
            raise ValueError(f"Waypoint name does not match its key: {name}")
        expected_joints = firmware_ik(waypoint.pose, bias)
        actual = _validate_joints(waypoint.joints)
        reconstructed = firmware_fk(actual, bias)
        if math.dist(waypoint.pose[:3], reconstructed[:3]) > 0.3:
            raise ValueError(f"{name}: stored raw joints disagree with firmware XYZ by more than 0.3 mm")
        if abs(_wrap_degrees(waypoint.pose[3]-reconstructed[3])) > 1.0:
            raise ValueError(f"{name}: stored yaw disagrees with raw joints by more than 1 degree")
        if any(abs(_wrap_degrees(a-b)) > 1.0 for a, b in zip(actual, expected_joints)):
            raise ValueError(f"{name}: recorded joints do not match the verified firmware IK branch")
        for previous in WAYPOINT_NAMES[:index]:
            if math.dist(waypoint.pose[:3], waypoints[previous].pose[:3]) < 1.0:
                raise ValueError(f"Duplicate or near-duplicate contact positions: {previous} and {name}")
        if waypoint.pose[2] > height:
            raise ValueError("A contact point is higher than the transfer plane")
    required = max(point.pose[2] for point in waypoints.values()) + 35.0
    if height < required-1e-9:
        raise ValueError(f"safe_z_mm must be at least {required:.6f} mm (35 mm above every contact)")


def load_plan(path, mode="refill", safe_z_mm=-10.0) -> TaskPlan:
    """Load this installation's taught task and validate every model route."""
    if mode not in ("refill", "chain"):
        raise ValueError("mode must be 'refill' or 'chain'")
    content = Path(path).expanduser().read_bytes()
    try:
        data = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Task file must be valid UTF-8 JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("Task file must contain a JSON object")
    if data.get("schema_version") != 1 or isinstance(data.get("schema_version"), bool):
        raise ValueError("Task schema_version must be 1")
    if data.get("coordinate_convention") != COORDINATE_CONVENTION:
        raise ValueError("Task coordinate_convention must explicitly specify firmware mm/degrees")
    if "units" in data and data["units"] != {"position": "mm", "angles": "degrees"}:
        raise ValueError("Task units must be position=mm and angles=degrees")
    if data.get("tool") != "suction_cup":
        raise ValueError("Task requires the suction_cup tool")
    if data.get("all_contact_waypoints_recorded") is not True:
        raise ValueError("All contact waypoints must be explicitly recorded")
    if data.get("robot_port") != EXPECTED_ROBOT_PORT:
        raise ValueError("Robot stable by-id path differs from this task's recorded robot")
    if data.get("destinations_in_order") != list(DESTINATION_NAMES):
        raise ValueError("Destination order must be place_B/place_C/place_D")
    bias = _bias(data.get("controller_tool_bias_mm"))
    if any(abs(a-b) > 0.05 for a, b in zip(bias, EXPECTED_BIAS_MM)):
        raise ValueError("Configured tool bias differs from the verified 59.7/0/0 mm geometry")
    waypoints = {}
    for name in WAYPOINT_NAMES:
        source = data.get(name)
        if not isinstance(source, dict) or source.get("status") not in _TAUGHT_STATUSES:
            raise ValueError(f"{name}: a confirmed manually taught contact is required")
        xyz = _vector(source.get("controller_xyz_mm"), 3, f"{name} XYZ")
        yaw = _finite_number(source.get("controller_yaw_deg"), f"{name} yaw")
        joints = _vector(source.get("vendor_joints_deg"), 4, f"{name} raw joints")
        if source.get("alarm_codes", []) != []:
            raise ValueError(f"{name}: recorded alarms must be empty")
        waypoints[name] = Waypoint(name, xyz+(yaw,), joints)
    _check_waypoints(waypoints, safe_z_mm, bias)
    routes = (("pick_A", "place_B"), ("pick_A", "place_C"), ("pick_A", "place_D"))
    if mode == "chain":
        routes = (("pick_A", "place_B"), ("place_B", "place_C"), ("place_C", "place_D"))
    plan = TaskPlan(mode, DESTINATION_NAMES, routes, float(safe_z_mm), waypoints,
                    hashlib.sha256(content).hexdigest(), data["robot_port"], bias)
    plan_summary(plan)  # No plan escapes without all dense route checks.
    return plan


def fixed_yaw_plan(plan: TaskPlan, yaw_deg: float | None = None) -> TaskPlan:
    """Copy a plan while holding its absolute firmware R throughout every leg.

    With no explicit reference, use the pick_A orientation. Runtime callers can
    supply the freshly measured R after grasping. XYZ, source-file fingerprint
    and taught objects are untouched; the wrist compensates for base rotation.
    """
    reference = _finite_number(
        plan.waypoints["pick_A"].pose[3] if yaw_deg is None else yaw_deg,
        "yaw reference",
    )
    waypoints = {}
    for name, point in plan.waypoints.items():
        pose = point.pose[:3] + (reference,)
        waypoints[name] = Waypoint(name, pose, firmware_ik(pose, plan.expected_bias_mm))
    held = replace(plan, waypoints=waypoints, yaw_policy="hold", yaw_reference_deg=reference)
    plan_summary(held)
    return held


def plan_summary(plan: TaskPlan) -> dict:
    """Return JSON-safe model diagnostics, never physical-transfer approval."""
    if plan.yaw_policy not in ("taught", "hold"):
        raise ValueError("yaw_policy must be 'taught' or 'hold'")
    if plan.yaw_policy == "hold":
        reference = _finite_number(plan.yaw_reference_deg, "yaw reference")
        if any(abs(_wrap_degrees(point.pose[3]-reference)) > 1e-8 for point in plan.waypoints.values()):
            raise ValueError("A held-yaw plan must use the same absolute R at every waypoint")
    _check_waypoints(plan.waypoints, plan.safe_z_mm, plan.expected_bias_mm)
    details, all_joints = [], []
    for source_name, destination_name in plan.routes:
        if source_name not in plan.waypoints or destination_name not in plan.waypoints:
            raise ValueError("Route contains an unknown waypoint")
        source, destination = plan.waypoints[source_name], plan.waypoints[destination_name]
        source_high, destination_high = _high(source, plan.safe_z_mm), _high(destination, plan.safe_z_mm)
        legs = ((source.pose, source_high), (source_high, destination_high),
                (destination_high, destination.pose))
        counts, length = [], 0.0
        for start, end in legs:
            segments = _interpolate(start, end, 5.0, 2.0, plan.expected_bias_mm)
            counts.append(len(segments))
            length += math.dist(start[:3], end[:3])
            dense = _interpolate(start, end, 1.0, 0.5, plan.expected_bias_mm)
            all_joints.extend(firmware_ik(point, plan.expected_bias_mm) for point in [start]+dense)
        details.append({
            "source": source_name, "destination": destination_name,
            "source_high_pose_mm_deg": list(source_high),
            "destination_high_pose_mm_deg": list(destination_high),
            "stages": ["vertical_lift", "high_plane_transfer", "vertical_descent"],
            "command_segment_counts": counts, "distance_mm": length,
            "model_geometry_pass": True, "physical_transfer_validated": False,
        })
    lower = [min(q[i] for q in all_joints) for i in range(4)]
    upper = [max(q[i] for q in all_joints) for i in range(4)]
    margin = min(
        min(*(distance for angle, low, high in zip(q, RAW_JOINT_LOWER_DEG, RAW_JOINT_UPPER_DEG)
              for distance in (angle-low, high-angle)), q[2]-(-15.0+max(q[1]-40.0, 0.0)))
        for q in all_joints
    )
    return {
        "schema_version": 1, "status": "model_geometry_pass_only", "model_geometry_pass": True,
        "coordinate_convention": COORDINATE_CONVENTION,
        "units": {"position": "mm", "angles": "degrees"},
        "assumed_link_lengths_mm": [135.0, 147.0],
        "minimum_contact_lift_mm": 35.0,
        "physical_transfer_validated": False, "automatic_pick_and_place_ready": False,
        "mode": plan.mode, "order": list(plan.order), "routes": [list(route) for route in plan.routes],
        "yaw_policy": plan.yaw_policy, "yaw_reference_deg": plan.yaw_reference_deg,
        "safe_z_mm": plan.safe_z_mm, "config_sha256": plan.config_sha256,
        "robot_port": plan.robot_port, "expected_bias_mm": list(plan.expected_bias_mm),
        "raw_joint_min_deg": lower, "raw_joint_max_deg": upper,
        "minimum_joint_limit_margin_deg": margin,
        "waypoints": {name: {"pose_mm_deg": list(point.pose), "raw_joints_deg": list(point.joints),
                             "lift_mm": plan.safe_z_mm-point.pose[2]} for name, point in plan.waypoints.items()},
        "route_details": details,
        "requires_refill_before_each_pick_A": plan.mode == "refill",
        "limitations": ["Ideal firmware geometry only; no physical collision or suction verification.",
                        "Configured bias is not a calibrated physical suction TCP.",
                        "Real hardware feedback, alarms, refill and clearance require runtime checks."],
    }
