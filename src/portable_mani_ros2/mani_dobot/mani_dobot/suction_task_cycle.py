"""Taught suction pick/place tasks; dry-run by default, with operator gates.

This is a fixed-waypoint controller, not camera localization or force control.
No ROS or serial interface is imported until --check or --execute is selected.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from .suction_task_plan import (
    TaskPlan, fixed_yaw_plan, interpolate_segments, load_plan, plan_summary,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskJournal:
    """Persist progress without allowing automatic resume of a held object."""

    def __init__(self, plan: TaskPlan, path: Path | None = None) -> None:
        self.path = path
        if path is not None and path.exists():
            raise ValueError("Run log already exists; use a new path, not automatic resume")
        self.data = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "mode": plan.mode,
            "routes": [list(route) for route in plan.routes],
            "config_sha256": plan.config_sha256,
            "robot_port": plan.robot_port,
            "safe_z_mm": plan.safe_z_mm,
            "yaw_policy": plan.yaw_policy,
            "yaw_reference_deg": plan.yaw_reference_deg,
            "status": "waiting_for_operator",
            "stage": "initial",
            "holding_assumption": "empty",
            "completed_destinations": [],
            "automatic_resume_allowed": False,
            "events": [],
        }

    def event(self, stage: str, **fields) -> None:
        self.data.update(fields)
        self.data["stage"] = stage
        event = {"at_utc": utc_now(), "stage": stage, **fields}
        self.data["events"].append(event)
        print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(self.data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def _yaw_difference(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _live_pose(state: dict) -> tuple[float, float, float, float]:
    pose = tuple(float(value) for value in state["pose"])
    if len(pose) != 4 or not all(math.isfinite(value) for value in pose):
        raise ValueError("Live pose must contain finite XYZ mm and yaw degrees")
    if state.get("alarms") != []:
        raise ValueError("Robot alarm state is missing or active")
    return pose


def _at_pose(actual, expected, *, distance_mm: float = 0.6, yaw_deg: float = 1.0) -> bool:
    return (
        math.dist(actual[:3], expected[:3]) <= distance_mm
        and abs(_yaw_difference(actual[3], expected[3])) <= yaw_deg
    )


def _above(pose, safe_z_mm: float) -> tuple[float, float, float, float]:
    return (pose[0], pose[1], safe_z_mm, pose[3])


def _lift(pose, distance_mm: float) -> tuple[float, float, float, float]:
    return (pose[0], pose[1], pose[2] + distance_mm, pose[3])


def _known_start(plan: TaskPlan, pose) -> bool:
    for waypoint in plan.waypoints.values():
        if _at_pose(pose, waypoint.pose, distance_mm=1.0):
            return True
        if _at_pose(pose, _above(waypoint.pose, plan.safe_z_mm), distance_mm=1.0):
            return True
        if _at_pose(pose, _lift(waypoint.pose, 10.0), distance_mm=1.0):
            return True
    return False


def _move(
    hardware, destination, journal: TaskJournal, stage: str,
    *, continuous: bool = False, speed_mm_s: float | None = None,
) -> None:
    start = _live_pose(hardware.state())
    if stage in ("above_source", "transfer_at_safe_height"):
        if abs(start[2] - destination[2]) > 0.6:
            raise RuntimeError("Horizontal transfer must start and end on the same high plane")
    elif (math.dist(start[:2], destination[:2]) > 0.6
          or abs(_yaw_difference(start[3], destination[3])) > 1.0):
        raise RuntimeError("Lift/descent/retract must be vertical with unchanged orientation")
    # Materialize and validate the whole short path before sending any segment.
    # Leave headroom for the hardware's 0.25 mm / 0.3 degree settling
    # tolerance; a perfect nominal 5 mm step could be too long from reality.
    segments = interpolate_segments(
        start, tuple(destination), max_step_mm=4.5, max_yaw_step_deg=1.5
    )
    use_continuous = (
        continuous and stage in ("above_source", "transfer_at_safe_height")
        and math.dist(start[:2], destination[:2]) > 0.25
    )
    if speed_mm_s is not None:
        hardware.set_speed(speed_mm_s)
    journal.event(
        stage, goal_controller_mm_deg=list(destination),
        segments=1 if use_continuous else len(segments),
        validation_segment_count=len(segments),
        motion_profile="continuous" if use_continuous else "segmented",
        requested_speed_mm_s=speed_mm_s,
    )
    if use_continuous:
        hardware.move_transfer(tuple(destination))
    else:
        for segment in segments:
            hardware.move_segment(segment)
    final = _live_pose(hardware.state())
    if not _at_pose(final, destination):
        raise RuntimeError(f"Did not reach {stage}; no later action is permitted")


def run_task(
    plan: TaskPlan, hardware, journal: TaskJournal | None = None,
    *, continuous_transfers: bool = False,
    vertical_speed_mm_s: float | None = None,
    transfer_speed_mm_s: float | None = None,
    confirm_each_action: bool = True,
) -> dict:
    """Run one finite target list, never automatically retry or skip a gate.

    The hardware object is injectable so the entire sequence can be tested
    without ROS, a camera or a serial port. Hardware exceptions propagate after
    a motion stop; suction is deliberately not disabled on failure.
    """
    journal = journal or TaskJournal(plan)
    armed = False
    try:
        if continuous_transfers and plan.yaw_policy != "hold":
            raise ValueError("Continuous transfers require the fixed grasp-yaw policy")
        for index, (source_name, target_name) in enumerate(plan.routes):
            source = plan.waypoints[source_name].pose
            target = plan.waypoints[target_name].pose
            source_label = source_name.removeprefix("pick_").removeprefix("place_")
            target_label = target_name.removeprefix("pick_").removeprefix("place_")
            journal.event(
                "wait_for_source",
                status="waiting_for_operator",
                target_index=index,
                source=source_name,
                target=target_name,
                continuous_transfers=continuous_transfers,
            )
            hardware.confirm(
                f"Target {index + 1}/{len(plan.routes)}: {source_label} -> {target_label}. "
                f"Confirm a block is centred at taught {source_label}, target {target_label} "
                "is empty, sheet/base/tool are unchanged, the full raised route is clear, "
                "and hands/cables have left the workspace. Do not reach in while running.",
                f"READY {source_label}",
            )
            state = hardware.state()
            current = _live_pose(state)
            if state.get("suction_on") is not False:
                raise RuntimeError("Suction must be confirmed off before starting a pick")
            if not _known_start(plan, current):
                raise RuntimeError("Initial pose is not a taught contact/high point; inspect first")
            if not armed:
                armed = True
                hardware.arm()
                # Arm must not reposition or change the taught coordinate reference.
                if not _at_pose(_live_pose(hardware.state()), current, distance_mm=0.5):
                    raise RuntimeError("Robot moved while arming")
            journal.event("running", status="running")

            # A cup already at the confirmed source contact can start the
            # suction test directly. Otherwise escape vertically before XY.
            if not _at_pose(current, source, distance_mm=0.25, yaw_deg=0.3):
                _move(hardware, _above(current, plan.safe_z_mm), journal, "raise_empty_at_current",
                      speed_mm_s=vertical_speed_mm_s)
                _move(hardware, _above(source, plan.safe_z_mm), journal, "above_source",
                      continuous=continuous_transfers, speed_mm_s=transfer_speed_mm_s)
            _move(hardware, source, journal, "descend_to_source", speed_mm_s=vertical_speed_mm_s)
            hardware.suction(True)
            journal.event("suction_on", holding_assumption="unverified_suction")
            grasp_pose = _live_pose(hardware.state())
            if not _at_pose(grasp_pose, source, distance_mm=0.4, yaw_deg=0.5):
                raise RuntimeError("Robot is not at the contact pose after suction activation")
            if plan.yaw_policy == "hold":
                # Capture the actual readback, not the taught destination yaw.
                locked_r = grasp_pose[3]
                source = (source[0], source[1], source[2], locked_r)
                target = (target[0], target[1], target[2], locked_r)
                journal.event("grasp_yaw_locked", locked_grasp_yaw_deg=locked_r)
            _move(hardware, _lift(source, 10.0), journal, "trial_lift_10mm",
                  speed_mm_s=vertical_speed_mm_s)
            if confirm_each_action:
                hardware.confirm(
                    "Observe from outside the workspace: the block has left the table "
                    "and remains attached without slipping. If not, cancel; do not confirm.",
                    "HELD",
                )
            held_state = hardware.state()
            if not _at_pose(_live_pose(held_state), _lift(source, 10.0)):
                raise RuntimeError("Robot moved while awaiting grasp confirmation")
            if held_state.get("suction_on") is not True:
                raise RuntimeError("Suction controller state changed after trial lift")
            journal.event(
                "grasp_confirmed" if confirm_each_action else "grasp_output_confirmed",
                holding_assumption="operator_confirmed_held" if confirm_each_action else "unverified_held",
                physical_grasp_verified=confirm_each_action,
            )
            _move(hardware, _above(source, plan.safe_z_mm), journal, "lift_loaded",
                  speed_mm_s=vertical_speed_mm_s)
            _move(hardware, _above(target, plan.safe_z_mm), journal, "transfer_at_safe_height",
                  continuous=continuous_transfers, speed_mm_s=transfer_speed_mm_s)
            _move(hardware, target, journal, "descend_to_target", speed_mm_s=vertical_speed_mm_s)
            hardware.suction(False)
            journal.event("suction_off", holding_assumption="release_unconfirmed")

            # Retract locally before asking whether release physically succeeded.
            # No XY transfer is allowed on an unconfirmed release.
            _move(hardware, _lift(target, 10.0), journal, "release_check_retract_10mm",
                  speed_mm_s=vertical_speed_mm_s)
            if confirm_each_action:
                hardware.confirm(
                    f"Observe that the block stays at {target_label} on the table and "
                    "the cup has separated. Do not reach into the workspace.",
                    "RELEASED",
                )
            released_state = hardware.state()
            if not _at_pose(_live_pose(released_state), _lift(target, 10.0)):
                raise RuntimeError("Robot moved while awaiting release confirmation")
            if released_state.get("suction_on") is not False:
                raise RuntimeError("Suction is not confirmed off after release")
            journal.event(
                "release_confirmed" if confirm_each_action else "release_output_confirmed",
                holding_assumption="empty" if confirm_each_action else "output_off_release_unverified",
                physical_release_verified=confirm_each_action,
            )
            _move(hardware, _above(target, plan.safe_z_mm), journal, "retract_empty",
                  speed_mm_s=vertical_speed_mm_s)
            completed = list(journal.data["completed_destinations"]) + [target_name]
            journal.event("target_complete", completed_destinations=completed)

        hardware.stop()
        journal.event("complete", status="complete", holding_assumption=(
            "empty" if confirm_each_action else "output_off_release_unverified"
        ))
        return journal.data
    except BaseException as exc:
        if armed:
            try:
                hardware.stop()
            except Exception as stop_error:
                print(f"Motion stop could not be verified: {stop_error}", file=sys.stderr)
        journal.event("stopped", status="stopped", error=f"{type(exc).__name__}: {exc}")
        raise


def finish_held_route(
    plan: TaskPlan, hardware, journal: TaskJournal | None = None, *, target_name: str,
    vertical_speed_mm_s: float = 30.0, transfer_speed_mm_s: float = 50.0,
    continuous_transfers: bool = True, confirm_each_action: bool = True,
) -> dict:
    """Explicit continuation of an inspected, stationary A+10 mm held block.

    This is never inferred from a prior run log and cannot start a new pick.
    The onsite operator must deliberately authorize the held continuation.
    """
    journal = journal or TaskJournal(plan)
    armed = False
    try:
        if plan.mode != "refill" or plan.yaw_policy != "hold":
            raise ValueError("Held continuation requires refill mode and fixed grasp yaw")
        if len(plan.routes) != 1 or plan.routes[0] != ("pick_A", target_name):
            raise ValueError("Held continuation requires exactly one selected A destination")
        state = hardware.state()
        current = _live_pose(state)
        source = plan.waypoints["pick_A"].pose
        if not _at_pose(current, _lift(source, 10.0), distance_mm=0.4, yaw_deg=0.5):
            raise RuntimeError("Held continuation must start at taught A plus 10 mm")
        if state.get("suction_on") is not True:
            raise RuntimeError("Held continuation requires suction already on")
        hardware.confirm(
            "Explicit held continuation: the block remains attached at A+10 mm, "
            "the selected destination and raised path are clear, and hands/cables are out. "
            "This does not restart a saved motion queue.",
            "CONTINUE HELD",
        )
        confirmed_state = hardware.state()
        if (not _at_pose(_live_pose(confirmed_state), current, distance_mm=0.25, yaw_deg=0.3)
                or confirmed_state.get("suction_on") is not True):
            raise RuntimeError("Held pose or suction changed during continuation confirmation")
        armed = True
        hardware.arm(holding_confirmed=True)
        armed_state = hardware.state()
        if (not _at_pose(_live_pose(armed_state), current, distance_mm=0.25, yaw_deg=0.3)
                or armed_state.get("suction_on") is not True):
            raise RuntimeError("Held pose or suction changed while arming continuation")
        locked_r = current[3]
        source = (*source[:3], locked_r)
        taught_target = plan.waypoints[target_name].pose
        target = (*taught_target[:3], locked_r)
        journal.event(
            "held_continuation", status="running", holding_assumption="operator_confirmed_held",
            source="pick_A", target=target_name, locked_grasp_yaw_deg=locked_r,
            automatic_resume_allowed=False,
        )
        _move(hardware, _above(source, plan.safe_z_mm), journal, "lift_loaded",
              speed_mm_s=vertical_speed_mm_s)
        _move(hardware, _above(target, plan.safe_z_mm), journal, "transfer_at_safe_height",
              continuous=continuous_transfers, speed_mm_s=transfer_speed_mm_s)
        _move(hardware, target, journal, "descend_to_target", speed_mm_s=vertical_speed_mm_s)
        hardware.suction(False)
        journal.event("suction_off", holding_assumption="release_unconfirmed")
        _move(hardware, _lift(target, 10.0), journal, "release_check_retract_10mm",
              speed_mm_s=vertical_speed_mm_s)
        if confirm_each_action:
            hardware.confirm("Observe the block remains at the destination and the cup is separate.", "RELEASED")
        released_state = hardware.state()
        if (not _at_pose(_live_pose(released_state), _lift(target, 10.0))
                or released_state.get("suction_on") is not False):
            raise RuntimeError("Released pose or suction does not match the commanded state")
        release_assumption = "empty" if confirm_each_action else "output_off_release_unverified"
        journal.event(
            "release_confirmed" if confirm_each_action else "release_output_confirmed",
            holding_assumption=release_assumption, physical_release_verified=confirm_each_action,
        )
        _move(hardware, _above(target, plan.safe_z_mm), journal, "retract_empty",
              speed_mm_s=vertical_speed_mm_s)
        hardware.stop()
        journal.event("complete", status="complete", holding_assumption=release_assumption,
                      completed_destinations=[target_name])
        return journal.data
    except BaseException as exc:
        if armed:
            try:
                hardware.stop()
            except Exception as stop_error:
                print(f"Motion stop could not be verified: {stop_error}", file=sys.stderr)
        journal.event("stopped", status="stopped", error=f"{type(exc).__name__}: {exc}")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Taught Dobot suction target mode. Default: offline plan only, no hardware access."
    )
    parser.add_argument("--waypoints", type=Path, required=True, help="This robot's taught task_waypoints.json")
    parser.add_argument("--mode", choices=("refill", "chain"), default="refill",
                        help="refill: A->B,A->C,A->D; chain: A->B,B->C,C->D")
    parser.add_argument("--targets", nargs="+", choices=("B", "C", "D"), default=["B", "C", "D"],
                        help="Finite destination order, e.g. --targets B C")
    parser.add_argument("--yaw-policy", choices=("hold", "taught"), default="hold",
                        help="hold: keep the actual grasp R while loaded; taught explicitly replays taught endpoint yaw")
    parser.add_argument("--transfer-mode", choices=("continuous", "segmented"), default="continuous",
                        help="continuous: one monitored high-plane MOVL; segmented: stop at short checkpoints")
    parser.add_argument("--safe-z-mm", type=float, default=-10.0,
                        help="Common controller Z for horizontal transfer, not tabletop height")
    parser.add_argument("--speed-mm-s", type=float, default=5.0)
    parser.add_argument("--vertical-speed-mm-s", type=float, default=30.0,
                        help="Lift/descent/retraction coordinate speed, separate from horizontal transfer")
    parser.add_argument("--acceleration-mm-s2", type=float, default=10.0)
    parser.add_argument("--confirmation-policy", choices=("each-action", "start-only"), default="each-action",
                        help="start-only omits grasp/release inspection gates; controller outputs do not prove a grasp")
    parser.add_argument("--continue-held", action="store_true",
                        help="Explicitly finish one held block at A+10 mm; never an automatic log resume")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Read-only live hardware check; no arming or movement")
    action.add_argument("--execute", action="store_true", help="Interactive real execution; every pick/grasp/release needs confirmation")
    parser.add_argument("--log", type=Path, help="New execution log path; existing logs cannot be resumed or overwritten")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.speed_mm_s) or not 5.0 <= args.speed_mm_s <= 50.0:
        raise ValueError("Speed must be finite and between 5 and 50 mm/s")
    if not math.isfinite(args.vertical_speed_mm_s) or not 5.0 <= args.vertical_speed_mm_s <= 50.0:
        raise ValueError("Vertical speed must be finite and between 5 and 50 mm/s")
    if not math.isfinite(args.acceleration_mm_s2) or not 5.0 <= args.acceleration_mm_s2 <= 30.0:
        raise ValueError("Acceleration must be finite and between 5 and 30 mm/s^2")
    plan = load_plan(args.waypoints, mode=args.mode, safe_z_mm=args.safe_z_mm)
    if len(set(args.targets)) != len(args.targets):
        raise ValueError("Destination list must not contain duplicates")
    destinations = tuple(f"place_{name}" for name in args.targets)
    sources = ("pick_A",) * len(destinations)
    if args.mode == "chain":
        sources = ("pick_A",) + destinations[:-1]
    plan = replace(plan, order=destinations, routes=tuple(zip(sources, destinations)))
    if args.yaw_policy == "hold":
        plan = fixed_yaw_plan(plan)
    if args.transfer_mode == "continuous" and args.yaw_policy != "hold":
        raise ValueError("Continuous transfer mode requires --yaw-policy hold")
    if args.continue_held and (
        not args.execute or len(plan.routes) != 1 or plan.mode != "refill" or plan.yaw_policy != "hold"
    ):
        raise ValueError("Held continuation requires --execute, hold/refill and exactly one target")
    print(json.dumps({"action": "offline_plan", **plan_summary(plan)}, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if not args.check and not args.execute:
        return 0
    if args.execute and not sys.stdin.isatty():
        raise ValueError("Real execution requires an interactive terminal; piped confirmations are not accepted")

    # Importing and constructing the hardware is deliberately after dry-run exit.
    from .suction_task_hardware import RosDobotHardware
    hardware = RosDobotHardware(
        plan.robot_port, plan.expected_bias_mm,
        velocity_mm_s=args.speed_mm_s,
        acceleration_mm_s2=args.acceleration_mm_s2,
    )
    try:
        if args.check:
            state = hardware.state()
            _live_pose(state)
            print(json.dumps({"mode": "read_only_hardware_check", "state": state, "motion_started": False}, ensure_ascii=False, allow_nan=False), flush=True)
            return 0
        log_path = args.log
        if log_path is None:
            name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
            log_path = args.waypoints.resolve().parent / "task_runs" / name
        journal = TaskJournal(plan, log_path)
        common = dict(
            continuous_transfers=args.transfer_mode == "continuous",
            vertical_speed_mm_s=args.vertical_speed_mm_s,
            transfer_speed_mm_s=args.speed_mm_s,
            confirm_each_action=args.confirmation_policy == "each-action",
        )
        if args.continue_held:
            finish_held_route(plan, hardware, journal, target_name=destinations[0], **common)
        else:
            run_task(plan, hardware, journal, **common)
        print(f"Task completed. Log: {log_path.resolve()}", flush=True)
        return 0
    except BaseException:
        if args.execute:
            print("Execution stopped. Suction is not automatically disabled; inspect any held block before intervening.", file=sys.stderr, flush=True)
        raise
    finally:
        hardware.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Task stopped: {type(exc).__name__}: {exc}. Suction has NOT been automatically disabled.", file=sys.stderr)
        raise SystemExit(1)
