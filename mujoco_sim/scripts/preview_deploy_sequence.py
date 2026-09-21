#!/usr/bin/env python3
"""Preview a conservative hardware-coordinate demo in the MuJoCo model."""

from __future__ import annotations

import argparse
import json
import math
import time

from mani_core.models import Pose, TaskAxes
from mani_dobot.backends.mujoco_backend import DobotMujocoBackend
from dobot_mujoco.kinematics import (
    BASE_HEIGHT,
    FLANGE_TO_TOOL_TIP,
    WRIST_TO_FLANGE,
)


XYZ_YAW_AXES = TaskAxes(True, True, True, False, False, True)


def hardware_wrist_to_sim_tcp(
    radius_mm: float, angle_deg: float, z_mm: float, yaw_deg: float = 0.0
) -> Pose:
    """Map Dobot firmware wrist coordinates to the model's gripper-tip TCP."""

    angle = math.radians(angle_deg)
    tcp_radius_m = radius_mm * 0.001 + WRIST_TO_FLANGE
    tcp_z_m = BASE_HEIGHT + z_mm * 0.001 - FLANGE_TO_TOOL_TIP
    return Pose.from_xyz_yaw(
        tcp_radius_m * math.cos(angle),
        tcp_radius_m * math.sin(angle),
        tcp_z_m,
        math.radians(yaw_deg),
        frame_id="magician_base_link",
    )


SEQUENCE = (
    ("reset", 150.0, 0.0, 100.0),
    ("right", 160.0, -30.0, 120.0),
    ("center", 160.0, 0.0, 120.0),
    ("left", 160.0, 30.0, 120.0),
    ("center", 160.0, 0.0, 120.0),
    ("nod_up", 160.0, 0.0, 140.0),
    ("nod_down", 160.0, 0.0, 100.0),
    ("reset", 150.0, 0.0, 100.0),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--dwell", type=float, default=0.5)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    args = parser.parse_args()

    import mujoco.viewer

    backend = DobotMujocoBackend()
    records = []
    with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
        def sync_realtime() -> None:
            viewer.sync()
            time.sleep(1.0 / backend.CONTROL_RATE_HZ)

        backend.set_step_callback(sync_realtime)
        for name, radius_mm, angle_deg, z_mm in SEQUENCE:
            target = hardware_wrist_to_sim_tcp(radius_mm, angle_deg, z_mm)
            result = backend.move_to_pose(
                "arm",
                target,
                controlled_axes=XYZ_YAW_AXES,
                duration_s=args.duration,
            )
            state = result.final_state or backend.state()
            record = {
                "point": name,
                "hardware_wrist_target_mm_deg": [
                    round(radius_mm * math.cos(math.radians(angle_deg)), 3),
                    round(radius_mm * math.sin(math.radians(angle_deg)), 3),
                    z_mm,
                    0.0,
                ],
                "sim_tcp_m": state.tcp_pose.position.round(6).tolist(),
                "success": result.success,
                "message": result.message,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            if not result.success:
                return 1
            dwell_until = time.monotonic() + args.dwell
            while viewer.is_running() and time.monotonic() < dwell_until:
                viewer.sync()
                time.sleep(0.02)

        backend.set_step_callback(None)
        hold_until = time.monotonic() + args.hold_seconds
        while viewer.is_running() and time.monotonic() < hold_until:
            viewer.sync()
            time.sleep(0.02)

    print(json.dumps({"status": "PREVIEW_PASS", "points": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
