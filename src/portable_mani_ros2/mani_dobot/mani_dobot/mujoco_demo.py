"""Run the generic task-space API against the Dobot MuJoCo backend."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from mani_core.models import Pose, TaskAxes

from .backends.mujoco_backend import DobotMujocoBackend


def _execute(backend: DobotMujocoBackend, args: argparse.Namespace):
    target = Pose.from_xyz_yaw(
        args.x,
        args.y,
        args.z,
        np.deg2rad(args.yaw_deg),
        frame_id="magician_base_link",
    )
    return backend.move_to_pose(
        "arm",
        target,
        controlled_axes=TaskAxes(True, True, True, False, False, True),
        duration_s=args.duration,
    )


def _print_result(backend: DobotMujocoBackend, result) -> None:
    state = result.final_state or backend.state()
    print(
        json.dumps(
            {
                "success": result.success,
                "message": result.message,
                "tcp_position_m": state.tcp_pose.position.round(6).tolist(),
                "tcp_yaw_deg": round(float(np.rad2deg(state.tcp_pose.yaw)), 4),
                "vendor_joints_deg": np.rad2deg(state.vendor_position).round(4).tolist(),
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=0.22)
    parser.add_argument("--y", type=float, default=0.04)
    parser.add_argument("--z", type=float, default=0.13)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the interactive MuJoCo viewer and play the trajectory in real time",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="close the viewer after N seconds; 0 keeps it open until you close it",
    )
    args = parser.parse_args()

    backend = DobotMujocoBackend()
    if not args.viewer:
        result = _execute(backend, args)
        _print_result(backend, result)
        return 0 if result.success else 1

    import mujoco.viewer

    with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
        viewer.sync()

        def sync_realtime() -> None:
            viewer.sync()
            time.sleep(1.0 / backend.CONTROL_RATE_HZ)

        backend.set_step_callback(sync_realtime)
        result = _execute(backend, args)
        backend.set_step_callback(None)
        viewer.sync()
        _print_result(backend, result)

        finished_at = time.monotonic()
        while viewer.is_running() and (
            args.hold_seconds <= 0.0
            or time.monotonic() - finished_at < args.hold_seconds
        ):
            viewer.sync()
            time.sleep(0.02)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
