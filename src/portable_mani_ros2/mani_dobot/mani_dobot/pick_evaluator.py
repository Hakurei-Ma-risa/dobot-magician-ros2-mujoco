"""Randomized physical-pick evaluation for the Dobot MuJoCo backend."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from mani_tasks import PickExecutor

from .backends.mujoco_backend import DobotMujocoBackend


def _run(args: argparse.Namespace, backend: DobotMujocoBackend) -> dict:
    rng = np.random.default_rng(args.seed)
    failures: list[dict] = []
    successes = 0
    for trial in range(args.trials):
        x = float(rng.uniform(args.x_min, args.x_max))
        y = float(rng.uniform(args.y_min, args.y_max))
        yaw = float(rng.uniform(-np.pi, np.pi))
        target = backend.reset_scene(x, y, yaw=yaw)
        report = PickExecutor(backend, backend.scene_object).execute(target)
        if report.success:
            successes += 1
        else:
            failures.append(
                {
                    "trial": trial,
                    "x_m": round(x, 5),
                    "y_m": round(y, 5),
                    "stage": report.stage.value,
                    "message": report.message,
                }
            )
    success_rate = successes / args.trials
    return {
        "seed": args.seed,
        "trials": args.trials,
        "successes": successes,
        "success_rate": success_rate,
        "required_success_rate": args.min_success_rate,
        "workspace_m": {
            "x": [args.x_min, args.x_max],
            "y": [args.y_min, args.y_max],
        },
        "failures": failures,
        "passed": success_rate >= args.min_success_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--x-min", type=float, default=0.20)
    parser.add_argument("--x-max", type=float, default=0.28)
    parser.add_argument("--y-min", type=float, default=-0.12)
    parser.add_argument("--y-max", type=float, default=0.12)
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    args = parser.parse_args()

    if args.trials <= 0:
        parser.error("--trials must be positive")
    if not (0.0 <= args.min_success_rate <= 1.0):
        parser.error("--min-success-rate must be in [0, 1]")
    if args.x_min >= args.x_max or args.y_min >= args.y_max:
        parser.error("workspace minima must be smaller than maxima")

    backend = DobotMujocoBackend()
    if not args.viewer:
        summary = _run(args, backend)
        print(json.dumps(summary, indent=2))
        return 0 if summary["passed"] else 1

    import mujoco.viewer

    with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
        def sync_realtime() -> None:
            viewer.sync()
            time.sleep(1.0 / backend.CONTROL_RATE_HZ)

        backend.set_step_callback(sync_realtime)
        summary = _run(args, backend)
        backend.set_step_callback(None)
        print(json.dumps(summary, indent=2))
        finished_at = time.monotonic()
        while viewer.is_running() and (
            args.hold_seconds <= 0.0
            or time.monotonic() - finished_at < args.hold_seconds
        ):
            viewer.sync()
            time.sleep(0.02)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
