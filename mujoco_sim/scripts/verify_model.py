#!/usr/bin/env python3
"""Compile the model and verify conventions, kinematics and control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from dobot_mujoco.kinematics import (
    RAW_JOINT_LOWER,
    RAW_JOINT_UPPER,
    analytic_tcp_position,
    analytic_wrist_position,
    model_to_raw,
    raw_to_model,
)
from dobot_mujoco.model import (
    control_raw_pose,
    joint_qpos_address,
    load_model,
    raw_pose_from_data,
    set_raw_pose,
    site_position,
    site_rotation,
)


def verify(samples: int, seed: int, output: Path | None = None) -> dict[str, float | int | str]:
    model = load_model()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)

    max_roundtrip_error = 0.0
    max_wrist_error = 0.0
    max_tcp_error = 0.0
    max_tool_vertical_error = 0.0

    for _ in range(samples):
        raw = rng.uniform(RAW_JOINT_LOWER, RAW_JOINT_UPPER)
        roundtrip = model_to_raw(raw_to_model(raw))
        max_roundtrip_error = max(max_roundtrip_error, float(np.max(np.abs(roundtrip - raw))))

        set_raw_pose(model, data, raw, gripper=rng.uniform(0.0, 0.0135))
        max_wrist_error = max(
            max_wrist_error,
            float(np.linalg.norm(site_position(model, data, "wrist_site") - analytic_wrist_position(raw))),
        )
        max_tcp_error = max(
            max_tcp_error,
            float(np.linalg.norm(site_position(model, data, "tcp_site") - analytic_tcp_position(raw))),
        )
        tool_z = site_rotation(model, data, "tcp_site")[:, 2]
        max_tool_vertical_error = max(
            max_tool_vertical_error,
            float(np.linalg.norm(tool_z - np.array([0.0, 0.0, 1.0]))),
        )

    # Exercise the actual position actuators and equality constraints for 3 s.
    start = np.deg2rad([0.0, 30.0, 20.0, 0.0])
    target = np.deg2rad([35.0, 50.0, 30.0, 55.0])
    set_raw_pose(model, data, start, gripper=0.0135)
    control_raw_pose(model, data, target, gripper=0.004)
    for _ in range(round(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    reached = raw_pose_from_data(model, data)
    control_error = float(np.max(np.abs(reached - target)))
    q2 = data.qpos[joint_qpos_address(model, "joint_2")]
    q3 = data.qpos[joint_qpos_address(model, "joint_3_relative")]
    mimic1 = data.qpos[joint_qpos_address(model, "joint_mimic_1")]
    mimic2 = data.qpos[joint_qpos_address(model, "joint_mimic_2")]
    max_constraint_error = float(max(abs(mimic1 + q2), abs(mimic2 + q3)))

    metrics: dict[str, float | int | str] = {
        "model": "dobot_magician_validation",
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "equality_constraints": int(model.neq),
        "random_samples": samples,
        "max_roundtrip_error_rad": max_roundtrip_error,
        "max_wrist_fk_error_m": max_wrist_error,
        "max_tcp_fk_error_m": max_tcp_error,
        "max_tool_vertical_error": max_tool_vertical_error,
        "dynamic_control_error_rad": control_error,
        "dynamic_constraint_error_rad": max_constraint_error,
    }

    tolerances = {
        "max_roundtrip_error_rad": 1e-12,
        "max_wrist_fk_error_m": 1e-9,
        "max_tcp_fk_error_m": 1e-9,
        "max_tool_vertical_error": 1e-8,
        "dynamic_control_error_rad": 0.04,
        "dynamic_constraint_error_rad": 1e-4,
    }
    failures = [name for name, tolerance in tolerances.items() if float(metrics[name]) > tolerance]
    metrics["status"] = "PASS" if not failures else "FAIL: " + ", ".join(failures)
    serialized = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(serialized)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "verification.json",
    )
    args = parser.parse_args()
    verify(args.samples, args.seed, args.output)


if __name__ == "__main__":
    main()
