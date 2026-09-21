#!/usr/bin/env python3
"""Render a deterministic MuJoCo scene to artifacts/magician_mujoco.png."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from dobot_mujoco.kinematics import analytic_tcp_position
from dobot_mujoco.model import load_model, set_raw_pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "magician_mujoco.png",
    )
    args = parser.parse_args()

    model = load_model()
    data = mujoco.MjData(model)
    pose = np.deg2rad([28.0, 43.0, 18.0, -35.0])
    set_raw_pose(model, data, pose, gripper=0.010)

    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
    target_mocap = int(model.body_mocapid[target_body])
    target_pose = np.deg2rad([-18.0, 38.0, 25.0, 0.0])
    data.mocap_pos[target_mocap] = analytic_tcp_position(target_pose)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=960)
    renderer.update_scene(data, camera="overview")
    pixels = renderer.render()
    renderer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

