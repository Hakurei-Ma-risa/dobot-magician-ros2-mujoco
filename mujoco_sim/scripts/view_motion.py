#!/usr/bin/env python3
"""Open a viewer and exercise all four raw joints with safe trajectories."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from dobot_mujoco.model import control_raw_pose, load_model, set_raw_pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    model = load_model()
    data = mujoco.MjData(model)
    home = np.deg2rad([0.0, 30.0, 20.0, 0.0])
    amplitude = np.deg2rad([35.0, 12.0, 10.0, 70.0])
    phase = np.array([0.0, 0.8, 1.6, 2.4])
    set_raw_pose(model, data, home, gripper=0.0135)

    started = time.monotonic()
    simulation_time = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and time.monotonic() - started < args.seconds:
            loop_start = time.monotonic()
            target = home + amplitude * np.sin(0.55 * simulation_time + phase)
            gripper = 0.00675 * (1.0 + np.sin(0.8 * simulation_time))
            control_raw_pose(model, data, target, gripper=gripper)
            mujoco.mj_step(model, data)
            simulation_time += model.opt.timestep
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()

