#!/usr/bin/env python3
"""Short smoke run for the Gymnasium environment."""

from __future__ import annotations

import argparse

from dobot_mujoco.env import DobotMagicianReachEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    env = DobotMagicianReachEnv()
    observation, info = env.reset(seed=11)
    total_reward = 0.0
    for step in range(args.steps):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            observation, info = env.reset(seed=11 + step)
    env.close()
    print(
        f"PASS steps={args.steps} observation_shape={observation.shape} "
        f"last_distance_m={info['distance']:.5f} total_reward={total_reward:.3f}"
    )


if __name__ == "__main__":
    main()

