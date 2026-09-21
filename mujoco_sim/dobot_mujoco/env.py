"""A small Gymnasium reach task for validating the Magician control pipeline."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .kinematics import (
    GRIPPER_MAX_OPENING_JOINT,
    RAW_JOINT_LOWER,
    RAW_JOINT_UPPER,
    analytic_tcp_position,
)
from .model import (
    control_raw_pose,
    driven_joint_velocities,
    load_model,
    raw_pose_from_data,
    set_raw_pose,
    site_position,
)


class DobotMagicianReachEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Joint-delta reaching environment using firmware-format joint angles.

    Actions are normalized deltas for theta1..theta4. Observations contain raw
    joints, the four driven joint velocities, TCP position and target position.
    This environment is deliberately compact: it validates interfaces before
    adding camera observations or an ACT/LeRobot policy.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 50  # 50 * 0.001 s = the ROS driver's 20 Hz update period.
        self.max_delta = np.deg2rad(np.array([4.0, 3.0, 3.0, 6.0]))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        observation_low = np.concatenate(
            [RAW_JOINT_LOWER, np.full(4, -50.0), np.full(3, -1.0), np.full(3, -1.0)]
        )
        observation_high = np.concatenate(
            [RAW_JOINT_UPPER, np.full(4, 50.0), np.full(3, 1.0), np.full(3, 1.0)]
        )
        self.observation_space = spaces.Box(
            low=observation_low.astype(np.float32),
            high=observation_high.astype(np.float32),
            dtype=np.float32,
        )
        self.target = np.zeros(3, dtype=np.float64)
        self.steps = 0
        self._renderer: mujoco.Renderer | None = None

    def _observation(self) -> NDArray[np.float32]:
        return np.concatenate(
            [
                raw_pose_from_data(self.model, self.data),
                driven_joint_velocities(self.model, self.data),
                site_position(self.model, self.data, "tcp_site"),
                self.target,
            ]
        ).astype(np.float32)

    def _distance(self) -> float:
        return float(np.linalg.norm(site_position(self.model, self.data, "tcp_site") - self.target))

    def _move_target_marker(self) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        mocap_id = int(self.model.body_mocapid[body_id])
        self.data.mocap_pos[mocap_id] = self.target
        mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        mujoco.mj_resetData(self.model, self.data)

        start = np.deg2rad(np.array([0.0, 30.0, 20.0, 0.0]))
        start += self.np_random.uniform(-1.0, 1.0, size=4) * np.deg2rad([8.0, 5.0, 5.0, 10.0])
        start = np.clip(start, RAW_JOINT_LOWER, RAW_JOINT_UPPER)
        set_raw_pose(self.model, self.data, start, gripper=GRIPPER_MAX_OPENING_JOINT)

        target_joints = self.np_random.uniform(RAW_JOINT_LOWER, RAW_JOINT_UPPER)
        # Keep training targets away from extreme singular/limit configurations.
        target_joints = 0.75 * target_joints + 0.25 * np.deg2rad([0.0, 30.0, 20.0, 0.0])
        self.target = analytic_tcp_position(target_joints)
        self._move_target_marker()
        self.steps = 0
        distance = self._distance()
        return self._observation(), {"distance": distance, "is_success": distance < 0.015}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        action_array = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if action_array.shape != (4,):
            raise ValueError(f"Expected action shape (4,), got {action_array.shape}")

        target_raw = raw_pose_from_data(self.model, self.data) + action_array * self.max_delta
        target_raw = np.clip(target_raw, RAW_JOINT_LOWER, RAW_JOINT_UPPER)
        control_raw_pose(self.model, self.data, target_raw, gripper=GRIPPER_MAX_OPENING_JOINT)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1
        distance = self._distance()
        success = distance < 0.015
        reward = -distance - 0.002 * float(np.square(action_array).sum()) + (1.0 if success else 0.0)
        truncated = self.steps >= self.max_episode_steps
        info = {"distance": distance, "is_success": success, "target_raw": target_raw.copy()}
        return self._observation(), reward, success, truncated, info

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=600, width=800)
        self._renderer.update_scene(self.data, camera="overview")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
