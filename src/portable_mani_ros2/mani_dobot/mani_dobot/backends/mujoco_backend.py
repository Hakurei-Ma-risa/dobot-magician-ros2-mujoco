"""Dobot Magician MuJoCo implementation of the generic embodiment protocol."""

from __future__ import annotations

from time import monotonic
from typing import Callable

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from dobot_mujoco.model import (
    PICK_OBJECT_HALF_HEIGHT,
    PICK_OBJECT_RADIUS,
    body_position,
    body_yaw,
    control_raw_pose,
    driven_joint_velocities,
    joint_qpos_address,
    load_model,
    raw_pose_from_data,
    reset_pick_object,
    set_raw_pose,
    site_position,
    site_rotation,
)
from mani_core.models import (
    EmbodimentCapabilities,
    EndEffectorCapability,
    ExecutionResult,
    ManipulatorCapability,
    Pose,
    RobotState,
    SceneObjectState,
    TaskAxes,
)
from mani_core.safety import SafetyEnvelope, SafetyViolation
from mani_core.trajectory import minimum_jerk

from ..kinematics import (
    GRIPPER_MAX_OPENING_JOINT,
    RAW_JOINT_LOWER,
    RAW_JOINT_UPPER,
    inverse_tcp_pose,
    model_to_raw,
    raw_to_model,
)


class DobotMujocoBackend:
    """Reference backend used to prove sim/real interface equivalence."""

    CONTROL_RATE_HZ = 20.0
    FRAME_SKIP = 50
    GROUP = "arm"
    END_EFFECTOR = "gripper"

    def __init__(self) -> None:
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self._step_callback: Callable[[], None] | None = None
        self._gripper_position = GRIPPER_MAX_OPENING_JOINT
        self._stopped = False
        self._capabilities = EmbodimentCapabilities(
            name="dobot_magician",
            manipulators=(
                ManipulatorCapability(
                    name=self.GROUP,
                    root_frame="magician_base_link",
                    tip_frame="magician_tcp",
                    joint_names=(
                        "magician_joint_1",
                        "magician_joint_2",
                        "magician_joint_3",
                        "magician_joint_4",
                    ),
                    task_axes=TaskAxes(True, True, True, False, False, True),
                    control_modes=("joint_position", "task_pose"),
                    end_effector=self.END_EFFECTOR,
                    command_rate_hz=self.CONTROL_RATE_HZ,
                ),
            ),
            end_effectors=(
                EndEffectorCapability(
                    name=self.END_EFFECTOR,
                    kind="parallel_gripper",
                    command_modes=("open", "close"),
                    max_opening_m=2.0 * GRIPPER_MAX_OPENING_JOINT,
                ),
            ),
            max_payload_kg=0.5,
        )
        self.safety = SafetyEnvelope(
            joint_lower=RAW_JOINT_LOWER,
            joint_upper=RAW_JOINT_UPPER,
            workspace_lower=np.array([-0.35, -0.35, 0.015]),
            workspace_upper=np.array([0.35, 0.35, 0.38]),
            supported_axes=TaskAxes(True, True, True, False, False, True),
            max_state_age_s=1.0,
        )
        self.reset()

    @property
    def capabilities(self) -> EmbodimentCapabilities:
        return self._capabilities

    def set_step_callback(self, callback: Callable[[], None] | None) -> None:
        """Register a hook after each 20 Hz control step (for viewers/loggers)."""

        self._step_callback = callback

    def reset(self, raw_pose: ArrayLike | None = None) -> RobotState:
        mujoco.mj_resetData(self.model, self.data)
        raw = (
            np.deg2rad([0.0, 30.0, 20.0, 0.0])
            if raw_pose is None
            else self.safety.validate_joints(raw_pose)
        )
        self._gripper_position = GRIPPER_MAX_OPENING_JOINT
        set_raw_pose(self.model, self.data, raw, gripper=self._gripper_position)
        self._stopped = False
        return self.state()

    def state(self) -> RobotState:
        raw = raw_pose_from_data(self.model, self.data)
        rotation = site_rotation(self.model, self.data, "tcp_site")
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        return RobotState(
            joint_names=self._capabilities.manipulator(self.GROUP).joint_names,
            position=raw_to_model(raw),
            velocity=driven_joint_velocities(self.model, self.data),
            tcp_pose=Pose.from_xyz_yaw(
                *site_position(self.model, self.data, "tcp_site"),
                yaw,
                frame_id="magician_base_link",
            ),
            end_effector_position=2.0
            * float(self.data.qpos[joint_qpos_address(self.model, "gripper_left")]),
            stamp_monotonic=monotonic(),
            vendor_position=raw,
        )

    def scene_object(self) -> SceneObjectState:
        position = body_position(self.model, self.data, "pick_object")
        yaw = body_yaw(self.model, self.data, "pick_object")
        return SceneObjectState(
            uuid="pick-object-0",
            class_id="cylinder",
            pose=Pose.from_xyz_yaw(
                *position,
                yaw,
                frame_id="magician_base_link",
            ),
            size=np.array(
                [2.0 * PICK_OBJECT_RADIUS, 2.0 * PICK_OBJECT_RADIUS, 2.0 * PICK_OBJECT_HALF_HEIGHT]
            ),
            confidence=1.0,
            source="mujoco_ground_truth",
        )

    def reset_scene(self, x: float, y: float, *, yaw: float = 0.0) -> SceneObjectState:
        self.reset()
        reset_pick_object(self.model, self.data, [x, y], yaw=yaw)
        # Let the object settle onto the table while holding the home pose.
        raw = raw_pose_from_data(self.model, self.data)
        control_raw_pose(
            self.model,
            self.data,
            raw,
            gripper=self._gripper_position,
        )
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        return self.scene_object()

    def _advance_raw_trajectory(self, raw_targets: np.ndarray) -> None:
        for raw_target in raw_targets:
            if self._stopped:
                raise RuntimeError("backend is stopped")
            control_raw_pose(
                self.model,
                self.data,
                self.safety.validate_joints(raw_target),
                gripper=self._gripper_position,
            )
            for _ in range(self.FRAME_SKIP):
                mujoco.mj_step(self.model, self.data)
            if self._step_callback is not None:
                self._step_callback()

    def move_joints(
        self, group: str, target: ArrayLike, *, duration_s: float
    ) -> ExecutionResult:
        if group != self.GROUP:
            return ExecutionResult(False, f"unknown manipulator group: {group}")
        started = monotonic()
        try:
            target_raw = model_to_raw(target)
            current_raw = raw_pose_from_data(self.model, self.data)
            trajectory = minimum_jerk(
                current_raw,
                target_raw,
                duration_s=duration_s,
                rate_hz=self.CONTROL_RATE_HZ,
            )
            self._advance_raw_trajectory(trajectory.position[1:])
            for _ in range(int(0.25 / self.model.opt.timestep)):
                mujoco.mj_step(self.model, self.data)
        except (ValueError, SafetyViolation, RuntimeError) as exc:
            return ExecutionResult(False, str(exc), self.state(), monotonic() - started)
        final_state = self.state()
        error = float(np.max(np.abs(final_state.vendor_position - target_raw)))
        return ExecutionResult(
            # Loaded grasps deflect the simple position actuators slightly;
            # task-space commands apply their own tighter TCP acceptance test.
            success=error < np.deg2rad(2.0),
            message=f"max raw-joint error {np.rad2deg(error):.3f} deg",
            final_state=final_state,
            duration_s=monotonic() - started,
        )

    def move_to_pose(
        self,
        group: str,
        target: Pose,
        *,
        controlled_axes: TaskAxes,
        duration_s: float,
    ) -> ExecutionResult:
        if group != self.GROUP:
            return ExecutionResult(False, f"unknown manipulator group: {group}")
        started = monotonic()
        try:
            self.safety.validate_pose(target, controlled_axes)
            current = self.state()
            target_raw = inverse_tcp_pose(target, seed_raw=current.vendor_position)
        except (ValueError, SafetyViolation) as exc:
            return ExecutionResult(False, str(exc), self.state(), monotonic() - started)
        result = self.move_joints(group, raw_to_model(target_raw), duration_s=duration_s)
        if result.final_state is None:
            return result
        position_error = float(np.linalg.norm(result.final_state.tcp_pose.position - target.position))
        yaw_error = abs(
            float(
                (result.final_state.tcp_pose.yaw - target.yaw + np.pi)
                % (2.0 * np.pi)
                - np.pi
            )
        )
        return ExecutionResult(
            success=result.success and position_error < 0.003 and yaw_error < np.deg2rad(1.0),
            message=(
                f"position error {position_error * 1000.0:.2f} mm, "
                f"yaw error {np.rad2deg(yaw_error):.2f} deg"
            ),
            final_state=result.final_state,
            duration_s=monotonic() - started,
        )

    def command_end_effector(self, name: str, command: str) -> ExecutionResult:
        if name != self.END_EFFECTOR:
            return ExecutionResult(False, f"unknown end effector: {name}", self.state())
        if command not in ("open", "close"):
            return ExecutionResult(False, f"unsupported gripper command: {command}", self.state())
        started = monotonic()
        start = self._gripper_position
        goal = GRIPPER_MAX_OPENING_JOINT if command == "open" else 0.0
        trajectory = minimum_jerk([start], [goal], duration_s=0.5, rate_hz=self.CONTROL_RATE_HZ)
        raw = raw_pose_from_data(self.model, self.data)
        for sample in trajectory.position[1:, 0]:
            self._gripper_position = float(sample)
            control_raw_pose(self.model, self.data, raw, gripper=self._gripper_position)
            for _ in range(self.FRAME_SKIP):
                mujoco.mj_step(self.model, self.data)
            if self._step_callback is not None:
                self._step_callback()
        return ExecutionResult(True, command, self.state(), monotonic() - started)

    def stop(self) -> None:
        self._stopped = True
