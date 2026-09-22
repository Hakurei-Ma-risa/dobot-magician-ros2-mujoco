"""Dobot Magician MuJoCo implementation of the generic embodiment protocol."""

from __future__ import annotations

from dataclasses import dataclass
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
    reset_free_object,
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


@dataclass(frozen=True)
class ClutterItem:
    body: str
    joint: str
    uuid: str
    class_id: str
    size_xyz_m: tuple[float, float, float]
    anchor_xy_m: tuple[float, float]

    @property
    def half_height_m(self) -> float:
        return self.size_xyz_m[2] * 0.5

    @property
    def footprint_radius_m(self) -> float:
        return max(self.size_xyz_m[0], self.size_xyz_m[1]) * 0.5


CLUTTER_ITEMS = (
    ClutterItem(
        "pick_object", "pick_object_free", "pick-object-0", "cylinder",
        (0.024, 0.024, 0.050), (0.245, 0.015),
    ),
    ClutterItem(
        "can_red", "can_red_free", "can-red-0", "can",
        (0.032, 0.032, 0.090), (0.300, -0.105),
    ),
    ClutterItem(
        "can_blue", "can_blue_free", "can-blue-0", "can",
        (0.032, 0.032, 0.090), (0.295, 0.120),
    ),
    ClutterItem(
        "block_green", "block_green_free", "block-green-0", "block",
        (0.036, 0.036, 0.036), (0.160, -0.100),
    ),
    ClutterItem(
        "block_purple", "block_purple_free", "block-purple-0", "block",
        (0.036, 0.036, 0.036), (0.165, 0.105),
    ),
)


class DobotMujocoBackend:
    """Reference backend used to prove sim/real interface equivalence."""

    CONTROL_RATE_HZ = 20.0
    FRAME_SKIP = 50
    GROUP = "arm"
    END_EFFECTOR = "gripper"

    def __init__(self, *, scene_mode: str = "single") -> None:
        if scene_mode not in ("single", "clutter"):
            raise ValueError(f"unknown scene mode: {scene_mode}")
        self.scene_mode = scene_mode
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
        if self.scene_mode == "clutter":
            self.reset_clutter(seed=0)

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

    def scene_objects(self) -> tuple[SceneObjectState, ...]:
        if self.scene_mode == "single":
            return (self.scene_object(),)
        objects = []
        for item in CLUTTER_ITEMS:
            position = body_position(self.model, self.data, item.body)
            yaw = body_yaw(self.model, self.data, item.body)
            objects.append(
                SceneObjectState(
                    uuid=item.uuid,
                    class_id=item.class_id,
                    pose=Pose.from_xyz_yaw(
                        *position, yaw, frame_id="magician_base_link"
                    ),
                    size=np.array(item.size_xyz_m),
                    confidence=1.0,
                    source="mujoco_ground_truth",
                )
            )
        return tuple(objects)

    def reset_clutter(self, *, seed: int = 0) -> tuple[SceneObjectState, ...]:
        """Place five free bodies using a reproducible, collision-free layout."""

        self.reset()
        marker_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker"
        )
        self.data.mocap_pos[self.model.body_mocapid[marker_id]] = [2.0, 2.0, 2.0]
        rng = np.random.default_rng(seed)
        placed: list[tuple[np.ndarray, float]] = []
        for item in CLUTTER_ITEMS:
            anchor = np.asarray(item.anchor_xy_m, dtype=np.float64)
            for _ in range(100):
                xy = anchor + rng.uniform(-0.020, 0.020, size=2)
                radius = item.footprint_radius_m
                if not (0.11 + radius <= xy[0] <= 0.35 - radius):
                    continue
                if not (-0.18 + radius <= xy[1] <= 0.18 - radius):
                    continue
                if all(
                    np.linalg.norm(xy - other_xy) >= radius + other_radius + 0.015
                    for other_xy, other_radius in placed
                ):
                    break
            else:
                raise RuntimeError(f"failed to place clutter item {item.body}")
            reset_free_object(
                self.model,
                self.data,
                item.joint,
                xy,
                half_height=item.half_height_m,
                yaw=float(rng.uniform(-np.pi, np.pi)),
            )
            placed.append((xy, radius))

        raw = raw_pose_from_data(self.model, self.data)
        control_raw_pose(
            self.model, self.data, raw, gripper=self._gripper_position
        )
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        return self.scene_objects()

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
