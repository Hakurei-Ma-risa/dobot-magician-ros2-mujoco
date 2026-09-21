"""A deterministic top-down pick primitive shared by sim and real robots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from mani_core import EmbodimentBackend, Pose, SceneObjectState, TaskAxes


class PickStage(str, Enum):
    OPEN = "open"
    PREGRASP = "pregrasp"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    VERIFY = "verify"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PickConfig:
    group_name: str = "arm"
    end_effector_name: str = "gripper"
    approach_distance_m: float = 0.080
    lift_distance_m: float = 0.090
    tcp_table_clearance_m: float = 0.004
    required_object_lift_m: float = 0.035
    move_duration_s: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "approach_distance_m": self.approach_distance_m,
            "lift_distance_m": self.lift_distance_m,
            "tcp_table_clearance_m": self.tcp_table_clearance_m,
            "required_object_lift_m": self.required_object_lift_m,
            "move_duration_s": self.move_duration_s,
        }
        for name, value in positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PickReport:
    success: bool
    stage: PickStage
    message: str
    completed_stages: tuple[PickStage, ...]
    initial_object_z_m: float
    final_object_z_m: float


@dataclass(frozen=True)
class TopDownPickPlan:
    pregrasp_pose: Pose
    grasp_pose: Pose
    lift_pose: Pose


def plan_top_down_pick(target: SceneObjectState, config: PickConfig) -> TopDownPickPlan:
    initial_z = float(target.pose.position[2])
    table_z = initial_z - 0.5 * float(target.size[2])
    grasp_z = table_z + config.tcp_table_clearance_m
    x, y = (float(value) for value in target.pose.position[:2])
    # A cylinder is rotationally symmetric.  Radial yaw keeps the wrist near
    # neutral instead of inheriting a perception yaw with no task meaning.
    yaw = (
        float(np.arctan2(y, x))
        if target.class_id.lower() in {"cylinder", "bottle", "can"}
        else target.pose.yaw
    )
    return TopDownPickPlan(
        pregrasp_pose=Pose.from_xyz_yaw(
            x,
            y,
            grasp_z + config.approach_distance_m,
            yaw,
            frame_id=target.pose.frame_id,
        ),
        grasp_pose=Pose.from_xyz_yaw(
            x, y, grasp_z, yaw, frame_id=target.pose.frame_id
        ),
        lift_pose=Pose.from_xyz_yaw(
            x,
            y,
            grasp_z + config.lift_distance_m,
            yaw,
            frame_id=target.pose.frame_id,
        ),
    )


class PickExecutor:
    """Execute top-down pick from object state without vendor-specific APIs."""

    CONTROLLED_AXES = TaskAxes(True, True, True, False, False, True)

    def __init__(
        self,
        backend: EmbodimentBackend,
        get_object_state: Callable[[], SceneObjectState],
        *,
        config: PickConfig | None = None,
    ) -> None:
        self.backend = backend
        self.get_object_state = get_object_state
        self.config = config or PickConfig()

    def _report(
        self,
        success: bool,
        stage: PickStage,
        message: str,
        completed: list[PickStage],
        initial_z: float,
    ) -> PickReport:
        final_z = float(self.get_object_state().pose.position[2])
        return PickReport(success, stage, message, tuple(completed), initial_z, final_z)

    def execute(self, target: SceneObjectState | None = None) -> PickReport:
        target = target or self.get_object_state()
        initial_z = float(target.pose.position[2])
        completed: list[PickStage] = []
        cfg = self.config
        plan = plan_top_down_pick(target, cfg)

        result = self.backend.command_end_effector(cfg.end_effector_name, "open")
        if not result.success:
            return self._report(False, PickStage.OPEN, result.message, completed, initial_z)
        completed.append(PickStage.OPEN)

        for stage, pose in (
            (PickStage.PREGRASP, plan.pregrasp_pose),
            (PickStage.DESCEND, plan.grasp_pose),
        ):
            result = self.backend.move_to_pose(
                cfg.group_name,
                pose,
                controlled_axes=self.CONTROLLED_AXES,
                duration_s=cfg.move_duration_s,
            )
            if not result.success:
                return self._report(False, stage, result.message, completed, initial_z)
            completed.append(stage)

        result = self.backend.command_end_effector(cfg.end_effector_name, "close")
        if not result.success:
            return self._report(False, PickStage.CLOSE, result.message, completed, initial_z)
        completed.append(PickStage.CLOSE)

        result = self.backend.move_to_pose(
            cfg.group_name,
            plan.lift_pose,
            controlled_axes=self.CONTROLLED_AXES,
            duration_s=cfg.move_duration_s,
        )
        if not result.success:
            return self._report(False, PickStage.LIFT, result.message, completed, initial_z)
        completed.append(PickStage.LIFT)

        final_object = self.get_object_state()
        object_lift = float(final_object.pose.position[2]) - initial_z
        if object_lift < cfg.required_object_lift_m:
            return self._report(
                False,
                PickStage.VERIFY,
                f"object lifted {object_lift * 1000.0:.1f} mm; "
                f"required {cfg.required_object_lift_m * 1000.0:.1f} mm",
                completed,
                initial_z,
            )
        completed.extend((PickStage.VERIFY, PickStage.COMPLETE))
        return self._report(
            True,
            PickStage.COMPLETE,
            f"object lifted {object_lift * 1000.0:.1f} mm",
            completed,
            initial_z,
        )
