"""Deterministic, policy-independent command validation."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import Pose, RobotState, TaskAxes


class SafetyViolation(ValueError):
    """Raised when a command must be rejected rather than silently clipped."""


@dataclass(frozen=True)
class SafetyEnvelope:
    joint_lower: NDArray[np.float64]
    joint_upper: NDArray[np.float64]
    workspace_lower: NDArray[np.float64]
    workspace_upper: NDArray[np.float64]
    supported_axes: TaskAxes
    max_state_age_s: float = 0.25

    def __post_init__(self) -> None:
        joint_lower = np.asarray(self.joint_lower, dtype=np.float64)
        joint_upper = np.asarray(self.joint_upper, dtype=np.float64)
        workspace_lower = np.asarray(self.workspace_lower, dtype=np.float64)
        workspace_upper = np.asarray(self.workspace_upper, dtype=np.float64)
        if joint_lower.shape != joint_upper.shape or joint_lower.ndim != 1:
            raise ValueError("joint limits must be same-sized vectors")
        if workspace_lower.shape != (3,) or workspace_upper.shape != (3,):
            raise ValueError("workspace limits must have shape (3,)")
        if np.any(joint_lower >= joint_upper):
            raise ValueError("each joint lower limit must be smaller than upper limit")
        if np.any(workspace_lower >= workspace_upper):
            raise ValueError("each workspace lower limit must be smaller than upper limit")
        object.__setattr__(self, "joint_lower", joint_lower.copy())
        object.__setattr__(self, "joint_upper", joint_upper.copy())
        object.__setattr__(self, "workspace_lower", workspace_lower.copy())
        object.__setattr__(self, "workspace_upper", workspace_upper.copy())

    def validate_state_freshness(self, state: RobotState) -> None:
        age = monotonic() - state.stamp_monotonic
        if age > self.max_state_age_s:
            raise SafetyViolation(
                f"robot state is stale ({age:.3f}s > {self.max_state_age_s:.3f}s)"
            )

    def validate_joints(self, values: ArrayLike) -> NDArray[np.float64]:
        joints = np.asarray(values, dtype=np.float64)
        if joints.shape != self.joint_lower.shape:
            raise SafetyViolation(
                f"joint command shape {joints.shape} != expected {self.joint_lower.shape}"
            )
        if not np.all(np.isfinite(joints)):
            raise SafetyViolation("joint command contains NaN or infinity")
        if np.any(joints < self.joint_lower) or np.any(joints > self.joint_upper):
            raise SafetyViolation(f"joint command outside limits: {joints.tolist()}")
        return joints.copy()

    def validate_pose(self, pose: Pose, required_axes: TaskAxes) -> Pose:
        if not self.supported_axes.supports(required_axes):
            raise SafetyViolation(
                f"requested axes {required_axes.as_tuple()} exceed embodiment capability "
                f"{self.supported_axes.as_tuple()}"
            )
        if np.any(pose.position < self.workspace_lower) or np.any(
            pose.position > self.workspace_upper
        ):
            raise SafetyViolation(
                f"target {pose.position.tolist()} outside workspace "
                f"[{self.workspace_lower.tolist()}, {self.workspace_upper.tolist()}]"
            )
        return pose

