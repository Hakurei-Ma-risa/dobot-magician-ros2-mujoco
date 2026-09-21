"""Stable in-process models mirrored by ``mani_interfaces`` ROS messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector(values: ArrayLike, length: int, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result.copy()


@dataclass(frozen=True)
class TaskAxes:
    """Controllable task-space axes in x, y, z, roll, pitch, yaw order."""

    x: bool = False
    y: bool = False
    z: bool = False
    roll: bool = False
    pitch: bool = False
    yaw: bool = False

    @classmethod
    def from_iterable(cls, values: Iterable[bool]) -> "TaskAxes":
        axes = tuple(bool(value) for value in values)
        if len(axes) != 6:
            raise ValueError(f"TaskAxes requires six booleans, got {len(axes)}")
        return cls(*axes)

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool, bool]:
        return self.x, self.y, self.z, self.roll, self.pitch, self.yaw

    def supports(self, required: "TaskAxes") -> bool:
        return all((not need) or have for have, need in zip(self.as_tuple(), required.as_tuple()))


@dataclass(frozen=True)
class Pose:
    """Position in metres and quaternion in ROS xyzw convention."""

    position: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )
    frame_id: str = "base_link"

    def __post_init__(self) -> None:
        position = _vector(self.position, 3, "position")
        quaternion = _vector(self.quaternion_xyzw, 4, "quaternion_xyzw")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12:
            raise ValueError("quaternion_xyzw has zero norm")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_xyzw", quaternion / norm)

    @classmethod
    def from_xyz_yaw(
        cls, x: float, y: float, z: float, yaw: float, *, frame_id: str = "base_link"
    ) -> "Pose":
        half = 0.5 * float(yaw)
        return cls(
            position=np.array([x, y, z], dtype=np.float64),
            quaternion_xyzw=np.array([0.0, 0.0, np.sin(half), np.cos(half)]),
            frame_id=frame_id,
        )

    @property
    def yaw(self) -> float:
        x, y, z, w = self.quaternion_xyzw
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


@dataclass(frozen=True)
class ManipulatorCapability:
    name: str
    root_frame: str
    tip_frame: str
    joint_names: tuple[str, ...]
    task_axes: TaskAxes
    control_modes: tuple[str, ...]
    end_effector: str | None = None
    command_rate_hz: float = 0.0


@dataclass(frozen=True)
class EndEffectorCapability:
    name: str
    kind: str
    command_modes: tuple[str, ...]
    max_opening_m: float = 0.0


@dataclass(frozen=True)
class EmbodimentCapabilities:
    name: str
    manipulators: tuple[ManipulatorCapability, ...]
    end_effectors: tuple[EndEffectorCapability, ...]
    max_payload_kg: float

    def manipulator(self, name: str) -> ManipulatorCapability:
        for manipulator in self.manipulators:
            if manipulator.name == name:
                return manipulator
        raise KeyError(f"Unknown manipulator group: {name}")


@dataclass(frozen=True)
class RobotState:
    joint_names: tuple[str, ...]
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    tcp_pose: Pose
    end_effector_position: float
    stamp_monotonic: float = field(default_factory=monotonic)
    vendor_position: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        position = _vector(self.position, len(self.joint_names), "position")
        velocity = _vector(self.velocity, len(self.joint_names), "velocity")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)
        if self.vendor_position is not None:
            object.__setattr__(
                self,
                "vendor_position",
                _vector(self.vendor_position, len(self.joint_names), "vendor_position"),
            )


@dataclass(frozen=True)
class SceneObjectState:
    """Robot-independent object estimate used by task executors."""

    uuid: str
    class_id: str
    pose: Pose
    size: NDArray[np.float64]
    confidence: float = 1.0
    source: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "uuid", str(self.uuid))
        object.__setattr__(self, "class_id", str(self.class_id))
        object.__setattr__(self, "size", _vector(self.size, 3, "size"))
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    message: str
    final_state: RobotState | None = None
    duration_s: float = 0.0
    safety_modified: bool = False

    def __post_init__(self) -> None:
        # Backends often produce NumPy scalars while checking tolerances.  Keep
        # this public boundary JSON/ROS friendly and backend independent.
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "duration_s", float(self.duration_s))
        object.__setattr__(self, "safety_modified", bool(self.safety_modified))
