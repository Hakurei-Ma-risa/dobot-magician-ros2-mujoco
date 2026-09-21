"""Protocol implemented by every real and simulated embodiment."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.typing import ArrayLike

from .models import EmbodimentCapabilities, ExecutionResult, Pose, RobotState, TaskAxes


@runtime_checkable
class EmbodimentBackend(Protocol):
    @property
    def capabilities(self) -> EmbodimentCapabilities: ...

    def state(self) -> RobotState: ...

    def move_joints(
        self, group: str, target: ArrayLike, *, duration_s: float
    ) -> ExecutionResult: ...

    def move_to_pose(
        self,
        group: str,
        target: Pose,
        *,
        controlled_axes: TaskAxes,
        duration_s: float,
    ) -> ExecutionResult: ...

    def command_end_effector(self, name: str, command: str) -> ExecutionResult: ...

    def stop(self) -> None: ...

