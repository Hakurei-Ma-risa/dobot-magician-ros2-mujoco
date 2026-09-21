import numpy as np

from mani_core import ExecutionResult, Pose, RobotState, SceneObjectState, TaskAxes
from mani_tasks import PickExecutor, PickStage


class FakeBackend:
    def __init__(self) -> None:
        self.object_z = 0.075
        self.closed = False
        self.moves: list[Pose] = []

    def move_to_pose(self, group, target, *, controlled_axes, duration_s):
        assert group == "arm"
        assert controlled_axes == TaskAxes(True, True, True, False, False, True)
        self.moves.append(target)
        if self.closed and len(self.moves) == 3:
            self.object_z += 0.07
        return ExecutionResult(True, "ok")

    def command_end_effector(self, name, command):
        assert name == "gripper"
        self.closed = command == "close"
        return ExecutionResult(True, command)


def test_pick_executor_uses_task_space_and_verifies_lift() -> None:
    backend = FakeBackend()

    def get_object() -> SceneObjectState:
        return SceneObjectState(
            uuid="object",
            class_id="cylinder",
            pose=Pose.from_xyz_yaw(0.22, 0.01, backend.object_z, 0.0),
            size=np.array([0.024, 0.024, 0.05]),
        )

    report = PickExecutor(backend, get_object).execute()
    assert report.success
    assert report.stage is PickStage.COMPLETE
    assert len(backend.moves) == 3
    assert np.isclose(backend.moves[1].position[2], 0.054)
