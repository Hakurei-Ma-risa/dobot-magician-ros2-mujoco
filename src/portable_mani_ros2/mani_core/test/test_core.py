from time import monotonic

import numpy as np
import pytest

from mani_core.models import ExecutionResult, Pose, RobotState, TaskAxes
from mani_core.safety import SafetyEnvelope, SafetyViolation
from mani_core.trajectory import minimum_jerk


def test_minimum_jerk_endpoints_and_derivatives() -> None:
    trajectory = minimum_jerk([0.0, -1.0], [1.0, 2.0], duration_s=2.0, rate_hz=20.0)
    np.testing.assert_allclose(trajectory.position[0], [0.0, -1.0])
    np.testing.assert_allclose(trajectory.position[-1], [1.0, 2.0])
    np.testing.assert_allclose(trajectory.velocity[[0, -1]], 0.0, atol=1e-12)
    np.testing.assert_allclose(trajectory.acceleration[[0, -1]], 0.0, atol=1e-12)


def test_capability_and_workspace_are_rejected_explicitly() -> None:
    envelope = SafetyEnvelope(
        joint_lower=np.array([-1.0, -1.0]),
        joint_upper=np.array([1.0, 1.0]),
        workspace_lower=np.array([0.1, -0.2, 0.0]),
        workspace_upper=np.array([0.4, 0.2, 0.4]),
        supported_axes=TaskAxes(True, True, True, False, False, True),
    )
    with pytest.raises(SafetyViolation):
        envelope.validate_pose(
            Pose.from_xyz_yaw(0.2, 0.0, 0.1, 0.0),
            TaskAxes(True, True, True, True, False, True),
        )
    with pytest.raises(SafetyViolation):
        envelope.validate_pose(
            Pose.from_xyz_yaw(0.5, 0.0, 0.1, 0.0),
            TaskAxes(True, True, True, False, False, True),
        )


def test_stale_state_is_rejected() -> None:
    envelope = SafetyEnvelope(
        joint_lower=np.array([-1.0]),
        joint_upper=np.array([1.0]),
        workspace_lower=np.array([-1.0, -1.0, -1.0]),
        workspace_upper=np.array([1.0, 1.0, 1.0]),
        supported_axes=TaskAxes(True, True, True, False, False, True),
        max_state_age_s=0.1,
    )
    state = RobotState(
        joint_names=("joint",),
        position=np.array([0.0]),
        velocity=np.array([0.0]),
        tcp_pose=Pose.from_xyz_yaw(0.0, 0.0, 0.0, 0.0),
        end_effector_position=0.0,
        stamp_monotonic=monotonic() - 1.0,
    )
    with pytest.raises(SafetyViolation):
        envelope.validate_state_freshness(state)


def test_execution_result_normalizes_numpy_scalars() -> None:
    result = ExecutionResult(np.bool_(True), "ok", duration_s=np.float64(0.25))
    assert result.success is True
    assert isinstance(result.duration_s, float)
