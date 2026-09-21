import numpy as np
import pytest

from mani_core.models import Pose
from mani_dobot.kinematics import (
    RAW_JOINT_LOWER,
    RAW_JOINT_UPPER,
    forward_tcp_pose,
    inverse_tcp_pose,
    model_to_raw,
    raw_to_model,
)


def test_raw_model_round_trip() -> None:
    raw = np.deg2rad([35.0, 40.0, 12.0, -20.0])
    np.testing.assert_allclose(model_to_raw(raw_to_model(raw)), raw, atol=1e-12)


def test_inverse_round_trip_random_poses() -> None:
    rng = np.random.default_rng(20260918)
    for _ in range(200):
        raw = rng.uniform(RAW_JOINT_LOWER, RAW_JOINT_UPPER)
        pose = forward_tcp_pose(raw)
        solved = inverse_tcp_pose(pose, seed_raw=raw)
        solved_pose = forward_tcp_pose(solved)
        np.testing.assert_allclose(solved_pose.position, pose.position, atol=1e-10)
        assert abs((solved_pose.yaw - pose.yaw + np.pi) % (2.0 * np.pi) - np.pi) < 1e-10


def test_unreachable_pose_is_rejected() -> None:
    with pytest.raises(ValueError):
        inverse_tcp_pose(Pose.from_xyz_yaw(0.8, 0.0, 0.2, 0.0))

