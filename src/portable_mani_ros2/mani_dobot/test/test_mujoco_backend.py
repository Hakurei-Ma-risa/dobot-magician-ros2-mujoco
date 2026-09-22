import numpy as np
import pytest

pytest.importorskip("mujoco", reason="MuJoCo is installed in the dobot-mujoco Conda env")

from mani_core.models import Pose, TaskAxes
from mani_dobot.backends.mujoco_backend import (
    DobotMujocoBackend,
    cartesian_raw_waypoints,
)
from mani_dobot.kinematics import forward_tcp_pose
from mani_tasks import PickExecutor


def test_generic_pose_command_reaches_target() -> None:
    backend = DobotMujocoBackend()
    target = Pose.from_xyz_yaw(0.22, 0.04, 0.13, 0.0, frame_id="magician_base_link")
    result = backend.move_to_pose(
        "arm",
        target,
        controlled_axes=TaskAxes(True, True, True, False, False, True),
        duration_s=1.5,
    )
    assert result.success, result.message
    assert np.linalg.norm(result.final_state.tcp_pose.position - target.position) < 0.003


def test_pose_waypoints_follow_straight_tcp_line() -> None:
    seed = np.deg2rad([0.0, 30.0, 20.0, 0.0])
    start = forward_tcp_pose(seed)
    target = Pose.from_xyz_yaw(0.22, 0.04, 0.13, 0.0, frame_id=start.frame_id)
    waypoints = cartesian_raw_waypoints(
        start, target, seed_raw=seed, duration_s=1.5, rate_hz=20.0
    )
    actual = np.array([forward_tcp_pose(raw).position for raw in waypoints])
    path = target.position - start.position
    normalized = path / np.linalg.norm(path)
    cross_track = actual - start.position - (
        (actual - start.position) @ normalized
    )[:, None] * normalized
    assert np.max(np.linalg.norm(cross_track, axis=1)) < 1e-8
    np.testing.assert_allclose(actual[-1], target.position, atol=1e-8)


def test_capability_mask_rejects_roll() -> None:
    backend = DobotMujocoBackend()
    result = backend.move_to_pose(
        "arm",
        Pose.from_xyz_yaw(0.22, 0.04, 0.13, 0.0),
        controlled_axes=TaskAxes(True, True, True, True, False, True),
        duration_s=1.0,
    )
    assert not result.success
    assert "axes" in result.message


def test_gripper_semantic_commands() -> None:
    backend = DobotMujocoBackend()
    closed = backend.command_end_effector("gripper", "close")
    assert closed.success
    assert closed.final_state.end_effector_position < 1e-3
    opened = backend.command_end_effector("gripper", "open")
    assert opened.success
    assert opened.final_state.end_effector_position > 0.026


def test_physical_pick_lifts_free_object() -> None:
    backend = DobotMujocoBackend()
    target = backend.reset_scene(0.22, 0.0)
    report = PickExecutor(backend, backend.scene_object).execute(target)
    assert report.success, report.message
    assert report.final_object_z_m - report.initial_object_z_m > 0.035
