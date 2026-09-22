"""Loading and state/control helpers for the Magician MJCF model."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kinematics import GRIPPER_MAX_OPENING_JOINT, model_to_raw, raw_to_model

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "dobot_magician.xml"
TABLE_TOP_Z = 0.05
PICK_OBJECT_HALF_HEIGHT = 0.025
PICK_OBJECT_RADIUS = 0.012

DRIVEN_JOINTS = ("joint_1", "joint_2", "joint_3_relative", "joint_4")
MIMIC_JOINTS = ("joint_mimic_1", "joint_mimic_2")


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def joint_qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"Unknown MuJoCo joint: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def joint_dof_address(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"Unknown MuJoCo joint: {joint_name}")
    return int(model.jnt_dofadr[joint_id])


def site_position(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> NDArray[np.float64]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise KeyError(f"Unknown MuJoCo site: {site_name}")
    return data.site_xpos[site_id].copy()


def site_rotation(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> NDArray[np.float64]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise KeyError(f"Unknown MuJoCo site: {site_name}")
    return data.site_xmat[site_id].reshape(3, 3).copy()


def body_position(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str
) -> NDArray[np.float64]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"Unknown MuJoCo body: {body_name}")
    return data.xpos[body_id].copy()


def body_yaw(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> float:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"Unknown MuJoCo body: {body_name}")
    rotation = data.xmat[body_id].reshape(3, 3)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    data.qpos[joint_qpos_address(model, name)] = value


def control_raw_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    raw: ArrayLike,
    *,
    gripper: float = GRIPPER_MAX_OPENING_JOINT,
) -> None:
    """Set position-actuator targets from firmware-format radians."""

    model_joints = raw_to_model(raw)
    gripper_value = float(np.clip(gripper, 0.0, GRIPPER_MAX_OPENING_JOINT))
    targets = {
        "act_joint_1": model_joints[0],
        "act_joint_2": model_joints[1],
        "act_joint_3_relative": model_joints[2],
        "act_joint_4": model_joints[3],
        "act_gripper": gripper_value,
    }
    for actuator_name, target in targets.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise KeyError(f"Unknown MuJoCo actuator: {actuator_name}")
        data.ctrl[actuator_id] = target


def set_raw_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    raw: ArrayLike,
    *,
    gripper: float = GRIPPER_MAX_OPENING_JOINT,
    set_control: bool = True,
) -> None:
    """Set an exact, constraint-consistent model state from raw radians."""

    q1, q2, q3_relative, q4 = raw_to_model(raw)
    gripper_value = float(np.clip(gripper, 0.0, GRIPPER_MAX_OPENING_JOINT))

    values = {
        "joint_1": q1,
        "joint_2": q2,
        "joint_3_relative": q3_relative,
        "joint_mimic_1": -q2,
        "joint_mimic_2": -q3_relative,
        "joint_4": q4,
        "gripper_left": gripper_value,
        "gripper_right": -gripper_value,
    }
    for name, value in values.items():
        _set_joint_qpos(model, data, name, value)
    data.qvel[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    if set_control:
        control_raw_pose(model, data, raw, gripper=gripper_value)
    mujoco.mj_forward(model, data)


def raw_pose_from_data(model: mujoco.MjModel, data: mujoco.MjData) -> NDArray[np.float64]:
    model_joints = np.array(
        [data.qpos[joint_qpos_address(model, name)] for name in DRIVEN_JOINTS],
        dtype=np.float64,
    )
    return model_to_raw(model_joints, validate=False)


def driven_joint_velocities(model: mujoco.MjModel, data: mujoco.MjData) -> NDArray[np.float64]:
    return np.array(
        [data.qvel[joint_dof_address(model, name)] for name in DRIVEN_JOINTS],
        dtype=np.float64,
    )


def reset_pick_object(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    position_xy: ArrayLike,
    *,
    yaw: float = 0.0,
) -> None:
    """Reset the free pick object above the tabletop with zero velocity."""

    reset_free_object(
        model,
        data,
        "pick_object_free",
        position_xy,
        half_height=PICK_OBJECT_HALF_HEIGHT,
        yaw=yaw,
    )


def reset_free_object(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    position_xy: ArrayLike,
    *,
    half_height: float,
    yaw: float = 0.0,
) -> None:
    """Set a tabletop free body pose and clear its velocity."""

    xy = np.asarray(position_xy, dtype=np.float64)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError(f"position_xy must contain two finite values, got {xy}")
    qpos_address = joint_qpos_address(model, joint_name)
    dof_address = joint_dof_address(model, joint_name)
    half_yaw = 0.5 * float(yaw)
    data.qpos[qpos_address : qpos_address + 7] = [
        xy[0],
        xy[1],
        TABLE_TOP_Z + float(half_height) + 0.001,
        np.cos(half_yaw),
        0.0,
        0.0,
        np.sin(half_yaw),
    ]
    data.qvel[dof_address : dof_address + 6] = 0.0
    mujoco.mj_forward(model, data)
