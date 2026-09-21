"""Joint conventions and analytic kinematics used by magician_ros2.

The Dobot firmware reports four angles ``[theta1, theta2, theta3, theta4]``.
The ROS URDF does not use theta3 directly: its third revolute joint is
``theta3 - theta2``.  Keeping this conversion in one module prevents a common
double-subtraction bug when data moves between hardware, ROS and MuJoCo.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

BASE_HEIGHT = 0.131305
REAR_ARM_LENGTH = 0.135
FOREARM_LENGTH = 0.147
WRIST_TO_FLANGE = 0.060
# Normal gripper DAE geometry reaches 104.843 mm below the flange origin.
FLANGE_TO_TOOL_TIP = 0.10484327246161541
GRIPPER_MAX_OPENING_JOINT = 0.0135

# Firmware/raw joint order: theta1, theta2, theta3, theta4.
RAW_JOINT_LOWER = np.deg2rad(np.array([-125.0, -5.0, -15.0, -150.0]))
RAW_JOINT_UPPER = np.deg2rad(np.array([125.0, 90.0, 70.0, 150.0]))


def _as_joint_vector(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (4,):
        raise ValueError(f"Expected four joint values, got shape {result.shape}")
    return result


def validate_raw_joints(raw: ArrayLike, *, atol: float = 1e-9) -> NDArray[np.float64]:
    """Return a checked raw joint vector in radians."""

    result = _as_joint_vector(raw)
    if np.any(result < RAW_JOINT_LOWER - atol) or np.any(result > RAW_JOINT_UPPER + atol):
        raise ValueError(
            "Raw joints outside Dobot limits (degrees): "
            f"{np.rad2deg(result).round(3).tolist()}"
        )
    return result


def raw_to_model(raw: ArrayLike, *, validate: bool = True) -> NDArray[np.float64]:
    """Convert firmware angles to MuJoCo/URDF joints.

    Returns ``[q1, q2, q3_relative, q4]`` where
    ``q3_relative = theta3 - theta2``.
    """

    raw_array = validate_raw_joints(raw) if validate else _as_joint_vector(raw)
    return np.array(
        [raw_array[0], raw_array[1], raw_array[2] - raw_array[1], raw_array[3]],
        dtype=np.float64,
    )


def model_to_raw(model_joints: ArrayLike, *, validate: bool = True) -> NDArray[np.float64]:
    """Convert ``[q1, q2, q3_relative, q4]`` back to firmware angles."""

    model_array = _as_joint_vector(model_joints)
    raw = np.array(
        [model_array[0], model_array[1], model_array[1] + model_array[2], model_array[3]],
        dtype=np.float64,
    )
    return validate_raw_joints(raw) if validate else raw


def analytic_wrist_position(raw: ArrayLike, *, world: bool = True) -> NDArray[np.float64]:
    """Position of the forearm endpoint used by ``dobot_forward_kin.py``.

    ``world=False`` returns coordinates relative to ``magician_base_link``.
    The hardware API's TCP may include a configured tool offset; that offset is
    deliberately not folded into this reference point.
    """

    theta1, theta2, theta3, _ = validate_raw_joints(raw)
    radius = REAR_ARM_LENGTH * np.sin(theta2) + FOREARM_LENGTH * np.cos(theta3)
    position = np.array(
        [
            radius * np.cos(theta1),
            radius * np.sin(theta1),
            REAR_ARM_LENGTH * np.cos(theta2) - FOREARM_LENGTH * np.sin(theta3),
        ],
        dtype=np.float64,
    )
    if world:
        position[2] += BASE_HEIGHT
    return position


def analytic_tcp_position(raw: ArrayLike, *, world: bool = True) -> NDArray[np.float64]:
    """Position of this model's nominal gripper tip.

    The parallelogram keeps the flange level, so the 60 mm wrist link remains
    horizontal and the nominal gripper tip is 60 mm below the flange.
    """

    theta1 = validate_raw_joints(raw)[0]
    position = analytic_wrist_position(raw, world=world)
    position += np.array(
        [
            WRIST_TO_FLANGE * np.cos(theta1),
            WRIST_TO_FLANGE * np.sin(theta1),
            -FLANGE_TO_TOOL_TIP,
        ]
    )
    return position
