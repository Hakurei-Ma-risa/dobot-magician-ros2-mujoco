"""Analytic Dobot Magician FK/IK at the Portable Mani adapter boundary."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mani_core.models import Pose


BASE_HEIGHT = 0.131305
REAR_ARM_LENGTH = 0.135
FOREARM_LENGTH = 0.147
WRIST_TO_FLANGE = 0.060
FLANGE_TO_TOOL_TIP = 0.10484327246161541
GRIPPER_MAX_OPENING_JOINT = 0.0135

# Vendor/raw order: theta1, theta2, theta3 absolute, theta4.
RAW_JOINT_LOWER = np.deg2rad(np.array([-125.0, -5.0, -15.0, -150.0]))
RAW_JOINT_UPPER = np.deg2rad(np.array([125.0, 90.0, 70.0, 150.0]))


def _joints(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError(f"expected four finite joint values, got {result}")
    return result


def validate_raw(raw: ArrayLike, *, atol: float = 1e-9) -> NDArray[np.float64]:
    result = _joints(raw)
    if np.any(result < RAW_JOINT_LOWER - atol) or np.any(result > RAW_JOINT_UPPER + atol):
        raise ValueError(f"Dobot raw joints outside limits: {np.rad2deg(result).tolist()}")
    return result.copy()


def raw_to_model(raw: ArrayLike) -> NDArray[np.float64]:
    theta1, theta2, theta3, theta4 = validate_raw(raw)
    return np.array([theta1, theta2, theta3 - theta2, theta4])


def model_to_raw(model: ArrayLike) -> NDArray[np.float64]:
    q1, q2, q3_relative, q4 = _joints(model)
    return validate_raw([q1, q2, q2 + q3_relative, q4])


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def forward_tcp_pose(raw: ArrayLike, *, frame_id: str = "magician_base_link") -> Pose:
    theta1, theta2, theta3, theta4 = validate_raw(raw)
    radial = (
        REAR_ARM_LENGTH * np.sin(theta2)
        + FOREARM_LENGTH * np.cos(theta3)
        + WRIST_TO_FLANGE
    )
    position = np.array(
        [
            radial * np.cos(theta1),
            radial * np.sin(theta1),
            BASE_HEIGHT
            + REAR_ARM_LENGTH * np.cos(theta2)
            - FOREARM_LENGTH * np.sin(theta3)
            - FLANGE_TO_TOOL_TIP,
        ]
    )
    return Pose.from_xyz_yaw(*position, _wrap_pi(theta1 + theta4), frame_id=frame_id)


def inverse_tcp_pose(
    pose: Pose,
    *,
    seed_raw: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Solve the 4-DoF top-down pose and choose the solution nearest ``seed_raw``."""

    x, y, z = pose.position
    theta1 = float(np.arctan2(y, x))
    radial = float(np.hypot(x, y) - WRIST_TO_FLANGE)
    vertical = float(z - BASE_HEIGHT + FLANGE_TO_TOOL_TIP)
    distance_sq = radial * radial + vertical * vertical
    cosine_elbow = (
        distance_sq - REAR_ARM_LENGTH**2 - FOREARM_LENGTH**2
    ) / (2.0 * REAR_ARM_LENGTH * FOREARM_LENGTH)
    if cosine_elbow < -1.0 - 1e-9 or cosine_elbow > 1.0 + 1e-9:
        raise ValueError(f"unreachable TCP position: {pose.position.tolist()}")
    cosine_elbow = float(np.clip(cosine_elbow, -1.0, 1.0))

    seed = validate_raw(seed_raw) if seed_raw is not None else None
    candidates: list[NDArray[np.float64]] = []
    for relative_planar in (np.arccos(cosine_elbow), -np.arccos(cosine_elbow)):
        phi1 = np.arctan2(vertical, radial) - np.arctan2(
            FOREARM_LENGTH * np.sin(relative_planar),
            REAR_ARM_LENGTH + FOREARM_LENGTH * np.cos(relative_planar),
        )
        theta2 = float(np.pi / 2.0 - phi1)
        theta3 = float(-(phi1 + relative_planar))
        theta4 = _wrap_pi(pose.yaw - theta1)
        candidate = np.array([theta1, theta2, theta3, theta4])
        try:
            candidates.append(validate_raw(candidate))
        except ValueError:
            continue

    if not candidates:
        raise ValueError(
            "TCP pose is geometrically reachable but violates Dobot raw joint limits: "
            f"position={pose.position.tolist()}, yaw={pose.yaw:.4f}"
        )
    if seed is None:
        neutral = np.deg2rad([0.0, 30.0, 20.0, 0.0])
        return min(candidates, key=lambda candidate: float(np.linalg.norm(candidate - neutral)))
    return min(candidates, key=lambda candidate: float(np.linalg.norm(candidate - seed)))

