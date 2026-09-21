"""Dobot Magician embodiment adapter."""

from .kinematics import (
    RAW_JOINT_LOWER,
    RAW_JOINT_UPPER,
    forward_tcp_pose,
    inverse_tcp_pose,
    model_to_raw,
    raw_to_model,
)

__all__ = [
    "RAW_JOINT_LOWER",
    "RAW_JOINT_UPPER",
    "forward_tcp_pose",
    "inverse_tcp_pose",
    "model_to_raw",
    "raw_to_model",
]

