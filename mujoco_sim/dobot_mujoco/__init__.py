"""Dobot Magician MuJoCo model helpers.

The Gymnasium environment is imported lazily so model-only ROS processes do
not acquire a Gymnasium runtime dependency.
"""
from .kinematics import (
    RAW_JOINT_LOWER,
    RAW_JOINT_UPPER,
    analytic_tcp_position,
    analytic_wrist_position,
    model_to_raw,
    raw_to_model,
)
from .model import MODEL_PATH, load_model, set_raw_pose

__all__ = [
    "DobotMagicianReachEnv",
    "MODEL_PATH",
    "RAW_JOINT_LOWER",
    "RAW_JOINT_UPPER",
    "analytic_tcp_position",
    "analytic_wrist_position",
    "load_model",
    "model_to_raw",
    "raw_to_model",
    "set_raw_pose",
]


def __getattr__(name: str):
    if name == "DobotMagicianReachEnv":
        from .env import DobotMagicianReachEnv

        return DobotMagicianReachEnv
    raise AttributeError(name)
