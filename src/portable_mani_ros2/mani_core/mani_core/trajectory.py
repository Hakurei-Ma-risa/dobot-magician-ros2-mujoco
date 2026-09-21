"""Smooth trajectories with explicit position/velocity/acceleration outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MinimumJerkTrajectory:
    time_s: NDArray[np.float64]
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]


def minimum_jerk(
    start: ArrayLike,
    goal: ArrayLike,
    *,
    duration_s: float,
    rate_hz: float,
) -> MinimumJerkTrajectory:
    """Generate a quintic minimum-jerk trajectory including both endpoints."""

    start_array = np.asarray(start, dtype=np.float64)
    goal_array = np.asarray(goal, dtype=np.float64)
    if start_array.shape != goal_array.shape or start_array.ndim != 1:
        raise ValueError("start and goal must be same-sized one-dimensional vectors")
    if not np.all(np.isfinite(start_array)) or not np.all(np.isfinite(goal_array)):
        raise ValueError("start and goal must be finite")
    if duration_s <= 0.0 or rate_hz <= 0.0:
        raise ValueError("duration_s and rate_hz must be positive")

    sample_count = max(2, int(np.ceil(duration_s * rate_hz)) + 1)
    time_s = np.linspace(0.0, duration_s, sample_count)
    u = time_s / duration_s
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    blend_d = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration_s
    blend_dd = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / duration_s**2
    delta = goal_array - start_array
    return MinimumJerkTrajectory(
        time_s=time_s,
        position=start_array + blend[:, None] * delta,
        velocity=blend_d[:, None] * delta,
        acceleration=blend_dd[:, None] * delta,
    )

