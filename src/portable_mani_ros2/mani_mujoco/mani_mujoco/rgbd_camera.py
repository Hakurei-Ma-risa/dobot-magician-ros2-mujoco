"""MuJoCo RGB-D and instance-mask rendering for the ROS simulation backend."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    frame_id: str = "camera_color_optical_frame"

    @classmethod
    def from_vertical_fov(
        cls, width: int, height: int, fovy_deg: float
    ) -> "CameraCalibration":
        focal = 0.5 * height / math.tan(math.radians(fovy_deg) * 0.5)
        return cls(
            width=width,
            height=height,
            fx=focal,
            fy=focal,
            cx=(width - 1) * 0.5,
            cy=(height - 1) * 0.5,
        )


@dataclass(frozen=True)
class RgbdFrame:
    rgb: NDArray[np.uint8]
    depth_m: NDArray[np.float32]
    object_bbox_xywh: tuple[int, int, int, int] | None


class MujocoRgbdCamera:
    """Render aligned RGB, metric depth and one target-object mask."""

    CAMERA_NAME = "d435i_sim"
    FOVY_DEG = 43.2

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        width: int = 640,
        height: int = 480,
        target_geom: str = "pick_object_geom",
        camera_name: str = CAMERA_NAME,
        include_segmentation: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("camera width and height must be positive")
        self.model = model
        self.camera_name = camera_name
        self.include_segmentation = include_segmentation
        self.calibration = CameraCalibration.from_vertical_fov(
            width, height, self.FOVY_DEG
        )
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._target_geom_id = None
        if include_segmentation:
            self._target_geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, target_geom
            )
            if self._target_geom_id < 0:
                self._renderer.close()
                raise KeyError(f"unknown MuJoCo target geom: {target_geom}")

    def render(self, data: mujoco.MjData) -> RgbdFrame:
        self._renderer.update_scene(data, camera=self.camera_name)

        self._renderer.disable_depth_rendering()
        self._renderer.disable_segmentation_rendering()
        rgb = self._renderer.render().copy()

        self._renderer.enable_depth_rendering()
        depth_m = self._renderer.render().copy()

        bbox = None
        if self.include_segmentation:
            self._renderer.enable_segmentation_rendering()
            segmentation = self._renderer.render()
            mask = (
                (segmentation[:, :, 0] == self._target_geom_id)
                & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            )
            bbox = self._bbox(mask)
            self._renderer.disable_segmentation_rendering()
        return RgbdFrame(rgb=rgb, depth_m=depth_m, object_bbox_xywh=bbox)

    def optical_pose(
        self, data: mujoco.MjData
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return camera optical pose in the MuJoCo world/base frame.

        MuJoCo camera axes are +X right, +Y up and -Z forward. ROS optical
        axes are +X right, +Y down and +Z forward, hence the two sign flips.
        """

        camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        rotation_mujoco = data.cam_xmat[camera_id].reshape(3, 3)
        rotation_optical = rotation_mujoco @ np.diag([1.0, -1.0, -1.0])
        quaternion_wxyz = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion_wxyz, rotation_optical.ravel())
        quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
        return data.cam_xpos[camera_id].copy(), quaternion_xyzw

    @staticmethod
    def _bbox(mask: NDArray[np.bool_]) -> tuple[int, int, int, int] | None:
        rows, columns = np.nonzero(mask)
        if rows.size == 0:
            return None
        x0, x1 = int(columns.min()), int(columns.max())
        y0, y1 = int(rows.min()), int(rows.max())
        return x0, y0, x1 - x0 + 1, y1 - y0 + 1

    @staticmethod
    def depth_millimeters(depth_m: NDArray[np.float32]) -> NDArray[np.uint16]:
        valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m < 65.535)
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        depth_mm[valid] = np.rint(depth_m[valid] * 1000.0).astype(np.uint16)
        return depth_mm

    def close(self) -> None:
        self._renderer.close()
