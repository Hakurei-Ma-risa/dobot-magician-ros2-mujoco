"""Module 2 "find objects": depth only, no object classes.

Idea: turn every depth pixel into a 3-D point in the robot base frame,
find the table height, keep the points that stick up above the table inside
the reachable ring, and group touching pixels into blobs. Each blob is one
candidate object. Nothing here knows what the objects are; choosing one is
module 3's job.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FindConfig:
    min_height_m: float = 0.008  # ignore table noise thinner than this
    max_height_m: float = 0.20  # ignore things higher than this above the table
    ring_min_m: float = 0.12  # closer than this to the robot base = robot body
    ring_max_m: float = 0.34  # farther than this = out of reach
    min_area_px: int = 150  # smaller blobs are noise
    open_px: int = 5  # morphological opening kernel, removes thin noise
    table_z_m: float | None = None  # None = estimate from the depth image
    resting_gap_m: float = 0.02  # bottom closer than this to the table = resting


@dataclass(frozen=True)
class Candidate:
    index: int
    bbox_xywh: tuple[int, int, int, int]
    center_m: tuple[float, float, float]  # footprint centre; z = middle of the object
    size_m: tuple[float, float, float]  # long side, short side, height
    yaw_rad: float  # direction of the long side in the base frame
    bottom_z_m: float  # lowest visible part (rises when the object is lifted)
    area_px: int
    mask: np.ndarray  # bool mask of this blob, image sized


def depth_to_base_points(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Return an (H, W, 3) array of base-frame points (camera optical model)."""

    height, width = depth_m.shape
    v, u = np.mgrid[0:height, 0:width]
    optical = np.stack(
        ((u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m), axis=-1
    )
    return optical @ np.asarray(rotation).T + np.asarray(translation)


def footprint_rectangle(xy_m: np.ndarray) -> tuple[float, float, float]:
    """Smallest rectangle around the top-view footprint: (long, short, yaw).

    yaw is the direction of the long side. A parallel gripper turned to this
    yaw closes across the short side.
    """

    points = (np.asarray(xy_m, dtype=np.float64) * 1000.0).astype(np.float32)
    (_, _), (width, height), angle = cv2.minAreaRect(points)
    yaw = np.deg2rad(angle)
    if width < height:
        width, height = height, width
        yaw += np.pi / 2.0
    yaw = float((yaw + np.pi / 2.0) % np.pi - np.pi / 2.0)  # gripper is symmetric
    return float(width) / 1000.0, float(height) / 1000.0, yaw


def estimate_table_z(points: np.ndarray, valid: np.ndarray, ring: np.ndarray) -> float:
    """The table is the most common height among valid points in the ring."""

    heights = points[..., 2][valid & ring]
    if heights.size < 100:
        heights = points[..., 2][valid]
    if heights.size == 0:
        raise ValueError("no valid depth pixels")
    histogram, edges = np.histogram(heights, bins=400)
    best = int(histogram.argmax())
    return float(0.5 * (edges[best] + edges[best + 1]))


def find_objects(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    config: FindConfig = FindConfig(),
) -> tuple[float, list[Candidate]]:
    points = depth_to_base_points(depth_m, fx, fy, cx, cy, rotation, translation)
    valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 5.0)
    radius = np.hypot(points[..., 0], points[..., 1])
    ring = (radius > config.ring_min_m) & (radius < config.ring_max_m)
    table_z = (
        float(config.table_z_m)
        if config.table_z_m is not None
        else estimate_table_z(points, valid, ring)
    )
    above = (
        valid
        & ring
        & (points[..., 2] > table_z + config.min_height_m)
        & (points[..., 2] < table_z + config.max_height_m)
    )
    mask = above.astype(np.uint8) * 255
    if config.open_px > 1:
        kernel = np.ones((config.open_px, config.open_px), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates: list[Candidate] = []
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        if area < config.min_area_px:
            continue
        blob = (labels == label) & above
        blob_points = points[blob]
        if blob_points.shape[0] < 20:
            continue
        low_xy, high_xy = np.quantile(blob_points[:, :2], [0.02, 0.98], axis=0)
        long_m, short_m, yaw = footprint_rectangle(blob_points[:, :2])
        # Same grasp, less wrist twist: pick yaw or yaw + pi, whichever is
        # closer to the direction from the robot base to the object.
        radial = float(np.arctan2(0.5 * (low_xy[1] + high_xy[1]), 0.5 * (low_xy[0] + high_xy[0])))
        yaw = radial + float((yaw - radial + np.pi / 2.0) % np.pi - np.pi / 2.0)
        top = float(np.quantile(blob_points[:, 2], 0.97))
        bottom = float(np.quantile(blob_points[:, 2], 0.03))
        # Resting on the table: the hidden bottom is the table itself.
        # Lifted (e.g. held by the gripper): the bottom is what we see. It only
        # counts as lifted if we also see a vertical side; a camera looking
        # straight down sees just the top face of a resting object.
        lifted = bottom > table_z + config.resting_gap_m and top - bottom > 0.015
        base_z = bottom if lifted else table_z
        object_height = max(top - base_z, config.min_height_m)
        candidates.append(
            Candidate(
                index=-1,
                bbox_xywh=(x, y, w, h),
                center_m=(
                    float(0.5 * (low_xy[0] + high_xy[0])),
                    float(0.5 * (low_xy[1] + high_xy[1])),
                    float(base_z + 0.5 * object_height),
                ),
                size_m=(long_m, short_m, float(object_height)),
                yaw_rad=yaw,
                bottom_z_m=bottom,
                area_px=area,
                mask=blob,
            )
        )
    # Stable numbering for people: left to right in the image.
    candidates.sort(key=lambda item: (item.bbox_xywh[0], item.bbox_xywh[1]))
    numbered = [
        Candidate(
            index=i,
            bbox_xywh=item.bbox_xywh,
            center_m=item.center_m,
            size_m=item.size_m,
            yaw_rad=item.yaw_rad,
            bottom_z_m=item.bottom_z_m,
            area_px=item.area_px,
            mask=item.mask,
        )
        for i, item in enumerate(candidates)
    ]
    return table_z, numbered
