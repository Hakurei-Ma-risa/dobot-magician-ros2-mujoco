import numpy as np

from mani_perception.rgbd_localizer import (
    PinholeIntrinsics,
    localize_bbox,
    robust_roi_depth,
)


def test_robust_roi_depth_ignores_invalid_pixels() -> None:
    depth = np.full((20, 20), 0.6)
    depth[7:13, 7:13] = 0.5
    depth[9, 9] = 0.0
    assert robust_roi_depth(depth, (4, 4, 12, 12)) == 0.5


def test_localize_bbox_uses_support_plane_for_object_center() -> None:
    intrinsics = PinholeIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0)
    # Optical frame aligned with target frame for this compact analytic test.
    position = localize_bbox(
        (59, 49, 3, 3),
        0.5,
        intrinsics,
        np.array([0.0, 0.0, 0.0]),
        np.eye(3),
        support_plane_z_m=0.45,
        object_height_m=0.10,
    )
    np.testing.assert_allclose(position, [0.05, 0.0, 0.5], atol=1e-9)


def test_localize_bbox_can_anchor_visible_bottom_to_table() -> None:
    intrinsics = PinholeIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0)
    position = localize_bbox(
        (69, 25, 3, 6),
        0.9,
        intrinsics,
        np.array([0.0, 0.0, 1.0]),
        np.diag([1.0, -1.0, -1.0]),
        support_plane_z_m=0.0,
        object_height_m=0.10,
        support_anchor="bbox_bottom",
    )
    np.testing.assert_allclose(position, [0.20, 0.20, 0.05], atol=1e-9)


def test_bottom_anchor_corrects_near_rim_of_cylinder() -> None:
    intrinsics = PinholeIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0)
    position = localize_bbox(
        (69, 25, 3, 6),
        0.9,
        intrinsics,
        np.array([0.30, 0.0, 1.0]),
        np.diag([1.0, -1.0, -1.0]),
        support_plane_z_m=0.0,
        object_height_m=0.10,
        support_anchor="bbox_bottom",
        support_footprint_radius_m=0.01,
    )
    direction = np.array([0.20, 0.20]) / np.sqrt(0.20**2 + 0.20**2)
    np.testing.assert_allclose(
        position, [0.50 + direction[0] * 0.01, 0.20 + direction[1] * 0.01, 0.05],
        atol=1e-9,
    )
