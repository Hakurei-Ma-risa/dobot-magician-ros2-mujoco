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
