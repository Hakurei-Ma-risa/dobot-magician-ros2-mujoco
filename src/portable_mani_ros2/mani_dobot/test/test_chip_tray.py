import pytest
import cv2
import numpy as np

from mani_dobot.chip_tray import (
    cell_xyz, check_path, corner_report, fit_grid, grid_points, occupancy, plan_moves, project, refine_homography,
)


CORNERS = [
    {"row": 1, "col": 1, "pose": [250.0, 60.0, -60.0, 10.0]},
    {"row": 1, "col": 6, "pose": [250.0, 0.0, -60.0, 10.0]},
    {"row": 12, "col": 1, "pose": [140.0, 60.0, -61.1, 10.0]},
]


def test_grid_reproduces_taught_corners_and_interpolates():
    grid = fit_grid(CORNERS)
    for corner in CORNERS:
        assert np.allclose(cell_xyz(grid, corner["row"], corner["col"]), corner["pose"][:3])
    assert np.allclose(cell_xyz(grid, 1, 2) - cell_xyz(grid, 1, 1), [0.0, -12.0, 0.0])
    assert np.allclose(cell_xyz(grid, 12, 6), [140.0, 0.0, -61.1])


def test_plan_lifts_vertically_and_moves_xy_only_at_travel_height():
    grid = fit_grid(CORNERS)
    cells = [(cell_xyz(grid, 1, 1), cell_xyz(grid, 5, 3)), (cell_xyz(grid, 2, 6), cell_xyz(grid, 7, 2))]
    park = (200.0, -120.0, 20.0, 0.0)
    steps = plan_moves((210.0, -140.0, -10.0), park, cells, safe_z=-50.0, r=10.0,
                       xy_speed=100.0, z_speed=50.0, grip_ms=300, release_ms=300)
    moves = [v for k, v in steps if k == "movl"]
    assert all(m[3] == 10.0 for m in moves[:-2]) and moves[-1][3] == park[3]
    start = (210.0, -140.0, -10.0)
    for a, b in zip([start] + moves, moves):
        if not np.allclose(a[:2], b[:2]):
            assert a[2] == b[2] and a[2] in (-50.0, -10.0)  # tray plane, or safe_z + 40 towards the park
    assert moves[-1][:3] == park[:3] and moves[-2][:3] == (park[0], park[1], -10.0)
    suction = [v for k, v in steps if k == "suction"]
    assert suction == [True, False, True, False]
    contacts = [m[:3] for m in moves if m[2] < -55]
    expected = [c for pair in cells for c in pair]
    assert np.allclose(contacts, expected)


def synthetic_tray(rows=12, cols=6, full=()):
    true_h = cv2.getPerspectiveTransform(
        np.float32([(1, 1), (cols, 1), (cols, rows), (1, rows)]),
        np.float32([(200, 150), (340, 160), (380, 440), (170, 430)]))
    image = np.full((480, 640, 3), 230, np.uint8)
    outline = project(true_h, [(0.5, 0.5), (cols+0.5, 0.5), (cols+0.5, rows+0.5), (0.5, rows+0.5)])
    cv2.fillConvexPoly(image, np.round(outline).astype(np.int32), (25, 20, 20))
    for (u, v), (c, r) in zip(project(true_h, grid_points(rows, cols)), grid_points(rows, cols)):
        if (int(r), int(c)) not in full:
            cv2.rectangle(image, (int(u)-4, int(v)-2), (int(u)+4, int(v)+2), (235, 235, 235), -1)
    return image, true_h


def test_refined_grid_finds_chips_from_a_rough_click():
    full = {(1, 1), (3, 3), (7, 3), (7, 4), (8, 3), (12, 6)}
    image, true_h = synthetic_tray(full=full)
    rough = cv2.getPerspectiveTransform(
        np.float32([(1, 1), (6, 1), (6, 12), (1, 12)]),
        np.float32([(203, 147), (337, 163), (384, 437), (166, 433)]))
    h, matched = refine_homography(image, rough, 12, 6)
    assert matched >= 60
    assert np.abs(project(h, grid_points(12, 6)) - project(true_h, grid_points(12, 6))).max() < 1.0
    found = occupancy(image, h, 12, 6)
    assert {(r+1, c+1) for r, c in zip(*np.nonzero(found))} == full


def test_corner_report_flags_a_corner_taught_one_cell_off():
    rect = [{"row": r, "col": c, "pose": [160 + (c-1)*21.0, 175 - (r-1)*20.0, -63.0, 100.0]}
            for r, c in ((1, 1), (1, 6), (15, 6), (15, 1))]
    mismatch, pitch_row, pitch_col, angle = corner_report(rect)
    assert mismatch < 1e-9 and abs(pitch_row - 20.0) < 1e-9 and abs(pitch_col - 21.0) < 1e-9 and abs(angle - 90) < 1e-6
    rect[3]["pose"][1] += 20.0
    assert corner_report(rect)[0] > 19.0


def test_path_check_rejects_a_start_outside_joint_limits():
    park = (109.5, -135.2, 67.5, 2.7)  # J2 = -7.2 deg, beyond the -5 deg limit
    steps = [("movl", (203.0, 144.0, 67.5, 100.0))]
    try:
        check_path(park, steps)
    except ValueError as exc:
        assert "joint 2" in str(exc)
    else:
        raise AssertionError("expected a joint-limit error")
    check_path((200.0, 0.0, -20.0, 0.0), [("movl", (220.0, 30.0, -20.0, 0.0))])


def test_four_corner_grid_follows_a_warped_tray_height():
    corners = [{"row": r, "col": c, "pose": [160 + (c-1)*21.0, 150 - (r-1)*20.0, z, 96.0]}
               for (r, c), z in (((1, 1), -63.2), ((1, 6), -67.3), ((15, 6), -63.7), ((15, 1), -67.9))]
    grid = fit_grid(corners)
    for corner in corners:
        assert np.allclose(cell_xyz(grid, corner["row"], corner["col"]), corner["pose"][:3])
    assert abs(cell_xyz(grid, 8, 3.5)[2] - np.mean([-63.2, -67.3, -63.7, -67.9])) < 1e-9
    assert corner_report(corners)[0] < 1e-9  # height warp is not reported as a wrong cell


def test_empty_legs_keep_the_park_wrist_angle():
    grid = fit_grid(CORNERS)
    park = (68.0, -200.0, 60.0, -60.0)
    steps = plan_moves((*park[:3], park[3]), park, [(cell_xyz(grid, 1, 1), cell_xyz(grid, 3, 3))], safe_z=-50.0,
                       r=10.0, xy_speed=100.0, z_speed=50.0, grip_ms=300, release_ms=300)
    moves = [v for k, v in steps if k == "movl"]
    assert moves[0] == (68.0, -200.0, -10.0, -60.0)  # down above the park XY first, park wrist
    assert moves[-1] == park and moves[-2] == (68.0, -200.0, -10.0, -60.0)  # up only above the park XY
    assert all(m[3] == 10.0 for m in moves[1:-2])  # tray legs hold the grasp R


def test_joint_angles_reject_the_parallelogram_coupled_limit():
    from mani_dobot.chip_tray import joint_angles
    joint_angles((272.834, -171.735, 35.172, 15.829))  # taught by hand, accepted by the firmware
    with pytest.raises(ValueError, match="coupled"):
        joint_angles((272.834, -171.735, 75.2, 15.829))  # firmware planner raised alarm 0x12 here
