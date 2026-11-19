import numpy as np

from mani_vision.proposals import FindConfig, find_objects, footprint_rectangle
from mani_vision.selectors import Box, RuleSelector, colour_name, parse_choice


def _top_down_scene():
    """Camera 1 m above a table at z = 0, looking straight down."""

    depth = np.full((200, 200), 1.0)
    depth[60:80, 120:140] = 0.96  # 4 cm block at the right
    depth[120:150, 40:50] = 0.98  # 2 cm long bar at the left
    rotation = np.diag([1.0, -1.0, -1.0])
    translation = np.array([0.0, 0.0, 1.0])
    return depth, rotation, translation


def test_find_objects_sees_both_blocks_and_their_heights() -> None:
    depth, rotation, translation = _top_down_scene()
    config = FindConfig(ring_min_m=0.0, ring_max_m=10.0, min_area_px=20)
    table_z, found = find_objects(depth, 200.0, 200.0, 100.0, 100.0, rotation, translation, config)
    assert abs(table_z) < 0.01
    assert len(found) == 2
    heights = sorted(round(item.size_m[2], 2) for item in found)
    assert heights == [0.02, 0.04]


def test_footprint_rectangle_gives_long_side_direction() -> None:
    xs = np.linspace(-0.025, 0.025, 30)
    ys = np.linspace(-0.006, 0.006, 8)
    grid = np.array([(x, y) for x in xs for y in ys])
    angle = np.deg2rad(30.0)
    rotated = grid @ np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    long_m, short_m, yaw = footprint_rectangle(rotated)
    assert long_m > 0.045 and short_m < 0.015
    assert abs(((yaw - angle) + np.pi / 2) % np.pi - np.pi / 2) < np.deg2rad(2.0)


def test_parse_choice_is_forgiving_but_never_invents_a_box() -> None:
    assert parse_choice('{"index": 2, "reason": "red"}', {0, 1, 2})[0] == 2
    assert parse_choice('Sure! {"index": 1, "reason": "x"}', {0, 1})[0] == 1
    assert parse_choice('{"index": 7, "reason": "x"}', {0, 1})[0] == -1
    assert parse_choice("no idea", {0, 1})[0] == -1


def _image_with_two_squares():
    image = np.zeros((100, 200, 3), np.uint8)
    image[:] = (200, 130, 70)  # orange-brown table
    image[30:60, 20:50] = (220, 30, 30)  # red
    image[30:60, 140:170] = (40, 60, 220)  # blue
    return image


def test_colour_name_ignores_the_table() -> None:
    image = _image_with_two_squares()
    assert colour_name(image, (15, 25, 40, 40)) == "red"
    assert colour_name(image, (135, 25, 40, 40)) == "blue"


def test_rule_selector_uses_colour_then_position() -> None:
    image = _image_with_two_squares()
    boxes = [
        Box(0, (15, 25, 40, 40), (0.20, 0.10, 0.07), (0.03, 0.03, 0.04)),
        Box(1, (135, 25, 40, 40), (0.30, -0.10, 0.07), (0.03, 0.03, 0.04)),
    ]
    selector = RuleSelector()
    assert selector.select("the blue one", image, boxes).index == 1
    assert selector.select("the red block", image, boxes).index == 0
    assert selector.select("the one on the right", image, boxes).index == 1
    assert selector.select("anything", image, boxes).index == 0  # nearest
    assert selector.select("the green one", image, boxes).index == -1
