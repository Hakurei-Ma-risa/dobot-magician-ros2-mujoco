import argparse
import collections
import math
import struct

import pytest

from mani_dobot import conveyor_sort as cs


SETUP = {
    "robot1": {
        "corners": [
            {"row": 1, "col": 1, "pose": [260.0, 30.0, -40.0, 0.0]},
            {"row": 1, "col": 3, "pose": [260.0, -30.0, -40.0, 0.0]},
            {"row": 3, "col": 3, "pose": [200.0, -30.0, -42.0, 0.0]},
            {"row": 3, "col": 1, "pose": [200.0, 30.0, -42.0, 0.0]},
        ],
        "belt_drop": [230.0, 120.0, -30.0, 10.0],
    },
    "robot2": {
        "belt_pick": [220.0, -100.0, -30.0, 5.0],
        "colour": [180.0, 50.0, -20.0, 0.0],
        "storage": {"red": [200.0, 150.0, -45.0, 0.0], "green": [240.0, 120.0, -45.0, 0.0],
                    "blue": [260.0, 80.0, -45.0, 0.0]},
    },
}


def options(**overrides):
    values = dict(block_mm=25.0, stack_mm=25.0, box_mm=[0.0], per_colour=3, clearance=10.0, pick_depth=1.0, start=1, count=9,
                  speed=150.0, z_speed=80.0, grip_ms=400, release_ms=400)
    values.update(overrides)
    return argparse.Namespace(**values)


class FakeBot:
    def __init__(self):
        self.belt_speeds, self.runs = [], []

    def send(self, command_id, params=b"", **_):
        if command_id == cs.EMOTOR:
            self.belt_speeds.append(struct.unpack("<BBi", params)[2])
            return b""
        raise AssertionError(command_id)

    def pose(self):
        return (220.0, -100.0, 50.0, 5.0, 0.0, 0.0, 0.0, 0.0)

    def run(self, steps):
        self.runs.append(steps)


def test_material_points_interpolate_the_three_by_three_grid():
    points = cs.material_points(SETUP["robot1"]["corners"])
    assert len(points) == 9
    assert points[0] == pytest.approx((260.0, 30.0, -40.0))
    assert points[4] == pytest.approx((230.0, 0.0, -41.0))
    assert points[8] == pytest.approx((200.0, -30.0, -42.0))


def test_robot1_travel_height_clears_blocks_and_the_belt():
    height = cs.safe_height_robot1(SETUP, block_mm=25, clearance=10)
    assert height == pytest.approx(-30.0 + 25 + 10)  # belt drop is the highest robot 1 point


def test_robot1_cycle_picks_deeper_and_only_moves_sideways_at_travel_height():
    plan = cs.Plan(SETUP, options())
    steps = plan.robot1_cycle((230.0, 0.0, 80.0, 0.0), 4)
    moves = [value for kind, value in steps if kind == "movl"]
    assert moves[1][:3] == pytest.approx((230.0, 0.0, -42.0))   # cell 5 top minus 1 mm pick depth
    assert moves[-2][:3] == pytest.approx((230.0, 120.0, -30.0))
    assert [kind for kind, _ in steps].count("suction") == 2
    for a, b in zip(moves, moves[1:]):
        assert a[:2] == pytest.approx(b[:2]) or a[2] == b[2] == plan.safe1


def test_robot2_goes_from_the_belt_into_the_box_and_waits_above_it():
    plan = cs.Plan(SETUP, options())
    pick = plan.robot2_pick((220.0, -100.0, 50.0, 5.0), "red", 2)
    store = plan.robot2_store("red", 2)
    assert ("suction", True) in pick and pick[-1][0] == "movl" and pick[-1][1][:2] == pytest.approx((220.0, -100.0))
    release = store.index(("suction", False))
    assert store[release - 1][1][:3] == pytest.approx((200.0, 150.0, -45.0 + 50.0))
    across = [value for kind, value in store if kind == "movj"]
    assert across[0][:2] == pytest.approx((200.0, 150.0)) and across[0][2] == pytest.approx(-45.0 + 50.0 + 10.0)
    assert store[-1][1][:2] == pytest.approx((200.0, 150.0)) and store[-1][1][2] > -45.0 + 50.0


def test_robot2_leg_lifts_only_as_high_as_the_arm_reaches_near_the_base():
    plan = cs.Plan(SETUP, options())
    near, far = (150.0, 0.0, 0.0), (280.0, 0.0, 0.0)
    up, across, down = plan.leg(near, far, 0.0, 40.0)
    reach = cs.top_reachable_z(near[:2], plan.r2, 0.0)
    assert up[-1][1][2] == pytest.approx(min(40.0, reach - 2.0))
    assert across[0][0] == "movj" and across[0][1][2] == pytest.approx(40.0)
    with pytest.raises(ValueError, match="只能抬到"):
        plan.leg(near, far, reach + 5.0, 0.0)


def test_storage_approach_clears_the_box_wall():
    plan = cs.Plan(SETUP, options(box_mm=[0.0, 60.0, 0.0], stack_mm=0.0))
    across = [value for kind, value in plan.robot2_store("green", 0) if kind == "movj"]
    assert across[0][:2] == pytest.approx((240.0, 120.0))
    assert across[0][2] == pytest.approx(-45.0 + 60.0 + 10.0)  # above the wall, higher than the belt level
    low = cs.Plan(SETUP, options(box_mm=[0.0], stack_mm=0.0))
    assert [v for k, v in low.robot2_store("green", 0) if k == "movj"][0][2] == pytest.approx(low.pick_level)


def test_plan_check_passes_for_reachable_setup_and_rejects_out_of_reach():
    plan = cs.Plan(SETUP, options())
    plan.check((260.0, 30.0, plan.safe1, 0.0), (220.0, -100.0, plan.pick_level, 5.0), 3)
    far = {**SETUP, "robot2": {**SETUP["robot2"], "storage": {**SETUP["robot2"]["storage"],
                                                                 "red": [400.0, 0.0, -20.0, 0.0]}}}
    with pytest.raises(ValueError):
        cs.Plan(far, options()).check((260.0, 30.0, plan.safe1, 0.0), (220.0, -100.0, plan.pick_level, 5.0), 3)


def test_belt_runs_the_given_steps_and_stops():
    bot = FakeBot()
    assert cs.belt_run(bot, -50000, 2500) == pytest.approx(0.05)
    assert bot.belt_speeds == [-50000, 0]


def test_robot2_job_stores_the_given_colour_and_counts_it():
    plan = cs.Plan(SETUP, options())
    stacked = collections.Counter()
    job = cs.Robot2Job(FakeBot(), plan, "blue", stacked, "第 1 块")
    job.run()
    assert job.error is None and job.lifted.is_set() and stacked["blue"] == 1 and len(job.bot.runs) == 2


def test_robot2_keeps_joint4_instead_of_world_yaw():
    pick = (258.1, 56.0, 12.3, 135.5)  # taught with the cup turned far round
    joint4 = pick[3] - math.degrees(math.atan2(pick[1], pick[0]))
    for point in ((23.3, -227.3, 81.2), (123.9, -214.2, 78.2), (197.8, 0.0, 66.5)):
        r = cs.r_keeping_j4(point, pick)
        assert r - math.degrees(math.atan2(point[1], point[0])) == pytest.approx(joint4)
        cs.joint_angles((*point, r))  # holding R=135.5 here would put joint 4 past 150 deg
