"""Two-robot conveyor colour sorting with Gemini vision; prompts in Chinese.

Robot 1 picks the blocks of a taught 3x3 material area one by one and puts
each on the conveyor (robot 1 Stepper1). The belt runs a fixed number of
steps to robot 2's pick position and stops. A D435i frame of the belt goes to
Gemini Robotics ER, which says whether a cube is on the belt and its colour.
Robot 2 picks the block at its taught point and places it in that colour's
storage box, then waits above the box (out of the camera's view). While
robot 2 sorts, robot 1 already fetches the next block.

  ros2 launch mani_perception d435i.launch.py color_profile:=640x480x15 depth_profile:=640x480x15
  export GEMINI_API_KEY='...'
  ros2 run mani_dobot conveyor_sort teach 1   # material corners, belt drop point
  ros2 run mani_dobot conveyor_sort teach 2   # belt pick, red/green/blue storage
  ros2 run mani_dobot conveyor_sort vision    # check the camera view and Gemini
  ros2 run mani_dobot conveyor_sort run

Home both robots first (dobot_home); taught points are lost with every power
cycle that is not followed by homing.
"""

from __future__ import annotations

import argparse
import collections
from contextlib import suppress
import json
import math
from pathlib import Path
import struct
import threading
import time

from . import belt_vision
from .chip_tray import cell_xyz, check_path, corner_report, fit_grid, joint_angles
from .dobot_home import describe
from .fast_pick_place import TAUGHT_TOOL_BIAS_MM, Dobot
from .sensor_check import BELT_FORWARD_SPEED, EMOTOR, ROBOTS


DEFAULT_SETUP = Path("reports/conveyor_sort/setup.json")
SIZE = 3  # material area is SIZE x SIZE
# drop point -> robot 2 pick point: the image sensor stopped the belt after 11.7 s at 5000 steps/s,
# and robot 2's pick point was taught at that stop
BELT_PULSES = 58500
CN = {"red": "红", "green": "绿", "blue": "蓝"}
TYPED = {"红": "red", "r": "red", "绿": "green", "g": "green", "蓝": "blue", "b": "blue"}


# ------------------------------------------------------------------ hardware

def open_robot(number: int, accel: float = 150.0) -> Dobot:
    name, path = ROBOTS[number - 1]
    bot = Dobot(path, accel)
    bias = bot.tool_bias()
    if math.dist(bias, TAUGHT_TOOL_BIAS_MM) > 0.5:
        bot.close()
        raise SystemExit(f"{name} 的工具偏置 {bias} 不是吸盘的 {TAUGHT_TOOL_BIAS_MM}，请先设置吸盘工具")
    return bot


def clear_alarms(bot, number: int) -> None:
    if bot.alarms():
        bot.clear_alarms()
        time.sleep(0.2)
    codes = bot.alarms()
    if codes:
        raise SystemExit(f"{ROBOTS[number-1][0]} 有清不掉的报警：{describe(codes)}。"
                         f"先运行 ros2 run mani_dobot dobot_home {number}")


def suction(bot, on: bool) -> None:
    bot.send(62, struct.pack("<BB", on, on), write=True)


def belt(bot, speed: int) -> None:
    bot.send(EMOTOR, struct.pack("<BBi", 0, speed != 0, speed), write=True)  # Stepper1


def belt_run(bot, speed: int, pulses: int) -> float:
    """Run the belt for ``pulses`` steps and stop; seconds taken."""
    seconds = abs(pulses / speed)
    belt(bot, speed)
    try:
        time.sleep(seconds)
    finally:
        belt(bot, 0)
    return seconds


# ------------------------------------------------------------------ geometry and plans

def load_setup(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_setup(path: Path, setup: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(setup, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def material_points(corners) -> list[tuple[float, float, float]]:
    """Block-top contact points, row-major from (1, 1)."""
    grid = fit_grid(corners)
    return [tuple(float(v) for v in cell_xyz(grid, row, col))
            for row in range(1, SIZE + 1) for col in range(1, SIZE + 1)]


def safe_height_robot1(setup, *, block_mm, clearance) -> float:
    """Robot 1 travel Z: a carried block clears every material block top and the belt."""
    one = setup["robot1"]
    return max([c["pose"][2] for c in one["corners"]] + [one["belt_drop"][2]]) + block_mm + clearance


def top_reachable_z(xy, r, z_low, z_high=250.0) -> float:
    """Highest Z above ``xy`` inside the joint limits, searching up from the reachable ``z_low``."""
    def ok(z):
        try:
            joint_angles((*xy, z, r))
            return True
        except ValueError:
            return False
    if ok(z_high):
        return z_high
    low, high = z_low, z_high
    while high - low > 0.5:
        mid = (low + high) / 2
        low, high = (mid, high) if ok(mid) else (low, mid)
    return low


def r_keeping_j4(point, reference) -> float:
    """R at ``point`` that keeps joint 4 where it is at the taught ``reference`` pose.

    Holding R (the world yaw) instead would turn joint 4 by the base rotation;
    robot 2 swings far enough between belt and boxes to leave the +-150 deg
    range. Its round cup does not care how the block turns.
    """
    joint4 = reference[3] - math.degrees(math.atan2(reference[1], reference[0]))
    return math.degrees(math.atan2(point[1], point[0])) + (joint4 + 180.0) % 360.0 - 180.0


def check_steps(start, steps) -> None:
    """ValueError if a MOVL leaves the joint limits anywhere or a MOVJ ends outside them.

    A MOVJ interpolates joint angles, so with both ends inside the per-joint
    limits the whole move is inside them.
    """
    previous = tuple(start[:4])
    for kind, value in steps:
        if kind == "movl":
            check_path(previous, [("movl", value)])
        elif kind == "movj":
            joint_angles(value)
        else:
            continue
        previous = value


def pick_steps(start, pick, *, safe_z, r, xy_speed, z_speed, grip_ms) -> list[tuple]:
    above = (*pick[:2], safe_z, r)
    steps = []
    if start[2] < safe_z - 1.0:
        steps += [("speed", z_speed), ("movl", (*start[:2], safe_z, r))]
    return steps + [("speed", xy_speed), ("movl", above), ("speed", z_speed), ("movl", (*pick[:3], r)),
                    ("suction", True), ("wait", grip_ms), ("movl", above)]


def place_steps(place, *, safe_z, r, xy_speed, z_speed, release_ms) -> list[tuple]:
    above = (*place[:2], safe_z, r)
    return [("speed", xy_speed), ("movl", above), ("speed", z_speed), ("movl", (*place[:3], r)),
            ("suction", False), ("wait", release_ms), ("movl", above)]


class Plan:
    """All motion of one run, built from the taught setup and the CLI options.

    Robot 1 moves in straight lines at one travel height. Robot 2's points lie
    at very different heights and partly close to its base, where the arm
    cannot rise far, so each of its legs lifts only as high as the leg needs
    (and the arm reaches there), joint-moves across and lowers vertically.
    """

    def __init__(self, setup, args) -> None:
        one, two = setup["robot1"], setup["robot2"]
        self.safe1 = safe_height_robot1(setup, block_mm=args.block_mm, clearance=args.clearance)
        self.r1, self.r2 = one["corners"][0]["pose"][3], two["belt_pick"][3]
        depth = args.pick_depth
        points = material_points(one["corners"])[args.start - 1:args.start - 1 + args.count]
        self.material = [(x, y, z - depth) for x, y, z in points]
        self.belt_drop = one["belt_drop"]
        self.belt_pick = (*two["belt_pick"][:2], two["belt_pick"][2] - depth)
        self.pick_pose = tuple(two["belt_pick"][:4])
        self.storage = {name: tuple(p[:3]) for name, p in two["storage"].items()}
        walls = args.box_mm if len(args.box_mm) == 3 else args.box_mm[:1] * 3
        self.box_mm = dict(zip(("red", "green", "blue"), walls))
        self.stack_mm, self.clearance = args.stack_mm, args.clearance
        # carried block clears a block standing on the belt; into the box it is lowered vertically
        self.pick_level = two["belt_pick"][2] + args.block_mm + args.clearance
        self.move = dict(xy_speed=args.speed, z_speed=args.z_speed)
        self.grip_ms, self.release_ms = args.grip_ms, args.release_ms

    def robot1_cycle(self, start, index: int) -> list[tuple]:
        return (pick_steps(start, self.material[index], safe_z=self.safe1, r=self.r1, grip_ms=self.grip_ms,
                           **self.move)
                + place_steps(self.belt_drop, safe_z=self.safe1, r=self.r1, release_ms=self.release_ms,
                              **self.move))

    def storage_point(self, colour: str, stacked: int) -> tuple[float, float, float]:
        x, y, z = self.storage[colour]
        return x, y, z + stacked * self.stack_mm

    def r2_at(self, point) -> float:
        return r_keeping_j4(point, self.pick_pose)

    def leg(self, a, b, level_a: float, level_b: float) -> tuple[list, list, list]:
        """(up, across, down) robot 2 steps from ``a`` to ``b``; joint 4 stays as taught at the pick."""
        z_speed = self.move["z_speed"]
        r_a, r_b = self.r2_at(a), self.r2_at(b)
        if math.dist(a[:2], b[:2]) < 1.0:
            return [], [], [("speed", z_speed), ("movl", (*b[:3], r_b))]
        need = max(level_a, level_b)
        heights = []
        for point, level in ((a, level_a), (b, level_b)):
            height = min(need, top_reachable_z(point[:2], self.r2_at(point), point[2]) - 2.0)
            if height < level - 0.01:
                raise ValueError(f"2号在 ({point[0]:.0f}, {point[1]:.0f}) 上方只能抬到 z={height + 2:.0f}，"
                                 f"不够 z={level:.0f} 的安全高度")
            heights.append(height)
        up = [] if a[2] >= heights[0] - 0.05 else [("speed", z_speed), ("movl", (*a[:2], heights[0], r_a))]
        return (up, [("movj", (*b[:2], heights[1], r_b))], [("speed", z_speed), ("movl", (*b[:3], r_b))])

    def storage_level(self, colour: str, stacked: int) -> float:
        return self.storage_point(colour, stacked)[2] + self.box_mm[colour] + self.clearance  # clears the wall

    def robot2_pick(self, start, colour: str, stacked: int = 0) -> list[tuple]:
        """From ``start`` onto the block on the belt, grip, and lift clear of the belt."""
        up, across, down = self.leg(start, self.belt_pick, start[2], self.pick_level)
        lift, _, _ = self.leg(self.belt_pick, self.storage_point(colour, stacked), self.pick_level,
                              self.storage_level(colour, stacked))
        return up + across + down + [("suction", True), ("wait", self.grip_ms)] + lift

    def robot2_store(self, colour: str, stacked: int) -> list[tuple]:
        """Belt -> box, release, and wait above the box (out of the camera's view of the belt)."""
        place = self.storage_point(colour, stacked)
        level = self.storage_level(colour, stacked)
        _, across, down = self.leg(self.belt_pick, place, self.pick_level, level)
        lift, _, _ = self.leg(place, self.belt_pick, level, self.pick_level)
        return across + down + [("suction", False), ("wait", self.release_ms)] + lift

    def check(self, start1, start2, per_colour: int) -> None:
        """ValueError if any planned move leaves the joint limits."""
        start = start1
        for index in range(len(self.material)):
            check_path(start, self.robot1_cycle(start, index))
            start = (*self.belt_drop[:2], self.safe1, self.r1)
        for colour in self.storage:
            for stacked in sorted({0, per_colour - 1}):
                first = self.robot2_pick(start2, colour, stacked) + self.robot2_store(colour, stacked)
                waiting = next(value for kind, value in reversed(first) if kind in ("movl", "movj"))
                for following in self.storage:
                    check_steps(start2, first + self.robot2_pick(waiting, following)
                                + self.robot2_store(following, 0))


# ------------------------------------------------------------------ teach

def ask_point(bot, prompt: str) -> list[float]:
    """Pose when Enter is pressed, re-asked until it is inside the joint limits."""
    while True:
        input(prompt)
        pose = [round(v, 3) for v in bot.pose()[:4]]
        try:
            joint_angles(pose)
        except ValueError as exc:
            print(f"  这个位置超出关节限位（{exc}），请换个姿态再按一次。")
            continue
        print(f"  已记录 x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} r={pose[3]:.1f}")
        return pose


def teach_corners(bot) -> list[dict]:
    corners = []
    for row, col in ((1, 1), (1, SIZE), (SIZE, SIZE), (SIZE, 1)):
        pose = ask_point(bot, f"\n[1号 物料区] 在第 {row} 行第 {col} 列放一块方块。按住解锁键，把吸盘轻放在"
                              f"方块顶面中心，松开解锁键后按回车 > ")
        corners.append({"row": row, "col": col, "pose": pose})
    while True:
        mismatch, pitch_row, pitch_col, angle = corner_report(corners)
        print(f"\n  行距 {pitch_row:.1f} mm，列距 {pitch_col:.1f} mm，行列夹角 {angle:.1f}°，"
              f"四角偏差 {mismatch:.1f} mm")
        if mismatch <= 4.0:
            return corners
        labels = "，".join(f"{i+1}=第{c['row']}行第{c['col']}列" for i, c in enumerate(corners))
        answer = input(f"  四个角对不成矩形，可能有一个角放错了格。重新示教哪一个？{labels}；k 保留 > ").strip()
        if answer.lower() == "k":
            return corners
        if answer in {str(i + 1) for i in range(len(corners))}:
            corner = corners[int(answer) - 1]
            corner["pose"] = ask_point(bot, f"  把吸盘轻放在第 {corner['row']} 行第 {corner['col']} 列的方块"
                                            "顶面中心，按回车 > ")


def teach_material(args) -> None:
    """Re-teach only robot 1's material area; the belt drop point stays."""
    setup = load_setup(args.setup)
    if "robot1" not in setup:
        raise SystemExit("还没示教过 1号，先运行 conveyor_sort teach 1")
    bot = open_robot(1)
    try:
        clear_alarms(bot, 1)
        setup["robot1"]["corners"] = teach_corners(bot)
    finally:
        bot.close()
    save_setup(args.setup, setup)
    print(f"\n已保存到 {args.setup}（传送带放料点没变）")


def teach_robot1(args) -> None:
    bot = open_robot(1)
    corners, drop = [], None
    try:
        clear_alarms(bot, 1)
        corners = teach_corners(bot)
        input("\n[1号 传送带放料点] 把吸盘轻放在任意一块方块的顶面上，按回车后吸盘开启 > ")
        suction(bot, True)
        drop = ask_point(bot, "  带着方块移到传送带上的放料位置，让方块底面贴住传送带，松开解锁键后按回车 > ")
        suction(bot, False)
        print("  吸盘已关闭，方块留在传送带上。")
    finally:
        with suppress(Exception):
            suction(bot, False)
        bot.close()
    setup = load_setup(args.setup)
    setup["robot1"] = {"corners": corners, "belt_drop": drop}
    save_setup(args.setup, setup)
    print(f"\n已保存到 {args.setup}")


def report_headroom(point, r: float, clearance: float = 10.0) -> None:
    """Tell how tall a box wall at a just-taught storage point may be."""
    headroom = top_reachable_z(point[:2], r, point[2]) - 2.0 - point[2]
    wall = headroom - clearance
    print(f"  这个位置上方最多还能抬高 {headroom:.0f} mm，盒壁要低于 {max(wall, 0):.0f} mm"
          f"（从方块底面算起）。")
    if wall < 20.0:
        print("  余量很小：盒壁高的话会撞。可以把盒子往 2号 底座方向挪近一点，再按 r 重新示教。")


def release_suction(number: int) -> None:
    bot = open_robot(number)
    try:
        input(f"用手托住 {ROBOTS[number-1][0]} 吸着的方块，按回车关闭吸盘 > ")
        suction(bot, False)
        print("吸盘已关闭。")
    finally:
        bot.close()


def teach_storage(args) -> None:
    """Re-teach one storage box of robot 2, for example after moving the box."""
    setup = load_setup(args.setup)
    if "robot2" not in setup:
        raise SystemExit("还没示教过 2号，先运行 conveyor_sort teach 2")
    bot = open_robot(2)
    try:
        clear_alarms(bot, 2)
        input(f"\n[2号 {CN[args.only]}色存储盒] 按住解锁键，把吸盘轻放在一块方块顶面上，按回车后吸盘开启 > ")
        suction(bot, True)
        while True:
            point = ask_point(bot, f"  带着方块移到{CN[args.only]}色存储盒的放置位置，让方块底面贴住盒底，"
                                   "松开解锁键后按回车 > ")
            report_headroom(point, r_keeping_j4(point, setup["robot2"]["belt_pick"]))
            if input("  r 重新示教这个点，回车保留 > ").strip().lower() != "r":
                break
        suction(bot, False)
        print("  吸盘已关闭。")
    finally:
        with suppress(Exception):
            suction(bot, False)
        bot.close()
    setup["robot2"]["storage"][args.only] = point
    save_setup(args.setup, setup)
    print(f"\n已保存到 {args.setup}")


def robot1_feed(r1, setup) -> None:
    """Robot 1 puts the block of material cell 1 on the belt, slowly, as a run would."""
    one = setup["robot1"]
    safe = safe_height_robot1(setup, block_mm=25.0, clearance=10.0)
    r, start = one["corners"][0]["pose"][3], r1.pose()[:4]
    move = dict(safe_z=safe, r=r, xy_speed=80.0, z_speed=50.0)
    steps = (pick_steps(start, material_points(one["corners"])[0], grip_ms=400, **move)
             + place_steps(one["belt_drop"], release_ms=400, **move))
    check_path(start, steps)
    r1.start()
    try:
        r1.run(steps)
    except BaseException:
        with suppress(Exception):
            r1.stop_and_clear()
        raise


def belt_block_to_pick(r1, args, setup) -> None:
    """Bring a block to robot 2's pick position the way a run does."""
    answer = input("\n[传送带] 回车 = 1号 从物料区第 1 格取一块放上传送带（第 1 格要有方块），再运到 2号取料位置；"
                   "m = 我已经手动把方块放在 1号放料点；s = 方块已经在取料位置 > ").strip().lower()
    if answer == "s":
        return
    if answer != "m":
        if "robot1" not in setup:
            raise SystemExit("还没示教过 1号，先运行 conveyor_sort teach 1")
        clear_alarms(r1, 1)
        robot1_feed(r1, setup)
    seconds = belt_run(r1, args.belt_speed, args.belt_pulses)
    print(f"  传送带走了 {seconds:.1f} 秒，已停。")


def ask_belt_pick(r2) -> list[float]:
    return ask_point(r2, "\n[2号 传送带取料点] 按住 2号 的解锁键，把吸盘轻放在这块方块顶面中心，"
                         "松开解锁键后按回车 > ")


def teach_pick(args) -> None:
    """Re-teach only robot 2's belt pick point on a block the belt stopped at the image sensor."""
    setup = load_setup(args.setup)
    if "robot2" not in setup:
        raise SystemExit("还没示教过 2号，先运行 conveyor_sort teach 2")
    r1, r2 = open_robot(1), open_robot(2)
    try:
        clear_alarms(r2, 2)
        belt_block_to_pick(r1, args, setup)
        pick = ask_belt_pick(r2)
    finally:
        with suppress(Exception):
            belt(r1, 0)
        r1.close()
        r2.close()
    setup["robot2"]["belt_pick"] = pick
    save_setup(args.setup, setup)
    print(f"\n已保存到 {args.setup}")


def teach_robot2(args) -> None:
    r1, r2 = open_robot(1), open_robot(2)
    try:
        clear_alarms(r2, 2)
        belt_block_to_pick(r1, args, load_setup(args.setup))
        pick = ask_belt_pick(r2)
        suction(r2, True)
        print("  吸盘已开启。")
        storage = {}
        for name in ("red", "green", "blue"):
            while True:
                storage[name] = ask_point(r2, f"\n[2号 {CN[name]}色存储盒] 带着方块移到{CN[name]}色存储盒的放置位置，"
                                              "让方块底面贴住盒底，松开解锁键后按回车 > ")
                report_headroom(storage[name], r_keeping_j4(storage[name], pick))
                if input("  r 重新示教这个点，回车保留 > ").strip().lower() != "r":
                    break
        suction(r2, False)
        print("  吸盘已关闭，方块留在蓝色存储盒。")
    finally:
        with suppress(Exception):
            belt(r1, 0)
        with suppress(Exception):
            suction(r2, False)
        r1.close()
        r2.close()
    setup = load_setup(args.setup)
    setup["robot2"] = {"belt_pick": pick, "storage": storage}
    save_setup(args.setup, setup)
    print(f"\n已保存到 {args.setup}")


# ------------------------------------------------------------------ run

class Robot2Job(threading.Thread):
    """Robot 2: pick the block from the belt and store it; ``lifted`` is set once the belt is free."""

    def __init__(self, bot, plan: Plan, colour: str, stacked: dict, label: str) -> None:
        super().__init__(daemon=True)
        self.bot, self.plan, self.colour, self.stacked, self.label = bot, plan, colour, stacked, label
        self.lifted = threading.Event()
        self.error = None

    def run(self) -> None:
        try:
            colour, count = self.colour, self.stacked[self.colour]
            self.bot.run(self.plan.robot2_pick(self.bot.pose()[:4], colour, count))
            self.lifted.set()
            self.bot.run(self.plan.robot2_store(colour, count))
            self.stacked[colour] += 1
            print(f"  {self.label}：已放入{CN[colour]}色盒（第 {self.stacked[colour]} 块）", flush=True)
        except BaseException as exc:
            self.error = exc
            self.lifted.set()

    def wait_lifted(self) -> None:
        self.lifted.wait()
        if self.error:
            raise self.error

    def finish(self) -> None:
        self.join()
        if self.error:
            raise self.error


def classify(r1, args, key: str, label: str, tag: str) -> str:
    """Colour of the block now on the belt, from Gemini, or typed in when Gemini is not sure."""
    while True:
        try:
            view = belt_vision.look(model=args.model, key=key, timeout_s=args.vision_timeout, label=tag)
        except Exception as exc:  # network, quota, camera: fall back to the person at the bench
            view, problem = None, f"出错（{exc}）"
        else:
            if view.colour:
                print(f"  {label}：Gemini 识别为{CN[view.colour]}色（{view.seconds:.1f} 秒）", flush=True)
                return view.colour
            problem = "颜色拿不准" if view.present else "没看到传送带上的方块"
            problem += f"（图片 {view.image}）"
        answer = input(f"  {label}：Gemini {problem}。回车重拍，c 传送带再走一点，"
                       "输入 红/绿/蓝 手动指定，q 停止 > ").strip().lower()
        if answer == "q":
            raise RuntimeError("已在识别这一步停止")
        if answer == "c":
            belt_run(r1, args.belt_speed, args.nudge_pulses)
        elif answer in TYPED:
            return TYPED[answer]


def vision_test(args) -> None:
    key = belt_vision.api_key()
    models = belt_vision.COMPARE_MODELS if args.compare else (args.model,)
    print(f"[视觉测试] 模型：{'、'.join(models)}；图片和回答存到 {belt_vision.DEFAULT_DIR}/")
    while input("\n把方块放到传送带上（或拿走），按回车拍照让 Gemini 识别（q 退出）> ").strip().lower() != "q":
        try:
            image, path = belt_vision.snapshot(label="test")
        except RuntimeError as exc:
            print(f"  拍照失败：{exc}")
            continue
        print(f"  图片 {path}")
        for model in models:
            try:
                view = belt_vision.ask(image, path, model=model, key=key, timeout_s=args.vision_timeout)
            except Exception as exc:
                print(f"  {model}：出错 {str(exc)[:160]}")
                continue
            verdict = ((f"有方块，{CN[view.colour]}色" if view.colour else "有方块，颜色拿不准")
                       if view.present else "没看到方块")
            print(f"  {model}：{verdict}（{view.seconds:.1f} 秒）")


def run(args) -> int:
    setup = load_setup(args.setup)
    missing = [key for key in ("robot1", "robot2") if key not in setup]
    if missing:
        raise SystemExit(f"还没示教：{', '.join(missing)}。先运行 conveyor_sort teach 1 / teach 2")
    if not (1 <= args.start and args.start + args.count - 1 <= SIZE * SIZE and args.count >= 1):
        raise SystemExit(f"--start/--count 超出物料区的 {SIZE * SIZE} 格")
    plan = Plan(setup, args)
    print(f"共 {args.count} 块（物料区第 {args.start}–{args.start + args.count - 1} 格，按行编号）；"
          f"1号移动高度 z={plan.safe1:.1f}，2号按每段需要抬升")
    if args.dry_run:
        start1 = (*plan.material[0][:2], plan.safe1, plan.r1)
        start2 = (*plan.belt_pick[:2], plan.pick_level, plan.r2)
        try:
            plan.check(start1, start2, args.per_colour)
        except ValueError as exc:
            raise SystemExit(f"路径超出关节限位：{exc}") from exc
        print("空跑检查通过：所有路径都在关节限位内。")
        return 0

    key = belt_vision.api_key()
    r1, r2 = open_robot(1, args.accel), open_robot(2, args.accel)
    job = None
    try:
        clear_alarms(r1, 1)
        clear_alarms(r2, 2)
        for number, bot in ((1, r1), (2, r2)):
            if bot.send(62)[1:2] == b"\x01":
                raise SystemExit(f"{ROBOTS[number-1][0]} 的吸盘还开着，可能还吸着方块。"
                                 f"先运行 ros2 run mani_dobot conveyor_sort release {number}")
        try:
            plan.check(r1.pose()[:4], r2.pose()[:4], args.per_colour)
        except ValueError as exc:
            raise SystemExit(f"路径超出关节限位：{exc}") from exc
        try:
            belt_vision.grab_rgb(timeout_s=10.0)
        except RuntimeError as exc:
            raise SystemExit("收不到 D435i 画面，先启动相机：ros2 launch mani_perception d435i.launch.py "
                             "color_profile:=640x480x15 depth_profile:=640x480x15") from exc
        if input("请确认：物料区放好方块、传送带和存储区已清空。按回车开始（q 退出）> ").strip().lower() == "q":
            return 0
        r1.start()
        r2.start()
        joint = (args.joint_speed,) * 4
        r2.send(80, struct.pack("<8f", *joint, *joint), write=True)  # MOVJ joint speed/accel
        started = time.monotonic()
        stacked = collections.Counter()
        r1.run(plan.robot1_cycle(r1.pose()[:4], 0))
        for index in range(args.count):
            label = f"第 {index + 1} 块"
            if job and job.error:
                raise job.error
            seconds = belt_run(r1, args.belt_speed, args.belt_pulses)
            print(f"  {label}：传送带 {seconds:.1f} 秒到达取料位置", flush=True)
            colour = classify(r1, args, key, label, f"block{index + 1}")
            if job:
                job.finish()  # robot 2 has stored the previous block
            if stacked[colour] >= args.per_colour:
                raise RuntimeError(f"{CN[colour]}色盒已放满 {args.per_colour} 块")
            job = Robot2Job(r2, plan, colour, stacked, label)
            job.start()
            job.wait_lifted()
            if index + 1 < args.count:
                r1.run(plan.robot1_cycle(r1.pose()[:4], index + 1))  # while robot 2 sorts
        job.finish()
        summary = "，".join(f"{CN[c]} {stacked[c]}" for c in ("red", "green", "blue"))
        print(f"完成 {args.count} 块，用时 {time.monotonic() - started:.1f} 秒（{summary}）")
        return 0
    except BaseException as exc:
        with suppress(Exception):
            belt(r1, 0)
        for bot in (r1, r2):
            with suppress(Exception):
                bot.stop_and_clear()
        print("\n已停止传送带和两台机械臂，吸盘保持原状。要取下吸着的方块：ros2 run mani_dobot conveyor_sort release 1 或 2")
        if isinstance(exc, KeyboardInterrupt):
            return 130
        raise
    finally:
        r1.close()
        r2.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP, help="示教结果文件")
    belt_options = argparse.ArgumentParser(add_help=False)
    belt_options.add_argument("--belt-speed", type=int, default=BELT_FORWARD_SPEED, help="传送带速度（脉冲/秒）")
    belt_options.add_argument("--belt-pulses", type=int, default=BELT_PULSES,
                              help="放料点到 2号取料位置的传送带步数（示教和运行要用同一个值）")
    belt_options.add_argument("--nudge-pulses", type=int, default=1500, help="识别不到时传送带再走的步数")
    vision_options = argparse.ArgumentParser(add_help=False)
    vision_options.add_argument("--model", default=belt_vision.DEFAULT_MODEL, help="Gemini 模型")
    vision_options.add_argument("--vision-timeout", type=float, default=30.0, help="Gemini 单次请求超时（秒）")
    sub = parser.add_subparsers(dest="command", required=True)
    teach = sub.add_parser("teach", parents=[belt_options], help="示教：1 = 物料区和放料点，2 = 取料点和存储盒")
    teach.add_argument("robot", choices=("1", "2"))
    teach.add_argument("--only", choices=("material", "pick", "red", "green", "blue"),
                       help="只重新示教一项：1号 material = 物料区四角；2号 pick = 传送带取料点，"
                            "red/green/blue = 存储盒")
    release = sub.add_parser("release", help="托住方块后关闭吸盘")
    release.add_argument("robot", choices=("1", "2"))
    go = argparse.ArgumentParser(add_help=False)  # run options
    go.add_argument("--count", type=int, default=SIZE * SIZE, help="搬几块")
    go.add_argument("--start", type=int, default=1, help="从物料区第几格开始（按行编号 1–9）")
    go.add_argument("--speed", type=float, default=150.0, help="水平速度 mm/s")
    go.add_argument("--z-speed", type=float, default=80.0, help="升降速度 mm/s")
    go.add_argument("--accel", type=float, default=150.0, help="加速度 mm/s^2")
    go.add_argument("--joint-speed", type=float, default=100.0, help="2号关节插补的速度和加速度 °/s")
    go.add_argument("--grip-ms", type=int, default=400, help="吸盘开启后等待毫秒")
    go.add_argument("--release-ms", type=int, default=400, help="吸盘关闭后等待毫秒")
    go.add_argument("--pick-depth", type=float, default=0.0, help="吸取时比示教点再往下压的毫米数")
    go.add_argument("--block-mm", type=float, default=25.0, help="方块高度 mm")
    go.add_argument("--box-mm", type=float, nargs="+", default=[30.0, 30.0, 10.0], metavar="MM",
                    help="存储盒的盒壁比示教时方块底面高出多少毫米，按 红 绿 蓝 给三个数，或一个数用于全部；"
                         "2号先抬过盒壁再横移")
    go.add_argument("--stack-mm", type=float, default=0.0, help="同色方块每块叠高的毫米数；存储区是盒子时为 0")
    go.add_argument("--per-colour", type=int, default=SIZE * SIZE, help="每个存储区最多放几块")
    go.add_argument("--clearance", type=float, default=10.0, help="移动时方块底面比最高障碍再高的毫米数")
    run_parser = sub.add_parser("run", parents=[belt_options, vision_options, go], help="运行分拣")
    run_parser.add_argument("--dry-run", action="store_true", help="只检查路径是否在关节限位内，不连机器")
    vision = sub.add_parser("vision", parents=[vision_options], help="视觉测试：拍传送带让 Gemini 识别")
    vision.add_argument("--compare", action="store_true",
                        help=f"同一张照片同时问 {' 和 '.join(belt_vision.COMPARE_MODELS)}，比较准确度和速度")
    args = parser.parse_args(argv)
    if args.command == "vision":
        vision_test(args)
        return 0
    if args.command == "release":
        release_suction(int(args.robot))
        return 0
    if args.command == "teach":
        if args.only and (args.only == "material") != (args.robot == "1"):
            raise SystemExit("1号 只能用 --only material；2号 用 --only pick/red/green/blue")
        if args.only == "material":
            teach_material(args)
        elif args.only == "pick":
            teach_pick(args)
        else:
            (teach_storage if args.only else teach_robot1 if args.robot == "1" else teach_robot2)(args)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
