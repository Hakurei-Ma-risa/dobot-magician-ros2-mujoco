"""Manual homing for a Dobot Magician, prompts in Chinese.

After power-up the arm droops past its joint limits (alarms 0x43/0x44/0x49)
and the base angle J1 is unknown (incremental encoder). Step 1 shows J2/J3
live while the user lifts the arm by hand with the forearm unlock button, and
clears the alarms once the posture is legal. Step 2 runs the controller's
homing after Enter and ends at a raised pose. Taught positions recorded
before homing are in a different J1 frame and must be re-taught.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import time

from .fast_pick_place import Dobot
from .sensor_check import ROBOTS


HOME = (200.0, 0.0, 80.0, 0.0)  # about 145 mm above the chip tray, inside the joint limits
ALARM_NAMES = {
    0x01: "未定义指令（清不掉，需断电重启）",
    0x40: "J1 正向限位", 0x41: "J1 负向限位",
    0x42: "J2 正向限位（大臂太往前）", 0x43: "J2 负向限位（大臂太往后）",
    0x44: "J3 正向限位（小臂太往下）", 0x45: "J3 负向限位（小臂太往上）",
    0x48: "平行四边形正向限位", 0x49: "平行四边形负向限位（小臂相对大臂太低）",
}


def describe(codes) -> str:
    return "、".join(ALARM_NAMES.get(code, hex(code)) for code in codes) or "无"


def wait_for_legal_posture(bot) -> bool:
    print("第 1 步：如果手臂垂着，按住小臂上的解锁键，用手把大臂往前推、小臂抬起来"
          "（大约 J2≈10°、J3≈30°；允许 J2 -5~90、J3 -15~95），然后松开解锁键。\n"
          "  姿态合法后会自动清除报警（Ctrl+C 退出）。")
    shown = None
    while True:
        bot.clear_alarms()
        time.sleep(0.2)
        codes = bot.alarms()
        joints = bot.pose()[4:]
        state = (round(joints[1]), round(joints[2]), tuple(codes))
        if state != shown:  # a new line only when something changed; \r breaks on wrapped lines
            print(f"  J2 {joints[1]:5.1f}°  J3 {joints[2]:5.1f}°  报警：{describe(codes)}", flush=True)
            shown = state
        if 0x01 in codes:
            print("  有“未定义指令”报警，请给这台断电重启后再运行本程序。")
            return False
        if not codes:
            print("  报警已清除，姿态正常。\n")
            return True
        time.sleep(0.3)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("robot", choices=("1", "2"), help="回零哪一台：1 或 2")
    args = parser.parse_args(argv)
    name, path = ROBOTS[int(args.robot) - 1]

    bot = Dobot(path, 150)
    try:
        print(f"{name} 手动回零")
        if not wait_for_legal_posture(bot):
            return 1
        if input(f"第 2 步：回零。底座会先转动找限位开关，然后手臂移到 "
                 f"({HOME[0]:g}, {HOME[1]:g}, {HOME[2]:g})。\n"
                 "  请清空手臂周围，按回车开始（q 退出）> ").strip().lower() == "q":
            return 0
        print("  回零中…")
        started = time.monotonic()
        bot.start()
        bot.home(HOME)
        pose = bot.pose()
        print(f"  回零完成，用时 {time.monotonic() - started:.0f} 秒。"
              f"位姿 {[round(v, 1) for v in pose[:4]]}，关节 {[round(v, 1) for v in pose[4:]]}")
        print("  回零前示教的点位（芯片盘四角、A/B/C/D）都要重新示教。")
        return 0
    except BaseException as exc:
        with suppress(Exception):
            bot.stop_and_clear()
        print("\n已停止并清空运动队列。")
        if isinstance(exc, KeyboardInterrupt):
            return 130
        raise
    finally:
        bot.close()


if __name__ == "__main__":
    raise SystemExit(main())
