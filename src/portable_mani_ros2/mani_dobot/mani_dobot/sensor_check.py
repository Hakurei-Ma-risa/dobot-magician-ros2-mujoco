"""Manual sensor check for two Dobot Magicians.

The user triggers one sensor at a time and presses Enter; every input of both
robots is sampled and compared with an idle baseline, so the result also tells
which robot a sensor is plugged into. Read on each robot: the GP2 colour
sensor, GP4 as an IR / switch input, and the digital level of EIO1-20. The arm
never moves. The optional last step runs a conveyor on Stepper1 for a few
seconds, only after Enter.
"""

from __future__ import annotations

import argparse
import collections
from contextlib import suppress
import struct
import time

from .fast_pick_place import Dobot


ROBOTS = (
    ("1号", "/dev/serial/by-id/usb-1a86_USB_Single_Serial_58EA056287-if00"),
    ("2号", "/dev/serial/by-id/usb-1a86_USB_Single_Serial_598C000140-if00"),
)
# Sensor port numbers as firmware 4.0.5 accepts them: 0 GP1, 1 GP2, 3 GP4, 4 GP5
# (2 and 5 are rejected).
GP2, GP4 = 1, 3
COLOUR, IR, DI, EMOTOR, IO_MUX = 137, 138, 133, 135, 130
GP4_SIGNAL_EIO, DI_PULL_UP = 5, 5  # GP4 signal pin and the IO function 'input with pull-up'
# Measured with this tool on 2026-10-17:
#   1号 GP4 image sensor -> EIO5 level, 1 idle, 0 blocked; Stepper1 drives the conveyor
#   2号 GP4 pressure switch -> EIO5 level, 0 idle, 1 pressed; GP2 colour sensor (version 1)
BELT_FORWARD_SPEED = -5000  # pulses/s; a positive speed runs this conveyor backwards
EIO = range(1, 21)
COLOUR_NAMES = {(0, 0, 0): "无", (1, 0, 0): "红", (0, 1, 0): "绿", (0, 0, 1): "蓝"}
ACTIONS = (
    ("图像传感器", "请遮挡 图像传感器 并保持住"),
    ("压感", "请移开手，然后按住 压感 并保持住"),
    ("红色块", "请松开压感，把 红色块 放到颜色传感器下面"),
    ("绿色块", "换成 绿色块"),
    ("蓝色块", "换成 蓝色块"),
)


def read_all(bots) -> dict[str, object]:
    values = {}
    for name, bot in bots:
        colour = tuple(bot.send(COLOUR)[:3])
        values[f"{name} GP2 颜色"] = COLOUR_NAMES.get(colour, str(colour))
        values[f"{name} GP4 红外/开关"] = bot.send(IR, bytes((GP4,)))[:1].hex() or "-"
        for addr in EIO:
            values[f"{name} EIO{addr} 电平"] = bot.send(DI, bytes((addr, 0)))[1:2].hex() or "-"
    return values


def sample(bots, seconds: float) -> dict[str, object]:
    """Most frequent value of every channel over ``seconds``."""
    counts = collections.defaultdict(collections.Counter)
    end = time.monotonic() + seconds
    while True:
        for key, value in read_all(bots).items():
            counts[key][value] += 1
        if time.monotonic() >= end:
            return {key: c.most_common(1)[0][0] for key, c in counts.items()}


def changes(baseline: dict, snapshot: dict) -> list[str]:
    return [f"{key}: {baseline[key]} → {value}" for key, value in snapshot.items()
            if key in baseline and value != baseline[key]]


def ask(prompt: str) -> str:
    return input(f"{prompt}\n  按回车开始采样（s 跳过，q 退出）> ").strip().lower()


def run_belt(bot, speed: int, seconds: float) -> None:
    bot.send(EMOTOR, struct.pack("<BBi", 0, 1, speed), write=True)  # Stepper1 on
    try:
        time.sleep(seconds)
    finally:
        bot.send(EMOTOR, struct.pack("<BBi", 0, 0, 0), write=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensor-version", type=int, choices=(0, 1), default=1,
                        help="传感器版本参数：1 = V2.0，0 = V1.0")
    parser.add_argument("--sample-s", type=float, default=1.5, help="每一步的采样时长（秒）")
    parser.add_argument("--belt-speed", type=int, default=BELT_FORWARD_SPEED,
                        help="传送带速度（脉冲/秒）；默认值是正向，改变正负号则反向")
    parser.add_argument("--belt-s", type=float, default=2.0, help="传送带测试时长（秒）")
    parser.add_argument("--belt-only", action="store_true", help="只测传送带，跳过传感器")
    args = parser.parse_args(argv)

    print("Dobot 传感器手动检测（机械臂全程不动）")
    bots = []
    try:
        for name, path in ROBOTS:
            try:
                bot = Dobot(path, 150)
                serial_no = bot.send(0).rstrip(b"\0").decode(errors="replace")
            except Exception as exc:
                print(f"  {name}: 连接失败（{exc}），本次跳过")
                continue
            bots.append((name, bot))
            print(f"  {name}: 序列号 {serial_no}，端口 {path}")
        if not bots:
            print("没有可用的 Dobot。")
            return 1
        results = {}
        if not args.belt_only:
            for _, bot in bots:
                bot.send(COLOUR, bytes((1, GP2, args.sensor_version)), write=True)
                bot.send(IR, bytes((1, GP4, args.sensor_version)), write=True)
                # the image sensor reads 0 forever without the pull-up (seen after a power cycle)
                bot.send(IO_MUX, bytes((GP4_SIGNAL_EIO, DI_PULL_UP)), write=True)
            time.sleep(0.2)
            print("已开启：每台的 GP2 颜色传感器、GP4 红外/开关输入。\n")
            if input("[基准] 请让所有传感器保持空闲：不遮挡、不按压、颜色传感器下面不放东西。\n"
                     "  按回车开始采样（q 退出）> ").strip().lower() == "q":
                return 0
            baseline = sample(bots, args.sample_s)
            print("  已记录基准。\n")

        for index, (label, prompt) in enumerate(() if args.belt_only else ACTIONS, 1):
            while True:
                answer = ask(f"[{index}/{len(ACTIONS)}] {prompt}")
                if answer == "q":
                    return 0
                if answer == "s":
                    results[label] = None
                    break
                found = changes(baseline, sample(bots, args.sample_s))
                print("  有变化：" if found else "  没有检测到变化。", *found, sep="\n    ")
                retry = input("  r 重测这一步，回车继续下一步 > ").strip().lower()
                if retry != "r":
                    results[label] = found
                    break
            print()

        names = [name for name, _ in bots]
        answer = input(f"[传送带] 先把带上的东西拿开。按回车让 {names[0]} 的 Stepper1 转 {args.belt_s:g} 秒"
                       f"{'，输入 2 改用 ' + names[1] if len(bots) > 1 else ''}，s 跳过 > ").strip().lower()
        if answer != "s":
            name, bot = bots[1] if answer == "2" and len(bots) > 1 else bots[0]
            print(f"  {name} Stepper1 转动中…")
            run_belt(bot, args.belt_speed, args.belt_s)
            moved = input("  传送带动了吗？(y/n) > ").strip().lower() == "y"
            results["传送带"] = [f"{name} Stepper1：{'动了' if moved else '没动'}"]

        print("\n== 汇总")
        for label, found in results.items():
            if found is None:
                print(f"  {label}：已跳过")
            elif found:
                print(f"  {label}：", *found, sep="\n    ")
            else:
                print(f"  {label}：没有检测到变化")
        return 0
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
    finally:
        for _, bot in bots:
            for cid, params in ((EMOTOR, struct.pack("<BBi", 0, 0, 0)),
                                (COLOUR, bytes((0, GP2, args.sensor_version))),
                                (IR, bytes((0, GP4, args.sensor_version)))):
                with suppress(Exception):
                    bot.send(cid, params, write=True)
            bot.close()


if __name__ == "__main__":
    raise SystemExit(main())
