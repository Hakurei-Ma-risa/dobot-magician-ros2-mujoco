"""Fast taught-waypoint suction pick-and-place for the Dobot Magician.

A whole A->target cycle (MOVL up / across / down, suction, dwell) is queued on
the controller at once, so the moves run back to back without host round
trips. The serial port is used directly with the same inter-process lock as
``dobot_driver.interface.Interface``, so this runs next to the ROS state
publisher. No ROS import.

Coordinates are Dobot controller XYZ in mm and R in degrees, as taught in
``task_waypoints.json``. R is held at the taught pick_A value for every move,
so the block does not rotate while carried.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, suppress
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import threading
import time


DEFAULT_PORT = os.environ.get(
    "DOBOT_PORT", "/dev/serial/by-id/usb-1a86_USB_Single_Serial_58EA056287-if00"
)
DEFAULT_WAYPOINTS = Path("reports/calibration/2026-10-17/task_waypoints.json")
TAUGHT_TOOL_BIAS_MM = (59.7, 0.0, 0.0)
MOVJ_XYZ, MOVL_XYZ = 1, 2
MAX_PENDING = 8  # unexecuted commands kept in the controller queue


def packet(command_id: int, params: bytes = b"", *, write=False, queued=False) -> bytes:
    payload = bytes((command_id, int(write) | int(queued) << 1)) + params
    return b"\xaa\xaa" + bytes((len(payload),)) + payload + bytes(((-sum(payload)) & 0xFF,))


def alarm_codes(state: bytes) -> list[int]:
    return [8*i + bit for i, byte in enumerate(state) for bit in range(8) if byte >> bit & 1]


def load_waypoints(path) -> dict[str, tuple[float, float, float, float]]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    points = {}
    for name in ("pick_A", "place_B", "place_C", "place_D"):
        x, y, z = data[name]["controller_xyz_mm"]
        points[name.split("_")[1]] = (float(x), float(y), float(z), float(data[name]["controller_yaw_deg"]))
    return points


def build_cycle(start, pick, place, *, safe_z, xy_speed, z_speed, grip_ms, release_ms) -> list[tuple]:
    """Steps for one start -> pick -> place cycle, ending above ``place``.

    Steps are ("speed", mm/s), ("movl", (x, y, z, r)), ("suction", bool) and
    ("wait", ms). R stays at the pick's R throughout.
    """
    r = pick[3]
    above_pick, above_place = (*pick[:2], safe_z, r), (*place[:2], safe_z, r)
    steps = []
    if start[2] < safe_z - 1.0:
        steps += [("speed", z_speed), ("movl", (*start[:2], safe_z, r))]
    steps += [
        ("speed", xy_speed), ("movl", above_pick),
        ("speed", z_speed), ("movl", (*pick[:3], r)),
        ("suction", True), ("wait", grip_ms),
        ("movl", above_pick),
        ("speed", xy_speed), ("movl", above_place),
        ("speed", z_speed), ("movl", (*place[:3], r)),
        ("suction", False), ("wait", release_ms),
        ("movl", above_place),
    ]
    return steps


class Dobot:
    def __init__(self, port: str, accel: float) -> None:
        import serial

        lock_id = hashlib.sha256(os.path.realpath(port).encode("utf-8")).hexdigest()[:16]
        self._lock_fd = os.open(f"/tmp/dobot-magician-{lock_id}.lock", os.O_CREAT | os.O_RDWR, 0o600)
        self.serial = serial.Serial(port, 115200, timeout=0.25, write_timeout=0.25)
        self.accel = accel
        self.speed = None
        self.last_index = 0
        self.executed = 0
        self._thread_lock = threading.Lock()  # flock does not exclude threads sharing the fd

    def close(self) -> None:
        self.serial.close()
        os.close(self._lock_fd)

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def send(self, command_id: int, params: bytes = b"", *, write=False, queued=False) -> bytes:
        request = packet(command_id, params, write=write, queued=queued)
        with self._locked():
            self.serial.reset_input_buffer()
            self.serial.write(request)
            self.serial.flush()
            for _ in range(3):  # skip a stale reply another client left behind
                header = self.serial.read(3)
                if len(header) != 3 or header[:2] != b"\xaa\xaa":
                    break
                body = self.serial.read(header[2] + 1)
                if len(body) == header[2] + 1 and body[0] == command_id and not sum(body) & 0xFF:
                    return body[2:-1]
        raise RuntimeError(f"No valid reply to Dobot command {command_id}")

    def pose(self) -> tuple[float, ...]:
        return struct.unpack("<8f", self.send(10))

    def alarms(self) -> list[int]:
        return alarm_codes(self.send(20))

    def clear_alarms(self) -> None:
        self.send(21, write=True)

    def tool_bias(self) -> tuple[float, ...]:
        return struct.unpack("<3f", self.send(60))

    def queue_index(self) -> int:
        return struct.unpack("<Q", self.send(246))[0]

    def stop_and_clear(self) -> None:
        self.send(242, write=True)  # force stop
        self.send(245, write=True)  # clear queue

    def start(self) -> None:
        self.stop_and_clear()
        self.send(83, struct.pack("<2f", 100.0, 100.0), write=True)  # common ratios 100 %
        self.send(240, write=True)
        self.speed = None

    def home(self, pose, timeout_s: float = 120.0) -> None:
        """Controller homing (J1 from its home switch), ending at ``pose``. Call start() first."""
        self.send(30, struct.pack("<4f", *pose), write=True)
        self._queue(31, struct.pack("<I", 0))
        self._wait_index(self.last_index, stall_s=timeout_s)
        self._wait_pose(pose, tolerance_mm=2.0, timeout_s=10.0)

    def _queue(self, command_id: int, params: bytes) -> None:
        if self.last_index - self.executed >= MAX_PENDING:
            self._wait_index(self.last_index - MAX_PENDING + 1)
        reply = self.send(command_id, params, write=True, queued=True)
        self.last_index = struct.unpack("<Q", reply)[0]

    def _wait_index(self, target: int, stall_s: float = 20.0) -> None:
        last, progressed, checked = None, time.monotonic(), 0.0
        while True:
            current = self.executed = self.queue_index()
            if current >= target:
                return
            now = time.monotonic()
            if current != last:
                last, progressed = current, now
            elif now - progressed > stall_s:
                raise RuntimeError(f"Dobot queue stalled at index {current}")
            if now - checked > 0.25:
                checked = now
                codes = self.alarms()
                if codes:
                    raise RuntimeError(f"Dobot alarm {[hex(c) for c in codes]}")
            time.sleep(0.02)

    def run(self, steps) -> None:
        final = None
        for kind, value in steps:
            if kind == "speed":
                if value != self.speed:
                    self._queue(81, struct.pack("<4f", value, value, self.accel, self.accel))
                    self.speed = value
            elif kind in ("movl", "movj"):  # movj: joint-interpolated, path not straight
                self._queue(84, struct.pack("<B4f", MOVL_XYZ if kind == "movl" else MOVJ_XYZ, *value))
                final = value
            elif kind == "suction":
                self._queue(62, struct.pack("<BB", value, value))
            elif kind == "wait":
                self._queue(110, struct.pack("<I", int(value)))
        self._queue(110, struct.pack("<I", 0))  # end marker: everything before it is done
        self._wait_index(self.last_index)
        if final is not None:
            self._wait_pose(final)

    def _wait_pose(self, target, tolerance_mm: float = 1.0, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if math.dist(self.pose()[:3], target[:3]) <= tolerance_mm:
                return
            time.sleep(0.02)
        raise RuntimeError(f"Dobot did not reach {target[:3]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", choices=("B", "C", "D"), default=["B", "C", "D"],
                        help="A->target cycles in order; Challenge 1 = --targets B")
    parser.add_argument("--waypoints", type=Path, default=DEFAULT_WAYPOINTS)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--safe-z", type=float, default=-10.0, help="transfer height, controller Z mm")
    parser.add_argument("--speed", type=float, default=150.0, help="horizontal speed mm/s")
    parser.add_argument("--z-speed", type=float, default=80.0, help="lift/descent speed mm/s")
    parser.add_argument("--accel", type=float, default=150.0, help="acceleration mm/s^2")
    parser.add_argument("--grip-ms", type=int, default=400, help="dwell after suction on")
    parser.add_argument("--release-ms", type=int, default=400, help="dwell after suction off")
    parser.add_argument("--refill-delay", type=float, default=0.0,
                        help="seconds to wait for the next block at A; 0 = press Enter")
    parser.add_argument("--dry-run", action="store_true", help="print the queued steps, no robot")
    parser.add_argument("--check", action="store_true", help="read pose/alarms/tool bias only")
    args = parser.parse_args(argv)

    points = load_waypoints(args.waypoints)
    motion = dict(safe_z=args.safe_z, xy_speed=args.speed, z_speed=args.z_speed,
                  grip_ms=args.grip_ms, release_ms=args.release_ms)
    if args.dry_run:
        start = (*points["A"][:2], args.safe_z)
        for name in args.targets:
            print(f"== A -> {name}")
            for step in build_cycle(start, points["A"], points[name], **motion):
                print("  ", *step)
            start = (*points[name][:2], args.safe_z)
        return 0

    bot = Dobot(args.port, args.accel)
    try:
        bias = bot.tool_bias()
        if math.dist(bias, TAUGHT_TOOL_BIAS_MM) > 0.5:
            raise SystemExit(f"Tool bias {bias} != taught {TAUGHT_TOOL_BIAS_MM}; waypoints would be wrong")
        codes = bot.alarms()
        if codes and not args.check:
            print(f"Clearing latched alarms {[hex(c) for c in codes]}")
            bot.clear_alarms()
            codes = bot.alarms()
            if codes:
                raise SystemExit(f"Alarms persist after clearing: {[hex(c) for c in codes]}")
        if args.check:
            print(json.dumps({"pose": bot.pose(), "alarms": [hex(c) for c in codes],
                              "tool_bias_mm": bias, "queue_index": bot.queue_index()}))
            bot.close()
            return 0
    except BaseException:
        bot.close()
        raise

    try:
        bot.start()
        started = time.monotonic()
        for index, name in enumerate(args.targets):
            if index:
                if args.refill_delay > 0:
                    print(f"Next block at A in {args.refill_delay:.1f} s ...", flush=True)
                    time.sleep(args.refill_delay)
                elif input("Put the next block at A, then Enter (q = quit): ").strip().lower() == "q":
                    break
            cycle_start = time.monotonic()
            pose = bot.pose()
            bot.run(build_cycle(pose[:3], points["A"], points[name], **motion))
            print(f"A -> {name} done in {time.monotonic() - cycle_start:.1f} s", flush=True)
        print(f"Finished in {time.monotonic() - started:.1f} s")
        return 0
    except BaseException as exc:
        with suppress(Exception):
            bot.stop_and_clear()
        print("Stopped and cleared the motion queue; suction left as is.")
        if isinstance(exc, KeyboardInterrupt):
            return 130
        raise
    finally:
        bot.close()


if __name__ == "__main__":
    raise SystemExit(main())
