"""Finite-duration health check for the Portable Mani RGB-D contract."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from threading import Thread
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class StreamStats:
    arrivals: list[float] = field(default_factory=list)
    source_stamps: list[float] = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    encoding: str | None = None

    def add(self, message: Image, arrival: float) -> None:
        self.arrivals.append(arrival)
        self.source_stamps.append(
            message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        )
        self.width = message.width
        self.height = message.height
        self.encoding = message.encoding

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "frames": len(self.arrivals),
            "width": self.width,
            "height": self.height,
            "encoding": self.encoding,
        }
        if len(self.arrivals) < 2:
            result.update(receive_hz=0.0, source_hz=0.0, max_gap_sec=None)
            return result

        gaps = [b - a for a, b in zip(self.arrivals, self.arrivals[1:])]
        arrival_span = self.arrivals[-1] - self.arrivals[0]
        stamp_span = self.source_stamps[-1] - self.source_stamps[0]
        result.update(
            receive_hz=round((len(self.arrivals) - 1) / arrival_span, 2),
            source_hz=round((len(self.source_stamps) - 1) / stamp_span, 2)
            if stamp_span > 0.0
            else 0.0,
            max_gap_sec=round(max(gaps), 3),
        )
        return result


class CameraHealthcheck(Node):
    def __init__(self) -> None:
        super().__init__("camera_healthcheck")
        self.declare_parameter("duration_sec", 10.0)
        self.declare_parameter("min_color_hz", 8.0)
        self.declare_parameter("min_depth_hz", 8.0)
        self.declare_parameter("max_gap_sec", 0.75)
        self.declare_parameter("require_d435_usb", True)
        self.declare_parameter(
            "color_topic", "/camera/camera/color/image_raw"
        )
        self.declare_parameter(
            "depth_topic", "/camera/camera/depth/image_rect_raw"
        )
        self.declare_parameter(
            "aligned_depth_topic",
            "/camera/camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info"
        )

        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.streams = {
            "color": StreamStats(),
            "depth_raw": StreamStats(),
            "depth_aligned": StreamStats(),
        }
        self.camera_info: CameraInfo | None = None
        # Keep explicit references so the subscriptions live for the full probe.
        # ``Node.subscriptions`` is a read-only rclpy property.
        self._subscription_handles = [
            self.create_subscription(
                Image,
                str(self.get_parameter("color_topic").value),
                lambda message: self._on_image("color", message),
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                Image,
                str(self.get_parameter("depth_topic").value),
                lambda message: self._on_image("depth_raw", message),
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                Image,
                str(self.get_parameter("aligned_depth_topic").value),
                lambda message: self._on_image("depth_aligned", message),
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                CameraInfo,
                str(self.get_parameter("camera_info_topic").value),
                self._on_camera_info,
                qos_profile_sensor_data,
            ),
        ]

    def _on_image(self, name: str, message: Image) -> None:
        self.streams[name].add(message, time.monotonic())

    def _on_camera_info(self, message: CameraInfo) -> None:
        self.camera_info = message

    def report(self) -> tuple[dict[str, Any], bool]:
        streams = {name: stats.summary() for name, stats in self.streams.items()}
        min_color_hz = float(self.get_parameter("min_color_hz").value)
        min_depth_hz = float(self.get_parameter("min_depth_hz").value)
        max_gap_sec = float(self.get_parameter("max_gap_sec").value)

        failures: list[str] = []
        for name, minimum in (
            ("color", min_color_hz),
            ("depth_raw", min_depth_hz),
            ("depth_aligned", min_depth_hz),
        ):
            stream = streams[name]
            if stream["receive_hz"] < minimum:
                failures.append(
                    f"{name} receive_hz {stream['receive_hz']} < {minimum}"
                )
            if stream["max_gap_sec"] is None or stream["max_gap_sec"] > max_gap_sec:
                failures.append(
                    f"{name} max_gap_sec {stream['max_gap_sec']} > {max_gap_sec}"
                )

        if self.camera_info is None:
            intrinsics = None
            failures.append("no color CameraInfo received")
        else:
            intrinsics = {
                "width": self.camera_info.width,
                "height": self.camera_info.height,
                "fx": round(self.camera_info.k[0], 4),
                "fy": round(self.camera_info.k[4], 4),
                "cx": round(self.camera_info.k[2], 4),
                "cy": round(self.camera_info.k[5], 4),
                "frame_id": self.camera_info.header.frame_id,
            }

        usb = find_d435i_usb_state()
        require_d435_usb = bool(self.get_parameter("require_d435_usb").value)
        if require_d435_usb and usb is None:
            failures.append("D435i USB device 8086:0b3a not found")
        elif require_d435_usb and usb is not None:
            if usb.get("speed_mbps", 0) < 5000:
                failures.append(f"D435i USB speed is only {usb.get('speed_mbps')} Mbps")
            if usb.get("power_control") != "on":
                failures.append("D435i USB power/control is not 'on'")

        report = {
            "passed": not failures,
            "duration_sec": self.duration_sec,
            "streams": streams,
            "color_intrinsics": intrinsics,
            "usb": usb,
            "failures": failures,
        }
        return report, not failures


def find_d435i_usb_state() -> dict[str, Any] | None:
    for device in Path("/sys/bus/usb/devices").glob("*"):
        try:
            vendor = (device / "idVendor").read_text().strip()
            product = (device / "idProduct").read_text().strip()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if (vendor, product) != ("8086", "0b3a"):
            continue
        try:
            speed = int(float((device / "speed").read_text().strip()))
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            speed = 0
        try:
            power_control = (device / "power/control").read_text().strip()
        except (FileNotFoundError, PermissionError, OSError):
            power_control = "unknown"
        return {
            "sysfs_path": str(device),
            "speed_mbps": speed,
            "power_control": power_control,
        }
    return None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CameraHealthcheck()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, name="camera_healthcheck_spin")
    spin_thread.start()
    start = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - start < node.duration_sec:
            time.sleep(0.05)
        report, passed = node.report()
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main(sys.argv)
