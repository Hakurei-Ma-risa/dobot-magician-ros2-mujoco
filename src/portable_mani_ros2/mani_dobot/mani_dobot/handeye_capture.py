"""Read-only RGB-D/TCP point collection for a fixed camera and Dobot."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import rclpy
from dobot_msgs.msg import DobotAlarmCodes
from geometry_msgs.msg import PoseStamped
from mani_perception.rgbd_localizer import decode_depth_m
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def timestamp(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def depth_point(
    depth_m: np.ndarray, pixel_xy, k, distortion=(), *, radius: int = 2
) -> tuple[np.ndarray, float]:
    """Deproject a small, sufficiently valid and smooth surface patch."""
    pixel = np.asarray(pixel_xy, dtype=float)
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        raise ValueError("Pixel must contain two finite coordinates")
    if depth_m.ndim != 2 or not isinstance(radius, int) or radius < 0:
        raise ValueError("Expected a 2D depth array and a nonnegative patch radius")
    u, v = pixel
    x, y = int(round(u)), int(round(v))
    if not (radius <= x < depth_m.shape[1] - radius and radius <= y < depth_m.shape[0] - radius):
        raise ValueError("Point is too close to the image border")
    patch = depth_m[y-radius:y+radius+1, x-radius:x+radius+1]
    valid = patch[np.isfinite(patch) & (patch > 0.10) & (patch < 2.0)]
    if valid.size < math.ceil(patch.size * 0.8):
        raise ValueError("Less than 80% valid depth around the point")
    spread = float(np.std(valid))
    if spread > 0.003:
        raise ValueError("Depth patch varies by more than 3 mm; avoid edges")
    camera_matrix = np.asarray(k, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(camera_matrix)) or min(camera_matrix[0, 0], camera_matrix[1, 1]) <= 0:
        raise ValueError("Camera intrinsics are invalid")
    coefficients = np.asarray(distortion, dtype=float).reshape(-1)
    if coefficients.size not in (0, 4, 5, 8, 12, 14) or not np.all(np.isfinite(coefficients)):
        raise ValueError("Camera distortion coefficients are invalid")
    try:
        ray = cv2.undistortPoints(
            np.array([[[u, v]]], dtype=np.float64), camera_matrix,
            coefficients if coefficients.size else None,
        ).reshape(2)
    except cv2.error as exc:
        raise ValueError("Camera distortion could not be inverted") from exc
    if not np.all(np.isfinite(ray)):
        raise ValueError("Camera ray is not finite")
    z = float(np.median(valid))
    return np.array([ray[0] * z, ray[1] * z, z]), spread


def stable_tcp(history, now: float) -> tuple[np.ndarray, float, float]:
    recent = [item for item in history if now - item[0] <= 0.65]
    if not recent or now - recent[-1][0] > 0.25:
        raise ValueError("TCP state is missing or older than 250 ms")
    if len(recent) < 6 or recent[-1][0] - recent[0][0] < 0.4:
        raise ValueError("Wait for at least 0.4 s of stable TCP state")
    positions = np.array([item[2] for item in recent])
    if not np.all(np.isfinite(positions)):
        raise ValueError("TCP position is not finite")
    center = np.median(positions, axis=0)
    if float(np.linalg.norm(positions - center, axis=1).max()) > 0.001:
        raise ValueError("TCP is moving by more than 1 mm")
    yaws = np.array([item[3] for item in recent])
    if not np.all(np.isfinite(yaws)):
        raise ValueError("TCP yaw is invalid")
    yaws = np.unwrap(yaws)
    if np.ptp(yaws) > math.radians(1):
        raise ValueError("TCP yaw is invalid or moving by more than 1 degree")
    return center, float(np.median(yaws)), recent[-1][1]


def write_dataset(path: Path, dataset: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(dataset, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


class HandeyeCapture(Node):
    def __init__(self, args) -> None:
        super().__init__("mani_handeye_capture")
        self.args = args
        self.rgb = deque(maxlen=12)
        self.depth = deque(maxlen=12)
        self.info = None
        self.info_received = 0.0
        self.tcp = deque(maxlen=30)
        self.alarm_codes = None
        self.alarm_received = 0.0
        self.frozen = None
        self.selected = None
        self.corners = None
        self.last_corner_check = 0.0
        self.status = "F: freeze RGB-D, then click a fixed target"
        self.window = "D435i / Dobot point calibration (read-only)"
        self.output = Path(args.output).expanduser().resolve()
        self.dataset = {
            "schema_version": 1,
            "method": "rgbd_tcp_point_correspondence",
            "units": "m",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "camera_frame": "camera_color_optical_frame",
            "base_frame": args.base_frame,
            "camera_serial": args.camera_serial,
            "robot_port": args.robot_port,
            "tcp_reference_verified": args.tcp_reference_verified,
            "tcp_reference_note": args.tcp_reference_note,
            "samples": [],
        }
        if self.output.exists():
            if not args.resume:
                raise ValueError("Output already exists; use another path or --resume")
            with self.output.open(encoding="utf-8") as stream:
                previous = json.load(stream)
            for key in ("schema_version", "method", "units", "camera_frame", "base_frame", "camera_serial", "robot_port", "tcp_reference_verified", "tcp_reference_note"):
                if previous.get(key) != self.dataset[key]:
                    raise ValueError(f"Resume metadata differs: {key}")
            self.dataset = previous
        self.initial_count = len(self.dataset["samples"])
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self.on_info, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/dobot_TCP", self.on_tcp, 10)
        self.create_subscription(DobotAlarmCodes, "/dobot_alarms", self.on_alarms, 10)

    def on_rgb(self, message) -> None:
        if message.encoding not in ("rgb8", "bgr8"):
            self.status = "Expected rgb8/bgr8 camera image"
            return
        frame = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
        frame = frame[:, :message.width * 3].reshape(message.height, message.width, 3)
        if message.encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.rgb.append((time.monotonic(), timestamp(message.header), message.header.frame_id, frame.copy()))

    def on_depth(self, message) -> None:
        try:
            depth = decode_depth_m(message)
        except ValueError as exc:
            self.status = str(exc)
            return
        self.depth.append((time.monotonic(), timestamp(message.header), message.header.frame_id, depth))

    def on_info(self, message) -> None:
        self.info = message
        self.info_received = time.monotonic()

    def on_tcp(self, message) -> None:
        if message.header.frame_id not in ("", self.args.base_frame):
            self.status = "Unexpected TCP base frame"
            return
        p, q = message.pose.position, message.pose.orientation
        quaternion = np.array([q.x, q.y, q.z, q.w], dtype=float)
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm < 1e-8:
            return
        x, y, z, w = quaternion / norm
        yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
        self.tcp.append((time.monotonic(), timestamp(message.header), np.array([p.x, p.y, p.z]), yaw))

    def on_alarms(self, message) -> None:
        self.alarm_codes = list(message.alarms_list)
        self.alarm_received = time.monotonic()

    def freeze(self) -> None:
        now = time.monotonic()
        if self.info is None or now - self.info_received > 1.0 or not self.rgb or not self.depth:
            raise ValueError("Fresh RGB, aligned depth and CameraInfo are required")
        pairs = [(abs(a[1]-b[1]), a, b) for a in self.rgb for b in self.depth if now-a[0] < 1 and now-b[0] < 1]
        if not pairs:
            raise ValueError("RGB-D images are stale")
        difference, rgb, depth = min(pairs, key=lambda pair: (pair[0], -pair[1][0]))
        if difference > 0.04:
            raise ValueError("RGB/depth timestamp gap exceeds 40 ms; retry F")
        if rgb[2] != self.dataset["camera_frame"] or depth[2] != rgb[2] or self.info.header.frame_id != rgb[2]:
            raise ValueError("RGB-D optical frames do not match")
        if depth[3].shape != rgb[3].shape[:2] or (self.info.height, self.info.width) != depth[3].shape:
            raise ValueError("RGB-D dimensions and intrinsics do not match")
        if self.info.distortion_model not in ("plumb_bob", "rational_polynomial"):
            raise ValueError("Unsupported camera distortion model")
        self.frozen = {"at": now, "rgb": rgb, "depth": depth, "info": self.info, "sync_ms": difference*1000}
        self.selected = None
        self.corners = None
        self.status = "Frozen BEFORE contact. Click a fixed physical point."

    def mouse(self, event, x, y, flags, parameter) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.frozen is None:
            return
        image = self.frozen["rgb"][3]
        if x >= image.shape[1] or y >= image.shape[0]:
            return
        pixel = np.array([float(x), float(y)])
        if self.corners is not None:
            points = self.corners.reshape(-1, 2)
            closest = points[np.argmin(np.linalg.norm(points-pixel, axis=1))]
            if np.linalg.norm(closest-pixel) <= 6:
                pixel = closest
        try:
            info = self.frozen["info"]
            point, spread = depth_point(self.frozen["depth"][3], pixel, info.k, info.d)
            recent_depths = [d for d in self.depth if abs(d[1]-self.frozen["depth"][1]) <= 0.4]
            repeated = [depth_point(d[3], pixel, info.k, info.d)[0][2] for d in recent_depths]
            if len(repeated) < 3 or float(np.ptp(repeated)) > 0.005:
                raise ValueError("Need 3 stable depth frames within 5 mm; retry F")
            self.selected = {"pixel_xy": pixel.tolist(), "camera_point_m": point.tolist(), "depth_patch_std_m": spread}
            self.status = (
                "Point ready. Align verified TCP to SAME point; S fit / V check"
                if self.args.tcp_reference_verified and self.alarm_codes == []
                else "Point preview ready. Recording blocked until TCP and alarms are resolved."
            )
        except ValueError as exc:
            self.selected = None
            self.status = str(exc)

    def gate(self) -> tuple[np.ndarray, float, float]:
        now = time.monotonic()
        if not self.args.tcp_reference_verified or not self.args.tcp_reference_note.strip():
            raise ValueError("BLOCKED: physical TCP contact point is not verified")
        if self.alarm_codes is None or now - self.alarm_received > 0.5:
            raise ValueError("BLOCKED: robot alarm state is missing/stale")
        if self.alarm_codes:
            raise ValueError(f"BLOCKED: robot alarms {self.alarm_codes}")
        for topic, expected in (("/dobot_TCP", "dobot_state_publisher"), ("/camera/camera/color/image_raw", "camera")):
            publishers = self.get_publishers_info_by_topic(topic)
            if len(publishers) != 1 or publishers[0].node_name != expected:
                raise ValueError(f"Unexpected or ambiguous publisher on {topic}")
        return stable_tcp(self.tcp, now)

    def save(self, role: str) -> None:
        if self.frozen is None or self.selected is None:
            raise ValueError("Freeze and select a target point first")
        if time.monotonic()-self.frozen["at"] > 120:
            raise ValueError("Frozen target older than 120 s; start a fresh point")
        base, yaw, stamp = self.gate()
        for other in self.dataset["samples"]:
            if np.linalg.norm(base-np.array(other["base_point_m"])) < 0.005:
                raise ValueError("New robot point must be at least 5 mm from existing points")
        info = self.frozen["info"]
        self.dataset["camera_info"] = {"width": info.width, "height": info.height, "k": list(info.k), "d": list(info.d), "distortion_model": info.distortion_model}
        sample = dict(self.selected, role=role, base_point_m=base.tolist(), tcp_yaw_rad=yaw,
                      camera_stamp_s=self.frozen["rgb"][1], robot_stamp_s=stamp,
                      depth_stamp_s=self.frozen["depth"][1], rgbd_sync_ms=self.frozen["sync_ms"],
                      captured_utc=datetime.now(timezone.utc).isoformat(), pairing="fixed_target_before_tcp_contact")
        self.dataset["samples"].append(sample)
        try:
            write_dataset(self.output, self.dataset)
        except Exception:
            self.dataset["samples"].pop()
            raise
        self.frozen = self.selected = self.corners = None
        self.status = f"Saved {role} sample #{len(self.dataset['samples'])}. Move to another target."
        print(self.status, flush=True)

    def canvas(self) -> np.ndarray:
        image = self.frozen["rgb"][3] if self.frozen is not None else (self.rgb[-1][3] if self.rgb else np.zeros((480, 640, 3), np.uint8))
        display = image.copy()
        if time.monotonic()-self.last_corner_check > 0.8:
            self.last_corner_check = time.monotonic()
            found, corners = cv2.findChessboardCornersSB(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), tuple(self.args.checkerboard_inner), cv2.CALIB_CB_NORMALIZE_IMAGE)
            self.corners = corners if found else None
        if self.corners is not None:
            cv2.drawChessboardCorners(display, tuple(self.args.checkerboard_inner), self.corners, True)
        if self.selected is not None:
            cv2.drawMarker(display, tuple(int(round(v)) for v in self.selected["pixel_xy"]), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        panel = np.zeros((max(480, image.shape[0]), image.shape[1]+520, 3), dtype=np.uint8)
        panel[:image.shape[0], :image.shape[1]] = display
        fit = sum(s["role"] == "fit" for s in self.dataset["samples"])
        check = sum(s["role"] == "check" for s in self.dataset["samples"])
        tcp_xyz = "unavailable" if not self.tcp else ", ".join(f"{value:.4f}" for value in self.tcp[-1][2])
        tcp_yaw = "unavailable" if not self.tcp else f"{math.degrees(self.tcp[-1][3]):.1f} deg"
        lines = ["CAMERA -> ROBOT BASE CALIBRATION", f"Camera: {self.args.camera_serial}",
                 f"TCP verified: {self.args.tcp_reference_verified}", f"Alarms: {self.alarm_codes}",
                 f"TCP XYZ (m): {tcp_xyz}", f"TCP yaw: {tcp_yaw}",
                 f"Fit points: {fit}/6   Check points: {check}/3", f"View: {'FROZEN' if self.frozen is not None else 'LIVE'}",
                 f"Checkerboard corners: {0 if self.corners is None else len(self.corners)}", "",
                 "F: freeze paired RGB / aligned depth", "Left click: select visible surface point",
                 ("Align TCP to this SAME fixed point" if self.args.tcp_reference_verified and self.alarm_codes == [] else "PREPARATION ONLY: sampling is blocked"), "S: save fit point   V: save check point",
                 "C: cancel frozen point   U: undo last", "Q / close window: exit", "",
                 "No robot motion commands are sent."]
        for index, line in enumerate(lines):
            cv2.putText(panel, line, (image.shape[1]+12, 24+index*24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, self.status[:130], (8, panel.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        return panel

    def run(self) -> None:
        if self.args.check_only:
            end = time.monotonic()+self.args.duration
            while rclpy.ok() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.02)
            try:
                self.gate()
                gate = "ready"
            except ValueError as exc:
                gate = str(exc)
            print(json.dumps({"rgb_frames": len(self.rgb), "depth_frames": len(self.depth), "tcp_frames": len(self.tcp), "alarms": self.alarm_codes, "sample_gate": gate, "dataset_written": self.output.exists()}, ensure_ascii=False))
            return
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.mouse)
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.005)
                cv2.imshow(self.window, self.canvas())
                key = cv2.waitKey(1) & 0xff
                try:
                    # GTK builds may return -1 for unsupported VISIBLE even
                    # while the window is open. AUTOSIZE is supported here.
                    if key in (ord("q"), 27) or cv2.getWindowProperty(self.window, cv2.WND_PROP_AUTOSIZE) < 0:
                        break
                    if key == ord("f"):
                        self.freeze()
                    elif key == ord("s"):
                        self.save("fit")
                    elif key == ord("v"):
                        self.save("check")
                    elif key == ord("c"):
                        self.frozen = self.selected = self.corners = None
                        self.status = "Live view; fixed target must not move during pairing"
                    elif key == ord("u") and len(self.dataset["samples"]) > self.initial_count:
                        self.dataset["samples"].pop()
                        write_dataset(self.output, self.dataset)
                        self.status = "Removed this session's last sample"
                except (ValueError, OSError) as exc:
                    self.status = str(exc)
                    print(self.status, flush=True)
        finally:
            cv2.destroyWindow(self.window)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--base-frame", default="magician_base_link")
    parser.add_argument("--tcp-reference-verified", action="store_true")
    parser.add_argument("--tcp-reference-note", default="")
    parser.add_argument("--checkerboard-inner", nargs=2, type=int, default=[5, 8], metavar=("COLS", "ROWS"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.tcp_reference_verified and not args.tcp_reference_note.strip():
        parser.error("--tcp-reference-verified requires an actual --tcp-reference-note")
    if min(args.checkerboard_inner) < 3 or args.duration <= 0:
        parser.error("Checkerboard inner dimensions must be >=3 and duration positive")
    rclpy.init(args=[])
    node = None
    try:
        node = HandeyeCapture(args)
        print("Point collector ready. Sample recording requires verified TCP and no robot alarms.", flush=True)
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
