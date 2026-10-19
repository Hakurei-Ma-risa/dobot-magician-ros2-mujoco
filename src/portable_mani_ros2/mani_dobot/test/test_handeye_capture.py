"""Read-only tests of calibration measurement validity and recording interlocks.

These tests never construct a ROS node, use a camera/serial port, or open a GUI.
"""

from collections import deque
import json
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mani_dobot import handeye_capture as capture
from mani_perception.point_calibration import evaluate_dataset


K = np.array([[600., 0., 320.], [0., 610., 240.], [0., 0., 1.]])
NOW = 10.0


def tcp_history(*, now=NOW, duration=0.5, count=11, position=(0.22, 0.02, 0.12), yaw=0.3):
    return [(float(received), 100.0 + float(received), np.array(position, dtype=float), yaw)
            for received in np.linspace(now-duration, now, count)]


def fake_collector(*, verified=True, note="Suction contact reference confirmed"):
    def publishers(topic):
        return [SimpleNamespace(node_name="camera" if "camera" in topic else "dobot_state_publisher")]
    return SimpleNamespace(
        args=SimpleNamespace(tcp_reference_verified=verified, tcp_reference_note=note),
        alarm_codes=[], alarm_received=NOW, tcp=tcp_history(),
        get_publishers_info_by_topic=publishers,
    )


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    monkeypatch.setattr(capture.time, "monotonic", lambda: NOW)


def test_depth_deprojection_has_correct_optical_ray_and_metric_units():
    point, spread = capture.depth_point(np.full((480, 640), 0.50), (380.0, 301.0), K)
    np.testing.assert_allclose(point, [0.05, 0.05, 0.50], atol=1e-12)
    assert spread == 0.0


def test_distorted_pixel_recovers_known_undistorted_ray():
    coefficients = np.array([0.16, -0.05, 0.002, -0.003, 0.01])
    camera_point = np.array([0.08, -0.045, 0.5])
    distorted, _ = cv2.projectPoints(camera_point.reshape(1, 3), np.zeros(3), np.zeros(3), K, coefficients)
    recovered, _ = capture.depth_point(np.full((480, 640), 0.5), distorted.reshape(2), K, coefficients)
    np.testing.assert_allclose(recovered, camera_point, atol=1e-8)


@pytest.mark.parametrize("pixel", [(0, 20), (639, 20), (20, 0), (20, 479), (-1, 30), (700, 30)])
def test_depth_image_edges_and_outside_pixels_are_rejected(pixel):
    with pytest.raises(ValueError, match="border"):
        capture.depth_point(np.full((480, 640), 0.5), pixel, K)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), 0.10, 2.0])
def test_invalid_depth_is_rejected(value):
    with pytest.raises(ValueError, match="valid depth"):
        capture.depth_point(np.full((30, 30), value), (15, 15), K)


def test_patch_requires_eighty_percent_valid_samples():
    depth = np.full((30, 30), 0.5)
    depth[13, 13:18] = 0.0
    capture.depth_point(depth, (15, 15), K)
    depth[14, 13] = 0.0
    with pytest.raises(ValueError, match="80%"):
        capture.depth_point(depth, (15, 15), K)


def test_depth_discontinuity_cannot_be_averaged_into_a_false_surface():
    depth = np.full((30, 30), 0.5)
    depth[13:18, 16:18] = 0.52
    with pytest.raises(ValueError, match="3 mm"):
        capture.depth_point(depth, (15, 15), K)


@pytest.mark.parametrize("focal", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_intrinsics_are_rejected(focal):
    intrinsics = K.copy()
    intrinsics[0, 0] = focal
    with pytest.raises(ValueError, match="intrinsics"):
        capture.depth_point(np.full((30, 30), 0.5), (15, 15), intrinsics)


@pytest.mark.parametrize("pixel", [(float("inf"), 15), (15, -float("inf")), (float("nan"), 15)])
def test_nonfinite_pixel_is_a_recoverable_validation_error(pixel):
    with pytest.raises(ValueError):
        capture.depth_point(np.full((30, 30), 0.5), pixel, K)


@pytest.mark.parametrize("coefficients", [[float("nan"), 0, 0, 0, 0], [float("inf"), 0, 0, 0, 0], [0, 0, 0]])
def test_invalid_distortion_is_rejected_before_opencv(coefficients):
    with pytest.raises(ValueError):
        capture.depth_point(np.full((480, 640), 0.5), (380, 301), K, coefficients)


def test_nonfinite_undistorted_ray_is_rejected(monkeypatch):
    monkeypatch.setattr(capture.cv2, "undistortPoints", lambda *args, **kwargs: np.array([[[np.nan, 0.]]]))
    with pytest.raises(ValueError):
        capture.depth_point(np.full((30, 30), 0.5), (15, 15), K)


def test_stable_tcp_returns_center_yaw_and_latest_robot_stamp():
    history = tcp_history()
    history[0][2][0] += 0.0002
    position, yaw, stamp = capture.stable_tcp(history, NOW)
    np.testing.assert_allclose(position, [0.22, 0.02, 0.12])
    assert yaw == pytest.approx(0.3)
    assert stamp == history[-1][1]


@pytest.mark.parametrize("history", [[], tcp_history(now=NOW-0.26),
                                   tcp_history(count=5), tcp_history(duration=0.35)])
def test_tcp_requires_fresh_data_and_at_least_point_four_seconds_stopped(history):
    with pytest.raises(ValueError):
        capture.stable_tcp(history, NOW)


def test_tcp_translation_motion_is_rejected():
    history = tcp_history()
    history[-1][2][0] += 0.0011
    with pytest.raises(ValueError, match="moving"):
        capture.stable_tcp(history, NOW)


def test_tcp_yaw_motion_is_rejected():
    history = tcp_history()
    history[-1] = (*history[-1][:3], history[-1][3] + math.radians(1.1))
    with pytest.raises(ValueError, match="yaw"):
        capture.stable_tcp(history, NOW)


def test_stationary_yaw_near_pi_wrap_is_not_mistaken_for_motion():
    history = tcp_history(yaw=math.pi - 0.0001)
    history[-1] = (*history[-1][:3], -math.pi + 0.0001)
    _, yaw, _ = capture.stable_tcp(history, NOW)
    assert abs(yaw - math.pi) < 0.001


@pytest.mark.parametrize("component", ["position", "yaw"])
def test_nonfinite_tcp_measurements_are_rejected(component):
    history = tcp_history()
    if component == "position":
        history[-1][2][0] = np.nan
    else:
        history[-1] = (*history[-1][:3], np.nan)
    with pytest.raises(ValueError, match="finite|invalid"):
        capture.stable_tcp(history, NOW)


@pytest.mark.parametrize("verified,note", [(False, "unknown"), (False, ""), (True, " ")])
def test_recording_gate_requires_confirmed_physical_tcp(verified, note):
    node = fake_collector(verified=verified, note=note)
    with pytest.raises(ValueError, match="physical TCP"):
        capture.HandeyeCapture.gate(node)


@pytest.mark.parametrize("alarm_codes,received", [(None, NOW), ([0x43], NOW),
                                                ([], NOW-0.51), ([], 0.0)])
def test_recording_gate_rejects_alarm_missing_alarm_and_stale_alarm(alarm_codes, received):
    node = fake_collector()
    node.alarm_codes, node.alarm_received = alarm_codes, received
    with pytest.raises(ValueError, match="alarm"):
        capture.HandeyeCapture.gate(node)


def test_recording_gate_rejects_stale_tcp_even_with_fresh_clean_alarms():
    node = fake_collector()
    node.tcp = tcp_history(now=NOW-0.26)
    with pytest.raises(ValueError, match="older than 250"):
        capture.HandeyeCapture.gate(node)


@pytest.mark.parametrize("publishers", [[], [SimpleNamespace(node_name="simulator")],
                                     [SimpleNamespace(node_name="camera"), SimpleNamespace(node_name="camera")]])
def test_recording_gate_rejects_ambiguous_or_wrong_data_source(publishers):
    node = fake_collector()
    node.get_publishers_info_by_topic = lambda topic: publishers
    with pytest.raises(ValueError, match="publisher"):
        capture.HandeyeCapture.gate(node)


def test_recording_gate_accepts_fresh_stationary_verified_tcp_with_clean_alarms():
    position, yaw, stamp = capture.HandeyeCapture.gate(fake_collector())
    np.testing.assert_allclose(position, [0.22, 0.02, 0.12])
    assert yaw == pytest.approx(0.3)
    assert stamp == pytest.approx(110.0)


def fake_saver(tmp_path):
    node = fake_collector()
    node.dataset = {
        "schema_version": 1, "method": "rgbd_tcp_point_correspondence", "units": "m",
        "camera_frame": "camera_color_optical_frame", "base_frame": "magician_base_link",
        "camera_serial": "test-only", "robot_port": "/dev/test-only",
        "tcp_reference_verified": True, "tcp_reference_note": node.args.tcp_reference_note,
        "samples": [],
    }
    node.output = tmp_path / "dataset.json"
    node.gate = lambda: capture.HandeyeCapture.gate(node)
    node.selected = {"pixel_xy": [350., 260.], "camera_point_m": [0.02, 0.01, 0.4],
                     "depth_patch_std_m": 0.0001}
    node.frozen = {
        "at": NOW, "rgb": (NOW, 200., "camera_color_optical_frame", None),
        "depth": (NOW, 200., "camera_color_optical_frame", None), "sync_ms": 0.0,
        "info": SimpleNamespace(width=640, height=480, k=K.ravel(), d=np.zeros(5), distortion_model="plumb_bob"),
    }
    node.corners = None
    return node


def test_save_calls_gate_and_never_records_unverified_tool(tmp_path):
    node = fake_saver(tmp_path)
    node.args.tcp_reference_verified = False
    with pytest.raises(ValueError, match="physical TCP"):
        capture.HandeyeCapture.save(node, "fit")
    assert not node.dataset["samples"]
    assert not node.output.exists()


def test_save_rejects_an_expired_frozen_target(tmp_path):
    node = fake_saver(tmp_path)
    node.frozen["at"] = NOW-121
    with pytest.raises(ValueError, match="120 s"):
        capture.HandeyeCapture.save(node, "fit")
    assert not node.output.exists()


def test_recorded_sample_matches_solver_schema_and_metric_direction(tmp_path):
    node = fake_saver(tmp_path)
    frozen = node.frozen
    random = np.random.default_rng(61)
    points = random.uniform([-0.10, -0.10, 0.30], [0.10, 0.10, 0.55], (9, 3))
    rotation = np.diag([1., -1., -1.])
    translation = np.array([0.21, 0.01, 0.65])
    for index, camera_point in enumerate(points):
        base_point = rotation @ camera_point + translation
        node.tcp = tcp_history(position=base_point)
        node.frozen = frozen
        node.selected = {"pixel_xy": [350., 260.], "camera_point_m": camera_point.tolist(),
                         "depth_patch_std_m": 0.0001}
        capture.HandeyeCapture.save(node, "fit" if index < 6 else "check")
    dataset = json.loads(node.output.read_text())
    result = evaluate_dataset(dataset)
    assert result["accepted"], result["rejection_reasons"]
    assert result["sample_count_fit"] == 6 and result["sample_count_check"] == 3
    np.testing.assert_allclose(result["R_base_camera"], rotation, atol=1e-12)
    np.testing.assert_allclose(result["translation_m"], translation, atol=1e-12)


def test_save_rolls_back_in_memory_measurement_when_disk_write_fails(tmp_path, monkeypatch):
    node = fake_saver(tmp_path)
    def fail(*args):
        raise OSError("simulated full disk")
    monkeypatch.setattr(capture, "write_dataset", fail)
    with pytest.raises(OSError):
        capture.HandeyeCapture.save(node, "fit")
    assert not node.dataset["samples"]
    assert node.selected is not None
