"""Physical invariants and independent acceptance of RGB-D rigid registration."""

import json

import numpy as np
import pytest

from mani_perception.point_calibration import (
    apply_transform,
    evaluate_dataset,
    fit_rigid_transform,
    main,
    rotation_to_quaternion_xyzw,
)


def known_transform():
    axis = np.array([1.0, -2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = 1.17
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
    return rotation, np.array([0.12, -0.08, 0.42])


def sample_dataset(noise_m=0.0):
    random = np.random.default_rng(37)
    camera = random.uniform([-0.12, -0.10, 0.32], [0.10, 0.12, 0.57], (15, 3))
    rotation, translation = known_transform()
    base = apply_transform(camera, rotation, translation)
    base += random.normal(0.0, noise_m, base.shape)
    samples = [
        {"role": "fit" if i < 10 else "check", "camera_point_m": a.tolist(),
         "base_point_m": b.tolist(), "pixel_xy": [200 + i, 220 + i],
         "tcp_yaw_rad": 0.2, "camera_stamp_s": 100.0 + i,
         "robot_stamp_s": 100.02 + i, "depth_patch_std_m": 0.0003}
        for i, (a, b) in enumerate(zip(camera, base))
    ]
    return {"schema_version": 1, "method": "rgbd_tcp_point_correspondence", "units": "m",
            "camera_frame": "camera_color_optical_frame", "base_frame": "magician_base_link",
            "camera_serial": "test-only", "robot_port": "/dev/test-only",
            "tcp_reference_verified": True,
            "tcp_reference_note": "same marked reference point measured in both frames",
            "samples": samples}


def test_known_se3_with_measurement_noise_and_direction():
    dataset = sample_dataset(noise_m=0.0003)
    result = evaluate_dataset(dataset)
    assert result["accepted"], result["rejection_reasons"]
    rotation, translation = known_transform()
    np.testing.assert_allclose(result["R_base_camera"], rotation, atol=0.003)
    np.testing.assert_allclose(result["translation_m"], translation, atol=0.001)
    assert result["check_errors"]["max_m"] < 0.002
    matrix = np.asarray(result["T_base_camera"])
    sample = dataset["samples"][-1]
    transformed = matrix @ np.r_[sample["camera_point_m"], 1.0]
    np.testing.assert_allclose(transformed[:3], sample["base_point_m"], atol=0.001)
    assert np.linalg.norm(np.linalg.inv(matrix) @ np.r_[sample["camera_point_m"], 1.0]
                          - np.r_[sample["base_point_m"], 1.0]) > 0.1
    assert result["scale_fitted"] is False


def test_three_noncollinear_planar_points_have_exact_rigid_solution():
    camera = np.array([[0.0, 0.0, 0.4], [0.1, 0.0, 0.4], [0.0, 0.1, 0.4]])
    rotation, translation = known_transform()
    base = apply_transform(camera, rotation, translation)
    fitted_r, fitted_t = fit_rigid_transform(camera, base)
    np.testing.assert_allclose(fitted_r, rotation, atol=1e-12)
    np.testing.assert_allclose(fitted_t, translation, atol=1e-12)
    with pytest.raises(ValueError, match="planar"):
        fit_rigid_transform(camera, base, allow_planar=False)


def test_planar_acceptance_requires_explicit_override_and_reports_limitation():
    dataset = sample_dataset()
    rotation, translation = known_transform()
    for sample in dataset["samples"]:
        if sample["role"] == "fit":
            sample["camera_point_m"][2] = 0.42
            sample["base_point_m"] = apply_transform(sample["camera_point_m"], rotation, translation).tolist()
    rejected = evaluate_dataset(dataset)
    assert not rejected["accepted"]
    assert "planar" in " ".join(rejected["rejection_reasons"])
    accepted = evaluate_dataset(dataset, allow_planar=True)
    assert accepted["accepted"]
    assert accepted["limitations"]
    assert accepted["check_errors"]["max_m"] < 1e-12


@pytest.mark.parametrize("camera", [
    [[0, 0, 0.4], [0.1, 0, 0.4], [0.2, 0, 0.4]],
    [[0, 0, 0.4], [0, 0, 0.4], [0, 0.1, 0.4]],
    [[0, 0, 0.4], [0.1, float("nan"), 0.4], [0, 0.1, 0.4]],
    [[0, 0, 0.4], [0.1, 0, 0.4], [0, 0.1, float("inf")]],
])
def test_line_duplicate_and_nonfinite_points_are_rejected(camera):
    with pytest.raises(ValueError):
        fit_rigid_transform(camera, camera)


def test_independent_bad_check_point_cannot_be_hidden_by_perfect_fit():
    dataset = sample_dataset()
    dataset["samples"][-1]["base_point_m"][0] += 0.025
    result = evaluate_dataset(dataset)
    assert result["fit_errors"]["max_m"] < 1e-12
    assert result["check_errors"]["max_m"] > 0.024
    assert not result["accepted"]
    assert "independent check" in " ".join(result["rejection_reasons"])


def test_scale_mismatch_is_not_fitted_away():
    dataset = sample_dataset()
    rotation, translation = known_transform()
    for sample in dataset["samples"]:
        sample["base_point_m"] = (rotation @ (np.asarray(sample["camera_point_m"]) * 1.15)
                                  + translation).tolist()
    result = evaluate_dataset(dataset)
    assert not result["accepted"]
    assert result["scale_fitted"] is False
    np.testing.assert_allclose(result["geometry"]["median_base_to_camera_pair_distance_ratio"], 1.15)
    np.testing.assert_allclose(np.linalg.det(result["R_base_camera"]), 1.0, atol=1e-12)
    assert result["fit_errors"]["rmse_m"] > 0.01


@pytest.mark.parametrize("rotation", [known_transform()[0], np.diag([1., -1., -1.]),
                                     np.diag([-1., 1., -1.]), np.diag([-1., -1., 1.])])
def test_xyzw_quaternion_reconstructs_rotation_including_half_turns(rotation):
    x, y, z, w = rotation_to_quaternion_xyzw(rotation)
    reconstructed = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    np.testing.assert_allclose(reconstructed, rotation, atol=1e-12)
    np.testing.assert_allclose(x*x + y*y + z*z + w*w, 1.0, atol=1e-12)


def test_repeated_fit_point_is_not_independent_acceptance():
    dataset = sample_dataset()
    dataset["samples"][-1].update({key: dataset["samples"][0][key]
                                   for key in ("camera_point_m", "base_point_m")})
    result = evaluate_dataset(dataset)
    assert not result["accepted"]
    assert "duplicate" in " ".join(result["rejection_reasons"])


def test_check_within_one_millimetre_is_not_independent():
    dataset = sample_dataset()
    rotation, translation = known_transform()
    point = np.asarray(dataset["samples"][0]["camera_point_m"]) + [0.0005, 0, 0]
    dataset["samples"][-1]["camera_point_m"] = point.tolist()
    dataset["samples"][-1]["base_point_m"] = apply_transform(point, rotation, translation).tolist()
    result = evaluate_dataset(dataset)
    assert not result["accepted"]
    assert not result["independent_check"]["camera"]["nonduplicate"]


@pytest.mark.parametrize("verified", [False, None, "true", 1])
def test_unverified_tool_reference_always_rejects_but_preserves_diagnostics(verified):
    dataset = sample_dataset()
    dataset["tcp_reference_verified"] = verified
    result = evaluate_dataset(dataset)
    assert not result["accepted"]
    assert result["T_base_camera"] is not None
    assert "TCP reference point is not verified" in " ".join(result["rejection_reasons"])
    assert result["source_metadata"]["tcp_reference_verified"] == verified


def test_cli_saves_rejected_diagnostics_and_does_not_overwrite_raw_input(tmp_path):
    dataset = sample_dataset()
    dataset["samples"][-1]["base_point_m"][0] += 0.02
    raw_path, output_path = tmp_path / "samples.json", tmp_path / "result.json"
    raw_path.write_text(json.dumps(dataset))
    assert main(["--dataset", str(raw_path), "--output", str(output_path)]) == 2
    result = json.loads(output_path.read_text())
    assert not result["accepted"]
    assert result["T_base_camera"] is not None
    assert result["camera_serial"] == "test-only"
    assert result["sample_count_fit"] == 10 and result["sample_count_check"] == 5
    with pytest.raises(SystemExit):
        main([str(raw_path), "--output", str(raw_path)])
    assert json.loads(raw_path.read_text()) == dataset


def test_missing_independent_samples_does_not_pass():
    dataset = sample_dataset()
    dataset["samples"] = dataset["samples"][:10]
    result = evaluate_dataset(dataset)
    assert not result["accepted"]
    assert result["fit_errors"]["rmse_m"] < 1e-12
    assert result["check_errors"] is None


def test_invalid_nonfinite_threshold_still_writes_failure_diagnostic(tmp_path):
    raw_path, output_path = tmp_path / "samples.json", tmp_path / "result.json"
    raw_path.write_text(json.dumps(sample_dataset()))
    assert main([str(raw_path), "--output", str(output_path), "--fit-rmse-mm", "nan"]) == 2
    result = json.loads(output_path.read_text())
    assert not result["accepted"]
    assert result["thresholds"]["fit_rmse_m"] is None


def test_reflection_cannot_be_reported_as_valid_rigid_motion():
    dataset = sample_dataset()
    for sample in dataset["samples"]:
        sample["base_point_m"] = (np.asarray(sample["camera_point_m"]) * [-1., 1., 1.]).tolist()
    result = evaluate_dataset(dataset)
    assert not result["accepted"]
    np.testing.assert_allclose(np.linalg.det(result["R_base_camera"]), 1.0, atol=1e-12)


@pytest.mark.parametrize("key,value", [("units", "mm"), ("schema_version", 2),
                                      ("camera_serial", ""), ("method", "similarity")])
def test_metadata_mismatch_is_rejected(key, value):
    dataset = sample_dataset()
    dataset[key] = value
    result = evaluate_dataset(dataset)
    assert not result["accepted"]
    assert result["rejection_reasons"]
