"""Rigid RGB-D point registration with independent acceptance measurements.

The transform always maps *camera optical coordinates to robot base coordinates*:
``p_base = R_base_camera @ p_camera + translation_m``.  Scale is never fitted.
This module has no ROS dependency and does not publish a transform.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


METHOD = "rgbd_tcp_point_correspondence"
METADATA_FIELDS = (
    "camera_frame", "base_frame", "camera_serial", "robot_port", "tcp_reference_note"
)


def _json_safe(value: Any) -> Any:
    """Keep failure diagnostics serializable even for malformed NaN input."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _points(value: Any, name: str, *, minimum: int = 1) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum:
        raise ValueError(f"{name} must have shape (N, 3), with N >= {minimum}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains non-finite coordinates")
    return points


def geometry_diagnostics(points_m: Any, *, min_spatial_ratio: float = 0.02) -> dict:
    """Describe span and degeneracy without estimating or changing scale."""
    points = _points(points_m, "points", minimum=3)
    if not 0.0 < min_spatial_ratio < 1.0:
        raise ValueError("min_spatial_ratio must lie strictly between 0 and 1")
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    pairs = distances[np.triu_indices(len(points), k=1)]
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    first = float(singular[0])
    ratios = singular / first if first > 0.0 else np.zeros(3)
    return {
        "singular_values_m": singular.tolist(),
        "second_to_first_ratio": float(ratios[1]),
        "third_to_first_ratio": float(ratios[2]),
        "min_pair_distance_m": float(np.min(pairs)),
        "max_pair_distance_m": float(np.max(pairs)),
        "duplicate_points": bool(np.min(pairs) <= 1e-6),
        "collinear": bool(first <= 1e-9 or ratios[1] <= 1e-6),
        "planar_or_shallow": bool(ratios[2] < min_spatial_ratio),
        "minimum_third_to_first_ratio": float(min_spatial_ratio),
    }


def _check_geometry(points: np.ndarray, name: str, allow_planar: bool,
                    min_spatial_ratio: float) -> dict:
    diagnostic = geometry_diagnostics(points, min_spatial_ratio=min_spatial_ratio)
    if diagnostic["duplicate_points"]:
        raise ValueError(f"{name} contains duplicate points (separation <= 1 micrometre)")
    if diagnostic["collinear"]:
        raise ValueError(f"{name} is collinear; at least 3 non-collinear points are required")
    if diagnostic["planar_or_shallow"] and not allow_planar:
        raise ValueError(
            f"{name} is planar or has insufficient depth variation: third/first "
            f"singular value ratio {diagnostic['third_to_first_ratio']:.5g} "
            f"< {min_spatial_ratio}; collect points at different heights or "
            "explicitly use --allow-planar"
        )
    return diagnostic


def fit_rigid_transform(camera_points_m: Any, base_points_m: Any, *,
                        allow_planar: bool = True,
                        min_spatial_ratio: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Fit proper SE(3) using Kabsch, returning ``(R_base_camera, t_base_camera)``.

    Three non-collinear points are mathematically sufficient.  The CLI enforces
    stronger sample counts and 3D coverage by default.  This function never fits
    a similarity transform and therefore cannot hide a depth-scale error.
    """
    camera = _points(camera_points_m, "camera_points_m", minimum=3)
    base = _points(base_points_m, "base_points_m", minimum=3)
    if camera.shape != base.shape:
        raise ValueError("camera and base point arrays must have identical shape")
    _check_geometry(camera, "camera fit points", allow_planar, min_spatial_ratio)
    _check_geometry(base, "base fit points", allow_planar, min_spatial_ratio)
    camera_mean, base_mean = camera.mean(axis=0), base.mean(axis=0)
    u, _, vt = np.linalg.svd((camera - camera_mean).T @ (base - base_mean))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = base_mean - rotation @ camera_mean
    return rotation, translation


def apply_transform(points_m: Any, rotation: Any, translation_m: Any) -> np.ndarray:
    """Transform one point of shape (3,) or many points of shape (N, 3)."""
    points = np.asarray(points_m, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation_m, dtype=float)
    if points.ndim not in (1, 2) or points.shape[-1] != 3:
        raise ValueError("points must have shape (3,) or (N, 3)")
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("rotation must be (3, 3), translation must be (3,)")
    if not all(np.all(np.isfinite(v)) for v in (points, rotation, translation)):
        raise ValueError("transform arguments must be finite")
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7)):
        raise ValueError("rotation must be a proper orthonormal rotation (determinant +1)")
    return points @ rotation.T + translation


def rotation_to_quaternion_xyzw(rotation: Any) -> np.ndarray:
    """Return a unit XYZW quaternion, including stable 180-degree rotations."""
    r = np.asarray(rotation, dtype=float)
    apply_transform(np.zeros(3), r, np.zeros(3))  # Validate SO(3).
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([(r[2, 1] - r[1, 2]) / s,
                      (r[0, 2] - r[2, 0]) / s,
                      (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(0.0, 1.0 + r[i, i] - r[j, j] - r[k, k])) * 2.0
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
        q[3] = (r[k, j] - r[j, k]) / s
    q /= np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return q


def _error_report(camera: np.ndarray, base: np.ndarray,
                  rotation: np.ndarray, translation: np.ndarray) -> dict:
    predictions = apply_transform(camera, rotation, translation)
    residuals = predictions - base
    errors = np.linalg.norm(residuals, axis=1)
    return {
        "sample_count": int(len(errors)),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "mean_m": float(np.mean(errors)),
        "max_m": float(np.max(errors)),
        "errors_m": errors.tolist(),
        "residuals_base_m": residuals.tolist(),
    }


def _validate_dataset(dataset: dict) -> tuple[list, list]:
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a JSON object")
    if dataset.get("schema_version") != 1 or isinstance(dataset.get("schema_version"), bool):
        raise ValueError("schema_version must be 1")
    if dataset.get("method") != METHOD:
        raise ValueError(f"method must be {METHOD!r}")
    if dataset.get("units") != "m":
        raise ValueError("units must be 'm'; convert millimetres explicitly before solving")
    for key in METADATA_FIELDS:
        if not isinstance(dataset.get(key), str) or not dataset[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if dataset["camera_frame"] == dataset["base_frame"]:
        raise ValueError("camera_frame and base_frame must be different")
    samples = dataset.get("samples")
    if not isinstance(samples, list):
        raise ValueError("samples must be a JSON array")
    fit, check = [], []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("role") not in ("fit", "check"):
            raise ValueError(f"sample {index}: role must be 'fit' or 'check'")
        for key in ("camera_point_m", "base_point_m"):
            point = np.asarray(sample.get(key), dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError(f"sample {index}: {key} must be 3 finite numbers")
        pixel = np.asarray(sample.get("pixel_xy"), dtype=float)
        if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
            raise ValueError(f"sample {index}: pixel_xy must be 2 finite numbers")
        for key in ("tcp_yaw_rad", "camera_stamp_s", "robot_stamp_s", "depth_patch_std_m"):
            value = sample.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"sample {index}: {key} must be a finite number")
        if sample["depth_patch_std_m"] < 0.0:
            raise ValueError(f"sample {index}: depth_patch_std_m cannot be negative")
        (fit if sample["role"] == "fit" else check).append(sample)
    return fit, check


def evaluate_dataset(dataset: dict, *, min_fit: int = 6, min_check: int = 3,
                     fit_rmse_limit_m: float = 0.003,
                     fit_max_limit_m: float = 0.005,
                     check_max_limit_m: float = 0.005,
                     allow_planar: bool = False,
                     min_spatial_ratio: float = 0.02) -> dict:
    """Return a JSON-safe diagnostic, including rejected and malformed datasets.

    ``accepted`` must be true before a consumer may publish ``T_base_camera``.
    Check samples are never used to estimate the transform.  Points repeated
    across fit/check roles are rejected, preserving independent acceptance.
    """
    result = {
        "schema_version": 1, "method": METHOD, "units": "m",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "transform_direction": "camera_to_base",
        "point_equation": "p_base = R_base_camera @ p_camera + translation_m",
        "accepted": False, "rejection_reasons": [], "limitations": [],
        "warnings": [], "independent_check": {},
        "sample_count_fit": 0, "sample_count_check": 0,
        "T_base_camera": None, "R_base_camera": None,
        "translation_m": None, "quaternion_xyzw": None,
        "fit_errors": None, "check_errors": None, "geometry": {},
        "scale_fitted": False,
        "thresholds": {
            "minimum_fit_samples": min_fit, "minimum_check_samples": min_check,
            "fit_rmse_m": fit_rmse_limit_m, "fit_max_m": fit_max_limit_m,
            "check_max_m": check_max_limit_m,
            "minimum_third_to_first_ratio": min_spatial_ratio,
            "allow_planar": allow_planar,
        },
    }
    if isinstance(dataset, dict):
        result.update({key: dataset.get(key) for key in METADATA_FIELDS})
        result["tcp_reference_verified"] = dataset.get("tcp_reference_verified") is True
        result["source_metadata"] = {
            key: dataset.get(key) for key in (*METADATA_FIELDS, "tcp_reference_verified")
        }
    try:
        if isinstance(min_fit, bool) or not isinstance(min_fit, int) or min_fit < 3:
            raise ValueError("min_fit must be an integer >= 3")
        if isinstance(min_check, bool) or not isinstance(min_check, int) or min_check < 1:
            raise ValueError("min_check must be an integer >= 1")
        for threshold in (fit_rmse_limit_m, fit_max_limit_m, check_max_limit_m):
            if not np.isfinite(threshold) or threshold <= 0.0:
                raise ValueError("error thresholds must be finite and positive")
        fit, check = _validate_dataset(dataset)
        if dataset.get("tcp_reference_verified") is not True:
            result["rejection_reasons"].append(
                "TCP reference point is not verified as the observed physical suction/contact "
                "reference; set tcp_reference_verified=true only after confirming the tool offset"
            )
        result["sample_count_fit"], result["sample_count_check"] = len(fit), len(check)
        if len(fit) < min_fit:
            result["rejection_reasons"].append(f"need at least {min_fit} fit samples; got {len(fit)}")
        if len(check) < min_check:
            result["rejection_reasons"].append(f"need at least {min_check} independent check samples; got {len(check)}")
        if len(fit) < 3:
            return _json_safe(result)
        camera = _points([s["camera_point_m"] for s in fit], "fit camera points", minimum=3)
        base = _points([s["base_point_m"] for s in fit], "fit base points", minimum=3)
        for name, points in (("camera_fit", camera), ("base_fit", base)):
            result["geometry"][name] = geometry_diagnostics(points, min_spatial_ratio=min_spatial_ratio)
        rotation, translation = fit_rigid_transform(
            camera, base, allow_planar=allow_planar,
            min_spatial_ratio=min_spatial_ratio,
        )
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, translation
        result.update({
            "T_base_camera": matrix.tolist(), "R_base_camera": rotation.tolist(),
            "translation_m": translation.tolist(),
            "quaternion_xyzw": rotation_to_quaternion_xyzw(rotation).tolist(),
            "fit_errors": _error_report(camera, base, rotation, translation),
        })
        pair_index = np.triu_indices(len(camera), k=1)
        camera_dist = np.linalg.norm(camera[:, None] - camera[None, :], axis=2)[pair_index]
        base_dist = np.linalg.norm(base[:, None] - base[None, :], axis=2)[pair_index]
        result["geometry"]["median_base_to_camera_pair_distance_ratio"] = float(
            np.median(base_dist / camera_dist)
        )
        if any(g["planar_or_shallow"] for key, g in result["geometry"].items()
               if key in ("camera_fit", "base_fit")):
            result["limitations"].append(
                "Planar or shallow training coverage was explicitly allowed; independent "
                "check points at other heights are necessary before using this transform "
                "away from the sampled plane."
            )
        if result["fit_errors"]["rmse_m"] > fit_rmse_limit_m:
            result["rejection_reasons"].append("fit RMSE exceeds threshold")
        if result["fit_errors"]["max_m"] > fit_max_limit_m:
            result["rejection_reasons"].append("fit maximum error exceeds threshold")
        if check:
            check_camera = _points([s["camera_point_m"] for s in check], "check camera points")
            check_base = _points([s["base_point_m"] for s in check], "check base points")
            result["check_errors"] = _error_report(check_camera, check_base, rotation, translation)
            for name, all_points in (("camera", np.vstack((camera, check_camera))),
                                     ("base", np.vstack((base, check_base)))):
                distances = np.linalg.norm(all_points[:, None] - all_points[None, :], axis=2)
                np.fill_diagonal(distances, np.inf)
                independent_distances = distances[len(camera):]
                minimum_distance = float(np.min(independent_distances))
                result["independent_check"][name] = {
                    "minimum_distance_from_other_measurements_m": minimum_distance,
                    "nonduplicate": bool(minimum_distance > 0.001),
                    "recommended_minimum_distance_m": 0.005,
                }
                if minimum_distance <= 0.001:
                    result["rejection_reasons"].append(
                        f"duplicate or near-duplicate {name} check points (<= 1 mm from "
                        "fit or another check point); acceptance requires independently placed points"
                    )
                elif minimum_distance < 0.005:
                    result["warnings"].append(
                        f"{name} check points are only {minimum_distance * 1000:.2f} mm "
                        "from another measurement; prefer independently placed points > 5 mm apart"
                    )
            if result["check_errors"]["max_m"] > check_max_limit_m:
                result["rejection_reasons"].append("independent check maximum error exceeds threshold")
        result["accepted"] = not result["rejection_reasons"]
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        result["rejection_reasons"].append(str(exc))
    return _json_safe(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", help="point-correspondence JSON dataset")
    parser.add_argument("--dataset", "--input", dest="dataset_option")
    parser.add_argument("--output", required=True, help="diagnostic/result JSON to write, including on failure")
    parser.add_argument("--min-fit", type=int, default=6)
    parser.add_argument("--min-check", type=int, default=3)
    parser.add_argument("--fit-rmse-mm", type=float, default=3.0)
    parser.add_argument("--fit-max-mm", type=float, default=5.0)
    parser.add_argument("--check-max-mm", type=float, default=5.0)
    parser.add_argument("--min-spatial-ratio", type=float, default=0.02)
    parser.add_argument("--allow-planar", action="store_true")
    args = parser.parse_args(argv)
    dataset_path = args.dataset_option or args.dataset
    if not dataset_path:
        parser.error("provide a dataset path or --dataset")
    output_path = Path(args.output).expanduser()
    if Path(dataset_path).expanduser().resolve() == output_path.resolve():
        parser.error("output must differ from input; retain the raw measurements")
    try:
        with Path(dataset_path).expanduser().open(encoding="utf-8") as stream:
            dataset = json.load(stream)
        result = evaluate_dataset(
            dataset, min_fit=args.min_fit, min_check=args.min_check,
            fit_rmse_limit_m=args.fit_rmse_mm / 1000.0,
            fit_max_limit_m=args.fit_max_mm / 1000.0,
            check_max_limit_m=args.check_max_mm / 1000.0,
            allow_planar=args.allow_planar, min_spatial_ratio=args.min_spatial_ratio,
        )
    except (OSError, ValueError, TypeError) as exc:
        result = evaluate_dataset({})
        result["rejection_reasons"] = [f"cannot load dataset: {exc}"]
    result["source_dataset"] = str(Path(dataset_path).expanduser().resolve())
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except OSError as exc:
        parser.exit(2, f"cannot write calibration diagnostic: {exc}\n")
    if result["accepted"]:
        print(f"ACCEPTED: camera -> base calibration saved to {output_path}")
    else:
        print(f"REJECTED: diagnostic saved to {output_path}; do not publish this transform")
        for reason in result["rejection_reasons"]:
            print(f"  - {reason}")
    for role in ("fit", "check"):
        report = result.get(f"{role}_errors")
        if report:
            print(f"{role}: N={report['sample_count']}, RMSE={report['rmse_m'] * 1000:.3f} mm, "
                  f"max={report['max_m'] * 1000:.3f} mm")
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
