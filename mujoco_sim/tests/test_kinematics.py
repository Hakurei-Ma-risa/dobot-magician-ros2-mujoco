import mujoco
import numpy as np

from dobot_mujoco.kinematics import (
    FLANGE_TO_TOOL_TIP,
    analytic_tcp_position,
    analytic_wrist_position,
    model_to_raw,
    raw_to_model,
)
from dobot_mujoco.model import load_model, set_raw_pose, site_position, site_rotation


def test_raw_model_roundtrip() -> None:
    raw = np.deg2rad([47.0, 36.0, 12.0, -80.0])
    np.testing.assert_allclose(model_to_raw(raw_to_model(raw)), raw, atol=1e-12)


def test_known_zero_pose() -> None:
    raw = np.zeros(4)
    expected_wrist = np.array([0.147, 0.0, 0.131305 + 0.135])
    expected_tcp = expected_wrist + np.array([0.060, 0.0, -FLANGE_TO_TOOL_TIP])
    np.testing.assert_allclose(analytic_wrist_position(raw), expected_wrist, atol=1e-12)
    np.testing.assert_allclose(analytic_tcp_position(raw), expected_tcp, atol=1e-12)


def test_mujoco_sites_match_analytic_fk() -> None:
    model = load_model()
    data = mujoco.MjData(model)
    raw = np.deg2rad([-83.0, 62.0, 41.0, 129.0])
    set_raw_pose(model, data, raw)
    np.testing.assert_allclose(site_position(model, data, "wrist_site"), analytic_wrist_position(raw), atol=1e-10)
    np.testing.assert_allclose(site_position(model, data, "tcp_site"), analytic_tcp_position(raw), atol=1e-10)
    np.testing.assert_allclose(site_rotation(model, data, "tcp_site")[:, 2], [0.0, 0.0, 1.0], atol=1e-10)


def test_upstream_cad_visual_parts_are_loaded() -> None:
    model = load_model()
    assert model.nmesh == 19
    # Includes the legacy scene, base camera housing, and parked clutter bodies.
    assert model.ngeom == 40
