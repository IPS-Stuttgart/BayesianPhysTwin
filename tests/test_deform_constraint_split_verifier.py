"""Synthetic checks of the second, production-independent arithmetic path."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def verifier():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/verify_deform_constraint_split_source.py"
    )
    spec = importlib.util.spec_from_file_location("split_second_arithmetic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_straight_qr_projection(verifier):
    points = np.zeros((12, 3))
    points[:, 0] = 0.05 * np.arange(12)
    delta = np.zeros_like(points)
    delta[2:10] = (0.01, 0.02, 0.03)
    value = verifier.qr_projection(points, delta, (0, 1, 10, 11))
    np.testing.assert_allclose(value[:, 0], 0, atol=1e-16)
    np.testing.assert_allclose(value[:, 1:], delta[:, 1:], atol=1e-16)


def test_qr_idempotence_and_complement(verifier):
    rng = np.random.default_rng(42)
    points = rng.normal(size=(12, 3))
    delta = rng.normal(size=(12, 3))
    delta[(0, 1, 10, 11), :] = 0
    value = verifier.qr_projection(points, delta, (0, 1, 10, 11))
    np.testing.assert_allclose(
        verifier.qr_projection(points, value, (0, 1, 10, 11)), value, atol=1e-14
    )
    np.testing.assert_allclose(np.sum(value * (delta - value)), 0, atol=1e-14)


def test_qr_degenerate_geometry_fails(verifier):
    with pytest.raises(ValueError, match="invalid nominal geometry"):
        verifier.qr_projection(np.zeros((12, 3)), np.ones((12, 3)), (0, 1, 10, 11))


def test_second_metrics_known_values(verifier):
    truth = np.zeros((2, 3, 3))
    predicted = np.zeros_like(truth)
    predicted[:, :, 0] = 0.003
    predicted[:, :, 1] = 0.004
    result = verifier.metric(predicted, truth)
    assert result["coordinate_l1_mm"] == pytest.approx(7 / 3)
    assert result["point_rmse_mm"] == pytest.approx(5)
    assert result["fde_mm"] == pytest.approx(5)


def test_second_content_binding(verifier, tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"artifact_id":"incorrect","passed":true}')
    with pytest.raises(ValueError, match="canonical digest"):
        verifier.read_bound(path)
