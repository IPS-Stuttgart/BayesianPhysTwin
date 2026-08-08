from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.dynamic_discrepancy import (
    DynamicDiscrepancyCorrection,
    fit_dimensionless_linearized_correction,
    load_dynamic_discrepancy_correction,
    prefix_position_velocity_coefficients,
    scale_coefficients_to_field_limit,
    write_dynamic_discrepancy_correction,
)


def _basis(node_count: int = 12) -> np.ndarray:
    values = np.column_stack(
        (
            np.ones(node_count),
            np.linspace(-1.0, 1.0, node_count),
            np.cos(np.linspace(0.0, np.pi, node_count)),
            np.sin(np.linspace(0.0, 2.0 * np.pi, node_count)),
        )
    )
    return np.linalg.qr(values, mode="reduced")[0]


def _correction() -> DynamicDiscrepancyCorrection:
    basis = _basis()
    zeros = np.zeros((4, 3))
    return DynamicDiscrepancyCorrection(
        case_id="case_a",
        graph_basis=basis,
        graph_eigenvalues=np.asarray((0.0, 0.1, 0.2, 0.3)),
        position_coefficients_m=zeros,
        velocity_coefficients_mps=zeros,
        generalized_force_coefficients_n=zeros,
        structural_coefficients_m=zeros,
        prefix_frame_start=19,
        prefix_frame_stop=26,
        frame_dt_s=0.05,
        information_boundary={
            "o_plus_prefix_frames": 6,
            "future_frames_used_for_fit_or_selection": False,
            "manual_tracks_used_for_fit_or_selection": False,
            "graph_rank": 4,
        },
        regularization={"dimensionless_ridge": 1e-4},
        source_checksums={"source": "a" * 64},
    )


def test_dynamic_discrepancy_roundtrip_and_fields(tmp_path) -> None:
    identity = _correction()
    coefficients = np.arange(12, dtype=float).reshape(4, 3) * 1e-4
    correction = replace(
        identity,
        position_coefficients_m=coefficients,
        velocity_coefficients_mps=2.0 * coefficients,
        generalized_force_coefficients_n=3.0 * coefficients,
        structural_coefficients_m=4.0 * coefficients,
    )
    written = write_dynamic_discrepancy_correction(tmp_path / "correction", correction)
    loaded = load_dynamic_discrepancy_correction(written["manifest_path"])

    assert loaded.artifact_id == correction.artifact_id
    np.testing.assert_allclose(loaded.position_field_m(), loaded.graph_basis @ coefficients)
    np.testing.assert_allclose(
        loaded.generalized_force_field_n(), loaded.graph_basis @ (3.0 * coefficients)
    )


def test_artifact_rejects_empty_or_negative_prefix_interval() -> None:
    with pytest.raises(ValueError, match="prefix interval"):
        replace(_correction(), prefix_frame_start=-1)


def test_artifact_enforces_six_frame_o_plus_boundary() -> None:
    with pytest.raises(ValueError, match="information boundary"):
        replace(
            _correction(),
            information_boundary={
                "o_plus_prefix_frames": 5,
                "future_frames_used_for_fit_or_selection": False,
                "manual_tracks_used_for_fit_or_selection": False,
                "graph_rank": 4,
            },
        )


def test_artifact_requires_exact_frozen_prefix_length() -> None:
    with pytest.raises(ValueError, match="exactly six O-plus frames"):
        replace(_correction(), prefix_frame_stop=27)


def test_prefix_position_and_velocity_recover_linear_graph_field() -> None:
    basis = _basis()
    frame_dt = 0.04
    position = np.asarray(
        ((0.01, 0.0, 0.0), (0.0, -0.005, 0.0), (0.0, 0.0, 0.003), (0.0, 0.0, 0.0))
    )
    velocity = np.asarray(
        ((0.02, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, -0.015), (0.0, 0.0, 0.0))
    )
    times = frame_dt * np.arange(7)
    coefficients = position[None] + (times - times[-1])[:, None, None] * velocity[None]
    residual = np.einsum("nr,trc->tnc", basis, coefficients)
    valid = np.ones(residual.shape[:2], dtype=bool)

    endpoint, estimated_velocity, history = prefix_position_velocity_coefficients(
        residual,
        valid,
        basis,
        frame_dt_s=frame_dt,
        ridge=1e-12,
    )

    np.testing.assert_allclose(endpoint, position, atol=1e-10)
    np.testing.assert_allclose(estimated_velocity, velocity, atol=1e-10)
    assert history.shape == (7, 4, 3)


def test_linearized_fit_uses_only_supplied_prefix() -> None:
    rng = np.random.default_rng(8)
    response = rng.normal(scale=0.002, size=(7, 12, 3, 5))
    expected = np.asarray((0.2, -0.4, 0.6, 0.0, 0.3))
    residual = np.einsum("tncp,p->tnc", response, expected)
    valid = np.ones(residual.shape[:2], dtype=bool)

    fitted, diagnostics = fit_dimensionless_linearized_correction(
        residual,
        valid,
        response,
        ridge=1e-12,
    )

    np.testing.assert_allclose(fitted, expected, atol=1e-8)
    assert diagnostics["linearized_prefix_rmse_m"] < 1e-10


def test_field_limit_preserves_direction() -> None:
    basis = _basis()
    coefficients = np.ones((4, 3))
    limited, diagnostics = scale_coefficients_to_field_limit(
        basis,
        coefficients,
        maximum_node_norm=0.01,
    )

    assert diagnostics["limit_applied"] is True
    assert np.max(np.linalg.norm(basis @ limited, axis=1)) == pytest.approx(0.01)
    ratios = limited[coefficients != 0.0] / coefficients[coefficients != 0.0]
    np.testing.assert_allclose(ratios, ratios[0])
