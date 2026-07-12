import numpy as np

from bayesian_phystwin.observation_model_audit import (
    cross_view_residual_audit,
    metric_agreement_audit,
    released_observation_capability_audit,
)


def _basis(node_count=20):
    values = np.column_stack(
        (
            np.ones(node_count),
            np.linspace(-1.0, 1.0, node_count),
            np.cos(np.linspace(0.0, np.pi, node_count)),
            np.sin(np.linspace(0.0, 2.0 * np.pi, node_count)),
        )
    )
    return np.linalg.qr(values, mode="reduced")[0]


def test_cross_view_audit_recognizes_transferable_physical_field() -> None:
    rng = np.random.default_rng(4)
    basis = _basis()
    baseline = rng.normal(scale=0.01, size=(7, 20, 3))
    coefficient = rng.normal(scale=0.004, size=(4, 3))
    field = basis @ coefficient
    observed = np.stack(
        [baseline + field[None] + rng.normal(scale=1e-5, size=baseline.shape) for _ in range(3)]
    )
    valid = np.ones(observed.shape[:3], dtype=bool)

    result = cross_view_residual_audit(
        observed,
        valid,
        baseline,
        basis,
        ridge=1e-10,
    )

    assert result["status"] == "available"
    assert result["mean_cross_view_error_ratio"] < 0.02
    assert result["relative_coefficient_dispersion"] < 0.01


def test_released_capability_audit_fails_closed_without_per_view_tracks() -> None:
    result = released_observation_capability_audit(
        {"object_points": np.zeros((3, 2, 3)), "object_visibilities": np.ones((3, 2))}
    )

    assert result["cross_view_residual_fields"]["available"] is False
    assert result["visibility_confidence_regression"]["available"] is False
    assert result["manual_track_and_chamfer_agreement"]["available"] is True


def test_metric_agreement_reports_framewise_correlation() -> None:
    result = metric_agreement_audit(
        np.asarray((1.0, 2.0, 3.0, 4.0)),
        np.asarray((2.0, 4.0, 6.0, 8.0)),
    )

    assert result["pearson_correlation"] == 1.0
    assert result["frame_count"] == 4
