from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.covariance_only_hybrid import (
    CovarianceOnlyHybridRecordV1,
    compose_covariance_only_hybrid,
)
from bayesian_phystwin.covariance_only_hybrid_analysis import (
    DONORS,
    HORIZONS,
    FoldSelection,
    _eigen_projection,
    _sign_test_pvalue,
    bootstrap_family,
    crossfit_select,
    effect_matrices,
    metric_for_fold,
    score_scale_grid,
    score_zero_covariance,
)


def _mean() -> np.ndarray:
    return np.zeros((2, 3), dtype=np.float64)


def _covariance() -> np.ndarray:
    return np.broadcast_to(
        np.eye(3, dtype=np.float64) * 1.0e-4,
        (2, 3, 3),
    ).copy()


def _record() -> CovarianceOnlyHybridRecordV1:
    return compose_covariance_only_hybrid(
        _mean(),
        _covariance(),
        reference_predictor_id="last_residual",
        covariance_donor_id="independent_endpoint_v1",
    ).record


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"mean_shape": "bad"}, "nonempty integer shape"),
        ({"mean_shape": 3}, "nonempty integer shape"),
        ({"mean_shape": ()}, "nonempty integer shape"),
        ({"mean_shape": (True, 3)}, "positive integer"),
        ({"mean_shape": (2, 0)}, "positive integer"),
        ({"covariance_shape": (2, 2)}, "incompatible"),
        ({"mean_object_identity_preserved": False}, "must remain true"),
        (
            {"minimum_covariance_scale": 2.0, "maximum_covariance_scale": 1.0},
            "must not be smaller",
        ),
        ({"reference_predictor_id": 1}, "canonical string"),
        ({"covariance_donor_id": "bad\nvalue"}, "single canonical line"),
        ({"metadata": {"bad": np.nan}}, "finite"),
    ],
)
def test_record_rejects_malformed_contract_fields(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        replace(_record(), **changes)


def test_record_accepts_its_exact_supplied_content_identity() -> None:
    record = _record()
    descriptor = record.descriptor()

    reconstructed = CovarianceOnlyHybridRecordV1(
        reference_predictor_id=str(descriptor["reference_predictor_id"]),
        covariance_donor_id=str(descriptor["covariance_donor_id"]),
        mean_shape=tuple(descriptor["mean_shape"]),
        covariance_shape=tuple(descriptor["covariance_shape"]),
        reference_mean_sha256=str(descriptor["reference_mean_sha256"]),
        donor_covariance_sha256=str(descriptor["donor_covariance_sha256"]),
        scale_schedule_sha256=str(descriptor["scale_schedule_sha256"]),
        output_covariance_sha256=str(descriptor["output_covariance_sha256"]),
        minimum_covariance_scale=float(descriptor["minimum_covariance_scale"]),
        maximum_covariance_scale=float(descriptor["maximum_covariance_scale"]),
        mean_object_identity_preserved=True,
        point_prediction_changed=False,
        metadata={},
        artifact_id=record.artifact_id,
    )

    assert reconstructed.artifact_id == record.artifact_id


@pytest.mark.parametrize(
    ("mean", "covariance", "scale", "tolerance", "match"),
    [
        (np.asarray(1.0), _covariance(), 1.0, 1.0e-10, "shape"),
        (np.zeros((2, 0)), np.zeros((2, 0, 0)), 1.0, 1.0e-10, "shape"),
        (_mean(), np.asarray(["bad"]), 1.0, 1.0e-10, "real numeric"),
        (
            _mean(),
            np.full((2, 3, 3), np.nan),
            1.0,
            1.0e-10,
            "finite",
        ),
        (_mean(), _covariance(), "bad", 1.0e-10, "real numeric"),
        (_mean(), _covariance(), np.nan, 1.0e-10, "finite"),
        (_mean(), _covariance(), 1.0, True, "finite number"),
        (_mean(), _covariance(), 1.0, np.nan, "finite number"),
        (_mean(), _covariance(), 1.0, 0.0, "positive"),
    ],
)
def test_composition_rejects_additional_malformed_inputs(
    mean: np.ndarray,
    covariance: object,
    scale: object,
    tolerance: object,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        compose_covariance_only_hybrid(
            mean,
            covariance,
            reference_predictor_id="last_residual",
            covariance_donor_id="independent_endpoint_v1",
            covariance_scale=scale,
            covariance_psd_tolerance=tolerance,  # type: ignore[arg-type]
        )


def _scoring_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    error = np.zeros((3, 2, 3), dtype=np.float64)
    covariance = np.broadcast_to(
        np.eye(3, dtype=np.float64) * 0.01,
        (3, 2, 3, 3),
    ).copy()
    valid = np.ones((3, 2), dtype=bool)
    return error, covariance, valid


@pytest.mark.parametrize(
    ("error", "covariance", "valid", "match"),
    [
        (
            np.zeros((2, 3)),
            np.zeros((2, 3, 3)),
            np.ones((2,), dtype=bool),
            "shapes differ",
        ),
        (
            np.zeros((2, 1, 3)),
            np.zeros((2, 1, 3, 3)),
            np.zeros((2, 1), dtype=bool),
            "no valid",
        ),
        (
            np.zeros((1, 1, 3)),
            np.asarray([[[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]]]),
            np.ones((1, 1), dtype=bool),
            "positive semidefinite",
        ),
    ],
)
def test_eigen_projection_rejects_invalid_scoring_inputs(
    error: np.ndarray,
    covariance: np.ndarray,
    valid: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _eigen_projection(error, covariance, valid)


def test_zero_covariance_score_uses_floor_and_rejects_empty_support() -> None:
    error, _, valid = _scoring_arrays()

    nll, coverage, width = score_zero_covariance(
        error,
        valid,
        observation_std_m=1.0e-6,
        eigenvalue_floor_m2=1.0e-4,
        marginal_coverage_z=1.0,
    )

    assert np.isfinite(nll)
    assert coverage == 1.0
    assert width == pytest.approx(0.02)
    with pytest.raises(ValueError, match="no valid"):
        score_zero_covariance(
            error,
            np.zeros_like(valid),
            observation_std_m=0.005,
            eigenvalue_floor_m2=1.0e-12,
            marginal_coverage_z=1.0,
        )


def test_scale_grid_uses_eigenvalue_floor() -> None:
    error, covariance, valid = _scoring_arrays()
    covariance.fill(0.0)

    nll, coverage, width = score_scale_grid(
        error,
        covariance,
        valid,
        scales=(0.5, 2.0),
        observation_std_m=1.0e-6,
        eigenvalue_floor_m2=1.0e-4,
        marginal_coverage_z=1.0,
    )

    np.testing.assert_allclose(nll[0], nll[1])
    np.testing.assert_allclose(coverage, 1.0)
    np.testing.assert_allclose(width, 0.02)


def test_crossfit_rejects_wrong_shape_and_nonfinite_grid() -> None:
    cases = ("case-0", "case-1")
    scales = (0.5, 1.0)
    good = np.ones((2, len(DONORS), len(HORIZONS), len(scales)))

    with pytest.raises(ValueError, match="shape"):
        crossfit_select(cases, good[..., :1], scales)
    bad = good.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        crossfit_select(cases, bad, scales)


def test_crossfit_can_select_dynamic_donor_and_nonraw_scale() -> None:
    cases = ("case-0", "case-1", "case-2")
    scales = (0.5, 1.0, 2.0)
    grid = np.full((3, len(DONORS), len(HORIZONS), len(scales)), 10.0)
    grid[:, 0, :, :] = 5.0
    grid[:, 1, :, 2] = 0.0

    folds, full = crossfit_select(cases, grid, scales)

    assert all(fold.selected_donor == "dynamic_endpoint_v2" for fold in folds)
    assert all(fold.selected_scales == (2.0, 2.0, 2.0) for fold in folds)
    assert full["selected_donor"] == "dynamic_endpoint_v2"


def _folds() -> tuple[FoldSelection, ...]:
    donor_scales = {
        "independent_endpoint_v1": (1.0, 1.0, 1.0),
        "dynamic_endpoint_v2": (2.0, 2.0, 2.0),
    }
    return (
        FoldSelection(
            held_case_id="case-0",
            selected_donor="independent_endpoint_v1",
            selected_scales=(1.0, 1.0, 1.0),
            donor_scales=donor_scales,
            donor_training_scores={name: float(index) for index, name in enumerate(DONORS)},
        ),
        FoldSelection(
            held_case_id="case-1",
            selected_donor="dynamic_endpoint_v2",
            selected_scales=(2.0, 2.0, 2.0),
            donor_scales=donor_scales,
            donor_training_scores={name: float(index) for index, name in enumerate(DONORS)},
        ),
    )


def test_effect_and_metric_read_each_fold_donor_and_scale() -> None:
    scales = (1.0, 2.0)
    reference = np.zeros((2, len(HORIZONS)))
    grid = np.empty((2, len(DONORS), len(HORIZONS), len(scales)))
    grid[:, 0, :, 0] = 1.0
    grid[:, 0, :, 1] = 2.0
    grid[:, 1, :, 0] = 3.0
    grid[:, 1, :, 1] = 4.0
    folds = _folds()

    effects = effect_matrices(reference, grid, scales, folds)
    selected = metric_for_fold(grid, folds, scales)

    np.testing.assert_allclose(effects["crossfit_selected_scaled_covariance"][0], 1.0)
    np.testing.assert_allclose(effects["crossfit_selected_scaled_covariance"][1], 4.0)
    np.testing.assert_allclose(selected[0], 1.0)
    np.testing.assert_allclose(selected[1], 4.0)


def test_sign_test_handles_exact_ties_and_mixed_signs() -> None:
    assert _sign_test_pvalue(np.zeros(4)) == 1.0
    assert 0.0 < _sign_test_pvalue(np.asarray([-1.0, -1.0, 1.0, 1.0])) <= 1.0


@pytest.mark.parametrize(
    ("replicates", "confidence"),
    [(999, 0.95), (1000, 0.5), (1000, 1.0)],
)
def test_bootstrap_rejects_invalid_configuration(
    replicates: int,
    confidence: float,
) -> None:
    with pytest.raises(ValueError, match="bootstrap"):
        bootstrap_family(
            {"arm": np.zeros((3, len(HORIZONS)))},
            arm_order=("arm",),
            replicates=replicates,
            seed=1,
            confidence=confidence,
        )


def test_bootstrap_rejects_bad_shapes_and_case_count_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        bootstrap_family(
            {"arm": np.zeros((3, 2))},
            arm_order=("arm",),
            replicates=1000,
            seed=1,
            confidence=0.95,
        )
    with pytest.raises(ValueError, match="different case counts"):
        bootstrap_family(
            {
                "first": np.zeros((3, len(HORIZONS))),
                "second": np.zeros((4, len(HORIZONS))),
            },
            arm_order=("first", "second"),
            replicates=1000,
            seed=1,
            confidence=0.95,
        )


def test_bootstrap_reports_better_worse_and_inconclusive() -> None:
    shape = (4, len(HORIZONS))
    rows = bootstrap_family(
        {
            "better": -np.ones(shape),
            "worse": np.ones(shape),
            "tie": np.zeros(shape),
        },
        arm_order=("better", "worse", "tie"),
        replicates=1000,
        seed=5,
        confidence=0.95,
    )

    overall = {
        str(row["arm"]): str(row["familywise_decision"])
        for row in rows
        if row["aggregation"] == "overall"
    }
    assert overall == {
        "better": "hybrid_better",
        "worse": "hybrid_worse",
        "tie": "inconclusive",
    }
