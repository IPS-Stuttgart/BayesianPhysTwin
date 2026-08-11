from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import bayesian_phystwin.covariance_only_hybrid as composition
import bayesian_phystwin.covariance_only_hybrid_analysis as analysis


def _mean() -> np.ndarray:
    return np.zeros((2, 3, 3), dtype=np.float64)


def _covariance() -> np.ndarray:
    return np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (2, 3, 3, 3),
    ).copy()


def _record() -> composition.CovarianceOnlyHybridRecordV1:
    return composition.compose_covariance_only_hybrid(
        _mean(),
        _covariance(),
        reference_predictor_id="last_residual",
        covariance_donor_id="independent_endpoint_v1",
    ).record


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (" leading-space", "canonical string"),
        ("two\nlines", "single canonical line"),
    ],
)
def test_identifiers_must_be_canonical_single_lines(
    value: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        composition.compose_covariance_only_hybrid(
            _mean(),
            _covariance(),
            reference_predictor_id=value,
            covariance_donor_id="independent_endpoint_v1",
        )


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (True, "finite number"),
        (float("nan"), "finite number"),
        (0.0, "positive"),
    ],
)
def test_psd_tolerance_must_be_a_finite_positive_number(
    value: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        composition.compose_covariance_only_hybrid(
            _mean(),
            _covariance(),
            reference_predictor_id="last_residual",
            covariance_donor_id="independent_endpoint_v1",
            covariance_psd_tolerance=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "value",
    [
        "3",
        3,
        (),
        (True, 3),
        (0, 3),
    ],
)
def test_shape_parser_rejects_noncanonical_shapes(value: object) -> None:
    with pytest.raises(ValueError, match="shape"):
        composition._shape(value, name="shape")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mean", "match"),
    [
        (np.asarray(0.0, dtype=np.float64), "shape"),
        (np.zeros((2, 3), dtype=np.float64)[:, ::-1], "C-contiguous"),
        (np.full((2, 3), np.nan, dtype=np.float64), "finite"),
    ],
)
def test_reference_mean_rejects_invalid_shape_layout_and_values(
    mean: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        composition.compose_covariance_only_hybrid(
            mean,
            np.zeros(mean.shape + (mean.shape[-1] if mean.ndim else 1,)),
            reference_predictor_id="last_residual",
            covariance_donor_id="independent_endpoint_v1",
        )


@pytest.mark.parametrize(
    ("covariance", "match"),
    [
        (np.full((2, 3, 3, 3), "not-a-number", dtype=object), "real numeric"),
        (np.full((2, 3, 3, 3), np.nan, dtype=np.float64), "finite"),
    ],
)
def test_donor_covariance_rejects_non_numeric_and_nonfinite_values(
    covariance: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        composition.compose_covariance_only_hybrid(
            _mean(),
            covariance,
            reference_predictor_id="last_residual",
            covariance_donor_id="independent_endpoint_v1",
        )


def test_scale_schedule_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="real numeric"):
        composition.compose_covariance_only_hybrid(
            _mean(),
            _covariance(),
            reference_predictor_id="last_residual",
            covariance_donor_id="independent_endpoint_v1",
            covariance_scale="large",
        )


def test_record_rejects_incompatible_shape_identity_and_scale_order() -> None:
    record = _record()

    with pytest.raises(ValueError, match="covariance_shape"):
        replace(record, covariance_shape=(2, 3, 3, 2))
    with pytest.raises(ValueError, match="mean_object_identity_preserved"):
        replace(record, mean_object_identity_preserved=False)
    with pytest.raises(ValueError, match="maximum_covariance_scale"):
        replace(
            record,
            minimum_covariance_scale=2.0,
            maximum_covariance_scale=1.0,
        )


def test_record_accepts_its_existing_exact_artifact_identity() -> None:
    record = _record()

    assert replace(record) == record


def test_composition_asserts_if_prediction_constructor_copies_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def copied_prediction(**values: object) -> SimpleNamespace:
        mean = values["mean_m"]
        assert isinstance(mean, np.ndarray)
        return SimpleNamespace(
            mean_m=mean.copy(),
            covariance_m2=values["covariance_m2"],
            record=values["record"],
        )

    monkeypatch.setattr(
        composition,
        "CovarianceOnlyHybridPredictionV1",
        copied_prediction,
    )
    with pytest.raises(AssertionError, match="copied the reference mean"):
        composition.compose_covariance_only_hybrid(
            _mean(),
            _covariance(),
            reference_predictor_id="last_residual",
            covariance_donor_id="independent_endpoint_v1",
        )


def _analysis_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    error = np.zeros((2, 2, 3), dtype=np.float64)
    covariance = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (2, 2, 3, 3),
    ).copy()
    valid = np.ones((2, 2), dtype=bool)
    return error, covariance, valid


def test_eigen_projection_rejects_shape_empty_and_indefinite_inputs() -> None:
    error, covariance, valid = _analysis_arrays()

    with pytest.raises(ValueError, match="shapes differ"):
        analysis._eigen_projection(error[..., :2], covariance, valid)
    with pytest.raises(ValueError, match="no valid"):
        analysis._eigen_projection(error, covariance, np.zeros_like(valid))
    covariance[..., 0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        analysis._eigen_projection(error, covariance, valid)


def test_zero_covariance_scoring_covers_valid_and_empty_horizons() -> None:
    error, _, valid = _analysis_arrays()

    nll, coverage, width = analysis.score_zero_covariance(
        error,
        valid,
        observation_std_m=0.005,
        eigenvalue_floor_m2=1e-12,
        marginal_coverage_z=1.6448536269514722,
    )

    assert np.isfinite(nll)
    assert coverage == 1.0
    assert width > 0.0
    with pytest.raises(ValueError, match="no valid"):
        analysis.score_zero_covariance(
            error,
            np.zeros_like(valid),
            observation_std_m=0.005,
            eigenvalue_floor_m2=1e-12,
            marginal_coverage_z=1.6448536269514722,
        )


@pytest.mark.parametrize("invalid", ["shape", "nonfinite"])
def test_crossfit_selection_rejects_invalid_grids(invalid: str) -> None:
    cases = ("case-0", "case-1")
    scales = (0.5, 1.0, 2.0)
    if invalid == "shape":
        grid = np.zeros((2, 2, 3, 2), dtype=np.float64)
    else:
        grid = np.zeros((2, 2, 3, 3), dtype=np.float64)
        grid[0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="nll_grid"):
        analysis.crossfit_select(cases, grid, scales)


def test_metric_for_fold_reads_each_selected_donor_and_scale() -> None:
    scales = (1.0, 2.0)
    grid = np.arange(2 * 2 * 3 * 2, dtype=np.float64).reshape(2, 2, 3, 2)
    folds = (
        analysis.FoldSelection(
            held_case_id="case-0",
            selected_donor="independent_endpoint_v1",
            selected_scales=(1.0, 2.0, 1.0),
            donor_scales={},
            donor_training_scores={},
        ),
        analysis.FoldSelection(
            held_case_id="case-1",
            selected_donor="dynamic_endpoint_v2",
            selected_scales=(2.0, 1.0, 2.0),
            donor_scales={},
            donor_training_scores={},
        ),
    )

    observed = analysis.metric_for_fold(grid, folds, scales)

    expected = np.asarray(
        [
            [grid[0, 0, 0, 0], grid[0, 0, 1, 1], grid[0, 0, 2, 0]],
            [grid[1, 1, 0, 1], grid[1, 1, 1, 0], grid[1, 1, 2, 1]],
        ]
    )
    np.testing.assert_array_equal(observed, expected)


def test_sign_test_returns_one_for_all_ties() -> None:
    assert analysis._sign_test_pvalue(np.zeros(5, dtype=np.float64)) == 1.0


def test_bootstrap_family_rejects_invalid_configuration_and_matrices() -> None:
    valid = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="bootstrap configuration"):
        analysis.bootstrap_family(
            {"first": valid},
            arm_order=("first",),
            replicates=999,
            seed=1,
            confidence=0.95,
        )
    with pytest.raises(ValueError, match="changed shape"):
        analysis.bootstrap_family(
            {"first": np.zeros((2, 2), dtype=np.float64)},
            arm_order=("first",),
            replicates=1000,
            seed=1,
            confidence=0.95,
        )
    with pytest.raises(ValueError, match="different case counts"):
        analysis.bootstrap_family(
            {
                "first": valid,
                "second": np.zeros((3, 3), dtype=np.float64),
            },
            arm_order=("first", "second"),
            replicates=1000,
            seed=1,
            confidence=0.95,
        )
