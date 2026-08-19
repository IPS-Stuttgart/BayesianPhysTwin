from __future__ import annotations

import hashlib

import numpy as np
import pytest

from bayesian_phystwin.predictive_query_mixture import (
    SameMeanGaussianMixtureCandidateV1,
    SameMeanGaussianMixtureSelectionV1,
    compose_same_mean_gaussian_mixture,
)
from bayesian_phystwin.query_density_calibration import (
    density_region_contains,
    fit_query_density_calibration,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _prediction(predictor_id: str):
    mean = np.zeros((2, 3), dtype=np.float64)
    covariance = np.broadcast_to(
        0.01 * np.eye(3, dtype=np.float64),
        (2, 3, 3),
    ).copy()
    return compose_same_mean_gaussian_mixture(
        mean,
        covariance,
        covariance,
        reference_predictor_id=predictor_id,
        nominal_covariance_id="nominal-covariance",
        tail_covariance_id="tail-covariance",
    )


def _fit(prediction, predictor_id: str):
    return fit_query_density_calibration(
        calibration_group_ids=["group-a"],
        residual_groups=[np.zeros_like(prediction.mean_m)],
        prediction_groups=[prediction],
        nominal_coverage=0.5,
        predictor_id=predictor_id,
        query_set_id=_digest("query-set"),
        grouping_rule_id=_digest("grouping-rule"),
        guard_id=_digest("guard"),
        calibration_evidence_id=_digest("calibration-evidence"),
    )


def test_selection_rejects_non_gaussian_reference_candidate() -> None:
    non_gaussian = SameMeanGaussianMixtureCandidateV1(
        tail_covariance_scale=4.0,
    )
    gaussian = SameMeanGaussianMixtureCandidateV1(
        tail_covariance_scale=1.0,
    )
    assert non_gaussian.candidate_id is not None

    with pytest.raises(ValueError, match="exact Gaussian candidate"):
        SameMeanGaussianMixtureSelectionV1(
            predictor_id=_digest("predictor"),
            query_set_id=_digest("query-set"),
            grouping_rule_id=_digest("grouping-rule"),
            development_evidence_id=_digest("development-evidence"),
            development_group_ids=["group-a"],
            candidates=[non_gaussian, gaussian],
            group_negative_log_scores=np.asarray([[1.0], [2.0]]),
            group_rms_marginal_standard_deviations=np.asarray([[1.0], [1.0]]),
            reference_candidate_id=non_gaussian.candidate_id,
            selected_candidate_id=None,
            maximum_worst_group_regret=0.0,
            maximum_width_ratio=2.0,
            density_floor_variance_m2=0.0,
            grid_frozen_before_development_scores=True,
            target_outcomes_used=False,
        )


def test_calibration_fit_rejects_prediction_from_other_predictor() -> None:
    prediction = _prediction(_digest("predictor-a"))

    with pytest.raises(ValueError, match="prediction does not match predictor_id"):
        _fit(prediction, _digest("predictor-b"))


def test_calibration_application_rejects_prediction_from_other_predictor() -> None:
    predictor_a = _digest("predictor-a")
    prediction_a = _prediction(predictor_a)
    calibration = _fit(prediction_a, predictor_a)
    prediction_b = _prediction(_digest("predictor-b"))

    with pytest.raises(ValueError, match="prediction does not match predictor_id"):
        density_region_contains(
            np.zeros_like(prediction_b.mean_m),
            prediction_b,
            calibration,
            predictor_id=predictor_a,
        )
