from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.interventional_cause_adequacy_calibration_v1 import (
    CAUSE_FAMILY_CALIBRATION_CLAIM_BOUNDARY,
    CauseFamilyAdequacyCalibrationV1,
)
from bayesian_phystwin_experiments.interventional_cause_adequacy_v1 import (
    CauseFamilyAdequacyStatus,
)

SHA = "a" * 64


def _calibration(
    scores: dict[str, float],
    *,
    alpha: float = 0.1,
    **kwargs: object,
) -> CauseFamilyAdequacyCalibrationV1:
    return CauseFamilyAdequacyCalibrationV1(
        cause_family_id=SHA,
        intervention_roster_id=SHA,
        whitening_id=SHA,
        grouping_rule_id=SHA,
        source_group_scores=scores,
        miscoverage_alpha=alpha,
        candidate_family_frozen_before_scores=True,
        target_outcomes_used=False,
        **kwargs,
    )


def test_exact_split_conformal_order_statistic() -> None:
    calibration = _calibration(
        {f"group-{index:02d}": float(index) for index in range(1, 20)},
        alpha=0.1,
    )

    assert calibration.quantile_index_one_based == 18
    assert calibration.noise_radius == 18.0
    assert calibration.finite_sample_coverage_lower_bound == pytest.approx(0.9)
    assert calibration.to_record()["claim_boundary"] == (
        CAUSE_FAMILY_CALIBRATION_CLAIM_BOUNDARY
    )


def test_group_order_does_not_change_calibration_identity() -> None:
    first = _calibration({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}, alpha=0.4)
    second = _calibration(
        {"d": 0.4, "c": 0.3, "b": 0.2, "a": 0.1},
        alpha=0.4,
    )

    assert second.noise_radius == first.noise_radius
    assert second.calibration_id == first.calibration_id


def test_calibration_applies_exact_frozen_radius() -> None:
    calibration = _calibration(
        {f"group-{index}": value for index, value in enumerate([0.1, 0.2, 0.3, 0.4])},
        alpha=0.4,
    )
    certificate = calibration.certify(
        residual_id=SHA,
        cause_signature_ids={"state": SHA},
        cause_signatures={"state": np.asarray([[1.0], [0.0]])},
        whitened_residual=np.asarray([1.0, 0.25]),
    )

    assert calibration.noise_radius == 0.4
    assert certificate.noise_radius == calibration.noise_radius
    assert certificate.status is CauseFamilyAdequacyStatus.ADEQUATE_UNIQUE
    assert certificate.metadata["cause_family_calibration_id"] == (
        calibration.calibration_id
    )


def test_too_few_groups_for_requested_alpha_fails_closed() -> None:
    with pytest.raises(ValueError, match="too few source groups"):
        _calibration({"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}, alpha=0.01)


def test_target_informed_or_post_score_family_is_rejected() -> None:
    common = dict(
        cause_family_id=SHA,
        intervention_roster_id=SHA,
        whitening_id=SHA,
        grouping_rule_id=SHA,
        source_group_scores={"a": 0.1, "b": 0.2},
        miscoverage_alpha=0.5,
    )
    with pytest.raises(ValueError, match="frozen before scores"):
        CauseFamilyAdequacyCalibrationV1(
            **common,
            candidate_family_frozen_before_scores=False,
            target_outcomes_used=False,
        )
    with pytest.raises(ValueError, match="target outcomes"):
        CauseFamilyAdequacyCalibrationV1(
            **common,
            candidate_family_frozen_before_scores=True,
            target_outcomes_used=True,
        )


def test_invalid_scores_and_identity_tampering_fail_closed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _calibration({"a": 0.1, "b": -0.2}, alpha=0.5)
    calibration = _calibration({"a": 0.1, "b": 0.2}, alpha=0.5)
    with pytest.raises(ValueError, match="calibration_id"):
        _calibration(
            {"a": 0.1, "b": 0.2},
            alpha=0.5,
            calibration_id="b" * 64,
        )
    assert calibration.calibration_id is not None
