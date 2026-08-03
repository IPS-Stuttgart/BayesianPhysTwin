from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.bias_aware_belief import SourceRegretCertificate
from bayesian_phystwin.pokeflex_baseline_relative_guard import (
    FEATURE_NAMES,
    apply_baseline_relative_guard,
    baseline_relative_guard_decision,
    certificate_from_payload,
    certificate_to_payload,
    extract_baseline_relative_guard_features,
    fit_baseline_relative_guard_certificate,
    summarize_guard_decisions,
)


def _certificate(intercept: float, *, support: float = 2.0) -> SourceRegretCertificate:
    dimension = len(FEATURE_NAMES)
    return SourceRegretCertificate(
        feature_center=np.zeros(dimension),
        feature_scale=np.ones(dimension),
        standardized_feature_lower=np.full(dimension, -support),
        standardized_feature_upper=np.full(dimension, support),
        coefficients=np.asarray([intercept, *np.zeros(dimension)]),
        upper_residual_quantile=0.0,
        nominal_coverage=0.75,
        minimum_improvement=0.0,
        ridge_penalty=0.0,
        support_margin_std=0.0,
        source_group_count=3,
        finite_sample_rank=3,
        finite_sample_coverage=0.75,
    )


def test_guard_features_measure_physical_alignment_without_an_outcome() -> None:
    source = np.zeros((4, 3))
    prior_motion = np.tile(np.asarray([0.001, 0.0, 0.0]), (4, 1))
    correction = 2.0 * prior_motion

    features = extract_baseline_relative_guard_features(
        correction,
        source,
        source + prior_motion,
        correction,
        association_count=99,
    )

    assert features.shape == (len(FEATURE_NAMES),)
    assert features[0] == pytest.approx(np.log1p(2.0))
    assert features[1] == pytest.approx(1.0)
    assert features[2] == pytest.approx(1.0)
    assert features[3] == pytest.approx(1.0)
    assert features[4] == pytest.approx(np.log1p(99))
    assert features[5] == pytest.approx(2.0)


def test_negative_upper_regret_admits_and_nonnegative_bound_falls_back() -> None:
    features = np.zeros(len(FEATURE_NAMES))
    baseline = np.arange(12, dtype=np.float64).reshape(4, 3)
    candidate = baseline + 0.001

    selected, accepted = apply_baseline_relative_guard(
        baseline, candidate, _certificate(-0.1), features
    )
    fallback, rejected = apply_baseline_relative_guard(
        baseline, candidate, _certificate(0.1), features
    )

    assert accepted["accepted"] is True
    assert np.array_equal(selected, candidate)
    assert rejected["accepted"] is False
    assert np.array_equal(fallback.view(np.uint64), baseline.view(np.uint64))


def test_out_of_support_candidate_uses_exact_fallback() -> None:
    features = np.full(len(FEATURE_NAMES), 3.0)
    baseline = np.zeros((2, 3))
    candidate = np.ones((2, 3))

    selected, decision = apply_baseline_relative_guard(
        baseline, candidate, _certificate(-100.0, support=1.0), features
    )

    assert decision["accepted"] is False
    assert decision["in_source_support"] is False
    assert decision["upper_regret_mm"] is None
    assert np.array_equal(selected.view(np.uint64), baseline.view(np.uint64))


def test_certificate_round_trip_preserves_decision() -> None:
    certificate = _certificate(-0.25)
    restored = certificate_from_payload(certificate_to_payload(certificate))
    features = np.zeros(len(FEATURE_NAMES))

    assert baseline_relative_guard_decision(certificate, features) == (
        baseline_relative_guard_decision(restored, features)
    )


def test_certificate_fit_uses_physical_objects_as_groups() -> None:
    rows = []
    for object_index, object_name in enumerate(("A", "B", "C")):
        for row_index in range(4):
            value = float(object_index + row_index) / 10.0
            rows.append(
                {
                    "object": object_name,
                    "features": dict.fromkeys(FEATURE_NAMES, value),
                    "regret_mm": -0.1 + 0.01 * value,
                }
            )

    certificate = fit_baseline_relative_guard_certificate(rows)

    assert certificate.source_group_count == 3
    assert certificate.feature_center.shape == (len(FEATURE_NAMES),)


def test_summary_counts_fully_unsupported_objects_as_exact_ties() -> None:
    decisions = [
        {
            "domain": "fresh",
            "object": "supported",
            "take_id": "supported_T1",
            "target_frame": 1,
            "take_target_frame_count": 2,
            "selected_regret_mm": -0.2,
            "accepted": True,
        }
    ]
    summary = summarize_guard_decisions(
        decisions,
        domain="fresh",
        object_baseline_mm={"supported": 2.0, "unsupported": 4.0},
        take_inventory={
            "supported_T1": {"object": "supported", "frame_count": 2},
            "unsupported_T1": {"object": "unsupported", "frame_count": 3},
        },
    )

    assert summary["object_count"] == 2
    assert summary["object_wins"] == 1
    assert summary["object_ties"] == 1
    assert summary["object_losses"] == 0
    assert summary["supported_object_count"] == 1
    assert summary["guarded_object_balanced_CD_UL1_mm"] == pytest.approx(2.95)


def test_summary_rejects_duplicate_frame_decisions() -> None:
    decision = {
        "domain": "fresh",
        "object": "object",
        "take_id": "object_T1",
        "target_frame": 1,
        "take_target_frame_count": 1,
        "selected_regret_mm": -0.1,
        "accepted": True,
    }

    with pytest.raises(ValueError, match="duplicate guard decision"):
        summarize_guard_decisions(
            [decision, decision],
            domain="fresh",
            object_baseline_mm={"object": 2.0},
            take_inventory={
                "object_T1": {"object": "object", "frame_count": 1}
            },
        )
