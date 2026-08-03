from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from bayesian_phystwin.deform360_pairwise_regret_guard import FEATURE_NAMES
from bayesian_phystwin.deform360_pairwise_regret_guard_source import (
    PRIMARY_METRICS,
    PairwiseRegretSourceConfig,
    evaluate_pairwise_regret_guard_source,
    pairwise_regret_certificate_from_dict,
)


def _payload() -> dict[str, object]:
    rows = []
    for object_index in range(5):
        intervals = []
        for interval_index, frame in enumerate((19, 38, 57)):
            features = np.linspace(0.1, 1.6, len(FEATURE_NAMES))
            features += object_index * 0.01 + interval_index * 0.001
            regret = -0.0015 - 0.0001 * interval_index
            intervals.append(
                {
                    "frame": frame,
                    "available": True,
                    "features": features.tolist(),
                    "regret": {metric: regret for metric in PRIMARY_METRICS},
                    "worst_regret_m": regret,
                }
            )
        rows.append(
            {
                "case": f"object-{object_index}-ep0001",
                "object_id": f"object-{object_index}",
                "scores": {
                    "baseline": {metric: 0.01 for metric in PRIMARY_METRICS}
                },
                "intervals": intervals,
            }
        )
    return {"rows": rows}


def _fast_config(**updates: object) -> PairwiseRegretSourceConfig:
    values = {
        "placebo_trial_count": 8,
        "maximum_placebo_pass_rate": 1.0,
    }
    values.update(updates)
    return PairwiseRegretSourceConfig(**values)


def test_source_qualification_emits_restorable_deployment_certificate() -> None:
    result = evaluate_pairwise_regret_guard_source(
        _payload(), config=_fast_config()
    )

    assert result["source"]["physical_object_count"] == 5
    assert result["cross_object"]["gate_passed"]
    assert result["controls"]["synthetic_positive"]["passed"]
    certificate = pairwise_regret_certificate_from_dict(
        result["deployment_artifact"]["candidate_certificate"]
    )
    assert certificate.source_group_count == 5
    assert certificate.finite_sample_coverage == pytest.approx(0.5)


def test_placebo_control_is_deterministic() -> None:
    config = _fast_config(placebo_seed=91)

    first = evaluate_pairwise_regret_guard_source(_payload(), config=config)
    second = evaluate_pairwise_regret_guard_source(_payload(), config=config)

    assert first["controls"]["placebo"] == second["controls"]["placebo"]


def test_unavailable_interval_must_be_exact_baseline_fallback() -> None:
    payload = _payload()
    payload["rows"][0]["intervals"][0]["available"] = False

    with pytest.raises(ValueError, match="exact baseline"):
        evaluate_pairwise_regret_guard_source(payload, config=_fast_config())


def test_worst_regret_must_match_coprimary_metrics() -> None:
    payload = deepcopy(_payload())
    payload["rows"][0]["intervals"][0]["worst_regret_m"] = 0.1

    with pytest.raises(ValueError, match="inconsistent"):
        evaluate_pairwise_regret_guard_source(payload, config=_fast_config())


def test_source_qualification_does_not_authorize_calibrated_safety_claim() -> None:
    result = evaluate_pairwise_regret_guard_source(
        _payload(), config=_fast_config()
    )

    assert result["fresh_accuracy_evaluation_allowed"]
    assert result["calibrated_safety_claim_allowed"] is False
    assert "deployment certificate provides 50%" in result["claim_boundary"]
