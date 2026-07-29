from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    INFLATED_FALLBACK_ARM,
    STRICT_ARM,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_synthetic import (
    run_adaptive_direct_depth_synthetic_v14,
    validate_adaptive_direct_depth_synthetic_v14,
    write_adaptive_direct_depth_synthetic_v14,
)


def test_v14_synthetic_controls_cover_both_arms_and_pass_the_gate() -> None:
    result = run_adaptive_direct_depth_synthetic_v14()

    assert result.gate_passed
    assert result.positive_detection_count == result.positive_trial_count == 12
    assert result.placebo_admission_count == 0
    assert result.placebo_exact_fallback_count == result.placebo_trial_count == 12
    assert result.positive_improvement_fraction >= 0.10
    assert {trial.carrier_arm for trial in result.trials} == {
        STRICT_ARM,
        INFLATED_FALLBACK_ARM,
    }


def test_v14_synthetic_result_is_deterministic_and_checksummed(
    tmp_path: Path,
) -> None:
    first = run_adaptive_direct_depth_synthetic_v14()
    second = run_adaptive_direct_depth_synthetic_v14()
    output = tmp_path / "summary.json"

    assert first.artifact_sha256 == second.artifact_sha256
    write_adaptive_direct_depth_synthetic_v14(output, first)
    loaded = validate_adaptive_direct_depth_synthetic_v14(output)
    assert loaded == first
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["information_boundary"]["real_object_observation_read"] is False
