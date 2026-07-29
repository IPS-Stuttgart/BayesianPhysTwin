from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_synthetic import (
    run_causal_response_synthetic_controls,
    write_causal_response_synthetic_result,
)


def test_synthetic_controls_detect_nonrigid_signal_and_reject_placebos() -> None:
    result = run_causal_response_synthetic_controls()

    assert result.positive_detection_count == result.config.trial_count
    assert result.placebo_admission_count == 0
    assert result.placebo_exact_fallback_count == result.config.trial_count
    assert result.positive_improvement_fraction >= 0.10
    assert all(
        trial.candidate_future_rmse_m < trial.baseline_future_rmse_m
        for trial in result.trials
        if trial.control_kind == "positive-nonrigid"
    )


def test_synthetic_controls_are_deterministic_and_checksummed(
    tmp_path: Path,
) -> None:
    first = run_causal_response_synthetic_controls()
    second = run_causal_response_synthetic_controls()
    path = tmp_path / "synthetic_controls.json"

    assert first.artifact_sha256 == second.artifact_sha256
    write_causal_response_synthetic_result(path, first)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_sha256"] == first.artifact_sha256
    assert payload["information_boundary"]["real_object_observation_read"] is False
