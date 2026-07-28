from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_fresh_pairwise_outcome import (
    ARMS,
    CANDIDATE_ARM,
    PRIMARY_METRICS,
    evaluate_fresh_pairwise_outcomes,
    score_fresh_pairwise_outcome_arrays,
    summarize_fresh_pairwise_outcomes,
)

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "configs/sota/deform360_fresh_pairwise_belief_v1.json"
COHORT = (
    REPO
    / "results/sota/deform360_fresh_source_lock_v1"
    / "deform360_fresh_object_cohort_lock_v1.json"
)
ANALYSIS = REPO / "configs/sota/deform360_fresh_pairwise_outcome_v1.json"


def _outcome_arrays() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8)
    target = rng.normal(0.0, 0.1, size=(76, 32, 3)).astype(np.float32)
    offsets = {
        "physical_prior": 0.006,
        "persistence": 0.008,
        "selected_raw_backbone_persistence_insufficient_default": 0.004,
        CANDIDATE_ARM: 0.001,
    }
    trajectories = {
        arm: (target + np.float32(offsets[arm])).astype(np.float32)
        for arm in ARMS
    }
    visible = np.ones(target.shape[:2], dtype=bool)
    return trajectories, target, visible


def test_outcome_scoring_excludes_measurement_identities_equally() -> None:
    trajectories, target, visible = _outcome_arrays()

    scores = score_fresh_pairwise_outcome_arrays(
        trajectories,
        target,
        visible,
        visible,
        center_ids=np.arange(16, dtype=np.int64),
    )

    assert set(scores) == set(ARMS)
    assert all(value["frame_count"] == 54 for value in scores.values())
    for metric in PRIMARY_METRICS:
        assert scores[CANDIDATE_ARM][metric] < scores["physical_prior"][metric]


def test_transfer_gate_requires_both_metrics_and_all_comparators() -> None:
    trajectories, target, visible = _outcome_arrays()
    scores = score_fresh_pairwise_outcome_arrays(
        trajectories,
        target,
        visible,
        visible,
        center_ids=np.arange(16, dtype=np.int64),
    )
    reports = [
        {
            "case": f"case-{index}",
            "object_id": f"object-{index}",
            "category": ("filament", "sheet", "volumetric")[index % 3],
            "scores": scores,
        }
        for index in range(12)
    ]

    summary = summarize_fresh_pairwise_outcomes(reports)

    assert summary["transfer_gate"]["passed"] is True
    assert all(summary["transfer_gate"]["checks"].values())
    assert all(
        comparison["joint_two_metric_wins"] == 12
        for comparison in summary["comparisons"].values()
    )

    reports[0]["scores"][CANDIDATE_ARM][PRIMARY_METRICS[0]] = 1.0
    failed = summarize_fresh_pairwise_outcomes(reports)
    assert failed["transfer_gate"]["passed"] is False


def test_invalid_barrier_aborts_before_future_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = tmp_path / "barrier.json"
    barrier.write_text(json.dumps({}), encoding="utf-8")
    future_opened = False

    def forbidden_pickle_load(_handle: object) -> object:
        nonlocal future_opened
        future_opened = True
        raise AssertionError("future payload was opened before the barrier")

    monkeypatch.setattr(
        "bayesian_phystwin.deform360_fresh_pairwise_outcome.pickle.load",
        forbidden_pickle_load,
    )

    with pytest.raises(ValueError, match="completeness barrier"):
        evaluate_fresh_pairwise_outcomes(
            repository_root=REPO,
            protocol_path=PROTOCOL,
            cohort_path=COHORT,
            admission_root=tmp_path / "admissions",
            prediction_root=tmp_path / "predictions",
            processed_root=tmp_path / "processed",
            barrier_path=barrier,
            analysis_path=ANALYSIS,
            output_dir=tmp_path / "outcome",
            operator_path=__file__,
        )
    assert future_opened is False
    assert not (tmp_path / "outcome").exists()
