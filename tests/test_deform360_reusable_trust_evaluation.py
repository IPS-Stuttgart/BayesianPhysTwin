from __future__ import annotations

from pathlib import Path

import pytest

import causal4d_public.deform360_reusable_trust_evaluation as evaluation
from causal4d_public.deform360_reusable_trust_protocol import (
    EXPECTED_SPLITS,
    load_reusable_trust_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "configs/causal4d_public/deform360_reusable_trust_fresh_v1.json"
ADDENDUM = (
    ROOT
    / "configs/causal4d_public/deform360_reusable_trust_physics_addendum_v1.json"
)


def _row(object_id: str, episode_id: int, improvement: float, seal: str) -> dict:
    baseline_track = 0.04
    baseline_chamfer = 0.03
    interval = {
        "frame_count": 25,
        "track_rmse_m": baseline_track * (1.0 - improvement),
        "chamfer_m": baseline_chamfer * (1.0 - improvement),
        "persistence_track_rmse_m": baseline_track,
        "persistence_chamfer_m": baseline_chamfer,
        "relative_score_vs_persistence": 1.0 - improvement,
        "track_improvement_fraction": improvement,
        "chamfer_improvement_fraction": improvement,
    }
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTwinFreshEvaluation",
        "object_id": object_id,
        "episode_id": episode_id,
        "episode_key": f"{object_id}/{episode_id}",
        "cohort_seal_result_sha256": seal,
        "metrics": {
            "future": {**interval, "frame_count": 75},
            "early": interval,
            "middle": interval,
            "late": interval,
        },
        "joint_future_win": improvement > 0.0,
    }
    payload["result_sha256"] = evaluation._result_sha256(payload)
    return payload


def test_fresh_gate_is_conjunctive_and_requires_all_twelve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_reusable_trust_protocol(PARENT, ADDENDUM)
    seal = "a" * 64
    cohort = {"result_sha256": seal}
    monkeypatch.setattr(
        evaluation,
        "validate_reusable_trust_prediction_cohort_seal",
        lambda *_args, **_kwargs: {"prediction_count": 12},
    )
    rows = [
        _row(object_id, episode_id, 0.08, seal)
        for object_id, split in EXPECTED_SPLITS.items()
        for episode_id in split["held_out_episode_ids"]
    ]

    passed = evaluation.aggregate_reusable_trust_fresh_gate(
        rows, cohort_seal=cohort, protocol=protocol
    )
    assert passed["passed"] is True
    assert passed["episode_count"] == 12

    failed_rows = list(rows)
    failed_rows[0] = _row("003-cable", 0, -0.2, seal)
    failed = evaluation.aggregate_reusable_trust_fresh_gate(
        failed_rows, cohort_seal=cohort, protocol=protocol
    )
    assert failed["passed"] is False
    assert failed["gates"]["maximum_episode_track_degradation"] is False

    with pytest.raises(ValueError, match="exactly the locked twelve"):
        evaluation.aggregate_reusable_trust_fresh_gate(
            rows[:-1], cohort_seal=cohort, protocol=protocol
        )
