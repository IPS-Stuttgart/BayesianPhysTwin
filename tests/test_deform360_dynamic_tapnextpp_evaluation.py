from __future__ import annotations

from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_dynamic_tapnextpp_assimilation import (
    CANDIDATE_ARM,
    PERSISTENCE_ARM,
    PHYSICAL_ARM,
    SELECTED_BACKBONE_ARM,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_evaluation import (
    aggregate_provider_source_gate,
    evaluate_guarded_assimilation_gate,
    load_source_evaluation_protocol,
    score_provider_case_arrays,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_EVALUATION_PROTOCOL = (
    ROOT
    / "configs"
    / "sota"
    / "deform360_dynamic_tapnextpp_source_evaluation_v1.json"
)


def _schedule() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entities = np.arange(72, dtype=np.int64)
    births = np.repeat(np.asarray([0, 6, 12, 20, 26, 32, 39, 45, 51]), 8)
    updates = np.repeat(np.asarray([19, 19, 19, 38, 38, 38, 57, 57, 57]), 8)
    return entities, births, updates


def test_provider_gate_accepts_precise_supported_calibrated_tracks() -> None:
    load_source_evaluation_protocol(SOURCE_EVALUATION_PROTOCOL)
    entities, births, updates = _schedule()
    target = np.zeros((76, 80, 3), dtype=np.float64)
    target[:, :, 0] = np.arange(76)[:, None] * 0.001
    target[:, :, 1] = np.arange(80)[None] * 0.002
    trajectory = target[:58, entities].copy()
    trajectory[:, :, 2] += 0.001
    covariance = np.repeat(
        (np.eye(3) * 1e-6)[None, None],
        58,
        axis=0,
    )
    covariance = np.repeat(covariance, 72, axis=1)
    score = score_provider_case_arrays(
        trajectory_world_m=trajectory,
        accepted_support=np.ones((58, 72), dtype=bool),
        local_covariance_m2=covariance,
        shared_bias_standard_deviation_m=0.005,
        target_m=target,
        target_visibility=np.ones((76, 80), dtype=bool),
        target_validity=np.ones((76, 80), dtype=bool),
        entity_ids=entities,
        birth_frames=births,
        update_frames=updates,
    )
    reports = [
        {
            **score,
            "object_hash": f"{index:064x}",
            "technical_failure": False,
        }
        for index in range(8)
    ]

    gate = aggregate_provider_source_gate(reports)

    assert score["supported_fraction"] == 1.0
    assert score["provider_rmse_m"] < 0.002
    assert score["relative_gain_over_persistence"] > 0.9
    assert gate["passed"] is True
    assert gate["calibration"]["worst_object_coverage"] == 1.0


def test_provider_score_accepts_explicit_variable_query_count() -> None:
    entities = np.arange(4, dtype=np.int64)
    births = np.asarray([0, 6, 20, 39])
    updates = np.asarray([19, 19, 38, 57])
    target = np.zeros((76, 8, 3), dtype=np.float64)
    target[:, :, 0] = np.arange(76)[:, None] * 0.001
    trajectory = target[:58, entities].copy()
    covariance = np.repeat(
        (np.eye(3) * 1e-6)[None, None],
        58,
        axis=0,
    )
    covariance = np.repeat(covariance, len(entities), axis=1)

    score = score_provider_case_arrays(
        trajectory_world_m=trajectory,
        accepted_support=np.ones((58, len(entities)), dtype=bool),
        local_covariance_m2=covariance,
        shared_bias_standard_deviation_m=0.005,
        target_m=target,
        target_visibility=np.ones((76, 8), dtype=bool),
        target_validity=np.ones((76, 8), dtype=bool),
        entity_ids=entities,
        birth_frames=births,
        update_frames=updates,
        expected_query_count=None,
    )

    assert score["scheduled_identity_count"] == len(entities)
    assert score["supported_fraction"] == 1.0


def _assimilation_case(index: int) -> dict:
    frame_zero = np.zeros((80, 3), dtype=np.float32)
    frame_zero[:, 1] = np.arange(80, dtype=np.float32) * 0.02
    target = np.repeat(frame_zero[None], 76, axis=0)
    physical = target.copy()
    persistence = target.copy()
    selected = target.copy()
    candidate = target.copy()
    for start, stop in ((20, 38), (39, 57), (58, 76)):
        physical[start:stop, :, 0] += 0.010
        selected[start:stop, :, 0] += 0.010
        persistence[start:stop, :, 0] += 0.012
        candidate[start:stop, :, 0] += 0.005
    report = {
        "center_ids": list(range(72)),
        "updates": [
            {
                "available_center_count": 72,
                "mean_prior_reliability": 0.9,
                "pairwise_gate": {
                    "accepted": True,
                    "inlier_fraction": 0.95,
                },
            }
            for _ in range(3)
        ],
    }
    return {
        "object_hash": f"{index + 100:064x}",
        "technical_failure": False,
        "arrays": {
            PHYSICAL_ARM: physical,
            PERSISTENCE_ARM: persistence,
            SELECTED_BACKBONE_ARM: selected,
            CANDIDATE_ARM: candidate,
        },
        "assimilation_report": report,
        "target_m": target,
        "visibility": np.ones((76, 80), dtype=bool),
        "validity": np.ones((76, 80), dtype=bool),
        "hidden_entity_ids": np.arange(72, 80),
    }


def test_crossfit_guard_accepts_consistent_hidden_identity_gain() -> None:
    result = evaluate_guarded_assimilation_gate(
        [_assimilation_case(index) for index in range(8)]
    )

    assert result["passed"] is True
    assert all(
        decision["guard_accepted"]
        for report in result["case_reports"]
        for decision in report["decisions"]
    )
    assert result["comparisons"][SELECTED_BACKBONE_ARM][
        "joint_object_wins"
    ] == 8


def test_crossfit_guard_keeps_technical_failure_in_denominator() -> None:
    cases = [_assimilation_case(index) for index in range(8)]
    cases[-1] = {
        "object_hash": cases[-1]["object_hash"],
        "technical_failure": True,
    }

    result = evaluate_guarded_assimilation_gate(cases)

    assert len(result["case_reports"]) == 8
    assert result["case_reports"][-1]["technical_failure"] is True
