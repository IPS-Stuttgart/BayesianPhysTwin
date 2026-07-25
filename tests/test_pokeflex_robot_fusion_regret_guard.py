from __future__ import annotations

from copy import deepcopy

import pytest

from bayesian_phystwin.pokeflex_robot_fusion_protocol import (
    EXPECTED_DEVELOPMENT_OBJECTS,
    POKEFLEX_ROBOT_FUSION_SOURCE_PROTOCOL_SHA256,
)
from bayesian_phystwin.pokeflex_robot_fusion_regret_guard import (
    EXPECTED_TAKES,
    evaluate_pokeflex_robot_fusion_cross_object,
    extract_pokeflex_robot_fusion_rows,
)


def _payload(
    object_name: str,
    take: str,
    *,
    candidate_regret_mm: float = -1.0,
) -> dict:
    take_id = f"{object_name}_{take}"
    targets = []
    for target_frame in range(6, 11):
        baseline = 5.0
        targets.append(
            {
                "target_frame": target_frame,
                "released_checkpoint_CD_UL1_mm": baseline,
                "robot_convex_scale_0_CD_UL1_mm": baseline,
                "robot_convex_scale_0.05_CD_UL1_mm": (
                    baseline + candidate_regret_mm
                ),
                "robot_convex_scale_0.1_CD_UL1_mm": (
                    baseline + candidate_regret_mm
                ),
                "robot_convex_scale_0.2_CD_UL1_mm": (
                    baseline + candidate_regret_mm
                ),
                "fusion_features": {
                    "baseline_deformation_rms_m": 0.01,
                    "robot_deformation_rms_m": 0.02,
                    "model_disagreement_rms_m": 0.01,
                    "deformation_cosine": 0.8,
                    "force_norm_n": 5.0,
                    "force_delta_norm_n": 1.0,
                    "tool_step_m": 0.002,
                },
            }
        )
    return {
        "artifact_kind": "PokeFlexRobotFusionSourceTake",
        "protocol": {
            "sha256": POKEFLEX_ROBOT_FUSION_SOURCE_PROTOCOL_SHA256,
        },
        "take": {
            "id": take_id,
            "object": object_name,
            "take": take,
        },
        "candidate_config": {
            "scales": [0.0, 0.05, 0.1, 0.2],
        },
        "causal_boundary": {
            "future_observation_used": False,
            "target_objects_opened": False,
        },
        "targets": targets,
    }


def _cohort(*, candidate_regret_mm: float = -1.0) -> list[dict]:
    return [
        _payload(
            object_name,
            take,
            candidate_regret_mm=candidate_regret_mm,
        )
        for object_name in EXPECTED_DEVELOPMENT_OBJECTS
        for take in EXPECTED_TAKES
    ]


def test_robot_fusion_source_rows_preserve_exact_fallback() -> None:
    rows, frames = extract_pokeflex_robot_fusion_rows(_cohort())

    assert len(frames) == 100
    assert len(rows) == 300
    assert {row["scale"] for row in rows} == {0.05, 0.1, 0.2}
    assert all(row["regret_mm"] == pytest.approx(-1.0) for row in rows)


def test_robot_fusion_source_rejects_nonexact_fallback() -> None:
    payloads = _cohort()
    payloads[0]["targets"][0]["robot_convex_scale_0_CD_UL1_mm"] += 1e-12

    with pytest.raises(ValueError, match="fallback outcome changed"):
        extract_pokeflex_robot_fusion_rows(payloads)


def test_robot_fusion_source_rejects_future_input() -> None:
    payloads = deepcopy(_cohort())
    payloads[0]["causal_boundary"]["future_observation_used"] = True

    with pytest.raises(ValueError, match="future observation"):
        extract_pokeflex_robot_fusion_rows(payloads)


def test_robot_fusion_cross_object_accepts_transferable_gain() -> None:
    result = evaluate_pokeflex_robot_fusion_cross_object(_cohort())

    assert result["cross_object"]["gate_passed"] is True
    assert result["cross_object"]["object_wins"] == 5
    assert result["cross_object"]["false_safe_rate"] == 0.0
    assert result["cross_object"]["object_balanced_relative_improvement"] == (
        pytest.approx(0.2)
    )
    assert result["source_fit_diagnostic"]["deployment_authorized"] is True


def test_robot_fusion_cross_object_rejects_unhelpful_gain() -> None:
    result = evaluate_pokeflex_robot_fusion_cross_object(
        _cohort(candidate_regret_mm=0.25)
    )

    assert result["cross_object"]["gate_passed"] is False
    assert result["cross_object"]["decision"].startswith("FAIL:")
    assert result["source_fit_diagnostic"]["deployment_authorized"] is False
