from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.prob4d_prospective_protocol import (
    DECISION_SCHEMA,
    PROTOCOL_SCHEMA,
    READINESS_SCHEMA,
    RESULT_SCHEMA,
    build_prob4d_prospective_protocol,
    check_prob4d_prospective_readiness,
    decide_prob4d_prospective_gates,
    load_prob4d_prospective_protocol,
    save_prob4d_prospective_protocol,
)


PROB4D_METHODS = (
    "prob4d_fused_gauge_marginalized",
    "prob4d_framewise_joint_gauge",
    "prob4d_tracklet_joint_gauge",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_artifacts(root: Path) -> list[dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    values = {
        "provider_evaluation_manifest": b"provider-evaluation-v1\n",
        "analysis_manifest": b"analysis-v1\n",
        "method_freeze": b"method-freeze-v1\n",
    }
    artifacts: list[dict[str, str]] = []
    for role, payload in values.items():
        path = root / f"{role}.json"
        path.write_bytes(payload)
        artifacts.append(
            {
                "role": role,
                "path": path.name,
                "sha256": _sha256(path),
                "access_stage": "source_only",
            }
        )
    return artifacts


def _methods() -> list[dict[str, object]]:
    return [
        {
            "method_id": "physical_baseline",
            "role": "physical_baseline",
            "observation_interface": "none",
            "gauge_treatment": "none",
            "sensor_assisted": False,
            "exact_fallback": True,
        },
        {
            "method_id": "simple_visual",
            "role": "visual_reference",
            "observation_interface": "simple_visual",
            "gauge_treatment": "fixed",
            "sensor_assisted": False,
            "exact_fallback": False,
        },
        {
            "method_id": "prob4d_fused_gauge_marginalized",
            "role": "prob4d_candidate",
            "observation_interface": "fused",
            "gauge_treatment": "marginalized",
            "sensor_assisted": False,
            "exact_fallback": False,
        },
        {
            "method_id": "prob4d_framewise_joint_gauge",
            "role": "prob4d_candidate",
            "observation_interface": "framewise_factors",
            "gauge_treatment": "explicit_joint_nuisance",
            "sensor_assisted": False,
            "exact_fallback": False,
        },
        {
            "method_id": "prob4d_tracklet_joint_gauge",
            "role": "prob4d_candidate",
            "observation_interface": "tracklet_factors",
            "gauge_treatment": "explicit_joint_nuisance",
            "sensor_assisted": False,
            "exact_fallback": False,
        },
    ]


def _criteria() -> list[dict[str, object]]:
    criteria: list[dict[str, object]] = []
    for method_id in PROB4D_METHODS:
        criteria.extend(
            [
                {
                    "criterion_id": f"provider.{method_id}.point_rmse",
                    "stage": "provider",
                    "method_id": method_id,
                    "reference_method_id": "simple_visual",
                    "metric": "paired_group_point_rmse_difference",
                    "statistic": "ci_upper",
                    "comparison": "<=",
                    "threshold": 0.0,
                },
                {
                    "criterion_id": f"physical.{method_id}.future_track",
                    "stage": "physical",
                    "method_id": method_id,
                    "reference_method_id": "physical_baseline",
                    "metric": "paired_group_future_track_error_difference",
                    "statistic": "ci_upper",
                    "comparison": "<=",
                    "threshold": 0.0,
                },
            ]
        )
    return criteria


def _configuration(artifact_root: Path) -> dict[str, object]:
    return {
        "schema_name": PROTOCOL_SCHEMA,
        "schema_version": 1,
        "protocol_id": "prob4d-bpt-prospective-v1",
        "claim_id": "prob4d.improves_bayesian_physical_twin",
        "frozen_at": "2026-08-03T12:00:00+00:00",
        "frozen_by": "independent-verifier",
        "split": {
            "development": [{"unit_id": "dev-1", "group_id": "object-dev"}],
            "calibration": [
                {"unit_id": "cal-1", "group_id": "object-calibration"}
            ],
            "target": [
                {"unit_id": "target-1", "group_id": "object-target-a"},
                {"unit_id": "target-2", "group_id": "object-target-b"},
            ],
        },
        "methods": _methods(),
        "software": {
            "prob4d_revision": "1" * 40,
            "prob4d_wheel_sha256": "2" * 64,
            "bayesian_phystwin_revision": "3" * 40,
            "bayesian_phystwin_wheel_sha256": "4" * 64,
            "motioncrafter_revision": "5" * 40,
            "motioncrafter_model_set_id": "6" * 64,
            "seed_policy": "derived-per-call",
            "python_version": "3.12.8",
            "numpy_version": "2.2.2",
        },
        "calibration": {
            "gauge_artifact_id": "7" * 64,
            "point_artifact_id": "8" * 64,
            "reliability_artifact_id": "9" * 64,
            "tracklet_policy_id": "a" * 64,
            "calibration_split_id": "object-calibration-v1",
            "grouping_definition": "one group per physical object",
        },
        "analysis": {
            "statistical_unit": "group_id",
            "primary_candidate_method_ids": list(PROB4D_METHODS),
            "bootstrap_resamples": 5000,
            "bootstrap_seed": 71,
            "exact_fallback_method_id": "physical_baseline",
            "harmful_update_definition": (
                "accepted candidate has higher future error than physical baseline"
            ),
            "rejection_treatment": "exact_physical_baseline_fallback",
            "target_outcomes_opened_before_freeze": False,
            "target_method_selection_allowed": False,
            "provider_and_physical_gates_separate": True,
            "causal4d_evaluation_before_bpt_gate": False,
        },
        "criteria": _criteria(),
        "frozen_artifacts": _write_source_artifacts(artifact_root),
    }


def _passing_statistics() -> dict[str, float]:
    return {
        criterion["criterion_id"]: -0.001
        for criterion in _criteria()
    }


def _result(
    protocol_sha256: str,
    *,
    statistics: dict[str, float],
    physical_update: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "schema_name": RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_sha256": protocol_sha256,
        "criterion_statistics": statistics,
        "physical_update": physical_update,
        "target_access": {
            "opened_after_freeze": True,
            "selection_performed_after_opening": False,
        },
    }


def test_protocol_round_trip_is_content_addressed_and_immutable(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    protocol = build_prob4d_prospective_protocol(_configuration(artifact_root))
    path = tmp_path / "protocol.json"
    save_prob4d_prospective_protocol(path, protocol)
    loaded = load_prob4d_prospective_protocol(path)

    assert loaded.protocol_sha256 == protocol.protocol_sha256
    assert loaded.primary_candidate_method_ids == PROB4D_METHODS
    assert tuple(loaded.split) == ("development", "calibration", "target")
    with pytest.raises(TypeError):
        loaded.analysis["bootstrap_seed"] = 0  # type: ignore[index]

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["analysis"]["bootstrap_seed"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content address mismatch"):
        load_prob4d_prospective_protocol(path)


def test_split_groups_must_be_disjoint(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path / "artifacts")
    configuration["split"]["target"][0]["group_id"] = "object-calibration"  # type: ignore[index]

    with pytest.raises(ValueError, match="group IDs must be disjoint"):
        build_prob4d_prospective_protocol(configuration)


def test_target_artifacts_cannot_enter_the_pre_target_freeze(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path / "artifacts")
    configuration["frozen_artifacts"].append(  # type: ignore[union-attr]
        {
            "role": "target_outcomes",
            "path": "target.json",
            "sha256": "b" * 64,
            "access_stage": "target",
        }
    )

    with pytest.raises(ValueError, match="target artifacts|target outcomes"):
        build_prob4d_prospective_protocol(configuration)


def test_readiness_hash_verifies_and_detects_tampering(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    protocol = build_prob4d_prospective_protocol(_configuration(artifact_root))

    ready = check_prob4d_prospective_readiness(protocol, artifact_root)
    assert ready["schema_name"] == READINESS_SCHEMA
    assert ready["ready_for_target_opening"] is True
    assert ready["causal4d_evaluation_admissible"] is False
    assert all(record["matched"] for record in ready["artifacts"])

    (artifact_root / "analysis_manifest.json").write_text(
        "tampered\n",
        encoding="utf-8",
    )
    rejected = check_prob4d_prospective_readiness(protocol, artifact_root)
    assert rejected["ready_for_target_opening"] is False
    assert any(
        record["role"] == "analysis_manifest" and not record["matched"]
        for record in rejected["artifacts"]
    )


def test_provider_failure_blocks_the_physical_gate(tmp_path: Path) -> None:
    protocol = build_prob4d_prospective_protocol(
        _configuration(tmp_path / "artifacts")
    )
    statistics = {
        key: value
        for key, value in _passing_statistics().items()
        if key.startswith("provider.")
    }
    statistics[
        "provider.prob4d_fused_gauge_marginalized.point_rmse"
    ] = 0.01
    decision = decide_prob4d_prospective_gates(
        protocol,
        _result(
            protocol.protocol_sha256,
            statistics=statistics,
            physical_update=None,
        ),
    )

    assert decision["schema_name"] == DECISION_SCHEMA
    assert decision["provider_gate"]["passed"] is False
    assert decision["physical_gate"]["admissible"] is False
    assert decision["physical_gate"]["passed"] is None
    assert decision["prob4d_supported_feeder"] is False
    assert decision["causal4d_evaluation_admissible"] is False


def test_provider_failure_rejects_opened_physical_evidence(tmp_path: Path) -> None:
    protocol = build_prob4d_prospective_protocol(
        _configuration(tmp_path / "artifacts")
    )
    statistics = _passing_statistics()
    statistics[
        "provider.prob4d_fused_gauge_marginalized.point_rmse"
    ] = 0.01

    with pytest.raises(ValueError, match="physical_update must be null"):
        decide_prob4d_prospective_gates(
            protocol,
            _result(
                protocol.protocol_sha256,
                statistics=statistics,
                physical_update={
                    "fallback_method_id": "physical_baseline",
                    "fallback_exact": True,
                    "evaluated_group_count": 2,
                    "accepted_update_count": 1,
                    "harmful_accepted_update_count": 0,
                },
            ),
        )


def test_both_gates_must_pass_before_causal4d_is_admissible(tmp_path: Path) -> None:
    protocol = build_prob4d_prospective_protocol(
        _configuration(tmp_path / "artifacts")
    )
    decision = decide_prob4d_prospective_gates(
        protocol,
        _result(
            protocol.protocol_sha256,
            statistics=_passing_statistics(),
            physical_update={
                "fallback_method_id": "physical_baseline",
                "fallback_exact": True,
                "evaluated_group_count": 2,
                "accepted_update_count": 1,
                "harmful_accepted_update_count": 0,
            },
        ),
    )

    assert decision["provider_gate"]["passed"] is True
    assert decision["physical_gate"]["passed"] is True
    assert decision["prob4d_supported_feeder"] is True
    assert decision["causal4d_evaluation_admissible"] is True
    assert decision["exact_fallback_method_id"] == "physical_baseline"


def test_target_informed_selection_is_rejected(tmp_path: Path) -> None:
    protocol = build_prob4d_prospective_protocol(
        _configuration(tmp_path / "artifacts")
    )
    result = _result(
        protocol.protocol_sha256,
        statistics=_passing_statistics(),
        physical_update={
            "fallback_method_id": "physical_baseline",
            "fallback_exact": True,
            "evaluated_group_count": 2,
            "accepted_update_count": 1,
            "harmful_accepted_update_count": 0,
        },
    )
    result["target_access"]["selection_performed_after_opening"] = True  # type: ignore[index]

    with pytest.raises(ValueError, match="target-informed method selection"):
        decide_prob4d_prospective_gates(protocol, result)
