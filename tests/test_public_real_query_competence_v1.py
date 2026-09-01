from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.public_real_query_competence_v1 import (
    EXACT_CONTEXT_SUPPORT_RULE,
    PUBLIC_REAL_QUERY_COMPETENCE_SCHEMA,
    action_family_v1,
    build_public_real_query_competence_evidence_v1,
    evaluate_tracking_cloth_action_support_v1,
    sha256_file,
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deform_result() -> dict[str, object]:
    objects = []
    for index in range(92):
        supported = index < 71
        harmful = index == 91
        candidate = 1.1 if harmful else 0.9
        source_action = "fold" if supported else "lift"
        legacy_accept = index < 80 or harmful
        objects.append(
            {
                "object_id": f"object-{index:03d}",
                "target_action_family": "shape",
                "source_actions": [source_action],
                "source_cv_active_rmse": {
                    "bayesian_action_ensemble": 0.9,
                    "persistence": 1.0,
                },
                "metrics": {
                    "bayesian_action_ensemble": {
                        "active_field_rmse": candidate,
                        "field_mae": 1.1,
                    },
                    "persistence": {
                        "active_field_rmse": 1.0,
                        "field_mae": 1.0,
                    },
                },
                "guard_accepts": legacy_accept,
            }
        )
    return {
        "schema": "bayesian-phystwin/deform360-untouched-confirmation-result-v5",
        "schema_version": 5,
        "status": "complete",
        "objects": objects,
    }


def _tracking_protocol(*, target_motion: str = "twist") -> dict[str, object]:
    return {"source_motion": "shake", "target_motion": target_motion}


def _tracking_metrics() -> dict[str, object]:
    return {"arms": {"guarded_bayesian_physics": {"rmse_mm": 2.5}}}


def _write_specimens(path: Path) -> Path:
    fieldnames = [
        "specimen",
        "arm",
        "rmse_mm",
        "mean_marker_error_mm",
        "coordinate_nll",
        "coordinate_90_coverage",
        "mean_full_90_width_mm",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(8):
            for arm, loss in (("persistence", 1.0), ("bayesian_physics", 2.0)):
                writer.writerow(
                    {
                        "specimen": f"specimen-{index}",
                        "arm": arm,
                        "rmse_mm": loss,
                        "mean_marker_error_mm": loss,
                        "coordinate_nll": 0.0,
                        "coordinate_90_coverage": 0.9,
                        "mean_full_90_width_mm": 1.0,
                    }
                )
    return path


def _pokeflex(*, same_profile: bool) -> dict[str, object]:
    if same_profile:
        objects = [
            {
                "object": "source-a",
                "baseline_mean_CD_UL1_mm": 6.0,
                "selected_mean_CD_UL1_mm": 5.7,
            },
            {
                "object": "source-b",
                "baseline_mean_CD_UL1_mm": 6.0,
                "selected_mean_CD_UL1_mm": 5.8,
            },
        ]
        return {
            "artifact_kind": "PokeFlexIndependentDepthRegretGuardProspectiveEvaluation",
            "schema_version": 1,
            "gate_passed": True,
            "baseline_object_mean_CD_UL1_mm": 6.0,
            "selected_object_mean_CD_UL1_mm": 5.75,
            "object_count": 2,
            "take_count": 3,
            "object_wins": 2,
            "object_losses": 0,
            "objects": objects,
        }
    objects = [
        {
            "object": f"target-{index}",
            "baseline_mean_CD_UL1_mm": 5.0,
            "selected_mean_CD_UL1_mm": 5.1,
        }
        for index in range(4)
    ]
    return {
        "artifact_kind": "PokeFlexIndependentDepthRegretGuardProspectiveEvaluation",
        "schema_version": 1,
        "gate_passed": False,
        "baseline_object_mean_CD_UL1_mm": 5.0,
        "selected_object_mean_CD_UL1_mm": 5.1,
        "false_safe_rate": 0.4,
        "object_count": 4,
        "objects": objects,
    }


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "deform360_result": _write_json(tmp_path / "deform.json", _deform_result()),
        "tracking_protocol": _write_json(
            tmp_path / "tracking-protocol.json", _tracking_protocol()
        ),
        "tracking_metrics": _write_json(
            tmp_path / "tracking-metrics.json", _tracking_metrics()
        ),
        "tracking_specimen_scores": _write_specimens(tmp_path / "specimens.csv"),
        "pokeflex_same_profile": _write_json(
            tmp_path / "pokeflex-same.json", _pokeflex(same_profile=True)
        ),
        "pokeflex_independent_object": _write_json(
            tmp_path / "pokeflex-shift.json", _pokeflex(same_profile=False)
        ),
    }


def _protocol(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "schema": "bayesian-phystwin.public-real-query-competence-protocol",
        "schema_version": 1,
        "inputs": {
            name: {"sha256": _sha(path)} for name, path in sorted(paths.items())
        },
        "statistics": {
            "bootstrap_replicates": 128,
            "bootstrap_seed": 260811,
            "confidence": 0.95,
            "target_harm_probability": 0.05,
        },
    }


def test_action_family_v1_matches_registered_vocabulary() -> None:
    assert action_family_v1("raise left") == "lift"
    assert action_family_v1("drag opposite directions") == "translate"
    assert action_family_v1("twist") == "shape"
    assert action_family_v1("compress") == "compress"
    assert action_family_v1("shake") == "dynamic"
    assert action_family_v1(None) == "other"


def test_build_public_real_evidence_separates_support_dimensions(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    result = build_public_real_query_competence_evidence_v1(
        protocol=_protocol(paths),
        deform360_result_path=paths["deform360_result"],
        tracking_protocol_path=paths["tracking_protocol"],
        tracking_metrics_path=paths["tracking_metrics"],
        tracking_specimen_scores_path=paths["tracking_specimen_scores"],
        pokeflex_same_profile_path=paths["pokeflex_same_profile"],
        pokeflex_independent_object_path=paths["pokeflex_independent_object"],
    )

    assert result["schema"] == PUBLIC_REAL_QUERY_COMPETENCE_SCHEMA
    assert result["exact_context_support_rule"] == EXACT_CONTEXT_SUPPORT_RULE
    assert result["artifact_id"] == content_id(
        {key: value for key, value in result.items() if key != "artifact_id"}
    )
    deform = result["deform360"]
    assert deform["exact_context_policy"]["accepted_count"] == 71
    assert deform["exact_context_policy"]["harmful_accepted_count"] == 0
    assert deform["exact_context_policy"]["retrospective_harm_gate_passed"] is True
    assert deform["ablations"]["always_candidate"]["harmful_accepted_count"] == 1
    assert deform["query_mismatch_control"]["policy"]["accepted_count"] == 0
    assert deform["query_rank_reversal"]["rank_reversal_observed"] is True
    assert (
        deform["query_rank_reversal"]["query_independent_routing_is_sufficient"]
        is False
    )
    assert len(deform["support_rows"]) == 92
    tracking = result["tracking_cloth"]
    assert tracking["exact_context_policy"]["accepted_count"] == 0
    assert tracking["always_candidate"]["candidate_wins_ties_losses"] == [0, 0, 8]
    assert tracking["decision"] == "exact-fallback-action-domain-out-of-scope"
    pokeflex = result["pokeflex"]
    assert pokeflex["same_profile_replication"]["relative_change"] < 0.0
    assert pokeflex["independent_object_stress"]["avoided_mean_regression_mm"] > 0.0


def test_tracking_action_match_is_admitted_in_retrospective_audit(
    tmp_path: Path,
) -> None:
    scores = _write_specimens(tmp_path / "specimens.csv")
    result = evaluate_tracking_cloth_action_support_v1(
        _tracking_protocol(target_motion="shake"),
        _tracking_metrics(),
        scores,
        bootstrap_replicates=16,
        bootstrap_seed=1,
        confidence=0.95,
        target_harm_probability=0.05,
    )
    assert result["exact_context_policy"]["accepted_count"] == 8
    assert result["exact_context_policy"]["harmful_accepted_count"] == 8


def test_input_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    protocol = _protocol(paths)
    protocol["inputs"]["tracking_metrics"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="tracking_metrics SHA-256 changed"):
        build_public_real_query_competence_evidence_v1(
            protocol=protocol,
            deform360_result_path=paths["deform360_result"],
            tracking_protocol_path=paths["tracking_protocol"],
            tracking_metrics_path=paths["tracking_metrics"],
            tracking_specimen_scores_path=paths["tracking_specimen_scores"],
            pokeflex_same_profile_path=paths["pokeflex_same_profile"],
            pokeflex_independent_object_path=paths["pokeflex_independent_object"],
        )


def test_sha256_file_preserves_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"exact\x00bytes\n")
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_duplicate_specimen_arm_is_rejected(tmp_path: Path) -> None:
    scores = _write_specimens(tmp_path / "specimens.csv")
    with scores.open("a", encoding="utf-8") as handle:
        handle.write("specimen-0,persistence,1,1,0,0.9,1\n")
    with pytest.raises(ValueError, match="duplicate specimen/arm score"):
        evaluate_tracking_cloth_action_support_v1(
            _tracking_protocol(),
            _tracking_metrics(),
            scores,
            bootstrap_replicates=8,
            bootstrap_seed=1,
            confidence=0.95,
            target_harm_probability=0.05,
        )


def test_bootstrap_result_is_deterministic(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    kwargs = {
        "protocol": _protocol(paths),
        "deform360_result_path": paths["deform360_result"],
        "tracking_protocol_path": paths["tracking_protocol"],
        "tracking_metrics_path": paths["tracking_metrics"],
        "tracking_specimen_scores_path": paths["tracking_specimen_scores"],
        "pokeflex_same_profile_path": paths["pokeflex_same_profile"],
        "pokeflex_independent_object_path": paths["pokeflex_independent_object"],
    }
    first = build_public_real_query_competence_evidence_v1(**kwargs)
    second = build_public_real_query_competence_evidence_v1(**kwargs)
    assert first == second
    assert np.isclose(
        first["deform360"]["gain_retained_fraction"],
        second["deform360"]["gain_retained_fraction"],
    )


def test_committed_public_real_evidence_is_content_bound() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (
            root / "evidence/public_real_query_competence_retrospective_v1.json"
        ).read_text(encoding="utf-8")
    )
    artifact_id = evidence.pop("artifact_id")
    assert artifact_id == content_id(evidence)
    assert artifact_id == (
        "6f77ec5658d4e77e5eb514e584f090410c940c201a0eaaf7f2923ffcafbb12f6"
    )
    assert evidence["deform360"]["exact_context_policy"]["accepted_count"] == 71
    assert evidence["deform360"]["query_rank_reversal"]["rank_reversal_observed"]
    assert evidence["information_boundary"]["new_measurements_collected"] is False
