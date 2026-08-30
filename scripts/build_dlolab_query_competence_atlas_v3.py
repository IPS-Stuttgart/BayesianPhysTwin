#!/usr/bin/env python3
"""Extend the frozen DLO-Lab competence atlas with separation v2."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import content_id, load_strict_json_object
from bayesian_phystwin.query_competence_atlas_v2 import (
    QueryCompetenceAtlasV2,
    QueryCompetenceStageV2,
    load_query_competence_atlas,
    save_query_competence_atlas,
)
from bayesian_phystwin.query_competence_certificate_v1 import SimulatorQueryScopeV1

ROOT = Path(__file__).resolve().parents[1]
ATLAS_V2 = Path("results/source/dlolab_query_competence_atlas_v2/atlas.json")
SEPARATION_SUMMARY = Path(
    "results/source/dlolab_separation_headroom_development_v2/summary.json"
)
ATLAS_V2_ID = "69bc63be221614750496fa1437fde462ad30f80dc0d37adb4a3c56638539252c"
ATLAS_V2_SHA256 = "6438b04e766c04e8f25b4a42123655724b5cd32f3b4c5bc88ac15f10ee0b6fa6"
SEPARATION_SUMMARY_ID = (
    "bcb1805bf16325c65d63ec6a59fab06ffd697e2daf2afe0b7f0274e66d410a4b"
)
SEPARATION_SUMMARY_SHA256 = (
    "cd4874d9ce638f4cc1d1fe31e6a3eb5a9ffcd81761b2717924d0bbc87b029a31"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_id(kind: str, **values: object) -> str:
    return cast(
        str,
        content_id(
            {
                "schema": "bayesian_phystwin.query_scope_component",
                "schema_version": 3,
                "kind": kind,
                **values,
            }
        ),
    )


def _prior_atlas() -> QueryCompetenceAtlasV2:
    path = ROOT / ATLAS_V2
    if _sha256(path) != ATLAS_V2_SHA256:
        raise ValueError("frozen atlas v2 bytes changed")
    atlas = load_query_competence_atlas(path)
    if atlas.artifact_id != ATLAS_V2_ID or len(atlas.entries) != 3:
        raise ValueError("frozen atlas v2 identity or roster changed")
    return atlas


def _separation_summary() -> Mapping[str, Any]:
    path = ROOT / SEPARATION_SUMMARY
    if _sha256(path) != SEPARATION_SUMMARY_SHA256:
        raise ValueError("separation summary bytes changed")
    value = cast(
        Mapping[str, Any],
        load_strict_json_object(path, label="separation summary"),
    )
    descriptor = dict(value)
    identity = descriptor.pop("artifact_id", None)
    expected = {
        "status": "native_qualification_failed",
        "source_revision": "d4a008afe29efa45d2f154bf288de18866d3370b",
        "result_id": "498951847f738bd7118cb5f5a3981f6c5a83071653f612687f555bef2f1da1bc",
        "attempted_worlds": 1,
        "completed_worlds": 1,
        "unrun_worlds": 8,
        "native_qualified_worlds": 0,
        "failed_checks": ["material_attachment"],
        "registered_attachment_threshold_m": 0.02,
        "reported_maximum_attachment_distance_m": 0.03045183085429911,
        "independent_final_attachment_distance_m": 0.03045183085429908,
        "ordinary_native_success": True,
        "development_value_analysis_performed": False,
        "development_gate_passed": False,
        "exact_fallback_retained": True,
        "retry_authorized": False,
        "protected_data_read": False,
    }
    if identity != SEPARATION_SUMMARY_ID or identity != content_id(descriptor):
        raise ValueError("separation summary content identity changed")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"separation summary field {key!r} changed")
    return value


def _separation_entry(
    summary: Mapping[str, Any], simulator_id: str
) -> QueryCompetenceStageV2:
    result_id = str(summary["result_id"])
    scope = SimulatorQueryScopeV1(
        simulator_id=simulator_id,
        task_id=_semantic_id("task", public_project="DLO-Lab", task="separation"),
        observation_policy_id=_semantic_id(
            "observation-policy",
            result_id=result_id,
            description="two-prefix-five-node-per-rope-development-v2",
        ),
        action_bank_id=_semantic_id(
            "action-bank",
            result_id=result_id,
            description="hold-plus-eight-symmetric-pulls-development-v2",
        ),
        metric_id=_semantic_id(
            "metric", task="separation", name="native-symmetric-nearest-distance"
        ),
        world_distribution_id=_semantic_id(
            "world-distribution",
            result_id=result_id,
            description="nine-fixed-rigid-in-plane-rotations-development-v2",
        ),
        statistical_unit="registered-development-world-before-value-analysis-v2",
        metadata={"public_simulator": "DLO-Lab", "task": "separation"},
    )
    return QueryCompetenceStageV2(
        query_scope=scope,
        evidence_role="source_screen",
        evidence_artifact_id=result_id,
        evidence_file_sha256=SEPARATION_SUMMARY_SHA256,
        independent_group_count=int(summary["completed_worlds"]),
        native_qualification="failed",
        action_headroom="not_evaluated",
        source_transfer="not_evaluated",
        prospective_risk="not_evaluated",
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=True,
        outcomes_used_for_selection=False,
        protected_data_read=False,
        terminal_reason="registered-material-attachment-qualification-failed",
        metadata={
            "task": "separation",
            "summary_id": summary["artifact_id"],
            "failed_checks": summary["failed_checks"],
            "ordinary_native_success": summary["ordinary_native_success"],
            "attachment_threshold_m": summary["registered_attachment_threshold_m"],
            "attachment_distance_m": summary[
                "independent_final_attachment_distance_m"
            ],
            "completed_worlds": summary["completed_worlds"],
            "unrun_worlds": summary["unrun_worlds"],
            "backend_wide_conclusion": False,
        },
    )


def build_atlas() -> QueryCompetenceAtlasV2:
    prior = _prior_atlas()
    tasks = {str(entry.query_scope.metadata["task"]) for entry in prior.entries}
    if tasks != {"wrapping", "slingshot", "coiling"}:
        raise ValueError("frozen atlas v2 task roster changed")
    simulator_ids = {entry.query_scope.simulator_id for entry in prior.entries}
    if len(simulator_ids) != 1:
        raise ValueError("frozen DLO-Lab simulator identity changed")
    summary = _separation_summary()
    separation = _separation_entry(summary, str(next(iter(simulator_ids))))
    return QueryCompetenceAtlasV2(
        entries=(*prior.entries, separation),
        metadata={
            **dict(prior.metadata),
            "atlas_release": 3,
            "prior_atlas_v2_id": ATLAS_V2_ID,
            "prior_atlas_v2_sha256": ATLAS_V2_SHA256,
            "separation_summary_id": SEPARATION_SUMMARY_ID,
            "separation_summary_sha256": SEPARATION_SUMMARY_SHA256,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atlas = build_atlas()
    save_query_competence_atlas(args.output, atlas)
    print(f"atlas_id={atlas.artifact_id}")


if __name__ == "__main__":
    main()
