#!/usr/bin/env python3
"""Extend the frozen DLO-Lab competence atlas with unknotting v1."""

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
ATLAS_V3 = Path("results/source/dlolab_query_competence_atlas_v3/atlas.json")
UNKNOTTING_SUMMARY = Path(
    "results/source/dlolab_unknotting_headroom_development_v1/summary.json"
)
ATLAS_V3_ID = "b81af983d4f52f4673f8c9d43a45183ad78f49a93d727ce47520481fa5dfbe35"
ATLAS_V3_SHA256 = "4b4fddb146f13688dbca7ce40d8cd44feeb04616f8a36626f8975d43d5b1e07b"
UNKNOTTING_SUMMARY_ID = (
    "53716b227b8cb6f8ffe2e3c5e788bd51a8615e3496c32faf60e4e4070e165113"
)
UNKNOTTING_SUMMARY_SHA256 = (
    "0641b1af7365bd37481d30dd3962e788579a2d9353a8682da4f37a681e8f0ff9"
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
    path = ROOT / ATLAS_V3
    if _sha256(path) != ATLAS_V3_SHA256:
        raise ValueError("frozen atlas v3 bytes changed")
    atlas = load_query_competence_atlas(path)
    if atlas.artifact_id != ATLAS_V3_ID or len(atlas.entries) != 4:
        raise ValueError("frozen atlas v3 identity or roster changed")
    return atlas


def _unknotting_summary() -> Mapping[str, Any]:
    path = ROOT / UNKNOTTING_SUMMARY
    if _sha256(path) != UNKNOTTING_SUMMARY_SHA256:
        raise ValueError("unknotting summary bytes changed")
    value = cast(
        Mapping[str, Any],
        load_strict_json_object(path, label="unknotting summary"),
    )
    descriptor = dict(value)
    identity = descriptor.pop("artifact_id", None)
    expected = {
        "status": "native_qualification_failed",
        "source_revision": "10857023f2964278bbbc6611b7b42db9e9fcc7fe",
        "result_id": "ad50ae325285894aa6b108d17b2d6d76faee95228d656397725a2b6935c68a07",
        "attempted_worlds": 1,
        "completed_worlds": 1,
        "unrun_worlds": 8,
        "failed_checks": ["segment_length"],
        "registered_maximum_segment_relative_error": 0.1,
        "reported_maximum_segment_relative_error": 0.1811057349566656,
        "independent_final_segment_relative_error": 0.15751209373415742,
        "maximum_attachment_offset_drift_m": 2.223918936468914e-07,
        "ordinary_native_success": True,
        "development_value_analysis_performed": False,
        "development_gate_passed": False,
        "exact_fallback_retained": True,
        "retry_authorized": False,
        "protected_data_read": False,
    }
    if identity != UNKNOTTING_SUMMARY_ID or identity != content_id(descriptor):
        raise ValueError("unknotting summary content identity changed")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"unknotting summary field {key!r} changed")
    return value


def _unknotting_entry(
    summary: Mapping[str, Any], simulator_id: str
) -> QueryCompetenceStageV2:
    result_id = str(summary["result_id"])
    scope = SimulatorQueryScopeV1(
        simulator_id=simulator_id,
        task_id=_semantic_id("task", public_project="DLO-Lab", task="unknotting"),
        observation_policy_id=_semantic_id(
            "observation-policy",
            result_id=result_id,
            description="two-prefix-five-node-development-v1",
        ),
        action_bank_id=_semantic_id(
            "action-bank",
            result_id=result_id,
            description="hold-plus-eight-axis-relative-pulls-development-v1",
        ),
        metric_id=_semantic_id(
            "metric", task="unknotting", name="native-exp-negative-penalty"
        ),
        world_distribution_id=_semantic_id(
            "world-distribution",
            result_id=result_id,
            description="nine-fixed-rigid-in-plane-rotations-development-v1",
        ),
        statistical_unit="registered-development-world-before-value-analysis-v1",
        metadata={"public_simulator": "DLO-Lab", "task": "unknotting"},
    )
    return QueryCompetenceStageV2(
        query_scope=scope,
        evidence_role="source_screen",
        evidence_artifact_id=result_id,
        evidence_file_sha256=UNKNOTTING_SUMMARY_SHA256,
        independent_group_count=int(summary["completed_worlds"]),
        native_qualification="failed",
        action_headroom="not_evaluated",
        source_transfer="not_evaluated",
        prospective_risk="not_evaluated",
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=True,
        outcomes_used_for_selection=False,
        protected_data_read=False,
        terminal_reason="registered-segment-length-qualification-failed",
        metadata={
            "task": "unknotting",
            "summary_id": summary["artifact_id"],
            "failed_checks": summary["failed_checks"],
            "ordinary_native_success": summary["ordinary_native_success"],
            "segment_relative_error_threshold": summary[
                "registered_maximum_segment_relative_error"
            ],
            "reported_maximum_segment_relative_error": summary[
                "reported_maximum_segment_relative_error"
            ],
            "independent_final_segment_relative_error": summary[
                "independent_final_segment_relative_error"
            ],
            "completed_worlds": summary["completed_worlds"],
            "unrun_worlds": summary["unrun_worlds"],
            "backend_wide_conclusion": False,
        },
    )


def build_atlas() -> QueryCompetenceAtlasV2:
    prior = _prior_atlas()
    tasks = {str(entry.query_scope.metadata["task"]) for entry in prior.entries}
    if tasks != {"wrapping", "slingshot", "coiling", "separation"}:
        raise ValueError("frozen atlas v3 task roster changed")
    simulator_ids = {entry.query_scope.simulator_id for entry in prior.entries}
    if len(simulator_ids) != 1:
        raise ValueError("frozen DLO-Lab simulator identity changed")
    summary = _unknotting_summary()
    unknotting = _unknotting_entry(summary, str(next(iter(simulator_ids))))
    return QueryCompetenceAtlasV2(
        entries=(*prior.entries, unknotting),
        metadata={
            **dict(prior.metadata),
            "atlas_release": 4,
            "prior_atlas_v3_id": ATLAS_V3_ID,
            "prior_atlas_v3_sha256": ATLAS_V3_SHA256,
            "unknotting_summary_id": UNKNOTTING_SUMMARY_ID,
            "unknotting_summary_sha256": UNKNOTTING_SUMMARY_SHA256,
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
