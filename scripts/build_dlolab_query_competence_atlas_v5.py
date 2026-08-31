#!/usr/bin/env python3
"""Extend the DLO-Lab competence atlas with certified Slingshot v4."""

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
ATLAS_V4 = Path("results/source/dlolab_query_competence_atlas_v4/atlas.json")
SLINGSHOT_V4_SUMMARY = Path(
    "results/source/dlolab_slingshot_policy_certificate_source_v4/summary.json"
)
ATLAS_V4_ID = "842941a296a055c78d17278e671de796a8f413bcb7ea30fb3d4ef4b232c460c2"
ATLAS_V4_SHA256 = "45890333ac292c0cd2bb5620b1e2bb572e297bddb923d5e570a4cb098adfd94b"
SLINGSHOT_V4_SUMMARY_ID = (
    "2882809b7265714a93be2d3f1455eeac527adbe681cc990cde762777fcaf3a85"
)
SLINGSHOT_V4_SUMMARY_SHA256 = (
    "cfbab2f371ec606fdbcf844cc8484f543a57829780f893f1b9bf3359dbae2564"
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
                "schema_version": 5,
                "kind": kind,
                **values,
            }
        ),
    )


def _prior_atlas() -> QueryCompetenceAtlasV2:
    path = ROOT / ATLAS_V4
    if _sha256(path) != ATLAS_V4_SHA256:
        raise ValueError("frozen atlas v4 bytes changed")
    atlas = load_query_competence_atlas(path)
    if atlas.artifact_id != ATLAS_V4_ID or len(atlas.entries) != 5:
        raise ValueError("frozen atlas v4 identity or roster changed")
    return atlas


def _slingshot_summary() -> Mapping[str, Any]:
    path = ROOT / SLINGSHOT_V4_SUMMARY
    if _sha256(path) != SLINGSHOT_V4_SUMMARY_SHA256:
        raise ValueError("Slingshot v4 summary bytes changed")
    value = cast(
        Mapping[str, Any],
        load_strict_json_object(path, label="Slingshot v4 summary"),
    )
    descriptor = dict(value)
    identity = descriptor.pop("artifact_id", None)
    if identity != SLINGSHOT_V4_SUMMARY_ID or identity != content_id(descriptor):
        raise ValueError("Slingshot v4 summary content identity changed")
    expected = {
        "status": "source_gate_passed",
        "source_revision": "e446285660991e4f2b83422c64788f4c410e0c97",
        "source_gate_passed": True,
        "prospective_policy_value_claim": True,
        "prospective_coverage_claim": True,
        "matched_comparison_scored": True,
        "complete_288_world_denominator_scored": True,
        "retry_authorized": False,
        "replacement_authorized": False,
        "protected_data_read": False,
        "new_recordings": False,
        "official_benchmark_or_sota_claim": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"Slingshot v4 summary field {key!r} changed")
    denominator = cast(Mapping[str, Any], value["denominator"])
    pre_future = cast(Mapping[str, Any], value["pre_future"])
    primary = cast(Mapping[str, Any], cast(Mapping[str, Any], value["arms"])["policy_gain_guard"])
    comparison = cast(Mapping[str, Any], value["comparisons"])
    coverage = cast(Mapping[str, Any], value["coverage"])
    if (
        denominator.get("evaluation_worlds") != 288
        or denominator.get("ordinary_action_processes_total") != 3328
        or denominator.get("technical_failures") != 0
        or pre_future.get("accepted_worlds") != 36
        or pre_future.get("fallback_worlds") != 252
        or pre_future.get("pre_future_gate_passed") is not True
        or primary.get("mean_gain_over_incumbent") != 0.003456773857275645
        or primary.get("mean_gain_ci95")
        != [0.001514428709116247, 0.005711325878898301]
        or primary.get("harm_probability_upper95") != 0.04070323881205934
        or comparison.get("policy_guard_gain_over_simultaneous_guard")
        != 0.004338043431440989
        or coverage.get("marginal_policy_gain") != 0.8958333333333334
    ):
        raise ValueError("Slingshot v4 registered result changed")
    return value


def _slingshot_v4_entry(
    summary: Mapping[str, Any], prior_slingshot: QueryCompetenceStageV2
) -> QueryCompetenceStageV2:
    result_id = str(cast(Mapping[str, Any], summary["identities"])["result_id"])
    prior_scope = prior_slingshot.query_scope
    execution = cast(Mapping[str, Any], summary["execution_contract"])
    scope = SimulatorQueryScopeV1(
        simulator_id=prior_scope.simulator_id,
        task_id=prior_scope.task_id,
        observation_policy_id=_semantic_id(
            "observation-policy",
            result_id=result_id,
            description=(
                "causal-three-frame-shared-bias-observation-plus-posterior-"
                "diagnostics-and-local-policy-gain-v4"
            ),
        ),
        action_bank_id=prior_scope.action_bank_id,
        metric_id=prior_scope.metric_id,
        world_distribution_id=_semantic_id(
            "world-distribution",
            result_id=result_id,
            description="fresh-continuous-slingshot-worlds-v4",
        ),
        statistical_unit=str(execution["statistical_unit"]),
        metadata={
            "public_simulator": "DLO-Lab",
            "task": "slingshot",
            "version": "reward-aligned-v4",
        },
    )
    primary = cast(Mapping[str, Any], cast(Mapping[str, Any], summary["arms"])["policy_gain_guard"])
    comparison = cast(Mapping[str, Any], summary["comparisons"])
    coverage = cast(Mapping[str, Any], summary["coverage"])
    pre_future = cast(Mapping[str, Any], summary["pre_future"])
    return QueryCompetenceStageV2(
        query_scope=scope,
        evidence_role="prospective_certificate",
        evidence_artifact_id=str(summary["artifact_id"]),
        evidence_file_sha256=SLINGSHOT_V4_SUMMARY_SHA256,
        independent_group_count=288,
        native_qualification="passed",
        action_headroom="passed",
        source_transfer="passed",
        prospective_risk="passed",
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=True,
        outcomes_used_for_selection=False,
        protected_data_read=False,
        terminal_reason="complete-reward-aligned-prospective-certificate-passed",
        metadata={
            "task": "slingshot",
            "version": "reward-aligned-v4",
            "summary_id": summary["artifact_id"],
            "result_id": result_id,
            "accepted_worlds": pre_future["accepted_worlds"],
            "fallback_worlds": pre_future["fallback_worlds"],
            "mean_gain": primary["mean_gain_over_incumbent"],
            "paired_gain_ci95": primary["mean_gain_ci95"],
            "harmful_worlds": primary["harmful_worlds_beyond_numeric_margin"],
            "harm_risk_upper": primary["harm_probability_upper95"],
            "marginal_gain_coverage": coverage["marginal_policy_gain"],
            "gain_over_matched_guard": comparison[
                "policy_guard_gain_over_simultaneous_guard"
            ],
            "paired_gain_vs_matched_guard_ci95": comparison[
                "policy_guard_paired_gain_vs_simultaneous_guard_ci95"
            ],
            "ordinary_action_processes": cast(Mapping[str, Any], summary["denominator"])[
                "ordinary_action_processes_total"
            ],
            "backend_wide_conclusion": False,
        },
    )


def build_atlas() -> QueryCompetenceAtlasV2:
    prior = _prior_atlas()
    prior_slingshots = [
        entry
        for entry in prior.entries
        if entry.query_scope.metadata.get("task") == "slingshot"
    ]
    if len(prior_slingshots) != 1 or prior_slingshots[0].decision != "rejected":
        raise ValueError("frozen rejected Slingshot query changed")
    summary = _slingshot_summary()
    slingshot_v4 = _slingshot_v4_entry(summary, prior_slingshots[0])
    return QueryCompetenceAtlasV2(
        entries=(*prior.entries, slingshot_v4),
        metadata={
            **dict(prior.metadata),
            "atlas_release": 5,
            "prior_atlas_v4_id": ATLAS_V4_ID,
            "prior_atlas_v4_sha256": ATLAS_V4_SHA256,
            "slingshot_v4_summary_id": SLINGSHOT_V4_SUMMARY_ID,
            "slingshot_v4_summary_sha256": SLINGSHOT_V4_SUMMARY_SHA256,
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
