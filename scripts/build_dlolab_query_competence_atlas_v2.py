#!/usr/bin/env python3
"""Build the staged DLO-Lab query-competence atlas v2."""

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
    save_query_competence_atlas,
)
from bayesian_phystwin.query_competence_certificate_v1 import (
    QueryCompetenceCertificateV1,
    SimulatorQueryScopeV1,
    load_query_competence_registry,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_V1 = Path("results/source/dlolab_query_competence_registry_v1/registry.json")
COILING_SUMMARY = Path("results/source/dlolab_coiling_offgrid_source_v2/summary.json")

BOUND_FILES = {
    Path("src/bayesian_phystwin/query_competence_atlas_v2.py"): (
        "78aeaeb6d4d73a2435f099c3aeba74d40c39f2a62025e48584e8e655c68f5bd8"
    ),
    REGISTRY_V1: "8f8b3dc7ab750420cbe8732d0a24679be772b21aff45abc69be0633b638e0159",
    COILING_SUMMARY: "50e7e5901cec2afedb9fb88d9c0a2cf95cd85e3d0f661685081c3e27063c9056",
    Path("docs/query_conditional_simulator_competence_v1.md"): (
        "377bddb53c0bb9e9eaf811e1a5687e3b57af1f430dca9abcfe60a72dc8e78b42"
    ),
    Path("docs/dlolab_coiling_offgrid_source_v2.md"): (
        "84293a6f4f255fba0b2d22aa247f553d6dc4c5d2a6f304128d63f358f2c3d528"
    ),
    Path("docs/dlolab_coiling_offgrid_source_v2_result.md"): (
        "e99b01dc67910ea3011a6c40a8caeb6d535704b16a66f12d679ef4dc04f12513"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_bound_files() -> None:
    for relative, expected in BOUND_FILES.items():
        if _sha256(ROOT / relative) != expected:
            raise ValueError(f"bound atlas evidence changed: {relative}")


def _expect(
    source: Mapping[str, Any], key: str, expected: object, *, label: str
) -> None:
    if source.get(key) != expected:
        raise ValueError(f"{label} field {key!r} changed")


def _semantic_id(kind: str, **values: object) -> str:
    return cast(
        str,
        content_id(
            {
                "schema": "bayesian_phystwin.query_scope_component",
                "schema_version": 2,
                "kind": kind,
                **values,
            }
        ),
    )


def _certificates_by_task() -> dict[str, QueryCompetenceCertificateV1]:
    registry = load_query_competence_registry(ROOT / REGISTRY_V1)
    if registry.artifact_id != (
        "017fe497894142cb5b4cffac933d8e1ff2ee6bd9e18463f43e1868b0ad731a4b"
    ):
        raise ValueError("query competence registry v1 identity changed")
    result = {
        str(certificate.query_scope.metadata["task"]): certificate
        for certificate in registry.certificates.values()
    }
    if set(result) != {"wrapping", "slingshot"}:
        raise ValueError("query competence registry v1 task roster changed")
    return result


def _wrapping_entry(
    certificate: QueryCompetenceCertificateV1,
) -> QueryCompetenceStageV2:
    if (
        not certificate.certified
        or certificate.mean_gain != 0.004721433249978326
        or certificate.harmful_group_count != 1
        or certificate.harm_risk_upper != 0.016365075773527655
        or certificate.group_count != 288
    ):
        raise ValueError("wrapping competence evidence changed")
    return QueryCompetenceStageV2(
        query_scope=certificate.query_scope,
        evidence_role="prospective_certificate",
        evidence_artifact_id=str(certificate.artifact_id),
        evidence_file_sha256=BOUND_FILES[REGISTRY_V1],
        independent_group_count=certificate.group_count,
        native_qualification="passed",
        action_headroom="passed",
        source_transfer="passed",
        prospective_risk="passed",
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=certificate.protocol_frozen_before_outcomes,
        outcomes_used_for_selection=certificate.outcomes_used_for_policy_or_gate_selection,
        protected_data_read=False,
        terminal_reason="prospective-query-certificate-passed",
        metadata={
            "task": "wrapping",
            "mean_gain": certificate.mean_gain,
            "paired_gain_ci95": list(certificate.paired_gain_ci95),
            "harm_risk_upper": certificate.harm_risk_upper,
            "source_certificate_id": certificate.artifact_id,
        },
    )


def _slingshot_entry(
    certificate: QueryCompetenceCertificateV1,
) -> QueryCompetenceStageV2:
    expected_failed = {
        "harm-risk-upper-bound-exceeded",
        "mean-gain-below-threshold",
        "oracle-headroom-below-threshold",
        "paired-gain-lower-bound-not-positive",
        "registered-source-gate-rejected",
        "retained-value-below-threshold",
    }
    if (
        certificate.certified
        or set(certificate.failed_checks) != expected_failed
        or certificate.mean_gain != 0.00022036606686823588
        or certificate.harmful_group_count != 14
        or certificate.harm_risk_upper != 0.07495200896418834
        or certificate.group_count != 288
    ):
        raise ValueError("Slingshot competence evidence changed")
    return QueryCompetenceStageV2(
        query_scope=certificate.query_scope,
        evidence_role="prospective_certificate",
        evidence_artifact_id=str(certificate.artifact_id),
        evidence_file_sha256=BOUND_FILES[REGISTRY_V1],
        independent_group_count=certificate.group_count,
        native_qualification="passed",
        action_headroom="failed",
        source_transfer="failed",
        prospective_risk="failed",
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=certificate.protocol_frozen_before_outcomes,
        outcomes_used_for_selection=certificate.outcomes_used_for_policy_or_gate_selection,
        protected_data_read=False,
        terminal_reason="prospective-value-and-harm-gates-failed",
        metadata={
            "task": "slingshot",
            "mean_gain": certificate.mean_gain,
            "paired_gain_ci95": list(certificate.paired_gain_ci95),
            "harm_risk_upper": certificate.harm_risk_upper,
            "source_certificate_id": certificate.artifact_id,
        },
    )


def _coiling_entry(
    summary: Mapping[str, Any], simulator_id: str
) -> QueryCompetenceStageV2:
    expected = {
        "status": "complete_source_gate_failed",
        "source_revision": "d46abfd681e807b85aef81319dd9793c1596592f",
        "lock_id": "bd5832df7286386b5f789e3d4f7e3c2ef386b80e556ce66220f0abbcb2dab062",
        "result_id": "23b0330f226d5ac5fd56d74922f22585e07b493375d39028be47ee6d7f310e06",
        "completed_worlds": 12,
        "native_qualified_worlds": 12,
        "unrun_worlds": 0,
        "oracle_headroom": 0.0016061967989121628,
        "crossfit_guarded_mean_gain": -0.0004303340638860791,
        "mean_observation_draw_harm_probability": 0.08333333333333333,
        "maximum_observation_draw_harm_probability": 1.0,
        "source_gate_passed": False,
        "prospective_worlds_selected": False,
        "prospective_execution_authorized": False,
        "retry_authorized": False,
        "protected_data_read": False,
    }
    for key, value in expected.items():
        _expect(summary, key, value, label="coiling summary")
    checks = cast(Mapping[str, object], summary["checks"])
    if (
        checks.get("adjusted_oracle_headroom_at_least_0_003") is not False
        or checks.get("crossfit_guarded_mean_gain_at_least_0_002") is not False
        or checks.get("mean_harm_probability_at_most_0_05") is not False
        or checks.get("maximum_harm_probability_at_most_0_20") is not False
    ):
        raise ValueError("coiling failed gate roster changed")
    lock_id = str(summary["lock_id"])
    scope = SimulatorQueryScopeV1(
        simulator_id=simulator_id,
        task_id=_semantic_id("task", public_project="DLO-Lab", task="coiling"),
        observation_policy_id=_semantic_id(
            "observation-policy",
            protocol_id=lock_id,
            description="three-frame-five-node-correlated-noise-v2",
        ),
        action_bank_id=_semantic_id(
            "action-bank",
            protocol_id=lock_id,
            description="seven-unique-coiling-waypoints-v2",
        ),
        metric_id=_semantic_id(
            "metric", task="coiling", name="unchanged-native-final-reward"
        ),
        world_distribution_id=_semantic_id(
            "world-distribution",
            protocol_id=lock_id,
            off_grid_source_worlds=12,
        ),
        statistical_unit="off-grid-source-world-mean-over-8192-sensor-draws-v2",
        metadata={"public_simulator": "DLO-Lab", "task": "coiling"},
    )
    return QueryCompetenceStageV2(
        query_scope=scope,
        evidence_role="source_screen",
        evidence_artifact_id=str(summary["result_id"]),
        evidence_file_sha256=BOUND_FILES[COILING_SUMMARY],
        independent_group_count=int(summary["completed_worlds"]),
        native_qualification="passed",
        action_headroom="failed",
        source_transfer="failed",
        prospective_risk="not_evaluated",
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=True,
        outcomes_used_for_selection=False,
        protected_data_read=False,
        terminal_reason="source-headroom-and-transfer-gates-failed",
        metadata={
            "task": "coiling",
            "oracle_headroom": summary["oracle_headroom"],
            "crossfit_guarded_mean_gain": summary["crossfit_guarded_mean_gain"],
            "mean_observation_draw_harm_probability": summary[
                "mean_observation_draw_harm_probability"
            ],
            "source_result_id": summary["result_id"],
        },
    )


def build_atlas() -> QueryCompetenceAtlasV2:
    _verify_bound_files()
    certificates = _certificates_by_task()
    wrapping = _wrapping_entry(certificates["wrapping"])
    slingshot = _slingshot_entry(certificates["slingshot"])
    if wrapping.query_scope.simulator_id != slingshot.query_scope.simulator_id:
        raise ValueError("DLO-Lab simulator identity changed across tasks")
    summary = cast(
        Mapping[str, Any],
        load_strict_json_object(ROOT / COILING_SUMMARY, label="coiling summary"),
    )
    coiling = _coiling_entry(summary, wrapping.query_scope.simulator_id)
    return QueryCompetenceAtlasV2(
        entries=(wrapping, slingshot, coiling),
        metadata={
            "public_simulator": "DLO-Lab",
            "policy_granularity": "exact-query-observation-action-metric-distribution",
            "stages": [
                "native_qualification",
                "action_headroom",
                "source_transfer",
                "prospective_risk",
            ],
            "source_registry_v1_id": (
                "017fe497894142cb5b4cffac933d8e1ff2ee6bd9e18463f43e1868b0ad731a4b"
            ),
            "source_sha256": {
                str(path): digest for path, digest in sorted(BOUND_FILES.items())
            },
            "backend_wide_competence_claim": False,
            "independent_human_review": False,
            "protected_data_read": False,
            "new_recordings": False,
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
