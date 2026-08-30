#!/usr/bin/env python3
"""Build the evidence-bound DLO-Lab query competence registry v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import (
    content_id,
    load_strict_json_object,
)
from bayesian_phystwin.query_competence_certificate_v1 import (
    QueryCompetenceGateV1,
    QueryCompetenceRegistryV1,
    SimulatorQueryScopeV1,
    build_query_competence_certificate,
    save_query_competence_registry,
)

ROOT = Path(__file__).resolve().parents[1]

WRAPPING_SUMMARY = Path(
    "results/sota/dlolab_wrapping_risk_certified_guard_source_v9/summary.json"
)
SLINGSHOT_SUMMARY = Path(
    "results/source/dlolab_slingshot_certified_guard_source_v2/summary.json"
)
SLINGSHOT_VERIFICATION = Path(
    "results/source/dlolab_slingshot_certified_guard_source_v2/verification.json"
)

BOUND_FILES = {
    Path("src/bayesian_phystwin/query_competence_certificate_v1.py"): (
        "3ddbe58808558681037bbb10ab2d3fe005413ae3949834d4d8fcb3f3d4a441de"
    ),
    WRAPPING_SUMMARY: "45877e22f1af55ba4f5c7e1a66ab213e148733c9a3da9d82a1dafe545b77a4d1",
    SLINGSHOT_SUMMARY: "85ea08c0a0f9bac17f39e40fa60f2734dd7a5fdcf7d63b30f23285ad353eaad3",
    SLINGSHOT_VERIFICATION: "f1f4639c196dac8618084cac059f2901a8c1db6fcf9bb725b7459dd67ea5011a",
    Path("docs/dlolab_wrapping_certified_guard_source_v9.md"): (
        "cbf609800f01500a3c565b973577ab23c169b8a6384dfa6eb42de290f7fddd72"
    ),
    Path("docs/dlolab_slingshot_certified_guard_source_v2.md"): (
        "4b86c545171a28fdb4904f21d87eae90cc1830e5029213fccca88d07c86432a3"
    ),
    Path("docs/dlolab_wrapping_risk_certified_guard_source_v9_result.md"): (
        "8df24ceee53ffd6560860e8fb3264cbf0041d1eabd8f860fffb6ae0c10673a28"
    ),
    Path("docs/dlolab_slingshot_certified_guard_source_v2_result.md"): (
        "472b01c13699dd90ff6644ecdf7b4342d19b98e10bae4c31ec5ad92647fc7e6a"
    ),
    Path("scripts/verify_dlolab_wrapping_certified_guard_source_v9.py"): (
        "1bee4b823a6636079140e44030e64ec751c6f19274711310b5cff8539435d76f"
    ),
    Path("scripts/verify_dlolab_slingshot_certified_guard_source_v2.py"): (
        "7f436ceec3101f94702d8de00e9c0749214d6a406198d1df2e5b633567de82dc"
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
        actual = _sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(f"bound source artifact changed: {relative}")


def _expect(
    source: Mapping[str, Any],
    key: str,
    expected: object,
    *,
    label: str,
) -> None:
    if source.get(key) != expected:
        raise ValueError(
            f"{label} field {key!r} changed: "
            f"expected {expected!r}, got {source.get(key)!r}"
        )


def _semantic_id(kind: str, **values: object) -> str:
    return content_id(
        {
            "schema": "bayesian_phystwin.query_scope_component",
            "schema_version": 1,
            "kind": kind,
            **values,
        }
    )


def _shared_simulator_id() -> str:
    return _semantic_id(
        "simulator-family",
        public_project="DLO-Lab",
        execution="native-qualified-task-runtime",
    )


def _wrapping_certificate(summary: Mapping[str, Any]):
    label = "wrapping summary"
    expected = {
        "artifact_id": "d3c577ce1ec215c6d56c4d405e7f9d886f38b7e6d021bb6d62f37da6bd4784b9",
        "status": "complete_source_gate_passed",
        "source_revision": "1b66630b939852547798a1d421a728b429cd7d88",
        "lock_id": "2f96bb2e52501a5e137e44faec4ed699b81dd828b00a98e3654f53e588e798ce",
        "result_id": "50801d4da518238ffc2e2d1995d7467f97286cb4535eff60633a5f3d0112b32d",
        "verified_tree_id": "ffca78f511c882446b95d283458aebb469942bc7d98dd66fb7b5844bcdecef5c",
        "ordinary_worlds": 288,
        "technical_failures": 0,
        "replacements": 0,
        "source_gate_passed": True,
        "guard_gain_over_fixed": 0.004721433249978326,
        "guard_gain_ci95": [0.0038941984708258824, 0.005597420470321563],
        "guard_harmed_worlds": 1,
        "guard_harm_risk_upper": 0.01636507577352747,
        "harm_risk_confidence": 0.95,
        "guard_downside_reduction_fraction": 0.9771841397191956,
        "continuous_gain_fraction_retained": 0.19430485277742182,
        "oracle_headroom_fraction_captured": 0.08704281718509738,
        "post_outcome_tuning": False,
        "independent_arithmetic_verified": True,
        "protected_data_read": False,
    }
    for key, value in expected.items():
        _expect(summary, key, value, label=label)
    if summary.get("failed_checks") != []:
        raise ValueError("wrapping registered source checks changed")
    lock_id = str(summary["lock_id"])
    decision_id = str(summary["decision_seal_id"])
    scope = SimulatorQueryScopeV1(
        simulator_id=_shared_simulator_id(),
        task_id=_semantic_id("task", public_project="DLO-Lab", task="wrapping"),
        observation_policy_id=_semantic_id(
            "observation-policy",
            protocol_id=lock_id,
            description="correlation-aware-noisy-prefix-v9",
        ),
        action_bank_id=_semantic_id(
            "action-bank",
            protocol_id=lock_id,
            description="registered-wrapping-waypoint-bank-v9",
        ),
        metric_id=_semantic_id(
            "metric",
            task="wrapping",
            name="native-reward",
            harm_margin=0.002,
        ),
        world_distribution_id=_semantic_id(
            "world-distribution",
            protocol_id=lock_id,
            fresh_worlds=288,
            seed=261910,
        ),
        statistical_unit="fresh-simulator-world-mean-over-4096-sensor-draws-v1",
        metadata={"public_simulator": "DLO-Lab", "task": "wrapping"},
    )
    gate = QueryCompetenceGateV1(
        expected_group_count=288,
        minimum_mean_gain=0.003,
        require_positive_paired_lower_bound=True,
        maximum_harm_risk_upper=0.05,
        minimum_downside_reduction_fraction=0.75,
        minimum_retained_candidate_gain_fraction=0.15,
        minimum_oracle_headroom_fraction=0.05,
        metadata={"registered_protocol": "wrapping-certified-guard-v9"},
    )
    return build_query_competence_certificate(
        query_scope=scope,
        gate=gate,
        candidate_policy_id=_semantic_id(
            "candidate-policy",
            task="wrapping",
            decision_id=decision_id,
            policy="posterior-chance-guard-0.975",
        ),
        baseline_policy_id=_semantic_id(
            "baseline-policy",
            task="wrapping",
            protocol_id=lock_id,
            policy="registered-fixed-action",
        ),
        protocol_id=lock_id,
        source_summary_artifact_id=str(summary["artifact_id"]),
        source_summary_sha256=BOUND_FILES[WRAPPING_SUMMARY],
        source_result_id=str(summary["result_id"]),
        verification_artifact_id=str(summary["verified_tree_id"]),
        verification_file_sha256=str(summary["verification_file_sha256"]),
        verified_tree_id=str(summary["verified_tree_id"]),
        group_count=int(summary["ordinary_worlds"]),
        technical_failures=int(summary["technical_failures"]),
        retries=0,
        replacements=int(summary["replacements"]),
        mean_gain=float(summary["guard_gain_over_fixed"]),
        paired_gain_ci95=list(summary["guard_gain_ci95"]),
        harmful_group_count=int(summary["guard_harmed_worlds"]),
        harm_confidence_level=float(summary["harm_risk_confidence"]),
        harm_risk_upper=float(summary["guard_harm_risk_upper"]),
        downside_reduction_fraction=float(
            summary["guard_downside_reduction_fraction"]
        ),
        retained_candidate_gain_fraction=float(
            summary["continuous_gain_fraction_retained"]
        ),
        oracle_headroom_fraction=float(
            summary["oracle_headroom_fraction_captured"]
        ),
        protocol_frozen_before_outcomes=True,
        outcomes_used_for_policy_or_gate_selection=bool(
            summary["post_outcome_tuning"]
        ),
        independent_implementation_replay=bool(
            summary["independent_arithmetic_verified"]
        ),
        source_gate_passed=bool(summary["source_gate_passed"]),
        metadata={
            "source_revision": summary["source_revision"],
            "source_schema": summary["schema"],
            "independent_human_review": summary["independent_human_review"],
            "result_doc_sha256": BOUND_FILES[
                Path("docs/dlolab_wrapping_risk_certified_guard_source_v9_result.md")
            ],
            "verifier_source_sha256": summary["verifier_file_sha256"],
            "claim_boundary": "public-simulator-query-only-not-robot-safety",
        },
    )


def _slingshot_certificate(
    summary: Mapping[str, Any],
    verification: Mapping[str, Any],
):
    label = "slingshot summary"
    expected = {
        "artifact_id": "f7947a626d6bf941704b532aebf6cde5447a7cde25e00a587ffe20a566f21086",
        "status": "complete_source_gate_failed",
        "frozen_source_revision": "7da610e3c321f605be29682d1360357496693c7e",
        "lock_id": "7008acbe9ab7fd805832df4e97794f5c6924d00153bb25b6a5b6a2aa9abd54ef",
        "result_id": "35388657b9d3e162a5dcadeb003f6943123b3f19a9d8ac04b2eccd1cdec32ba1",
        "tree_sha256": "9c66f1a3f241465966d1ba37e0de8fe91622a9d7f87d910436d18c021803424f",
        "ordinary_future_worlds": 288,
        "technical_failures": 0,
        "retries": 0,
        "replacements": 0,
        "source_gate_passed": False,
        "guard_mean_gain": 0.00022036606686823588,
        "guard_gain_ci95": [-0.00011082010377221094, 0.000529773406601129],
        "guard_harmed_worlds": 14,
        "guard_harm_risk_upper": 0.07495200896418834,
        "guard_downside_reduction_fraction": 0.9265427750335868,
        "guard_fraction_of_posterior_gain": 0.013783169271828296,
        "guard_fraction_of_oracle_headroom": 0.008592475029876073,
        "independent_implementation_replay": True,
        "protected_data_read": False,
    }
    for key, value in expected.items():
        _expect(summary, key, value, label=label)
    expected_failed = {
        "guard_captures_at_least_5pct_oracle_headroom",
        "guard_gain_at_least_0_001",
        "guard_harm_upper_at_most_0_05",
        "guard_retains_at_least_10pct_posterior_gain",
        "positive_paired_ci95_vs_incumbent",
    }
    if set(summary.get("failed_checks", [])) != expected_failed:
        raise ValueError("slingshot registered failed checks changed")
    _expect(
        verification,
        "artifact_id",
        "c5206613f48bfb68c682f3c63289046252432d5cb4a144ef4e48e2345bd4ce94",
        label="slingshot verification",
    )
    _expect(
        verification,
        "verification",
        "PASS",
        label="slingshot verification",
    )
    _expect(
        verification,
        "result_id",
        summary["result_id"],
        label="slingshot verification",
    )
    lock_id = str(summary["lock_id"])
    decision_id = str(summary["decision_id"])
    scope = SimulatorQueryScopeV1(
        simulator_id=_shared_simulator_id(),
        task_id=_semantic_id("task", public_project="DLO-Lab", task="slingshot"),
        observation_policy_id=_semantic_id(
            "observation-policy",
            protocol_id=lock_id,
            description="12-position-causal-prefix-shared-bias-v2",
        ),
        action_bank_id=_semantic_id(
            "action-bank",
            protocol_id=lock_id,
            description="registered-seven-action-slingshot-bank-v2",
        ),
        metric_id=_semantic_id(
            "metric",
            task="slingshot",
            name="native-reward",
            harm_margin=0.002,
        ),
        world_distribution_id=_semantic_id(
            "world-distribution",
            protocol_id=lock_id,
            fresh_worlds=288,
            distribution="registered-continuous-slingshot-worlds-v2",
        ),
        statistical_unit="fresh-simulator-world-mean-over-4096-sensor-draws-v1",
        metadata={"public_simulator": "DLO-Lab", "task": "slingshot"},
    )
    gate = QueryCompetenceGateV1(
        expected_group_count=288,
        minimum_mean_gain=0.001,
        require_positive_paired_lower_bound=True,
        maximum_harm_risk_upper=0.05,
        minimum_downside_reduction_fraction=0.75,
        minimum_retained_candidate_gain_fraction=0.10,
        minimum_oracle_headroom_fraction=0.05,
        metadata={"registered_protocol": "slingshot-certified-guard-v2"},
    )
    return build_query_competence_certificate(
        query_scope=scope,
        gate=gate,
        candidate_policy_id=_semantic_id(
            "candidate-policy",
            task="slingshot",
            decision_id=decision_id,
            policy="frozen-mean-regret-guard",
        ),
        baseline_policy_id=_semantic_id(
            "baseline-policy",
            task="slingshot",
            protocol_id=lock_id,
            policy="registered-incumbent-action-5",
        ),
        protocol_id=lock_id,
        source_summary_artifact_id=str(summary["artifact_id"]),
        source_summary_sha256=BOUND_FILES[SLINGSHOT_SUMMARY],
        source_result_id=str(summary["result_id"]),
        verification_artifact_id=str(verification["artifact_id"]),
        verification_file_sha256=BOUND_FILES[SLINGSHOT_VERIFICATION],
        verified_tree_id=str(summary["tree_sha256"]),
        group_count=int(summary["ordinary_future_worlds"]),
        technical_failures=int(summary["technical_failures"]),
        retries=int(summary["retries"]),
        replacements=int(summary["replacements"]),
        mean_gain=float(summary["guard_mean_gain"]),
        paired_gain_ci95=list(summary["guard_gain_ci95"]),
        harmful_group_count=int(summary["guard_harmed_worlds"]),
        harm_confidence_level=0.95,
        harm_risk_upper=float(summary["guard_harm_risk_upper"]),
        downside_reduction_fraction=float(
            summary["guard_downside_reduction_fraction"]
        ),
        retained_candidate_gain_fraction=float(
            summary["guard_fraction_of_posterior_gain"]
        ),
        oracle_headroom_fraction=float(
            summary["guard_fraction_of_oracle_headroom"]
        ),
        protocol_frozen_before_outcomes=True,
        outcomes_used_for_policy_or_gate_selection=False,
        independent_implementation_replay=bool(
            summary["independent_implementation_replay"]
        ),
        source_gate_passed=bool(summary["source_gate_passed"]),
        metadata={
            "source_revision": summary["frozen_source_revision"],
            "source_schema": summary["schema"],
            "independent_human_review": summary["independent_human_review"],
            "result_doc_sha256": BOUND_FILES[
                Path("docs/dlolab_slingshot_certified_guard_source_v2_result.md")
            ],
            "verifier_source_sha256": verification["verifier_source_sha256"],
            "claim_boundary": "public-simulator-query-only-not-robot-safety",
        },
    )


def build_registry() -> QueryCompetenceRegistryV1:
    _verify_bound_files()
    wrapping = load_strict_json_object(
        ROOT / WRAPPING_SUMMARY,
        label="wrapping source summary",
    )
    slingshot = load_strict_json_object(
        ROOT / SLINGSHOT_SUMMARY,
        label="slingshot source summary",
    )
    verification = load_strict_json_object(
        ROOT / SLINGSHOT_VERIFICATION,
        label="slingshot verification",
    )
    wrapping_certificate = _wrapping_certificate(wrapping)
    slingshot_certificate = _slingshot_certificate(slingshot, verification)
    return QueryCompetenceRegistryV1(
        certificates={
            str(wrapping_certificate.query_scope.query_id): wrapping_certificate,
            str(slingshot_certificate.query_scope.query_id): slingshot_certificate,
        },
        metadata={
            "evidence_family": "prospective-public-simulator-cross-task-v1",
            "simulator_family": "DLO-Lab",
            "policy_granularity": "exact-query-and-policy",
            "backend_wide_competence_claim": False,
            "failed_or_unknown_queries": "exact-complete-belief-fallback",
            "independent_human_review": False,
            "source_artifact_sha256s": {
                str(path): digest for path, digest in sorted(BOUND_FILES.items())
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/source/dlolab_query_competence_registry_v1/registry.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    registry = build_registry()
    save_query_competence_registry(
        registry,
        args.output,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "artifact_id": registry.artifact_id,
                "certified_query_count": len(registry.certified_query_ids),
                "failed_query_count": len(registry.failed_query_ids),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
