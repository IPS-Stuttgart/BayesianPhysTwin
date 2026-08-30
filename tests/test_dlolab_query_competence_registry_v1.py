from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from bayesian_phystwin.query_competence_certificate_v1 import (
    SimulatorQueryScopeV1,
    load_query_competence_registry,
    select_query_competent_belief,
)
from scripts.build_dlolab_query_competence_registry_v1 import build_registry

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = (
    ROOT / "results/source/dlolab_query_competence_registry_v1/registry.json"
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Belief:
    artifact_id: str
    state: tuple[float, ...]


def _belief(label: str) -> _Belief:
    return _Belief(_digest(label), (float(len(label)),))


def _by_task():
    registry = load_query_competence_registry(REGISTRY)
    return registry, {
        certificate.query_scope.metadata["task"]: certificate
        for certificate in registry.certificates.values()
    }


def test_committed_registry_is_exact_builder_output() -> None:
    committed = load_query_competence_registry(REGISTRY)
    rebuilt = build_registry()

    assert committed.artifact_id == (
        "017fe497894142cb5b4cffac933d8e1ff2ee6bd9e18463f43e1868b0ad731a4b"
    )
    assert committed.to_record() == rebuilt.to_record()
    assert hashlib.sha256(REGISTRY.read_bytes()).hexdigest() == (
        "8f8b3dc7ab750420cbe8732d0a24679be772b21aff45abc69be0633b638e0159"
    )


def test_cross_task_evidence_yields_one_pass_and_one_fail() -> None:
    registry, tasks = _by_task()
    wrapping = tasks["wrapping"]
    slingshot = tasks["slingshot"]

    assert wrapping.query_scope.simulator_id == slingshot.query_scope.simulator_id
    assert wrapping.query_scope.query_id != slingshot.query_scope.query_id
    assert wrapping.certified
    assert not slingshot.certified
    assert registry.certified_query_ids == (wrapping.query_scope.query_id,)
    assert registry.failed_query_ids == (slingshot.query_scope.query_id,)
    assert wrapping.mean_gain == 0.004721433249978326
    assert wrapping.harmful_group_count == 1
    assert wrapping.harm_risk_upper < 0.05
    assert slingshot.mean_gain == 0.00022036606686823588
    assert slingshot.harmful_group_count == 14
    assert slingshot.harm_risk_upper > 0.05
    assert not registry.metadata["backend_wide_competence_claim"]
    assert not registry.metadata["independent_human_review"]


def test_evidence_registry_routes_wrapping_only() -> None:
    registry, tasks = _by_task()
    baseline = _belief("baseline")
    candidate = _belief("candidate")
    wrapping = tasks["wrapping"]
    slingshot = tasks["slingshot"]

    selected_wrapping, wrapping_receipt = select_query_competent_belief(
        baseline,
        candidate,
        registry,
        query_scope=wrapping.query_scope,
        candidate_policy_id=wrapping.candidate_policy_id,
        baseline_policy_id=wrapping.baseline_policy_id,
        common_domain_id=_digest("public-simulator-domain"),
        inference_admissible=True,
    )
    selected_slingshot, slingshot_receipt = select_query_competent_belief(
        baseline,
        candidate,
        registry,
        query_scope=slingshot.query_scope,
        candidate_policy_id=slingshot.candidate_policy_id,
        baseline_policy_id=slingshot.baseline_policy_id,
        common_domain_id=_digest("public-simulator-domain"),
        inference_admissible=True,
    )
    unknown = SimulatorQueryScopeV1(
        simulator_id=wrapping.query_scope.simulator_id,
        task_id=_digest("unknown-task"),
        observation_policy_id=wrapping.query_scope.observation_policy_id,
        action_bank_id=wrapping.query_scope.action_bank_id,
        metric_id=wrapping.query_scope.metric_id,
        world_distribution_id=wrapping.query_scope.world_distribution_id,
        statistical_unit=wrapping.query_scope.statistical_unit,
    )
    selected_unknown, unknown_receipt = select_query_competent_belief(
        baseline,
        candidate,
        registry,
        query_scope=unknown,
        candidate_policy_id=wrapping.candidate_policy_id,
        baseline_policy_id=wrapping.baseline_policy_id,
        common_domain_id=_digest("public-simulator-domain"),
        inference_admissible=True,
    )

    assert selected_wrapping is candidate
    assert wrapping_receipt.selected_candidate
    assert selected_slingshot is baseline
    assert not slingshot_receipt.selected_candidate
    assert slingshot_receipt.metadata["routing_reason"] == (
        "query-competence-rejected"
    )
    assert selected_unknown is baseline
    assert not unknown_receipt.selected_candidate
    assert unknown_receipt.metadata["routing_reason"] == "unknown-query"
