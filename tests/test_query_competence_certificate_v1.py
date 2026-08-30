from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from bayesian_phystwin.guard_harm_risk import one_sided_binomial_upper_bound
from bayesian_phystwin.query_competence_certificate_v1 import (
    QueryCompetenceCertificateV1,
    QueryCompetenceGateV1,
    QueryCompetenceRegistryV1,
    SimulatorQueryScopeV1,
    build_query_competence_certificate,
    load_query_competence_registry,
    save_query_competence_registry,
    select_query_competent_belief,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _scope(label: str) -> SimulatorQueryScopeV1:
    return SimulatorQueryScopeV1(
        simulator_id=_digest("simulator"),
        task_id=_digest(f"task:{label}"),
        observation_policy_id=_digest(f"observation:{label}"),
        action_bank_id=_digest(f"actions:{label}"),
        metric_id=_digest("native-reward"),
        world_distribution_id=_digest(f"worlds:{label}"),
        statistical_unit="fresh-simulator-world-v1",
        metadata={"label": label},
    )


def _gate(*, retained: float = 0.10) -> QueryCompetenceGateV1:
    return QueryCompetenceGateV1(
        expected_group_count=100,
        minimum_mean_gain=0.003,
        require_positive_paired_lower_bound=True,
        maximum_harm_risk_upper=0.05,
        minimum_downside_reduction_fraction=0.75,
        minimum_retained_candidate_gain_fraction=retained,
        minimum_oracle_headroom_fraction=0.05,
    )


def _certificate(
    scope: SimulatorQueryScopeV1,
    *,
    source_passed: bool,
) -> QueryCompetenceCertificateV1:
    harmful = 0 if source_passed else 10
    return build_query_competence_certificate(
        query_scope=scope,
        gate=_gate(),
        candidate_policy_id=_digest(f"candidate:{scope.query_id}"),
        baseline_policy_id=_digest(f"baseline:{scope.query_id}"),
        protocol_id=_digest(f"protocol:{scope.query_id}"),
        source_summary_artifact_id=_digest(f"summary:{scope.query_id}"),
        source_summary_sha256=_digest(f"summary-file:{scope.query_id}"),
        source_result_id=_digest(f"result:{scope.query_id}"),
        verification_artifact_id=_digest(f"verification:{scope.query_id}"),
        verification_file_sha256=_digest(
            f"verification-file:{scope.query_id}"
        ),
        verified_tree_id=_digest(f"tree:{scope.query_id}"),
        group_count=100,
        technical_failures=0,
        retries=0,
        replacements=0,
        mean_gain=0.004 if source_passed else 0.0002,
        paired_gain_ci95=(0.003, 0.005) if source_passed else (-0.001, 0.001),
        harmful_group_count=harmful,
        harm_confidence_level=0.95,
        harm_risk_upper=one_sided_binomial_upper_bound(harmful, 100, 0.95),
        downside_reduction_fraction=0.90,
        retained_candidate_gain_fraction=0.20 if source_passed else 0.01,
        oracle_headroom_fraction=0.10 if source_passed else 0.01,
        protocol_frozen_before_outcomes=True,
        outcomes_used_for_policy_or_gate_selection=False,
        independent_implementation_replay=True,
        source_gate_passed=source_passed,
        metadata={"evidence": "synthetic-test"},
    )


@dataclass(frozen=True)
class _Belief:
    artifact_id: str
    state: tuple[float, ...]


def _belief(label: str) -> _Belief:
    return _Belief(artifact_id=_digest(label), state=(float(len(label)),))


def test_query_scope_changes_for_every_semantic_component() -> None:
    base = _scope("wrapping")
    fields = (
        "simulator_id",
        "task_id",
        "observation_policy_id",
        "action_bank_id",
        "metric_id",
        "world_distribution_id",
    )
    for field in fields:
        changed = replace(base, **{field: _digest(f"changed:{field}"), "query_id": None})
        assert changed.query_id != base.query_id
    changed_unit = replace(
        base,
        statistical_unit="fresh-session-v1",
        query_id=None,
    )
    assert changed_unit.query_id != base.query_id


def test_positive_and_negative_queries_coexist_without_pooling() -> None:
    wrapping = _certificate(_scope("wrapping"), source_passed=True)
    slingshot = _certificate(_scope("slingshot"), source_passed=False)
    registry = QueryCompetenceRegistryV1(
        certificates={
            str(wrapping.query_scope.query_id): wrapping,
            str(slingshot.query_scope.query_id): slingshot,
        }
    )

    assert registry.certified_query_ids == (wrapping.query_scope.query_id,)
    assert registry.failed_query_ids == (slingshot.query_scope.query_id,)
    assert wrapping.certified
    assert not slingshot.certified
    assert "registered-source-gate-rejected" in slingshot.failed_checks
    assert "harm-risk-upper-bound-exceeded" in slingshot.failed_checks


def test_exact_query_and_policy_match_admits_candidate() -> None:
    certificate = _certificate(_scope("wrapping"), source_passed=True)
    registry = QueryCompetenceRegistryV1(
        certificates={str(certificate.query_scope.query_id): certificate}
    )
    baseline = _belief("baseline-belief")
    candidate = _belief("candidate-belief")

    selected, receipt = select_query_competent_belief(
        baseline,
        candidate,
        registry,
        query_scope=certificate.query_scope,
        candidate_policy_id=certificate.candidate_policy_id,
        baseline_policy_id=certificate.baseline_policy_id,
        common_domain_id=_digest("common-domain"),
        inference_admissible=True,
    )

    assert selected is candidate
    assert receipt.selected_candidate
    assert receipt.metadata["routing_reason"] == "query-competence-authorized"


@pytest.mark.parametrize(
    "mode, expected_reason",
    [
        ("failed", "query-competence-rejected"),
        ("unknown", "unknown-query"),
        ("candidate-policy", "candidate-policy-mismatch"),
        ("baseline-policy", "baseline-policy-mismatch"),
        ("inference", "inference-rejected"),
    ],
)
def test_every_unqualified_route_returns_same_baseline_object(
    mode: str,
    expected_reason: str,
) -> None:
    positive = _certificate(_scope("wrapping"), source_passed=True)
    negative = _certificate(_scope("slingshot"), source_passed=False)
    registry = QueryCompetenceRegistryV1(
        certificates={
            str(positive.query_scope.query_id): positive,
            str(negative.query_scope.query_id): negative,
        }
    )
    scope = positive.query_scope
    candidate_policy = positive.candidate_policy_id
    baseline_policy = positive.baseline_policy_id
    inference = True
    if mode == "failed":
        scope = negative.query_scope
        candidate_policy = negative.candidate_policy_id
        baseline_policy = negative.baseline_policy_id
    elif mode == "unknown":
        scope = _scope("unseen-task")
    elif mode == "candidate-policy":
        candidate_policy = _digest("wrong-candidate-policy")
    elif mode == "baseline-policy":
        baseline_policy = _digest("wrong-baseline-policy")
    elif mode == "inference":
        inference = False
    baseline = _belief("baseline-belief")
    candidate = _belief("candidate-belief")

    selected, receipt = select_query_competent_belief(
        baseline,
        candidate,
        registry,
        query_scope=scope,
        candidate_policy_id=candidate_policy,
        baseline_policy_id=baseline_policy,
        common_domain_id=_digest("common-domain"),
        inference_admissible=inference,
    )

    assert selected is baseline
    assert not receipt.selected_candidate
    assert receipt.metadata["routing_reason"] == expected_reason


def test_registry_round_trip_and_content_tampering_fail_closed(tmp_path: Path) -> None:
    certificate = _certificate(_scope("wrapping"), source_passed=True)
    registry = QueryCompetenceRegistryV1(
        certificates={str(certificate.query_scope.query_id): certificate},
        metadata={"claim": "query-scoped-only"},
    )
    path = tmp_path / "registry.json"
    save_query_competence_registry(registry, path)
    loaded = load_query_competence_registry(path)
    assert loaded.artifact_id == registry.artifact_id
    assert loaded.to_record() == registry.to_record()

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["certificates"][0]["mean_gain"] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_id"):
        load_query_competence_registry(path)


def test_harm_bound_and_derived_decision_cannot_be_forged() -> None:
    certificate = _certificate(_scope("wrapping"), source_passed=True)
    with pytest.raises(ValueError, match="Clopper-Pearson"):
        replace(certificate, harm_risk_upper=0.0, artifact_id=None)
    with pytest.raises(ValueError, match="certified"):
        replace(certificate, certified=False, artifact_id=None)
    with pytest.raises(ValueError, match="failed_checks"):
        replace(
            certificate,
            failed_checks=("invented-failure",),
            artifact_id=None,
        )


def test_gate_and_metadata_are_defensively_frozen() -> None:
    metadata = {"nested": {"items": ["source"]}}
    gate = QueryCompetenceGateV1(
        expected_group_count=100,
        minimum_mean_gain=0.0,
        require_positive_paired_lower_bound=False,
        maximum_harm_risk_upper=0.10,
        minimum_downside_reduction_fraction=0.0,
        minimum_retained_candidate_gain_fraction=0.0,
        minimum_oracle_headroom_fraction=0.0,
        metadata=metadata,
    )
    metadata["nested"]["items"].append("mutated")
    assert list(gate.metadata["nested"]["items"]) == ["source"]
    with pytest.raises(TypeError, match="immutable"):
        gate.metadata["nested"]["items"].append("forbidden")
