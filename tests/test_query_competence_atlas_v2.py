from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from bayesian_phystwin.query_competence_atlas_v2 import (
    QueryCompetenceAtlasV2,
    QueryCompetenceStageV2,
    load_query_competence_atlas,
    save_query_competence_atlas,
    select_atlas_candidate,
)
from bayesian_phystwin.query_competence_certificate_v1 import SimulatorQueryScopeV1


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _scope(label: str) -> SimulatorQueryScopeV1:
    return SimulatorQueryScopeV1(
        simulator_id=_digest("simulator"),
        task_id=_digest(f"task-{label}"),
        observation_policy_id=_digest(f"observation-{label}"),
        action_bank_id=_digest(f"action-{label}"),
        metric_id=_digest(f"metric-{label}"),
        world_distribution_id=_digest(f"worlds-{label}"),
        statistical_unit="simulator-world",
        metadata={"task": label},
    )


def _entry(
    label: str,
    *,
    role: str = "prospective_certificate",
    statuses=("passed", "passed", "passed", "passed"),
) -> QueryCompetenceStageV2:
    return QueryCompetenceStageV2(
        query_scope=_scope(label),
        evidence_role=role,
        evidence_artifact_id=_digest(f"evidence-{label}"),
        evidence_file_sha256=_digest(f"file-{label}"),
        independent_group_count=10,
        native_qualification=statuses[0],
        action_headroom=statuses[1],
        source_transfer=statuses[2],
        prospective_risk=statuses[3],
        exact_fallback_retained=True,
        protocol_frozen_before_outcomes=True,
        outcomes_used_for_selection=False,
        protected_data_read=False,
        terminal_reason=f"terminal-{label}",
    )


def test_stage_order_prevents_later_pass_after_earlier_failure() -> None:
    with pytest.raises(ValueError, match="cannot pass"):
        _entry("invalid", statuses=("passed", "failed", "passed", "not_evaluated"))


def test_source_screen_cannot_claim_prospective_risk() -> None:
    with pytest.raises(ValueError, match="source evidence"):
        _entry("invalid", role="source_screen")
    source = _entry(
        "source",
        role="source_screen",
        statuses=("passed", "failed", "failed", "not_evaluated"),
    )
    assert source.decision == "rejected"
    assert source.first_failed_stage == "action_headroom"
    assert source.furthest_evaluated_stage == "source_transfer"


def test_atlas_roundtrip_rederives_decisions_and_identity(tmp_path) -> None:
    certified = _entry("certified")
    rejected = _entry("rejected", statuses=("passed", "passed", "failed", "failed"))
    atlas = QueryCompetenceAtlasV2(entries=(rejected, certified))
    path = tmp_path / "atlas.json"
    save_query_competence_atlas(path, atlas)
    loaded = load_query_competence_atlas(path)
    assert loaded.to_record() == atlas.to_record()
    assert loaded.certified_query_ids == (certified.query_scope.query_id,)
    assert loaded.rejected_query_ids == (rejected.query_scope.query_id,)
    with pytest.raises(FileExistsError):
        save_query_competence_atlas(path, atlas)


def test_deserializer_rejects_tampered_derived_decision() -> None:
    record = _entry("tamper").to_record()
    record["decision"] = "rejected"
    with pytest.raises(ValueError, match="derived field"):
        QueryCompetenceStageV2.from_mapping(record)


@dataclass(frozen=True)
class _Belief:
    artifact_id: str


def test_selection_admits_only_certified_exact_query_and_preserves_fallback_identity() -> (
    None
):
    certified = _entry("certified")
    rejected = _entry("rejected", statuses=("passed", "failed", "failed", "failed"))
    atlas = QueryCompetenceAtlasV2(entries=(certified, rejected))
    baseline = _Belief(_digest("baseline"))
    candidate = _Belief(_digest("candidate"))

    selected, receipt = select_atlas_candidate(
        baseline,
        candidate,
        atlas,
        query_id=str(certified.query_scope.query_id),
        inference_admissible=True,
    )
    assert selected is candidate
    assert receipt["reason"] == "query-certified"

    selected, receipt = select_atlas_candidate(
        baseline,
        candidate,
        atlas,
        query_id=str(rejected.query_scope.query_id),
        inference_admissible=True,
    )
    assert selected is baseline
    assert receipt["reason"] == "query-stage-rejected"

    selected, receipt = select_atlas_candidate(
        baseline,
        candidate,
        atlas,
        query_id=_digest("unknown"),
        inference_admissible=True,
    )
    assert selected is baseline
    assert receipt["reason"] == "unknown-query"

    selected, receipt = select_atlas_candidate(
        baseline,
        candidate,
        atlas,
        query_id=str(certified.query_scope.query_id),
        inference_admissible=False,
    )
    assert selected is baseline
    assert receipt["reason"] == "inference-rejected"


def test_atlas_rejects_duplicate_query_scopes() -> None:
    entry = _entry("duplicate")
    with pytest.raises(ValueError, match="unique"):
        QueryCompetenceAtlasV2(entries=(entry, entry))
