from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import pytest

from bayesian_phystwin.complete_belief_selection import (
    CompleteBeliefGuardDecisionV1,
)
from bayesian_phystwin.guarded_belief_selection_v2 import (
    CandidateBeliefConstructionReceiptV1,
    GuardedBeliefSelectionReceiptV2,
    bind_guarded_belief_selection_receipt,
    build_candidate_belief_construction_receipt,
)
from bayesian_phystwin.inference.v1 import finalize_guarded_update


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Belief:
    artifact_id: str


@dataclass(frozen=True)
class _Inference:
    candidate_id: str
    update_id: str
    admission_id: str
    observation_artifact_id: str
    linearization_artifact_id: str
    inference_admissible: bool


def _inference() -> _Inference:
    update_id = _digest("update")
    return _Inference(
        candidate_id=update_id,
        update_id=update_id,
        admission_id=_digest("admission"),
        observation_artifact_id=_digest("observation"),
        linearization_artifact_id=_digest("linearization"),
        inference_admissible=True,
    )


def _guarded(*, accepted: bool) -> tuple[_Inference, object, object, object]:
    inference = _inference()
    baseline = _Belief(_digest("baseline"))
    candidate = _Belief(_digest("candidate"))
    decision = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=_digest("domain"),
        certificate_id=_digest(f"certificate:{accepted}"),
        inference_admissible=True,
        regret_guard_accepted=accepted,
        reason="accepted" if accepted else "regret-guard-rejected",
    )
    guarded = finalize_guarded_update(
        inference,
        baseline,
        candidate,
        decision,
    )
    return inference, baseline, candidate, guarded


def test_guarded_selection_receipt_binds_construction_and_exact_selection() -> None:
    inference, baseline, candidate, guarded = _guarded(accepted=True)
    construction = build_candidate_belief_construction_receipt(
        inference,
        baseline,
        candidate,
        common_domain_id=guarded.guard_decision.common_domain_id,
    )
    receipt = bind_guarded_belief_selection_receipt(
        inference,
        guarded,
        construction,
    )

    assert receipt.candidate_construction_receipt_id == construction.receipt_id
    assert receipt.selected_belief_id == candidate.artifact_id
    assert receipt.selected_candidate
    assert not receipt.exact_fallback
    restored = GuardedBeliefSelectionReceiptV2.from_record(receipt.to_record())
    assert restored == receipt


def test_guard_rejection_receipt_selects_exact_baseline() -> None:
    inference, baseline, candidate, guarded = _guarded(accepted=False)
    construction = build_candidate_belief_construction_receipt(
        inference,
        baseline,
        candidate,
        common_domain_id=guarded.guard_decision.common_domain_id,
    )
    receipt = bind_guarded_belief_selection_receipt(
        inference,
        guarded,
        construction,
    )

    assert receipt.selected_belief_id == baseline.artifact_id
    assert not receipt.selected_candidate
    assert receipt.exact_fallback


def test_selection_rejects_candidate_or_domain_substitution() -> None:
    inference, baseline, candidate, guarded = _guarded(accepted=True)
    construction = build_candidate_belief_construction_receipt(
        inference,
        baseline,
        candidate,
        common_domain_id=guarded.guard_decision.common_domain_id,
    )

    wrong_candidate = replace(
        construction,
        candidate_belief_id=_digest("other-candidate"),
    )
    with pytest.raises(ValueError, match="guarded candidate"):
        bind_guarded_belief_selection_receipt(
            inference,
            guarded,
            wrong_candidate,
        )

    wrong_domain = replace(
        construction,
        common_domain_id=_digest("other-domain"),
    )
    with pytest.raises(ValueError, match="different common domains"):
        bind_guarded_belief_selection_receipt(
            inference,
            guarded,
            wrong_domain,
        )


def test_nested_receipt_round_trip_detects_tampering() -> None:
    inference, baseline, candidate, guarded = _guarded(accepted=True)
    construction = build_candidate_belief_construction_receipt(
        inference,
        baseline,
        candidate,
        common_domain_id=guarded.guard_decision.common_domain_id,
    )
    assert (
        CandidateBeliefConstructionReceiptV1.from_record(
            construction.to_record()
        )
        == construction
    )
    receipt = bind_guarded_belief_selection_receipt(
        inference,
        guarded,
        construction,
    )
    tampered = receipt.to_record()
    nested = dict(tampered["candidate_construction"])
    nested["candidate_belief_id"] = _digest("tampered")
    tampered["candidate_construction"] = nested
    with pytest.raises(ValueError):
        GuardedBeliefSelectionReceiptV2.from_record(tampered)
