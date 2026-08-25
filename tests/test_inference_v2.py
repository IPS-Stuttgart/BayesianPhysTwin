from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from bayesian_phystwin.inference.v2 import (
    CandidateProposalV1,
    CompleteBeliefGuardDecisionV1,
    InferenceSession,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "api/inference-session-public-api-v2.json"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DummyBelief:
    artifact_id: str


@dataclass(frozen=True)
class DummyInference:
    candidate_id: str
    inference_admissible: bool


@dataclass(frozen=True)
class DummyObservation:
    value: int


def _decision(
    inference: DummyInference,
    baseline: DummyBelief,
    candidate: DummyBelief,
    *,
    accepted: bool,
) -> CompleteBeliefGuardDecisionV1:
    return CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=_digest("domain"),
        certificate_id=_digest("certificate"),
        inference_admissible=inference.inference_admissible,
        regret_guard_accepted=accepted,
        reason="unit-test-guard",
        metadata={"source": "unit-test"},
    )


def _accepted_session() -> InferenceSession[DummyObservation, DummyBelief]:
    def candidate_factory(
        prior: DummyBelief,
        observation: DummyObservation,
        *,
        context: Mapping[str, Any],
    ) -> CandidateProposalV1[DummyBelief]:
        assert prior.artifact_id == _digest("baseline")
        assert observation.value == 7
        assert context["case_id"] == "case-7"
        return CandidateProposalV1(
            inference=DummyInference(_digest("candidate-inference"), True),
            candidate_belief=DummyBelief(_digest("candidate")),
            metadata={"provider": "dummy"},
        )

    def guard_policy(
        inference: DummyInference,
        baseline: DummyBelief,
        candidate: DummyBelief,
        *,
        context: Mapping[str, Any],
    ) -> CompleteBeliefGuardDecisionV1:
        assert context["case_id"] == "case-7"
        return _decision(inference, baseline, candidate, accepted=True)

    return InferenceSession(
        session_id=_digest("session"),
        candidate_factory=candidate_factory,
        guard_policy=guard_policy,
        metadata={"protocol": "unit-test"},
    )


def test_session_selects_the_exact_candidate_and_binds_metadata() -> None:
    baseline = DummyBelief(_digest("baseline"))
    session = _accepted_session()

    result = session.assimilate(
        baseline,
        DummyObservation(7),
        context={"case_id": "case-7"},
    )

    assert result.selected_candidate is True
    assert result.selected_belief is result.candidate_belief
    assert result.exact_fallback is False
    assert result.metadata["session_id"] == _digest("session")
    assert result.metadata["session"] == {"protocol": "unit-test"}
    assert result.metadata["proposal"] == {"provider": "dummy"}
    assert result.metadata["context"] == {"case_id": "case-7"}
    assert len(cast(str, result.metadata["proposal_id"])) == 64
    with pytest.raises(TypeError, match="immutable"):
        result.metadata["context"] = {}  # type: ignore[index]


def test_session_rejection_returns_the_exact_prior_object() -> None:
    baseline = DummyBelief(_digest("fallback-baseline"))
    candidate = DummyBelief(_digest("fallback-candidate"))
    inference = DummyInference(_digest("fallback-inference"), True)

    def candidate_factory(
        prior: DummyBelief,
        observation: object,
        *,
        context: Mapping[str, Any],
    ) -> CandidateProposalV1[DummyBelief]:
        del prior, observation, context
        return CandidateProposalV1(inference, candidate)

    def guard_policy(
        proposed: DummyInference,
        prior: DummyBelief,
        proposed_belief: DummyBelief,
        *,
        context: Mapping[str, Any],
    ) -> CompleteBeliefGuardDecisionV1:
        del context
        return _decision(proposed, prior, proposed_belief, accepted=False)

    result = InferenceSession(
        session_id=_digest("fallback-session"),
        candidate_factory=candidate_factory,
        guard_policy=guard_policy,
    ).assimilate(baseline, object())

    assert result.selected_candidate is False
    assert result.selected_belief is baseline
    assert result.baseline_belief is baseline
    assert result.exact_fallback is True
    assert result.selection.reason == "regret-guard-rejected"


def test_candidate_proposal_is_content_addressed_and_immutable() -> None:
    inference = DummyInference(_digest("proposal-inference"), True)
    belief = DummyBelief(_digest("proposal-belief"))
    proposal = CandidateProposalV1(
        inference,
        belief,
        metadata={"provider": "direct"},
    )

    assert proposal.proposal_id == proposal.proposal_id
    assert len(proposal.proposal_id) == 64
    assert proposal.to_record()["candidate_belief_id"] == belief.artifact_id
    changed = CandidateProposalV1(
        inference,
        belief,
        metadata={"provider": "other"},
    )
    assert changed.proposal_id != proposal.proposal_id
    with pytest.raises(TypeError, match="immutable"):
        proposal.metadata["provider"] = "tampered"  # type: ignore[index]


def test_session_fails_closed_on_invalid_boundaries() -> None:
    valid_factory = _accepted_session().candidate_factory
    valid_guard = _accepted_session().guard_policy

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        InferenceSession("not-a-digest", valid_factory, valid_guard)
    with pytest.raises(TypeError, match="candidate_factory"):
        InferenceSession(
            _digest("session"),
            cast(Any, object()),
            valid_guard,
        )
    with pytest.raises(TypeError, match="guard_policy"):
        InferenceSession(
            _digest("session"),
            valid_factory,
            cast(Any, object()),
        )
    with pytest.raises(TypeError, match="metadata"):
        InferenceSession(
            _digest("session"),
            valid_factory,
            valid_guard,
            metadata=cast(Any, 0),
        )

    session = _accepted_session()
    with pytest.raises(TypeError, match="context"):
        session.assimilate(
            DummyBelief(_digest("baseline")),
            DummyObservation(7),
            context=cast(Any, 0),
        )
    with pytest.raises(TypeError, match="prior must expose artifact_id"):
        session.assimilate(cast(Any, object()), DummyObservation(7))


def test_session_rejects_invalid_policy_outputs() -> None:
    baseline = DummyBelief(_digest("invalid-output-baseline"))
    candidate = DummyBelief(_digest("invalid-output-candidate"))
    inference = DummyInference(_digest("invalid-output-inference"), True)

    def invalid_factory(
        prior: DummyBelief,
        observation: object,
        *,
        context: Mapping[str, Any],
    ) -> object:
        del prior, observation, context
        return object()

    def valid_factory(
        prior: DummyBelief,
        observation: object,
        *,
        context: Mapping[str, Any],
    ) -> CandidateProposalV1[DummyBelief]:
        del prior, observation, context
        return CandidateProposalV1(inference, candidate)

    def invalid_guard(
        proposed: DummyInference,
        prior: DummyBelief,
        proposed_belief: DummyBelief,
        *,
        context: Mapping[str, Any],
    ) -> object:
        del proposed, prior, proposed_belief, context
        return object()

    with pytest.raises(TypeError, match="return CandidateProposalV1"):
        InferenceSession(
            _digest("invalid-factory-session"),
            cast(Any, invalid_factory),
            _accepted_session().guard_policy,
        ).assimilate(baseline, object())

    with pytest.raises(TypeError, match="CompleteBeliefGuardDecisionV1"):
        InferenceSession(
            _digest("invalid-guard-session"),
            valid_factory,
            cast(Any, invalid_guard),
        ).assimilate(baseline, object())


def test_session_preserves_finalizer_binding_checks() -> None:
    baseline = DummyBelief(_digest("binding-baseline"))
    candidate = DummyBelief(_digest("binding-candidate"))
    inference = DummyInference(_digest("binding-inference"), True)

    def candidate_factory(
        prior: DummyBelief,
        observation: object,
        *,
        context: Mapping[str, Any],
    ) -> CandidateProposalV1[DummyBelief]:
        del prior, observation, context
        return CandidateProposalV1(inference, candidate)

    def mismatched_guard(
        proposed: DummyInference,
        prior: DummyBelief,
        proposed_belief: DummyBelief,
        *,
        context: Mapping[str, Any],
    ) -> CompleteBeliefGuardDecisionV1:
        del context
        decision = _decision(proposed, prior, proposed_belief, accepted=True)
        return CompleteBeliefGuardDecisionV1(
            baseline_belief_id=_digest("other-baseline"),
            candidate_belief_id=decision.candidate_belief_id,
            common_domain_id=decision.common_domain_id,
            certificate_id=decision.certificate_id,
            inference_admissible=decision.inference_admissible,
            regret_guard_accepted=decision.regret_guard_accepted,
            reason=decision.reason,
        )

    with pytest.raises(ValueError, match="baseline belief"):
        InferenceSession(
            _digest("binding-session"),
            candidate_factory,
            mismatched_guard,
        ).assimilate(baseline, object())


def test_provider_neutral_api_matches_snapshot() -> None:
    import bayesian_phystwin.inference.v2 as inference_v2

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["package"] == "bayesian_phystwin.inference.v2"
    assert manifest["policy"] == "exact-provider-neutral-session-export-surface"
    assert inference_v2.__all__ == manifest["symbols"]
    assert all(hasattr(inference_v2, symbol) for symbol in inference_v2.__all__)
