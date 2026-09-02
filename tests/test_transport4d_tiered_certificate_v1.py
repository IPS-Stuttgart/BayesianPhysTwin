from __future__ import annotations

import copy

import numpy as np
import pytest

from bayesian_phystwin_experiments.transport4d_tiered_certificate_v1 import (
    TieredTransportCertificateV1,
    TransportCandidateV1,
    TransportTier,
)

SHA = "a" * 64


def candidate(
    tier: TransportTier,
    *,
    passed: bool = True,
    target_outcome_blind: bool = True,
    effect: float | None = 1.0,
    radius: float | None = 0.1,
) -> TransportCandidateV1:
    kwargs: dict[str, object] = {
        "tier": tier,
        "evidence_id": SHA,
        "checks": {"source_gate": passed, "query_support": passed},
        "target_outcome_blind": target_outcome_blind,
        "adaptation_dimension": 0,
        "transports_mean": True,
        "transports_uncertainty": True,
        "query_effect": np.asarray([effect], dtype=np.float64),
        "query_error_radius": radius,
    }
    if tier is TransportTier.LOW_DIMENSIONAL_CORRECTION:
        kwargs["adaptation_dimension"] = 1
    elif tier is TransportTier.UNCERTAINTY_ONLY:
        kwargs.update(
            transports_mean=False,
            transports_uncertainty=True,
            query_effect=None,
            query_error_radius=None,
        )
    elif tier is TransportTier.PROCEDURE_ONLY:
        kwargs.update(
            adaptation_dimension=8,
            transports_mean=False,
            transports_uncertainty=False,
            query_effect=None,
            query_error_radius=None,
        )
    return TransportCandidateV1(**kwargs)


def certificate(*candidates: TransportCandidateV1) -> TieredTransportCertificateV1:
    fallback = "hold"
    return TieredTransportCertificateV1(
        query_id="registered-target-effect",
        query_contract_id=SHA,
        baseline_belief_id=SHA,
        action_portfolio_id=SHA,
        baseline_query=np.asarray([0.0]),
        action_names=(fallback, "execute"),
        action_weights=np.asarray([[0.0], [-1.0]]),
        action_offsets=np.asarray([0.0, 0.0]),
        fallback_action_name=fallback,
        candidates=candidates,
        regret_tolerance=0.0,
    )


def test_selects_highest_structurally_and_decision_certified_tier() -> None:
    exact = candidate(TransportTier.EXACT_COEFFICIENTS)
    query_effect = candidate(TransportTier.QUERY_IDENTIFIABLE_EFFECT)

    result = certificate(query_effect, exact)

    assert result.selected_tier is TransportTier.EXACT_COEFFICIENTS
    assert result.selected_action_name == "execute"
    assert result.used_exact_fallback is False
    assert result.evaluation_for(TransportTier.EXACT_COEFFICIENTS).selected is True
    assert (
        result.evaluation_for(TransportTier.QUERY_IDENTIFIABLE_EFFECT).selected is False
    )


def test_descends_when_higher_tier_does_not_identify_an_action() -> None:
    uncertain_exact = candidate(
        TransportTier.EXACT_COEFFICIENTS,
        effect=0.05,
        radius=0.20,
    )
    query_effect = candidate(
        TransportTier.QUERY_IDENTIFIABLE_EFFECT,
        effect=0.8,
        radius=0.1,
    )

    result = certificate(uncertain_exact, query_effect)

    assert result.selected_tier is TransportTier.QUERY_IDENTIFIABLE_EFFECT
    exact_eval = result.evaluation_for(TransportTier.EXACT_COEFFICIENTS)
    assert exact_eval.structurally_eligible is True
    assert exact_eval.action_certified is False
    assert exact_eval.reason_code == "regret-budget-exceeded"


def test_uncertainty_only_and_procedure_only_keep_exact_fallback() -> None:
    fallback = "hold"
    uncertainty = candidate(TransportTier.UNCERTAINTY_ONLY)
    procedure = candidate(TransportTier.PROCEDURE_ONLY)
    result = TieredTransportCertificateV1(
        query_id="registered-target-effect",
        query_contract_id=SHA,
        baseline_belief_id=SHA,
        action_portfolio_id=SHA,
        baseline_query=np.asarray([0.0]),
        action_names=(fallback, "execute"),
        action_weights=np.asarray([[0.0], [-1.0]]),
        action_offsets=np.asarray([0.0, 0.0]),
        fallback_action_name=fallback,
        candidates=(procedure, uncertainty),
        regret_tolerance=0.0,
    )

    assert result.selected_tier is TransportTier.UNCERTAINTY_ONLY
    assert result.belief_transport_only is True
    assert result.used_exact_fallback is True
    assert result.selected_action_name is result.fallback_action_name

    procedure_result = certificate(
        candidate(TransportTier.UNCERTAINTY_ONLY, passed=False),
        procedure,
    )
    assert procedure_result.selected_tier is TransportTier.PROCEDURE_ONLY
    assert procedure_result.belief_transport_only is False
    assert (
        procedure_result.selected_action_name is procedure_result.fallback_action_name
    )


def test_all_rejected_returns_fallback_without_transport_tier() -> None:
    result = certificate(
        candidate(TransportTier.EXACT_COEFFICIENTS, passed=False),
        candidate(TransportTier.PROCEDURE_ONLY, passed=False),
    )

    assert result.selected_tier is None
    assert result.selected_candidate_id is None
    assert result.used_exact_fallback is True
    assert result.selected_action_name is result.fallback_action_name


def test_target_outcome_contamination_fails_closed() -> None:
    result = certificate(
        candidate(
            TransportTier.EXACT_COEFFICIENTS,
            target_outcome_blind=False,
        ),
        candidate(TransportTier.PROCEDURE_ONLY),
    )

    assert result.selected_tier is TransportTier.PROCEDURE_ONLY
    exact = result.evaluation_for(TransportTier.EXACT_COEFFICIENTS)
    assert exact.selectable is False
    assert exact.reason_code == "target-outcome-contaminated"


def test_affine_ball_regret_is_exact_in_one_dimension() -> None:
    exact = candidate(
        TransportTier.EXACT_COEFFICIENTS,
        effect=0.4,
        radius=0.1,
    )
    result = certificate(exact)
    evaluation = result.evaluation_for(TransportTier.EXACT_COEFFICIENTS)

    # Losses are hold=0 and execute=-q over q in [0.3, 0.5].
    # Execute has zero worst-case regret; hold has worst regret 0.5.
    assert evaluation.robust_regret_upper_by_action == pytest.approx((0.5, 0.0))
    assert evaluation.action_name == "execute"
    assert evaluation.minimax_regret_upper == pytest.approx(0.0)


def test_arrays_and_metadata_are_immutable_copies() -> None:
    effect = np.asarray([1.0])
    metadata = {"source": ["public"]}
    exact = TransportCandidateV1(
        tier=TransportTier.EXACT_COEFFICIENTS,
        evidence_id=SHA,
        checks={"source_gate": True},
        target_outcome_blind=True,
        adaptation_dimension=0,
        transports_mean=True,
        transports_uncertainty=True,
        query_effect=effect,
        query_error_radius=0.1,
        metadata=metadata,
    )
    result = certificate(exact)
    effect[0] = 99.0
    metadata["source"].append("mutated")

    assert exact.query_effect is not None
    assert exact.query_effect.tolist() == [1.0]
    assert exact.metadata == {"source": ["public"]}
    with pytest.raises(ValueError):
        exact.query_effect[0] = 2.0
    with pytest.raises(ValueError):
        result.baseline_query[0] = 2.0


def test_content_ids_detect_tampering() -> None:
    exact = candidate(TransportTier.EXACT_COEFFICIENTS)
    with pytest.raises(ValueError, match="candidate_id"):
        TransportCandidateV1(
            tier=exact.tier,
            evidence_id=exact.evidence_id,
            checks=exact.checks,
            target_outcome_blind=exact.target_outcome_blind,
            adaptation_dimension=exact.adaptation_dimension,
            transports_mean=exact.transports_mean,
            transports_uncertainty=exact.transports_uncertainty,
            query_effect=exact.query_effect,
            query_error_radius=exact.query_error_radius,
            candidate_id="b" * 64,
        )

    result = certificate(exact)
    kwargs = {
        "query_id": result.query_id,
        "query_contract_id": result.query_contract_id,
        "baseline_belief_id": result.baseline_belief_id,
        "action_portfolio_id": result.action_portfolio_id,
        "baseline_query": result.baseline_query,
        "action_names": result.action_names,
        "action_weights": result.action_weights,
        "action_offsets": result.action_offsets,
        "fallback_action_name": result.fallback_action_name,
        "candidates": result.candidates,
        "regret_tolerance": result.regret_tolerance,
        "artifact_id": "b" * 64,
    }
    with pytest.raises(ValueError, match="artifact_id"):
        TieredTransportCertificateV1(**kwargs)


def test_invalid_tier_semantics_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero target-fit"):
        TransportCandidateV1(
            tier=TransportTier.EXACT_COEFFICIENTS,
            evidence_id=SHA,
            checks={"source_gate": True},
            target_outcome_blind=True,
            adaptation_dimension=1,
            transports_mean=True,
            transports_uncertainty=False,
            query_effect=np.asarray([1.0]),
            query_error_radius=0.1,
        )
    with pytest.raises(ValueError, match="must transport uncertainty"):
        TransportCandidateV1(
            tier=TransportTier.UNCERTAINTY_ONLY,
            evidence_id=SHA,
            checks={"source_gate": True},
            target_outcome_blind=True,
            adaptation_dimension=0,
            transports_mean=False,
            transports_uncertainty=False,
        )


def test_duplicate_tier_is_rejected() -> None:
    first = candidate(TransportTier.EXACT_COEFFICIENTS)
    second = copy.deepcopy(first)
    with pytest.raises(ValueError, match="one candidate per transport tier"):
        certificate(first, second)
