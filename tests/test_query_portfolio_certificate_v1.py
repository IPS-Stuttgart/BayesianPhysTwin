"""Unit contracts for simultaneous query-portfolio certificates."""

from __future__ import annotations

import copy

import pytest

from bayesian_phystwin.guard_harm_risk import one_sided_binomial_upper_bound
from bayesian_phystwin.query_portfolio_certificate_v1 import (
    QueryPortfolioCertificateV1,
    QueryPortfolioMemberV1,
    load_query_portfolio_certificate,
    save_query_portfolio_certificate,
)


def _digest(character: str) -> str:
    return character * 64


def _deployed(
    character: str,
    *,
    harmful: int,
    unguarded: int,
    mean_gain: float,
    gain_lower: float,
) -> QueryPortfolioMemberV1:
    confidence = 1.0 - 0.05 / 3.0
    return QueryPortfolioMemberV1(
        query_id=_digest(character),
        decision="certified",
        prospective_risk_evaluated=True,
        candidate_deployed=True,
        exact_fallback_selected=False,
        evidence_artifact_id=_digest("d"),
        evidence_file_sha256=_digest("e"),
        independent_groups=288,
        harmful_groups=harmful,
        unguarded_harmful_groups=unguarded,
        mean_gain_over_fallback=mean_gain,
        familywise_gain_lower=gain_lower,
        familywise_harm_upper=one_sided_binomial_upper_bound(
            harmful,
            288,
            confidence,
        ),
        gain_vector_sha256=_digest("f"),
    )


def _rejected(character: str, *, risk_evaluated: bool) -> QueryPortfolioMemberV1:
    return QueryPortfolioMemberV1(
        query_id=_digest(character),
        decision="rejected",
        prospective_risk_evaluated=risk_evaluated,
        candidate_deployed=False,
        exact_fallback_selected=True,
        evidence_artifact_id=_digest("1"),
        evidence_file_sha256=_digest("2"),
    )


def _certificate() -> QueryPortfolioCertificateV1:
    return QueryPortfolioCertificateV1(
        atlas_id=_digest("3"),
        atlas_file_sha256=_digest("4"),
        members=(
            _deployed(
                "a",
                harmful=1,
                unguarded=15,
                mean_gain=0.0047,
                gain_lower=0.0038,
            ),
            _deployed(
                "b",
                harmful=6,
                unguarded=69,
                mean_gain=0.0034,
                gain_lower=0.0013,
            ),
            _rejected("c", risk_evaluated=True),
            _rejected("9", risk_evaluated=False),
        ),
        familywise_confidence=0.95,
        harm_risk_budget=0.05,
        component_trials_prospective=True,
        portfolio_synthesis_posthoc=True,
        selector_must_be_outcome_independent=True,
    )


def test_certificate_accounts_for_final_risk_family_and_exact_fallback() -> None:
    certificate = _certificate()
    assert certificate.risk_evaluable_count == 3
    assert len(certificate.deployed_members) == 2
    assert len(certificate.fallback_members) == 2
    assert certificate.per_query_confidence == pytest.approx(0.9833333333333333)
    assert certificate.simultaneous_harm_passed
    assert certificate.simultaneous_positive_gain_passed
    assert certificate.maximum_deployed_harm_upper < 0.05
    assert certificate.joint_value_and_harm_confidence_lower == pytest.approx(0.90)
    aggregate = certificate.to_record()["descriptive_equal_query_aggregate"]
    assert aggregate == {
        "evaluation_worlds": 576,
        "guarded_harmful_worlds": 7,
        "unguarded_harmful_worlds": 84,
        "harmful_world_reduction": 77,
        "harmful_world_reduction_fraction": pytest.approx(77 / 84),
        "cross_task_reward_gains_pooled": False,
    }


def test_certificate_round_trips_and_rejects_derived_field_tampering(tmp_path) -> None:
    certificate = _certificate()
    path = tmp_path / "certificate.json"
    save_query_portfolio_certificate(path, certificate)
    assert load_query_portfolio_certificate(path).to_record() == certificate.to_record()
    changed = copy.deepcopy(certificate.to_record())
    changed["maximum_deployed_harm_upper"] = 0.001
    with pytest.raises(ValueError, match="derived field"):
        QueryPortfolioCertificateV1.from_mapping(changed)


def test_certificate_rejects_unadjusted_harm_bound() -> None:
    certificate = _certificate()
    changed = copy.deepcopy(certificate.to_record())
    member = next(item for item in changed["members"] if item["candidate_deployed"])
    member["familywise_harm_upper"] = one_sided_binomial_upper_bound(
        member["harmful_groups"],
        member["independent_groups"],
        0.95,
    )
    member.pop("artifact_id")
    with pytest.raises(ValueError, match="familywise harm bound"):
        QueryPortfolioCertificateV1(
            atlas_id=certificate.atlas_id,
            atlas_file_sha256=certificate.atlas_file_sha256,
            members=tuple(
                QueryPortfolioMemberV1.from_mapping(item)
                if "artifact_id" in item
                else QueryPortfolioMemberV1(
                    query_id=item["query_id"],
                    decision=item["decision"],
                    prospective_risk_evaluated=item["prospective_risk_evaluated"],
                    candidate_deployed=item["candidate_deployed"],
                    exact_fallback_selected=item["exact_fallback_selected"],
                    evidence_artifact_id=item["evidence_artifact_id"],
                    evidence_file_sha256=item["evidence_file_sha256"],
                    independent_groups=item["independent_groups"],
                    harmful_groups=item["harmful_groups"],
                    unguarded_harmful_groups=item["unguarded_harmful_groups"],
                    mean_gain_over_fallback=item["mean_gain_over_fallback"],
                    familywise_gain_lower=item["familywise_gain_lower"],
                    familywise_harm_upper=item["familywise_harm_upper"],
                    gain_vector_sha256=item["gain_vector_sha256"],
                    metadata=item["metadata"],
                )
                for item in changed["members"]
            ),
            familywise_confidence=0.95,
            harm_risk_budget=0.05,
            component_trials_prospective=True,
            portfolio_synthesis_posthoc=True,
            selector_must_be_outcome_independent=True,
        )


def test_non_deployed_query_must_retain_exact_fallback() -> None:
    with pytest.raises(ValueError, match="exact fallback"):
        QueryPortfolioMemberV1(
            query_id=_digest("8"),
            decision="rejected",
            prospective_risk_evaluated=False,
            candidate_deployed=False,
            exact_fallback_selected=False,
            evidence_artifact_id=_digest("7"),
            evidence_file_sha256=_digest("6"),
        )
