from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin.complete_belief_selection import (
    CompleteBeliefGuardDecisionV1,
)
from bayesian_phystwin.posterior_covariance_semantics import (
    PosteriorCovarianceSemanticsV1,
    exact_prior_fallback_covariance_semantics,
    working_irls_covariance_semantics,
)
from bayesian_phystwin.prob4d_factor_stream import (
    apply_claim_bearing_prob4d_stream_update,
    start_claim_bearing_prob4d_stream_run,
)
from prob4d_factor_stream_contract_cases import (
    DOMAIN,
    _Belief,
    _claim_update,
    _linearization,
    _policy,
    _stream_tree,
)


def _apply_update(
    stream: Any,
    observation: Any,
    *,
    run: Any,
    baseline: _Belief,
    candidate: _Belief,
    policy: Any,
    inference_admissible: bool,
    guard_accepted: bool,
    covariance_semantics: PosteriorCovarianceSemanticsV1 | None = None,
):
    linearization = _linearization(
        observation,
        baseline.artifact_id,
        policy,
    )
    update = _claim_update(
        observation.artifact_id,
        linearization.artifact_id,
        admissible=inference_admissible,
    )
    decision = CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=DOMAIN,
        certificate_id="e" * 64,
        inference_admissible=inference_admissible,
        regret_guard_accepted=guard_accepted,
        reason="accepted" if guard_accepted else "rejected",
    )
    result = apply_claim_bearing_prob4d_stream_update(
        stream,
        run,
        baseline=baseline,
        candidate=candidate,
        observation=observation,
        linearization=linearization,
        claim_update=update,
        decision=decision,
        nuisance_policy=policy,
        covariance_semantics=covariance_semantics,
    )
    return (*result, update)


def _covariance_metadata(update: Any) -> dict[str, str]:
    return {
        "source": "ClaimBearingProb4DUpdateV1.result.posterior_covariance",
        "claim_update_id": update.update_id,
    }


def test_inference_rejection_records_exact_prior_fallback_semantics(
    tmp_path: Path,
) -> None:
    stream, observations, _ = _stream_tree(tmp_path)
    baseline = _Belief("a" * 64)
    candidate = _Belief("b" * 64)
    policy = _policy()
    run = start_claim_bearing_prob4d_stream_run(
        stream,
        baseline,
        nuisance_policy=policy,
    )

    selected, updated_run, step, update = _apply_update(
        stream,
        observations[0],
        run=run,
        baseline=baseline,
        candidate=candidate,
        policy=policy,
        inference_admissible=False,
        guard_accepted=False,
    )
    fallback = exact_prior_fallback_covariance_semantics(
        update.result.posterior_covariance,
        reason=update.result.reason,
        metadata=_covariance_metadata(update),
    )
    working = working_irls_covariance_semantics(
        update.result.posterior_covariance,
        metadata=_covariance_metadata(update),
    )

    assert selected is baseline
    assert step.exact_fallback
    assert step.covariance_semantics_id == fallback.artifact_id
    assert step.covariance_policy_id == working.policy_id
    assert updated_run.covariance_policy_id == working.policy_id


def test_rejected_first_member_does_not_block_later_accepted_update(
    tmp_path: Path,
) -> None:
    stream, observations, _ = _stream_tree(tmp_path, two_updates=True)
    baseline = _Belief("a" * 64)
    policy = _policy()
    run = start_claim_bearing_prob4d_stream_run(
        stream,
        baseline,
        nuisance_policy=policy,
    )

    selected0, run0, step0, update0 = _apply_update(
        stream,
        observations[0],
        run=run,
        baseline=baseline,
        candidate=_Belief("b" * 64),
        policy=policy,
        inference_admissible=False,
        guard_accepted=False,
    )
    selected1, run1, step1, update1 = _apply_update(
        stream,
        observations[1],
        run=run0,
        baseline=selected0,
        candidate=_Belief("f" * 64),
        policy=policy,
        inference_admissible=True,
        guard_accepted=True,
    )
    fallback0 = exact_prior_fallback_covariance_semantics(
        update0.result.posterior_covariance,
        reason=update0.result.reason,
        metadata=_covariance_metadata(update0),
    )
    working1 = working_irls_covariance_semantics(
        update1.result.posterior_covariance,
        metadata=_covariance_metadata(update1),
    )

    assert selected0 is baseline
    assert selected1.artifact_id == "f" * 64
    assert step0.covariance_semantics_id == fallback0.artifact_id
    assert step1.covariance_semantics_id == working1.artifact_id
    assert step1.previous_step_id == step0.step_id
    assert run1.covariance_policy_id == working1.policy_id


def test_recursive_update_rejects_working_semantics_for_rejected_inference(
    tmp_path: Path,
) -> None:
    stream, observations, _ = _stream_tree(tmp_path)
    baseline = _Belief("a" * 64)
    policy = _policy()
    run = start_claim_bearing_prob4d_stream_run(
        stream,
        baseline,
        nuisance_policy=policy,
    )

    with pytest.raises(ValueError, match="contradicts the admission decision"):
        _apply_update(
            stream,
            observations[0],
            run=run,
            baseline=baseline,
            candidate=_Belief("b" * 64),
            policy=policy,
            inference_admissible=False,
            guard_accepted=False,
            covariance_semantics=working_irls_covariance_semantics([[0.2]]),
        )


def test_recursive_update_rejects_fallback_semantics_for_accepted_inference(
    tmp_path: Path,
) -> None:
    stream, observations, _ = _stream_tree(tmp_path)
    baseline = _Belief("a" * 64)
    policy = _policy()
    run = start_claim_bearing_prob4d_stream_run(
        stream,
        baseline,
        nuisance_policy=policy,
    )

    fallback = exact_prior_fallback_covariance_semantics(
        [[0.2]],
        reason="accepted",
    )
    with pytest.raises(ValueError, match="contradicts the admission decision"):
        _apply_update(
            stream,
            observations[0],
            run=run,
            baseline=baseline,
            candidate=_Belief("b" * 64),
            policy=policy,
            inference_admissible=True,
            guard_accepted=True,
            covariance_semantics=fallback,
        )


def test_recursive_update_rejects_fallback_reason_drift(
    tmp_path: Path,
) -> None:
    stream, observations, _ = _stream_tree(tmp_path)
    baseline = _Belief("a" * 64)
    policy = _policy()
    run = start_claim_bearing_prob4d_stream_run(
        stream,
        baseline,
        nuisance_policy=policy,
    )

    fallback = exact_prior_fallback_covariance_semantics(
        [[0.2]],
        reason="different-rejection",
    )
    with pytest.raises(ValueError, match="rejected result reason"):
        _apply_update(
            stream,
            observations[0],
            run=run,
            baseline=baseline,
            candidate=_Belief("b" * 64),
            policy=policy,
            inference_admissible=False,
            guard_accepted=False,
            covariance_semantics=fallback,
        )
