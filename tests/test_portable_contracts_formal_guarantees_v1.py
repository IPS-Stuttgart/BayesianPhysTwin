from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from bayesian_phystwin.inference.v1 import (
    CompleteBeliefGuardDecisionV1,
    finalize_guarded_update,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _CompleteBelief:
    artifact_id: str
    mean_xyz_m: tuple[tuple[float, float, float], ...]
    covariance_diag_m2: tuple[tuple[float, float, float], ...]
    hypothesis_weights: tuple[float, ...]
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class _CandidateInference:
    candidate_id: str
    inference_admissible: bool


def _belief(label: str, *, offset: float) -> _CompleteBelief:
    return _CompleteBelief(
        artifact_id=_digest(label),
        mean_xyz_m=((offset, 0.0, 0.0), (0.1 + offset, 0.0, 0.0)),
        covariance_diag_m2=((1.0e-5, 1.1e-5, 1.2e-5),) * 2,
        hypothesis_weights=(0.4, 0.6),
        provenance=(label, "frozen-source-v1"),
    )


def _decision(
    baseline: _CompleteBelief,
    candidate: _CompleteBelief,
    *,
    inference_admissible: bool,
    regret_guard_accepted: bool,
) -> CompleteBeliefGuardDecisionV1:
    return CompleteBeliefGuardDecisionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id=_digest("formal-guarantee-domain"),
        certificate_id=_digest(
            f"formal-guarantee-certificate:{inference_admissible}:"
            f"{regret_guard_accepted}"
        ),
        inference_admissible=inference_admissible,
        regret_guard_accepted=regret_guard_accepted,
        reason="formal-guarantee-regression",
        metadata={"protocol": "formal-guarantees-v1"},
    )


def test_rejected_update_is_complete_belief_noninterference() -> None:
    baseline = _belief("baseline", offset=0.0)
    candidate = _belief("candidate", offset=0.01)
    inference = _CandidateInference(
        candidate_id=_digest("candidate-inference"),
        inference_admissible=True,
    )

    result = finalize_guarded_update(
        inference,
        baseline,
        candidate,
        _decision(
            baseline,
            candidate,
            inference_admissible=True,
            regret_guard_accepted=False,
        ),
        metadata={"case": "regret-rejection"},
    )

    assert result.selected_belief is baseline
    assert result.exact_fallback is True
    assert result.selected_candidate is False
    assert result.to_record()["selected_belief_id"] == baseline.artifact_id
    assert result.selected_belief.provenance is baseline.provenance
    assert result.selected_belief.hypothesis_weights is baseline.hypothesis_weights


def test_inference_rejection_and_acceptance_preserve_exact_objects() -> None:
    baseline = _belief("baseline-2", offset=0.0)
    candidate = _belief("candidate-2", offset=-0.005)

    rejected = finalize_guarded_update(
        _CandidateInference(_digest("inference-rejected"), False),
        baseline,
        candidate,
        _decision(
            baseline,
            candidate,
            inference_admissible=False,
            regret_guard_accepted=False,
        ),
    )
    accepted = finalize_guarded_update(
        _CandidateInference(_digest("inference-accepted"), True),
        baseline,
        candidate,
        _decision(
            baseline,
            candidate,
            inference_admissible=True,
            regret_guard_accepted=True,
        ),
    )

    assert rejected.selected_belief is baseline
    assert rejected.exact_fallback is True
    assert accepted.selected_belief is candidate
    assert accepted.exact_fallback is False


def _linear_gaussian_posterior(
    mean: np.ndarray,
    covariance: np.ndarray,
    design: np.ndarray,
    observation_covariance: np.ndarray,
    observation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    innovation_covariance = (
        design @ covariance @ design.T + observation_covariance
    )
    gain = np.linalg.solve(
        innovation_covariance,
        design @ covariance,
    ).T
    posterior_mean = mean + gain @ (observation - design @ mean)
    residual_map = np.eye(len(mean)) - gain @ design
    posterior_covariance = (
        residual_map @ covariance @ residual_map.T
        + gain @ observation_covariance @ gain.T
    )
    return posterior_mean, 0.5 * (
        posterior_covariance + posterior_covariance.T
    )


def test_retained_basis_reparameterization_preserves_posterior() -> None:
    mean = np.array([0.02, -0.01, 0.03], dtype=np.float64)
    covariance = np.array(
        [
            [4.0e-4, 0.7e-4, 0.2e-4],
            [0.7e-4, 3.0e-4, 0.4e-4],
            [0.2e-4, 0.4e-4, 5.0e-4],
        ],
        dtype=np.float64,
    )
    design = np.array(
        [[1.0, 0.2, -0.1], [0.0, 0.8, 0.4]],
        dtype=np.float64,
    )
    observation_covariance = np.array(
        [[8.0e-5, 1.0e-5], [1.0e-5, 6.0e-5]],
        dtype=np.float64,
    )
    observation = np.array([0.025, 0.002], dtype=np.float64)
    transform = np.array(
        [[1.3, 0.2, 0.0], [0.1, 0.9, 0.3], [0.0, 0.2, 1.1]],
        dtype=np.float64,
    )

    direct_mean, direct_covariance = _linear_gaussian_posterior(
        mean,
        covariance,
        design,
        observation_covariance,
        observation,
    )
    inverse_transform = np.linalg.solve(transform, np.eye(3))
    transformed_mean, transformed_covariance = _linear_gaussian_posterior(
        inverse_transform @ mean,
        inverse_transform @ covariance @ inverse_transform.T,
        design @ transform,
        observation_covariance,
        observation,
    )

    np.testing.assert_allclose(
        transform @ transformed_mean,
        direct_mean,
        atol=1.0e-13,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        transform @ transformed_covariance @ transform.T,
        direct_covariance,
        atol=1.0e-13,
        rtol=1.0e-12,
    )


def test_dense_and_low_rank_query_covariance_are_equivalent() -> None:
    conditional = np.array(
        [
            [4.0e-5, 0.2e-5, 0.0, 0.0],
            [0.2e-5, 5.0e-5, 0.1e-5, 0.0],
            [0.0, 0.1e-5, 6.0e-5, 0.3e-5],
            [0.0, 0.0, 0.3e-5, 7.0e-5],
        ],
        dtype=np.float64,
    )
    factor = np.array(
        [
            [0.003, 0.0],
            [0.003, 0.001],
            [0.0, 0.002],
            [0.0, 0.002],
        ],
        dtype=np.float64,
    )
    query = np.array(
        [[1.0, 0.0, -0.5, 0.0], [0.0, 0.5, 0.0, 1.0]],
        dtype=np.float64,
    )

    dense = conditional + factor @ factor.T
    projected_dense = query @ dense @ query.T
    projected_factorized = (
        query @ conditional @ query.T
        + (query @ factor) @ (query @ factor).T
    )

    np.testing.assert_allclose(
        projected_factorized,
        projected_dense,
        atol=1.0e-15,
        rtol=1.0e-13,
    )
