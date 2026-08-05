"""Complete-belief routing for one append-only Prob4D stream."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._portable_contracts import load_strict_json_object
from ._prob4d_recursive_policy import RecursiveNuisancePolicyV1
from ._prob4d_recursive_records import (
    ClaimBearingProb4DStreamRunV1,
    ClaimBearingProb4DStreamStepV1,
)
from ._prob4d_stream_binding import bind_prob4d_stream_observation
from ._prob4d_stream_common import (
    PROB4D_PROJECT_ID,
    ArtifactBelief,
    BeliefT,
    _sha256,
    _write_atomic_json,
)
from ._prob4d_stream_manifest import Prob4DObservationFactorStreamV1
from .complete_belief_selection import (
    CompleteBeliefGuardDecisionV1,
    select_complete_belief,
)
from .observation_belief import ObservationBeliefV1
from .physical_linearization import PhysicalLinearizationV1
from .posterior_covariance_semantics import (
    POSTERIOR_COVARIANCE_SEMANTICS_VERSION,
    PosteriorCovarianceSemanticsV1,
    working_irls_covariance_semantics,
)
from .prospective_prob4d_update import ClaimBearingProb4DUpdateV1


def start_claim_bearing_prob4d_stream_run(
    stream: Prob4DObservationFactorStreamV1,
    initial_belief: ArtifactBelief,
    *,
    nuisance_policy: RecursiveNuisancePolicyV1,
    metadata: Mapping[str, Any] | None = None,
) -> ClaimBearingProb4DStreamRunV1:
    """Start an empty immutable run bound to one stream and baseline belief."""

    if not isinstance(stream, Prob4DObservationFactorStreamV1):
        raise TypeError("stream must be a Prob4DObservationFactorStreamV1")
    if not isinstance(nuisance_policy, RecursiveNuisancePolicyV1):
        raise TypeError("nuisance_policy must be a RecursiveNuisancePolicyV1")
    initial_id = _sha256(initial_belief.artifact_id, name="initial_belief_id")
    return ClaimBearingProb4DStreamRunV1(
        stream_artifact_id=cast(str, stream.artifact_id),
        initial_belief_id=initial_id,
        recursive_nuisance_policy_id=cast(str, nuisance_policy.policy_id),
        metadata=metadata or {},
    )


def apply_claim_bearing_prob4d_stream_update(
    stream: Prob4DObservationFactorStreamV1,
    run: ClaimBearingProb4DStreamRunV1,
    *,
    baseline: BeliefT,
    candidate: BeliefT,
    observation: ObservationBeliefV1,
    linearization: PhysicalLinearizationV1,
    claim_update: ClaimBearingProb4DUpdateV1,
    decision: CompleteBeliefGuardDecisionV1,
    nuisance_policy: RecursiveNuisancePolicyV1,
    covariance_semantics: PosteriorCovarianceSemanticsV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[BeliefT, ClaimBearingProb4DStreamRunV1, ClaimBearingProb4DStreamStepV1]:
    """Append one exact accepted update or byte-identical complete-belief fallback."""

    if not isinstance(stream, Prob4DObservationFactorStreamV1):
        raise TypeError("stream must be a Prob4DObservationFactorStreamV1")
    if not isinstance(run, ClaimBearingProb4DStreamRunV1):
        raise TypeError("run must be a ClaimBearingProb4DStreamRunV1")
    if run.stream_artifact_id != stream.artifact_id:
        raise ValueError("run identifies a different Prob4D stream")
    index = len(run.steps)
    if index >= len(stream.updates):
        raise ValueError("the Prob4D stream has no unconsumed update")
    if baseline.artifact_id != run.final_belief_id:
        raise ValueError("baseline does not match the run's current belief")
    if not isinstance(linearization, PhysicalLinearizationV1):
        raise TypeError("linearization must be a PhysicalLinearizationV1")
    if not isinstance(claim_update, ClaimBearingProb4DUpdateV1):
        raise TypeError("claim_update must be a ClaimBearingProb4DUpdateV1")
    if not isinstance(decision, CompleteBeliefGuardDecisionV1):
        raise TypeError("decision must be a CompleteBeliefGuardDecisionV1")
    if not isinstance(nuisance_policy, RecursiveNuisancePolicyV1):
        raise TypeError("nuisance_policy must be a RecursiveNuisancePolicyV1")
    if nuisance_policy.policy_id != run.recursive_nuisance_policy_id:
        raise ValueError("recursive nuisance policy differs from the run lock")
    if decision.common_domain_id != nuisance_policy.state_domain_id:
        raise ValueError("guard common domain differs from nuisance policy")

    binding = bind_prob4d_stream_observation(stream, index, observation)
    if linearization.observation_artifact_id != observation.artifact_id:
        raise ValueError("linearization does not bind the stream observation")
    if linearization.baseline_belief_id != baseline.artifact_id:
        raise ValueError("linearization does not bind the current baseline belief")
    if (
        linearization.metadata.get("recursive_nuisance_policy_id")
        != nuisance_policy.policy_id
    ):
        raise ValueError(
            "linearization does not bind the recursive nuisance policy"
        )
    if claim_update.observation_artifact_id != observation.artifact_id:
        raise ValueError("claim update does not bind the stream observation")
    if claim_update.linearization_artifact_id != linearization.artifact_id:
        raise ValueError("claim update does not bind the physical linearization")
    if decision.baseline_belief_id != baseline.artifact_id:
        raise ValueError("guard decision does not bind the current baseline")
    if decision.candidate_belief_id != candidate.artifact_id:
        raise ValueError("guard decision does not bind the candidate belief")
    if decision.inference_admissible != claim_update.inference_admissible:
        raise ValueError("guard and claim update disagree on inference admissibility")

    semantics = covariance_semantics
    if semantics is None:
        semantics = working_irls_covariance_semantics(
            claim_update.result.posterior_covariance,
            metadata={
                "source": "ClaimBearingProb4DUpdateV1.result.posterior_covariance",
                "claim_update_id": claim_update.update_id,
            },
        )
    elif not isinstance(semantics, PosteriorCovarianceSemanticsV1):
        raise TypeError(
            "covariance_semantics must be a PosteriorCovarianceSemanticsV1"
        )
    if semantics.dimension != len(claim_update.result.posterior_covariance):
        raise ValueError("covariance semantics dimension differs from result")

    selected, selection = select_complete_belief(
        baseline,
        candidate,
        decision,
        metadata={
            "stream_artifact_id": stream.artifact_id,
            "stream_update_id": stream.updates[index].update_id,
            "update_index": index,
        },
    )
    exact_fallback = not selection.selected_candidate and selected is baseline
    if not selection.selected_candidate and not exact_fallback:
        raise AssertionError("rejected recursive update did not reuse the baseline")
    if selection.selected_candidate and selected is not candidate:
        raise AssertionError("accepted recursive update did not reuse the candidate")

    update = stream.updates[index]
    step_metadata = frozen_finite_json_mapping(
        metadata,
        name="claim-bearing Prob4D stream step metadata",
    )
    step = ClaimBearingProb4DStreamStepV1(
        stream_artifact_id=cast(str, stream.artifact_id),
        stream_update_id=cast(str, update.update_id),
        observation_binding_id=cast(str, binding.binding_id),
        update_index=index,
        admitted_frame_start=update.admitted_frame_start,
        causal_frame_stop=update.causal_frame_stop,
        prior_belief_id=baseline.artifact_id,
        observation_artifact_id=observation.artifact_id,
        linearization_artifact_id=linearization.artifact_id,
        claim_update_id=claim_update.update_id,
        candidate_belief_id=candidate.artifact_id,
        guard_decision_id=decision.decision_id,
        selection_id=selection.selection_id,
        selected_belief_id=selection.selected_belief_id,
        selected_candidate=selection.selected_candidate,
        exact_fallback=exact_fallback,
        provider_manifest_id=claim_update.provider_manifest_id,
        calibration_artifact_ids=claim_update.calibration_artifact_ids,
        runtime_revision_source=claim_update.runtime_revision_source,
        runtime_revision_independently_verified=(
            claim_update.runtime_revision_independently_verified
        ),
        covariance_semantics_id=cast(str, semantics.artifact_id),
        covariance_policy_id=semantics.policy_id,
        recursive_nuisance_policy_id=cast(
            str,
            nuisance_policy.policy_id,
        ),
        previous_step_id=None if not run.steps else run.steps[-1].step_id,
        reason=selection.reason,
        metadata={
            **plain_json(step_metadata),
            "prob4d_project_id": PROB4D_PROJECT_ID,
            "posterior_covariance_semantics_version": (
                POSTERIOR_COVARIANCE_SEMANTICS_VERSION
            ),
        },
    )
    if run.steps:
        provider_id = run.provider_manifest_id
        calibration_ids = run.calibration_artifact_ids
        runtime_source = run.runtime_revision_source
        covariance_policy_id = run.covariance_policy_id
    else:
        provider_id = claim_update.provider_manifest_id
        calibration_ids = claim_update.calibration_artifact_ids
        runtime_source = claim_update.runtime_revision_source
        covariance_policy_id = semantics.policy_id
    updated_run = replace(
        run,
        steps=(*run.steps, step),
        provider_manifest_id=provider_id,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source=runtime_source,
        runtime_revision_independently_verified=True,
        covariance_policy_id=covariance_policy_id,
        run_id=None,
    )
    return selected, updated_run, step


def write_claim_bearing_prob4d_stream_run(
    run: ClaimBearingProb4DStreamRunV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist one recursive complete-belief routing history."""

    if not isinstance(run, ClaimBearingProb4DStreamRunV1):
        raise TypeError("run must be a ClaimBearingProb4DStreamRunV1")
    return _write_atomic_json(run.to_record(), path, overwrite=overwrite)


def load_claim_bearing_prob4d_stream_run(
    path: str | Path,
) -> ClaimBearingProb4DStreamRunV1:
    """Load and independently revalidate a recursive routing history."""

    value = load_strict_json_object(
        path,
        label="claim-bearing Prob4D stream run",
    )
    return ClaimBearingProb4DStreamRunV1.from_mapping(value)
