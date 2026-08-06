from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.prob4d_factor_stream import (
    Prob4DObservationFactorStreamUpdateV1,
    Prob4DObservationFactorStreamV1,
    RecursiveNuisancePolicyV1,
    prob4d_observation_identity_summary,
)
from bayesian_phystwin.prob4d_visual_bias_stream import (
    prob4d_visual_bias_nuisance_family_id,
    validate_prob4d_visual_bias_nuisance_stream,
)
from bayesian_phystwin.prob4d_visual_bias_update import (
    PROB4D_VISUAL_BIAS_ORTHOGONALIZATION,
)

prob4d_visual_bias = pytest.importorskip("prob4d.visual_bias")
prob4d_visual_bias_stream = pytest.importorskip("prob4d.visual_bias_stream")

VisualBiasNuisanceV1 = prob4d_visual_bias.VisualBiasNuisanceV1
build_visual_bias_nuisance_stream = (
    prob4d_visual_bias_stream.build_visual_bias_nuisance_stream
)

REVISION = "1" * 40
REPOSITORY = "FlorianPfaff/Prob4D"


def _observation(*, start: int, stop: int) -> ObservationBeliefV1:
    count = stop - start
    frame_ids = np.arange(start, stop, dtype=np.int64)
    groups = np.arange(count, dtype=np.int64)
    mean = np.zeros((count, 3), dtype=np.float64)
    mean[:, 0] = np.arange(count, dtype=np.float64) * 0.01
    mean[:, 2] = 1.0
    return ObservationBeliefV1(
        case_id="case-a",
        stream_id="stream-a",
        causal_frame_stop=stop,
        view_names=("cam-0",),
        window_names=("window-0",),
        factor_names=(),
        source_repository=REPOSITORY,
        source_revision=REVISION,
        source_artifact_sha256="9" * 64,
        declared_frame_ids=frame_ids,
        mean_xyz_m=mean,
        frame_ids=frame_ids,
        entity_ids=np.arange(10 + start, 10 + stop, dtype=np.int64),
        view_indices=np.zeros(count, dtype=np.int64),
        window_indices=np.zeros(count, dtype=np.int64),
        correlation_group_ids=groups,
        factor_group_ids=groups,
        prior_reliability=np.full(count, 0.9),
        association_probability=np.ones(count),
        local_covariance_m2=np.repeat(
            np.eye(3, dtype=np.float64)[None],
            count,
            axis=0,
        )
        * 1e-4,
        low_rank_factor_m=np.zeros((count, 3, 0), dtype=np.float64),
        group_ids=groups,
        group_prior_nominal_probability=np.full(count, 0.9),
        group_composite_weight=np.ones(count),
        metadata={"causal_source": "cross-repository stream test"},
    )


def _factor_stream(
    observations: tuple[ObservationBeliefV1, ...],
) -> Prob4DObservationFactorStreamV1:
    updates: list[Prob4DObservationFactorStreamUpdateV1] = []
    previous: str | None = None
    for index, observation in enumerate(observations):
        persistent_count, observation_count, identity_sha = (
            prob4d_observation_identity_summary(observation)
        )
        update = Prob4DObservationFactorStreamUpdateV1(
            update_index=index,
            admitted_frame_start=int(observation.frame_ids[0]),
            causal_frame_stop=observation.causal_frame_stop,
            bundle_manifest_path=f"update-{index}/factors.json",
            bundle_manifest_sha256=str(index + 2) * 64,
            bundle_payload_sha256=str(index + 4) * 64,
            bundle_sequence_id="sequence-a",
            case_id=observation.case_id,
            stream_id=observation.stream_id,
            source_repository=observation.source_repository,
            source_revision=observation.source_revision,
            factor_count=1,
            observation_count=observation_count,
            persistent_identity_count=persistent_count,
            observation_identity_sha256=identity_sha,
            gauge_ids=observation.window_names,
            previous_update_id=previous,
        )
        updates.append(update)
        previous = update.update_id
    return Prob4DObservationFactorStreamV1(
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="stream-a",
        source_repository=REPOSITORY,
        source_revision=REVISION,
        updates=tuple(updates),
        metadata={"protocol": "cross-repository visual-bias stream"},
    )


def _sidecar(observation: ObservationBeliefV1) -> object:
    _, observation_count, identity_sha = prob4d_observation_identity_summary(
        observation
    )
    return VisualBiasNuisanceV1(
        observation_artifact_id=observation.artifact_id,
        observation_identity_sha256=identity_sha,
        bias_ids=("session",),
        basis_names=("tx", "ty", "tz"),
        row_bias_indices=np.zeros(observation_count, dtype=np.int64),
        bias_jacobian=np.repeat(
            np.eye(3, dtype=np.float64)[None],
            observation_count,
            axis=0,
        ),
        joint_bias_covariance=np.diag(
            np.array([0.04, 0.01, 0.01], dtype=np.float64)
        ),
        orthogonalization_semantics=(
            PROB4D_VISUAL_BIAS_ORTHOGONALIZATION
        ),
        maximum_gauge_projection=0.0,
        gauge_projection_tolerance=1e-6,
        metadata={"calibration": "source-only"},
    )


def _fixture(update_count: int = 2):
    observations = tuple(
        _observation(start=2 * index, stop=2 * index + 2)
        for index in range(update_count)
    )
    factor_stream = _factor_stream(observations)
    sidecars = tuple(_sidecar(observation) for observation in observations)
    stream = build_visual_bias_nuisance_stream(
        stream_key="case-a/session-a",
        nuisances=sidecars,
        observation_stream_update_ids=tuple(
            update.update_id for update in factor_stream.updates
        ),
        frame_intervals=tuple(
            (
                update.admitted_frame_start,
                update.causal_frame_stop,
            )
            for update in factor_stream.updates
        ),
        model_metadata={"calibration": "source-only"},
        metadata={"protocol": "cross-repository visual-bias stream"},
    )
    family_id = prob4d_visual_bias_nuisance_family_id(stream.bias_model_id)
    policy = RecursiveNuisancePolicyV1(
        mode="persistent_explicit_state",
        state_domain_id="d" * 64,
        nuisance_family_ids=("prob4d-gauge", family_id),
        metadata={"protocol": "cross-repository visual-bias stream"},
    )
    return observations, factor_stream, sidecars, stream, policy


def test_actual_prob4d_stream_is_reconstructed_and_bound_exactly() -> None:
    observations, factor_stream, sidecars, stream, policy = _fixture()

    binding = validate_prob4d_visual_bias_nuisance_stream(
        factor_stream,
        observations,
        sidecars,
        stream,
        policy,
    )

    assert binding.visual_bias_stream.artifact_id == stream.artifact_id
    assert binding.visual_bias_stream.bias_model_id == stream.bias_model_id
    assert binding.visual_bias_stream.observation_count == 4
    assert binding.visual_bias_stream.global_design().shape == (4, 3, 3)
    assert np.array_equal(
        binding.visual_bias_stream.joint_bias_covariance,
        stream.joint_bias_covariance,
    )
    assert not binding.claim_bearing_execution_admissible
    with pytest.raises(ValueError, match="duplicate the prior"):
        binding.require_claim_bearing_execution()


def test_actual_single_update_stream_remains_compatible_with_v2_solver() -> None:
    observations, factor_stream, sidecars, stream, policy = _fixture(
        update_count=1
    )

    binding = validate_prob4d_visual_bias_nuisance_stream(
        factor_stream,
        observations,
        sidecars,
        stream,
        policy,
    )

    assert binding.claim_bearing_execution_admissible
    binding.require_claim_bearing_execution()


def test_conditionally_independent_policy_is_rejected() -> None:
    observations, factor_stream, sidecars, stream, policy = _fixture()
    conditional = RecursiveNuisancePolicyV1(
        mode="conditionally_independent_increments",
        state_domain_id=policy.state_domain_id,
        nuisance_family_ids=policy.nuisance_family_ids,
        conditional_independence_evidence_id="e" * 64,
    )

    with pytest.raises(ValueError, match="duplicate the prior"):
        validate_prob4d_visual_bias_nuisance_stream(
            factor_stream,
            observations,
            sidecars,
            stream,
            conditional,
        )


def test_stream_sidecar_slice_tampering_is_rejected() -> None:
    observations, factor_stream, sidecars, stream, policy = _fixture()
    tampered_jacobian = np.array(stream.bias_jacobian, copy=True)
    tampered_jacobian[2, 0, 0] += 0.5
    tampered = replace(
        stream,
        bias_jacobian=tampered_jacobian,
        artifact_id=None,
    )

    with pytest.raises(ValueError, match="Jacobian differs"):
        validate_prob4d_visual_bias_nuisance_stream(
            factor_stream,
            observations,
            sidecars,
            tampered,
            policy,
        )


def test_stream_factor_update_mismatch_is_rejected() -> None:
    observations, factor_stream, sidecars, stream, policy = _fixture()
    first, second = stream.updates
    tampered_first = replace(
        first,
        observation_stream_update_id="f" * 64,
        update_id=None,
    )
    tampered_second = replace(
        second,
        previous_update_id=tampered_first.update_id,
        update_id=None,
    )
    tampered = replace(
        stream,
        updates=(tampered_first, tampered_second),
        artifact_id=None,
    )

    with pytest.raises(ValueError, match="different factor-stream update"):
        validate_prob4d_visual_bias_nuisance_stream(
            factor_stream,
            observations,
            sidecars,
            tampered,
            policy,
        )
