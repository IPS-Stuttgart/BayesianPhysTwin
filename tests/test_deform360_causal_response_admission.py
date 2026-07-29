from __future__ import annotations

from dataclasses import replace

import numpy as np

from bayesian_phystwin.deform360_causal_response_admission import (
    CausalResponseAdmissionConfig,
    direct_depth_observation_sha256,
    evaluate_causal_response_admission,
)
from bayesian_phystwin.deform360_direct_depth_provider import (
    DirectDepthEndpointConfig,
    DirectDepthEndpointObservations,
)


def _physical() -> np.ndarray:
    birth = np.asarray(
        [
            [-0.03, -0.02, 0.8],
            [0.00, -0.02, 0.8],
            [0.03, -0.02, 0.8],
            [-0.03, 0.02, 0.8],
            [0.00, 0.02, 0.8],
            [0.03, 0.02, 0.8],
        ],
        dtype=np.float64,
    )
    response = np.column_stack(
        (
            np.linspace(-0.002, 0.002, len(birth)),
            np.linspace(0.0015, -0.0015, len(birth)),
            np.zeros(len(birth)),
        )
    )
    return np.stack((birth, birth + response))


def _observation(
    physical: np.ndarray,
    residual_scale: float,
    *,
    global_update_bias: np.ndarray | None = None,
    global_update_scale: float = 1.0,
) -> DirectDepthEndpointObservations:
    points = physical.copy()
    deformation = (physical[1] - physical[0]) * np.asarray(
        [0.5, 1.5, 0.2, 2.0, 0.7, 1.3]
    )[:, None]
    points[1] += residual_scale * deformation
    update_centroid = np.mean(points[1], axis=0)
    points[1] = update_centroid + global_update_scale * (points[1] - update_centroid)
    if global_update_bias is not None:
        points[1] += np.asarray(global_update_bias, dtype=np.float64)
    count = physical.shape[1]
    covariance = np.repeat(
        (0.001**2 * np.eye(3))[None, None],
        2 * count,
        axis=0,
    ).reshape(2, count, 3, 3)
    return DirectDepthEndpointObservations(
        endpoint_frames=np.asarray([0, 1]),
        entity_ids=np.arange(count),
        point_world_m=points,
        covariance_m2=covariance,
        accepted_support=np.ones((2, count), dtype=bool),
        association_probability=np.full((2, count), 0.9),
        support_count=np.full((2, count), 3),
        maximum_view_scatter_m=np.zeros((2, count)),
        config=DirectDepthEndpointConfig(),
    )


def _evaluate(
    proposal: DirectDepthEndpointObservations,
    validation: DirectDepthEndpointObservations,
    *,
    tactile: float = 1.0,
    physical: np.ndarray | None = None,
    action_conditioning: np.ndarray | None = None,
    config: CausalResponseAdmissionConfig | None = None,
):
    state = _physical() if physical is None else physical
    return evaluate_causal_response_admission(
        "source-case",
        state,
        proposal,
        validation,
        np.full(state.shape[1], 0.8),
        proposal_camera_ids=("camera-0", "camera-1", "camera-2"),
        validation_camera_ids=("camera-3", "camera-4", "camera-5"),
        tactile_contact_probability=tactile,
        actuator_displacement_m=0.01,
        action_conditioning_positions_m=action_conditioning,
        config=config,
    )


def test_agreeing_causal_deformation_is_admitted_and_reproducible() -> None:
    physical = _physical()
    proposal = _observation(physical, 1.0)
    validation = _observation(physical, 0.95)

    first = _evaluate(proposal, validation, physical=physical)
    second = _evaluate(proposal, validation, physical=physical)

    assert first.admitted
    assert first.reason == "causal-cross-panel-response"
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.descriptor() == second.descriptor()
    assert first.metrics.supported_count == 6
    assert first.metrics.supported_group_count == 3
    assert first.metrics.effective_count == 3.0
    assert first.metrics.cross_panel_pairwise_cosine > 0.99
    assert first.metrics.validation_improvement_fraction > 0.99


def test_global_camera_translation_is_not_treated_as_deformation() -> None:
    physical = _physical()
    proposal = _observation(
        physical,
        0.0,
        global_update_bias=np.asarray([0.02, -0.01, 0.005]),
    )
    validation = _observation(
        physical,
        0.0,
        global_update_bias=np.asarray([0.02, -0.01, 0.005]),
    )

    result = _evaluate(proposal, validation, physical=physical)

    assert not result.admitted
    assert result.reason == "insufficient-nonrigid-update-headroom"
    assert result.metrics.proposal_pairwise_residual_rms_m < 1e-12
    assert result.metrics.validation_pairwise_residual_rms_m < 1e-12


def test_global_similarity_scale_is_not_treated_as_deformation() -> None:
    physical = _physical()
    proposal = _observation(
        physical,
        0.0,
        global_update_scale=1.1,
    )
    validation = _observation(
        physical,
        0.0,
        global_update_scale=1.1,
    )

    result = _evaluate(proposal, validation, physical=physical)

    assert not result.admitted
    assert result.reason == "insufficient-nonrigid-update-headroom"
    assert result.metrics.proposal_pairwise_residual_rms_m < 1e-12
    assert result.metrics.validation_pairwise_residual_rms_m < 1e-12


def test_cross_panel_disagreement_rejects_a_camera_specific_update() -> None:
    physical = _physical()
    proposal = _observation(physical, 1.0)
    validation = _observation(physical, -0.5)

    result = _evaluate(
        proposal,
        validation,
        physical=physical,
        config=CausalResponseAdmissionConfig(
            minimum_observed_centered_rms_m=0.0001,
            minimum_pairwise_residual_rms_m=0.0001,
        ),
    )

    assert not result.admitted
    assert result.reason in {
        "response-not-action-aligned",
        "cross-panel-pairwise-disagreement",
        "cross-panel-vector-disagreement",
        "no-heldout-prefix-improvement",
    }
    assert result.metrics.validation_improvement_fraction < 0.0


def test_tactile_contact_is_a_required_independent_causal_signal() -> None:
    physical = _physical()
    proposal = _observation(physical, 1.0)
    validation = _observation(physical, 1.0)

    result = _evaluate(proposal, validation, tactile=0.0, physical=physical)

    assert not result.admitted
    assert result.reason == "insufficient-tactile-contact"


def test_sparse_cross_panel_support_rejects_without_shape_failure() -> None:
    physical = _physical()
    proposal = _observation(physical, 1.0)
    validation = _observation(physical, 1.0)
    sparse = np.zeros_like(proposal.accepted_support)
    sparse[:, :2] = True
    points = np.asarray(proposal.point_world_m).copy()
    covariance = np.asarray(proposal.covariance_m2).copy()
    support_count = np.asarray(proposal.support_count).copy()
    points[~sparse] = np.nan
    covariance[~sparse] = np.nan
    support_count[~sparse] = 0
    proposal = replace(
        proposal,
        point_world_m=points,
        covariance_m2=covariance,
        accepted_support=sparse,
        support_count=support_count,
    )
    validation = replace(
        validation,
        point_world_m=points,
        covariance_m2=covariance,
        accepted_support=sparse,
        support_count=support_count,
    )

    result = _evaluate(proposal, validation, physical=physical)

    assert not result.admitted
    assert result.reason == "insufficient-cross-panel-support"
    assert result.metrics.supported_count == 2
    assert result.selected_entity_ids.shape == (2,)
    assert result.spatial_group_assignments.shape == (2,)


def test_persistence_baseline_can_use_separate_physical_action_support() -> None:
    action_conditioning = _physical()
    persistence = np.repeat(action_conditioning[:1], 2, axis=0)
    proposal = _observation(action_conditioning, 0.5)
    validation = _observation(action_conditioning, 0.5)

    result = _evaluate(
        proposal,
        validation,
        physical=persistence,
        action_conditioning=action_conditioning,
    )

    assert result.admitted
    assert result.metrics.physical_centered_rms_m > 0.0
    assert result.physical_prefix_sha256 != result.action_conditioning_prefix_sha256


def test_dense_support_does_not_inflate_effective_evidence_without_bound() -> None:
    physical = _physical()
    proposal = _observation(physical, 1.0)
    validation = _observation(physical, 1.0)
    config = CausalResponseAdmissionConfig(maximum_effective_count=2.0)

    result = evaluate_causal_response_admission(
        "source-case",
        physical,
        proposal,
        validation,
        np.full(physical.shape[1], 0.8),
        proposal_camera_ids=("camera-0", "camera-1"),
        validation_camera_ids=("camera-2", "camera-3"),
        tactile_contact_probability=1.0,
        actuator_displacement_m=0.01,
        config=config,
    )

    assert result.metrics.effective_count == 2.0


def test_state_residual_does_not_change_prior_support_or_effective_count() -> None:
    physical = _physical()
    proposal = _observation(physical, 1.0)
    validation = _observation(physical, 1.0)
    shifted_physical = physical.copy()
    shifted_physical[1, :, 2] += np.linspace(-0.02, 0.02, physical.shape[1])

    nominal = _evaluate(proposal, validation, physical=physical)
    shifted = _evaluate(proposal, validation, physical=shifted_physical)

    assert np.array_equal(
        nominal.selected_entity_ids,
        shifted.selected_entity_ids,
    )
    assert nominal.metrics.effective_count == shifted.metrics.effective_count
    assert nominal.metrics.supported_count == shifted.metrics.supported_count


def test_camera_panels_must_be_disjoint() -> None:
    physical = _physical()
    proposal = _observation(physical, 1.0)

    with np.testing.assert_raises_regex(ValueError, "must be disjoint"):
        evaluate_causal_response_admission(
            "source-case",
            physical,
            proposal,
            proposal,
            np.full(physical.shape[1], 0.8),
            proposal_camera_ids=("camera-0", "camera-1"),
            validation_camera_ids=("camera-1", "camera-2"),
            tactile_contact_probability=1.0,
            actuator_displacement_m=0.01,
        )


def test_observation_digest_binds_scatter_and_provider_config() -> None:
    observation = _observation(_physical(), 1.0)
    changed_scatter = replace(
        observation,
        maximum_view_scatter_m=(observation.maximum_view_scatter_m + 1e-4),
    )
    changed_config = replace(
        observation,
        config=replace(
            observation.config,
            depth_standard_deviation_m=(
                observation.config.depth_standard_deviation_m + 1e-4
            ),
        ),
    )

    digest = direct_depth_observation_sha256(observation)
    assert direct_depth_observation_sha256(changed_scatter) != digest
    assert direct_depth_observation_sha256(changed_config) != digest
