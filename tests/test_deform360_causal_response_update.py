from __future__ import annotations

import numpy as np

from bayesian_phystwin.deform360_causal_response_admission import (
    CausalResponseAdmissionConfig,
    evaluate_causal_response_admission,
)
from bayesian_phystwin.deform360_causal_response_update import (
    BASELINE_ARM,
    CANDIDATE_ARM,
    CausalResponseMeasurementConfig,
    build_causal_response_measurements,
    predict_causal_response_candidate,
)
from bayesian_phystwin.deform360_direct_depth_provider import (
    DirectDepthEndpointConfig,
    DirectDepthEndpointObservations,
)


def _baseline() -> np.ndarray:
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
    frames = [birth]
    for frame in range(1, 6):
        frames.append(birth + frame * response)
    return np.stack(frames)


def _observations(
    baseline: np.ndarray,
    *,
    scale: float = 1.0,
    translation: np.ndarray | None = None,
) -> DirectDepthEndpointObservations:
    frames = np.asarray([1, 2])
    physical = baseline[frames]
    local_scale = np.asarray([0.5, 1.5, 0.2, 2.0, 0.7, 1.3])
    deformation = (physical[1] - physical[0]) * local_scale[:, None]
    points = physical.copy()
    points[1] += deformation
    centroid = np.mean(points, axis=1, keepdims=True)
    points = centroid + scale * (points - centroid)
    if translation is not None:
        points += np.asarray(translation)[None, None]
    count = baseline.shape[1]
    covariance = np.repeat(
        (0.001**2 * np.eye(3))[None, None],
        2 * count,
        axis=0,
    ).reshape(2, count, 3, 3)
    return DirectDepthEndpointObservations(
        endpoint_frames=frames,
        entity_ids=np.arange(count),
        point_world_m=points,
        covariance_m2=covariance,
        accepted_support=np.ones((2, count), dtype=bool),
        association_probability=np.full((2, count), 0.9),
        support_count=np.full((2, count), 3),
        maximum_view_scatter_m=np.full((2, count), 0.001),
        config=DirectDepthEndpointConfig(),
    )


def _admission(
    baseline: np.ndarray,
    proposal: DirectDepthEndpointObservations,
    validation: DirectDepthEndpointObservations,
    *,
    tactile: float = 1.0,
):
    return evaluate_causal_response_admission(
        "source-case",
        baseline,
        proposal,
        validation,
        np.full(baseline.shape[1], 0.8),
        proposal_camera_ids=("camera-0", "camera-1", "camera-2"),
        validation_camera_ids=("camera-3", "camera-4", "camera-5"),
        tactile_contact_probability=tactile,
        actuator_displacement_m=0.01,
        config=CausalResponseAdmissionConfig(
            minimum_physical_centered_rms_m=0.0002,
            minimum_observed_centered_rms_m=0.0002,
            minimum_pairwise_residual_rms_m=0.0001,
        ),
    )


def test_admitted_measurements_are_metric_and_residual_independent_reliable() -> None:
    baseline = _baseline()
    proposal = _observations(
        baseline,
        scale=1.02,
        translation=np.asarray([0.01, -0.005, 0.002]),
    )
    validation = _observations(baseline)
    admission = _admission(baseline, proposal, validation)
    assert admission.admitted

    response = build_causal_response_measurements(
        baseline,
        proposal,
        admission,
    )

    assert response.accepted
    update = admission.update_frame
    entities = admission.selected_entity_ids
    covariance = response.measurements.covariance_m2[update, entities]
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    assert np.all(response.measurements.prior_reliability[update, entities] > 0.0)
    assert np.allclose(
        response.measurements.association_probability[update, entities],
        0.9,
    )
    assert abs(response.endpoint_nuisance[0].scale - 1.0 / 1.02) < 1e-6
    assert abs(response.endpoint_nuisance[1].scale - 1.0 / 1.02) < 0.01


def test_admitted_update_changes_only_the_future_and_reports_uncertainty() -> None:
    baseline = _baseline().astype(np.float32)
    proposal = _observations(baseline)
    validation = _observations(baseline)
    admission = _admission(baseline, proposal, validation)
    response = build_causal_response_measurements(
        baseline,
        proposal,
        admission,
    )

    report, arrays = predict_causal_response_candidate(
        baseline,
        response,
        admission,
    )

    assert report["candidate_applied"]
    assert np.array_equal(
        arrays[CANDIDATE_ARM][: admission.update_frame + 1],
        baseline[: admission.update_frame + 1],
    )
    assert not np.array_equal(
        arrays[CANDIDATE_ARM][admission.update_frame + 1 :],
        baseline[admission.update_frame + 1 :],
    )
    assert np.all(
        arrays["candidate_correction_variance_m2"][admission.update_frame + 1 :] > 0.0
    )
    assert arrays[CANDIDATE_ARM].dtype == baseline.dtype


def test_rejected_admission_is_bit_exact_baseline_fallback() -> None:
    baseline = _baseline().astype(np.float32)
    proposal = _observations(baseline)
    validation = _observations(baseline)
    admission = _admission(
        baseline,
        proposal,
        validation,
        tactile=0.0,
    )
    response = build_causal_response_measurements(
        baseline,
        proposal,
        admission,
    )

    report, arrays = predict_causal_response_candidate(
        baseline,
        response,
        admission,
    )

    assert not report["candidate_applied"]
    assert report["bit_exact_baseline_fallback"]
    assert np.array_equal(arrays[BASELINE_ARM], baseline)
    assert np.array_equal(arrays[CANDIDATE_ARM], baseline)
    assert not np.any(arrays["candidate_correction_variance_m2"])


def test_excessive_similarity_nuisance_falls_back_exactly() -> None:
    baseline = _baseline().astype(np.float32)
    proposal = _observations(
        baseline,
        translation=np.asarray([0.2, 0.0, 0.0]),
    )
    validation = _observations(baseline)
    admission = _admission(baseline, proposal, validation)
    assert admission.admitted
    response = build_causal_response_measurements(
        baseline,
        proposal,
        admission,
        config=CausalResponseMeasurementConfig(maximum_translation_m=0.05),
    )

    report, arrays = predict_causal_response_candidate(
        baseline,
        response,
        admission,
    )

    assert not response.accepted
    assert "outside-limits" in response.reason
    assert report["bit_exact_baseline_fallback"]
    assert np.array_equal(arrays[CANDIDATE_ARM], baseline)


def test_admission_hash_binds_the_exact_proposal_observations() -> None:
    baseline = _baseline()
    proposal = _observations(baseline)
    validation = _observations(baseline)
    admission = _admission(baseline, proposal, validation)
    changed = _observations(
        baseline,
        translation=np.asarray([1e-4, 0.0, 0.0]),
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "differ from the admitted artifact",
    ):
        build_causal_response_measurements(
            baseline,
            changed,
            admission,
        )


def test_zero_support_rejection_still_builds_an_exact_fallback_carrier() -> None:
    baseline = _baseline().astype(np.float32)
    proposal = _observations(baseline)
    validation = _observations(baseline)
    points = np.full_like(proposal.point_world_m, np.nan)
    covariance = np.full_like(proposal.covariance_m2, np.nan)
    unsupported = DirectDepthEndpointObservations(
        endpoint_frames=proposal.endpoint_frames,
        entity_ids=proposal.entity_ids,
        point_world_m=points,
        covariance_m2=covariance,
        accepted_support=np.zeros_like(proposal.accepted_support),
        association_probability=np.zeros_like(proposal.association_probability),
        support_count=np.zeros_like(proposal.support_count),
        maximum_view_scatter_m=np.zeros_like(proposal.maximum_view_scatter_m),
        config=proposal.config,
    )
    admission = _admission(baseline, unsupported, validation)
    assert not admission.admitted
    assert len(admission.selected_entity_ids) == 0

    response = build_causal_response_measurements(
        baseline,
        unsupported,
        admission,
    )
    report, arrays = predict_causal_response_candidate(
        baseline,
        response,
        admission,
    )

    assert len(response.measurements.entity_ids) == baseline.shape[1]
    assert report["bit_exact_baseline_fallback"]
    assert np.array_equal(arrays[CANDIDATE_ARM], baseline)


def test_candidate_rejects_a_different_physical_prefix() -> None:
    baseline = _baseline()
    proposal = _observations(baseline)
    validation = _observations(baseline)
    admission = _admission(baseline, proposal, validation)
    response = build_causal_response_measurements(
        baseline,
        proposal,
        admission,
    )
    changed = baseline.copy()
    changed[0, 0, 0] += 1e-4

    with np.testing.assert_raises_regex(
        ValueError,
        "baseline prefix differs",
    ):
        predict_causal_response_candidate(
            changed,
            response,
            admission,
        )
