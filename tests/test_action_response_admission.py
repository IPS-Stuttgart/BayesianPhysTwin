from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayesian_phystwin.action_response_admission import (
    ActionResponseAdmissionConfig,
    build_action_response_guard_decision,
    evaluate_action_response_admission,
)
from bayesian_phystwin.complete_belief_selection import select_complete_belief


def _fixture(
    *,
    sensor_count: int = 3,
    gain: float = 1.0,
    shared_bias_m: np.ndarray | None = None,
) -> dict[str, object]:
    frame_count = 5
    node_count = 6
    physical = np.zeros((frame_count, node_count, 3), dtype=np.float64)
    shape = np.asarray([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
    for frame, progress in enumerate(np.linspace(0.0, 1.0, frame_count)):
        physical[frame, :, 1] = 0.01 * progress * shape
    observed = np.repeat(physical[None] * gain, sensor_count, axis=0)
    if shared_bias_m is not None:
        bias = np.asarray(shared_bias_m, dtype=np.float64)
        for frame, progress in enumerate(np.linspace(0.0, 1.0, frame_count)):
            observed[:, frame] += progress * bias
    covariance = np.broadcast_to(
        np.eye(3) * 1e-8,
        (sensor_count, frame_count, node_count, 3, 3),
    ).copy()
    action = np.zeros((frame_count, 1, 3), dtype=np.float64)
    action[:, 0, 0] = np.linspace(0.0, 0.01, frame_count)
    return {
        "physical_positions_m": physical,
        "observed_positions_m": observed,
        "observation_validity": np.ones(
            (sensor_count, frame_count, node_count),
            dtype=bool,
        ),
        "observation_covariance_m2": covariance,
        "prior_reliability": np.full(
            (sensor_count, frame_count, node_count),
            0.9,
        ),
        "association_probability": np.full(
            (sensor_count, frame_count, node_count),
            0.95,
        ),
        "actuator_positions_m": action,
        "sensor_group_ids": tuple(f"camera-{index}" for index in range(sensor_count)),
        "correlation_cluster_ids": tuple(
            f"node-{index}" for index in range(node_count)
        ),
        "action_support": np.ones(node_count),
        "physical_prefix_id": "physical-prefix",
        "observation_prefix_id": "observation-prefix",
        "action_prefix_id": "action-prefix",
        "config": ActionResponseAdmissionConfig(
            minimum_response_gain=0.05,
            minimum_direction_cosine=0.8,
            minimum_observed_response_rms_m=0.0001,
            minimum_identifiable_physical_rms_m=0.0001,
        ),
    }


def test_true_shape_response_survives_shared_translation_bias() -> None:
    inputs = _fixture(shared_bias_m=np.asarray([0.004, -0.003, 0.002]))

    result = evaluate_action_response_admission(**inputs)

    assert result.admitted
    assert result.reason == "admitted-action-aligned-prefix-response"
    assert result.shared_bias_mode == "translation-invariant"
    assert result.passing_group_count == 3
    assert all(group.direction_cosine > 0.99 for group in result.groups)
    assert result.to_dict()["information_boundary"]["future_target_read"] is False
    assert result.artifact_id.startswith("sha256:")


def test_sensor_specific_physical_projections_are_supported() -> None:
    inputs = _fixture()
    shared = np.asarray(inputs["physical_positions_m"])
    projected = np.repeat(shared[None], 3, axis=0)
    projected[1] = projected[1][..., [1, 0, 2]]
    projected[2, ..., 2] = projected[2, ..., 1]
    projected[2, ..., 1] = 0.0
    inputs["physical_positions_m"] = projected
    inputs["observed_positions_m"] = projected.copy()

    result = evaluate_action_response_admission(**inputs)

    assert result.admitted
    assert all(group.direction_cosine > 0.99 for group in result.groups)


def test_static_object_with_coherent_translation_bias_is_rejected() -> None:
    inputs = _fixture(gain=0.0, shared_bias_m=np.asarray([0.0, 0.01, 0.0]))

    result = evaluate_action_response_admission(**inputs)

    assert not result.admitted
    assert result.reason == "insufficient-action-aligned-response"
    assert result.passing_group_count == 0


def test_no_measured_action_rejects_even_aligned_camera_response() -> None:
    inputs = _fixture()
    inputs["actuator_positions_m"] = np.zeros((5, 1, 3))

    result = evaluate_action_response_admission(**inputs)

    assert not result.admitted
    assert result.reason == "insufficient-measured-action"


def test_orthogonal_response_is_rejected() -> None:
    inputs = _fixture()
    observed = np.asarray(inputs["observed_positions_m"]).copy()
    observed[..., 2] = observed[..., 1]
    observed[..., 1] = 0.0
    inputs["observed_positions_m"] = observed

    result = evaluate_action_response_admission(**inputs)

    assert not result.admitted
    assert result.passing_group_count == 0


def test_duplicate_correlated_camera_does_not_add_independent_evidence() -> None:
    inputs = _fixture()
    original = evaluate_action_response_admission(**inputs)
    observed = np.asarray(inputs["observed_positions_m"])
    inputs["observed_positions_m"] = np.concatenate(
        (observed, observed[:1]),
        axis=0,
    )
    for name in (
        "observation_validity",
        "observation_covariance_m2",
        "prior_reliability",
        "association_probability",
    ):
        value = np.asarray(inputs[name])
        inputs[name] = np.concatenate((value, value[:1]), axis=0)
    inputs["sensor_group_ids"] = ("camera-0", "camera-1", "camera-2", "camera-0")

    duplicated = evaluate_action_response_admission(**inputs)

    assert duplicated.independent_group_count == original.independent_group_count
    np.testing.assert_allclose(
        [group.response_gain_lower for group in duplicated.groups],
        [group.response_gain_lower for group in original.groups],
    )
    assert duplicated.groups[0].sensor_count == 2
    assert duplicated.admitted == original.admitted


def test_state_residual_does_not_change_prior_reliability_summary() -> None:
    inputs = _fixture()
    first = evaluate_action_response_admission(**inputs)
    observed = np.asarray(inputs["observed_positions_m"]).copy()
    observed[:, -1, 0, 2] += 1.0
    inputs["observed_positions_m"] = observed

    second = evaluate_action_response_admission(**inputs)

    np.testing.assert_allclose(
        [group.mean_prior_reliability for group in first.groups],
        [group.mean_prior_reliability for group in second.groups],
    )


def test_repeated_cluster_identity_does_not_increase_effective_count() -> None:
    inputs = _fixture()
    inputs["correlation_cluster_ids"] = (
        "shared-a",
        "shared-a",
        "shared-b",
        "shared-b",
        "shared-c",
        "shared-c",
    )

    result = evaluate_action_response_admission(**inputs)

    assert not result.admitted
    assert all(group.supported_cluster_count == 3 for group in result.groups)
    assert all(group.effective_cluster_count == 3.0 for group in result.groups)


def test_reference_nodes_remove_bias_without_erasing_global_response() -> None:
    inputs = _fixture()
    physical = np.asarray(inputs["physical_positions_m"]).copy()
    observed = np.asarray(inputs["observed_positions_m"]).copy()
    physical[:, :3] = 0.0
    observed[:, :, :3] = 0.0
    for frame, progress in enumerate(np.linspace(0.0, 1.0, 5)):
        bias = progress * np.asarray([0.005, 0.003, -0.002])
        observed[:, frame] += bias
    inputs["physical_positions_m"] = physical
    inputs["observed_positions_m"] = observed
    inputs["action_support"] = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    inputs["shared_bias_reference_mask"] = np.asarray(
        [True, True, True, False, False, False]
    )
    inputs["correlation_cluster_ids"] = (
        "ref-0",
        "ref-1",
        "ref-2",
        "active-0",
        "active-1",
        "active-2",
    )
    inputs["config"] = ActionResponseAdmissionConfig(
        minimum_supported_cluster_count=3,
        minimum_response_gain=0.05,
        minimum_direction_cosine=0.8,
        minimum_observed_response_rms_m=0.0001,
        minimum_identifiable_physical_rms_m=0.0001,
    )

    result = evaluate_action_response_admission(**inputs)

    assert result.admitted
    assert result.shared_bias_mode == "reference-residual-translation"


def test_sensor_order_does_not_change_content_address() -> None:
    inputs = _fixture()
    first = evaluate_action_response_admission(**inputs)
    order = np.asarray([2, 0, 1])
    for name in (
        "observed_positions_m",
        "observation_validity",
        "observation_covariance_m2",
        "prior_reliability",
        "association_probability",
    ):
        inputs[name] = np.asarray(inputs[name])[order]
    groups = tuple(inputs["sensor_group_ids"])
    inputs["sensor_group_ids"] = tuple(groups[index] for index in order)

    second = evaluate_action_response_admission(**inputs)

    assert second.artifact_id == first.artifact_id


@dataclass(frozen=True)
class _Belief:
    artifact_id: str


def test_rejected_response_returns_exact_complete_baseline_object() -> None:
    inputs = _fixture()
    inputs["actuator_positions_m"] = np.zeros((5, 1, 3))
    admission = evaluate_action_response_admission(**inputs)
    baseline = _Belief("a" * 64)
    candidate = _Belief("b" * 64)
    decision = build_action_response_guard_decision(
        admission,
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        common_domain_id="c" * 64,
        regret_certificate_id="d" * 64,
        numerical_inference_admissible=True,
        regret_guard_accepted=True,
    )

    selected, record = select_complete_belief(baseline, candidate, decision)

    assert selected is baseline
    assert not decision.inference_admissible
    assert not decision.regret_guard_accepted
    assert record.selected_belief_id == baseline.artifact_id
    assert (
        decision.metadata["action_response_admission_id"]
        == admission.artifact_id
    )
