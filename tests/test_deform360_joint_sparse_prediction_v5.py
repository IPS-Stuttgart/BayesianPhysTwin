from __future__ import annotations

from dataclasses import replace

import numpy as np

from bayesian_phystwin._gauge_aware_contracts import GaugeAwareObservationBatch
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.deform360_joint_sparse_prediction_v5 import (
    B0_PHYSICAL_FALLBACK,
    B1_LAST_CAUSAL_RESIDUAL,
    T1_CONTACT_ONLY,
    V1_VISUAL_GUARDED,
    VT2_VISUOTACTILE_UNGUARDED,
    VT3_VISUOTACTILE_ANCHOR_BIAS,
    Deform360JointSparsePredictionInputV5,
    run_deform360_joint_sparse_prediction_v5,
)


def _batch(
    *,
    visual_shift_m: float = 0.012,
    contact_shift_m: float = 0.010,
    with_contact: bool = True,
) -> GaugeAwareObservationBatch:
    count = 4
    state = np.zeros((count, 3, 1), dtype=np.float64)
    state[:, 0, 0] = 1.0
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = visual_shift_m
    covariance = np.broadcast_to(
        np.eye(3, dtype=np.float64) * 0.001**2,
        (count, 3, 3),
    ).copy()
    propagation = _propagation()
    arguments: dict[str, object] = {}
    if with_contact:
        anchor_count = 3
        anchor_state = np.zeros((anchor_count, 3, 1), dtype=np.float64)
        anchor_state[:, 0, 0] = 1.0
        anchor_innovation = np.zeros((anchor_count, 3), dtype=np.float64)
        anchor_innovation[:, 0] = contact_shift_m
        anchor_bias = np.broadcast_to(
            np.eye(3, dtype=np.float64),
            (anchor_count, 3, 3),
        ).copy()
        arguments = {
            "anchor_innovation_m": anchor_innovation,
            "anchor_covariance_m2": np.broadcast_to(
                np.eye(3, dtype=np.float64) * 0.0015**2,
                (anchor_count, 3, 3),
            ).copy(),
            "anchor_state_jacobian": anchor_state,
            "anchor_correlation_group_ids": ("contact-a", "contact-b", "contact-c"),
            "anchor_prior_reliability": np.ones(anchor_count),
            "anchor_prior_nominal_probability": np.full(anchor_count, 0.99),
            "anchor_composite_weight": np.ones(anchor_count),
            "anchor_bias_jacobian": anchor_bias,
            "anchor_bias_prior_covariance": np.eye(3) * 0.020**2,
        }
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=covariance,
        state_jacobian=state,
        gauge_jacobian=np.zeros((count, 3, 1)),
        shared_bias_jacobian=np.zeros((count, 3, 0)),
        view_bias_jacobian=np.zeros((count, 3, 0)),
        query_state_jacobian=propagation[1:3].reshape(-1, 3, 1),
        gauge_prior_covariance=np.eye(1) * 0.005**2,
        correlation_group_ids=("visual-a", "visual-b", "visual-c", "visual-d"),
        prior_reliability=np.ones(count),
        association_probability=np.ones(count),
        prior_nominal_probability=np.full(count, 0.99),
        composite_weight=np.ones(count),
        state_prior_covariance_m2=np.eye(1) * 0.020**2,
        physical_response_scale_m=0.020,
        metadata={"observation_causal_frame_stop": 2},
        **arguments,
    )


def _propagation() -> np.ndarray:
    result = np.zeros((3, 2, 3, 1), dtype=np.float64)
    result[1:, :, 0, 0] = np.asarray([0.75, 1.0])[:, None]
    return result


def _problem(
    batch: GaugeAwareObservationBatch,
    *,
    factor_admitted: bool = True,
) -> Deform360JointSparsePredictionInputV5:
    physical = np.zeros((3, 2, 3), dtype=np.float32)
    persistence = np.zeros_like(physical)
    return Deform360JointSparsePredictionInputV5(
        object_id="001-test",
        episode_id=0,
        stratum="sheet",
        physical_prediction_m=physical,
        persistence_m=persistence,
        last_causal_residual_m=np.full((2, 3), 0.002),
        future_state_jacobian_m=_propagation(),
        observation_batch=batch,
        causal_frame_stop=1,
        evaluation_frame_range_half_open=(1, 3),
        factor_admitted=factor_admitted,
        physical_mode="warp_twin",
        source_artifact_ids={"fixture": "a" * 64},
    )


def _config() -> PriorAwareGaugeConfigV1:
    return PriorAwareGaugeConfigV1(
        effective_samples_per_correlation_group=1.0,
        effective_samples_per_anchor_correlation_group=1.0,
        minimum_conditional_information_fraction=0.0,
        minimum_identifiable_fraction=0.0 + 1e-12,
        minimum_query_sensitivity_fraction=0.0,
    )


def test_registered_arms_propagate_state_and_preserve_references() -> None:
    result = run_deform360_joint_sparse_prediction_v5(
        _problem(_batch()),
        config=_config(),
    )

    assert np.array_equal(
        result.trajectories_m[B0_PHYSICAL_FALLBACK],
        np.zeros((3, 2, 3)),
    )
    assert np.array_equal(
        result.trajectories_m[B1_LAST_CAUSAL_RESIDUAL],
        np.concatenate(
            (
                np.zeros((1, 2, 3), dtype=np.float32),
                np.full((2, 2, 3), 0.002, dtype=np.float32),
            )
        ),
    )
    for method in (
        V1_VISUAL_GUARDED,
        T1_CONTACT_ONLY,
        VT2_VISUOTACTILE_UNGUARDED,
        VT3_VISUOTACTILE_ANCHOR_BIAS,
    ):
        assert result.inference_results[method].inference_admissible
        assert not np.array_equal(
            result.trajectories_m[method],
            result.trajectories_m[B0_PHYSICAL_FALLBACK],
        )
        assert np.array_equal(
            result.trajectories_m[method][0],
            result.trajectories_m[B0_PHYSICAL_FALLBACK][0],
        )
        assert result.trajectories_m[method].dtype == np.float32
    assert result.risk_score >= 0.0
    assert len(result.result_id) == 64


def test_contact_only_arm_never_consumes_visual_residual() -> None:
    first = run_deform360_joint_sparse_prediction_v5(
        _problem(_batch(visual_shift_m=0.012)),
        config=_config(),
    )
    second = run_deform360_joint_sparse_prediction_v5(
        _problem(_batch(visual_shift_m=-0.080)),
        config=_config(),
    )

    assert np.array_equal(
        first.trajectories_m[T1_CONTACT_ONLY],
        second.trajectories_m[T1_CONTACT_ONLY],
    )
    assert not np.array_equal(
        first.trajectories_m[V1_VISUAL_GUARDED],
        second.trajectories_m[V1_VISUAL_GUARDED],
    )


def test_anchor_bias_control_is_a_distinct_inference_arm() -> None:
    result = run_deform360_joint_sparse_prediction_v5(
        _problem(_batch(visual_shift_m=0.0, contact_shift_m=0.020)),
        config=_config(),
    )

    unguarded = result.inference_results[VT2_VISUOTACTILE_UNGUARDED]
    biased = result.inference_results[VT3_VISUOTACTILE_ANCHOR_BIAS]
    assert len(unguarded.anchor_bias_coefficients) == 0
    assert len(biased.anchor_bias_coefficients) == 3
    assert not np.array_equal(
        result.trajectories_m[VT2_VISUOTACTILE_UNGUARDED],
        result.trajectories_m[VT3_VISUOTACTILE_ANCHOR_BIAS],
    )


def test_missing_contact_fails_closed_without_crashing() -> None:
    result = run_deform360_joint_sparse_prediction_v5(
        _problem(_batch(with_contact=False)),
        config=_config(),
    )
    baseline = result.trajectories_m[B0_PHYSICAL_FALLBACK]

    assert result.diagnostics["contact_available"] is False
    assert result.diagnostics["risk_score_candidate_method_id"] == (
        V1_VISUAL_GUARDED
    )
    assert not np.array_equal(result.trajectories_m[V1_VISUAL_GUARDED], baseline)
    assert not result.inference_results[T1_CONTACT_ONLY].inference_admissible
    for method in (
        T1_CONTACT_ONLY,
        VT2_VISUOTACTILE_UNGUARDED,
        VT3_VISUOTACTILE_ANCHOR_BIAS,
    ):
        assert np.array_equal(result.trajectories_m[method], baseline)


def test_visual_risk_score_does_not_depend_on_unregistered_contact() -> None:
    with_contact = run_deform360_joint_sparse_prediction_v5(
        _problem(_batch(with_contact=True)),
        config=_config(),
    )
    without_contact = run_deform360_joint_sparse_prediction_v5(
        _problem(_batch(with_contact=False)),
        config=_config(),
    )

    assert with_contact.risk_score == without_contact.risk_score


def test_failed_joint_admission_is_byte_exact_physical_fallback() -> None:
    result = run_deform360_joint_sparse_prediction_v5(
        _problem(_batch(), factor_admitted=False),
        config=_config(),
    )
    baseline = result.trajectories_m[B0_PHYSICAL_FALLBACK]

    for method in (
        V1_VISUAL_GUARDED,
        VT2_VISUOTACTILE_UNGUARDED,
        VT3_VISUOTACTILE_ANCHOR_BIAS,
    ):
        assert np.array_equal(result.trajectories_m[method], baseline)
        assert result.trajectories_m[method].dtype == baseline.dtype
    assert not np.array_equal(result.trajectories_m[T1_CONTACT_ONLY], baseline)


def test_query_jacobian_must_bind_exact_future_propagation() -> None:
    batch = _batch()
    changed = replace(
        batch,
        query_state_jacobian=np.asarray(batch.query_state_jacobian) * 2.0,
    )
    try:
        _problem(changed)
    except ValueError as error:
        assert "exactly bind" in str(error)
    else:
        raise AssertionError("mismatched query Jacobian was accepted")


def test_observation_values_are_bound_into_input_identity() -> None:
    first = _problem(_batch(visual_shift_m=0.012))
    second = _problem(_batch(visual_shift_m=0.013))

    assert first.input_id != second.input_id


def test_pre_cutoff_state_propagation_is_rejected() -> None:
    propagation = _propagation()
    propagation[0, 0, 0, 0] = 1.0
    batch = replace(
        _batch(),
        query_state_jacobian=propagation[1:3].reshape(-1, 3, 1),
    )
    physical = np.zeros((3, 2, 3), dtype=np.float32)
    try:
        Deform360JointSparsePredictionInputV5(
            object_id="001-test",
            episode_id=0,
            stratum="sheet",
            physical_prediction_m=physical,
            persistence_m=physical,
            last_causal_residual_m=np.zeros((2, 3)),
            future_state_jacobian_m=propagation,
            observation_batch=batch,
            causal_frame_stop=1,
            evaluation_frame_range_half_open=(1, 3),
            factor_admitted=True,
            physical_mode="warp_twin",
            source_artifact_ids={"fixture": "a" * 64},
        )
    except ValueError as error:
        assert "before the causal cutoff" in str(error)
    else:
        raise AssertionError("pre-cutoff state propagation was accepted")


def test_persistence_fallback_requires_exact_persistence() -> None:
    problem = _problem(_batch())
    try:
        replace(
            problem,
            physical_mode="persistence_fallback",
            physical_prediction_m=np.ones_like(problem.physical_prediction_m),
        )
    except ValueError as error:
        assert "byte-equivalent" in str(error)
    else:
        raise AssertionError("non-persistent fallback was accepted")
