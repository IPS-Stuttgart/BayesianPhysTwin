from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, strategies as st

from bayesian_phystwin.gauge_aware_belief import (
    GaugeAwareBeliefConfig,
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
    decode_gauge_aware_query,
    select_gauge_aware_candidate,
    update_gauge_aware_belief,
)


def _empty_design(count: int) -> np.ndarray:
    return np.zeros((count, 3, 0), dtype=np.float64)


def _permutation_batch() -> GaugeAwareObservationBatch:
    count = 8
    first = np.linspace(-1.0, 1.0, count)
    second = np.sin(np.linspace(0.0, 2.0 * np.pi, count, endpoint=False))
    state = np.zeros((count, 3, 2), dtype=np.float64)
    state[:, 0, 0] = first
    state[:, 1, 1] = second
    gauge = np.zeros((count, 3, 1), dtype=np.float64)
    gauge[:, 0, 0] = 1.0
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = 0.011 * first + 0.004
    innovation[:, 1] = -0.006 * second
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.tile(np.eye(3) * 2e-5, (count, 1, 1)),
        state_jacobian=state,
        gauge_jacobian=gauge,
        shared_bias_jacobian=_empty_design(count),
        view_bias_jacobian=_empty_design(count),
        query_state_jacobian=state,
        gauge_prior_covariance=np.asarray([[0.01]]),
        correlation_group_ids=("a", "a", "b", "b", "c", "c", "d", "d"),
        prior_reliability=np.linspace(0.65, 1.0, count),
        prior_nominal_probability=np.linspace(0.7, 0.95, count),
        composite_weight=np.asarray([0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 0.6, 0.6]),
        physical_response_scale_m=0.1,
        state_prior_covariance_m2=np.diag([0.01, 0.02]),
        metadata={"observation_artifact_id": "c" * 64},
    )


def _permute_rows(
    batch: GaugeAwareObservationBatch,
    permutation: tuple[int, ...],
) -> GaugeAwareObservationBatch:
    indices = np.asarray(permutation, dtype=np.int64)
    return replace(
        batch,
        innovation_m=batch.innovation_m[indices],
        observation_covariance_m2=batch.observation_covariance_m2[indices],
        state_jacobian=batch.state_jacobian[indices],
        gauge_jacobian=batch.gauge_jacobian[indices],
        shared_bias_jacobian=batch.shared_bias_jacobian[indices],
        view_bias_jacobian=batch.view_bias_jacobian[indices],
        correlation_group_ids=tuple(
            batch.correlation_group_ids[index] for index in indices
        ),
        prior_reliability=batch.prior_reliability[indices],
        prior_nominal_probability=batch.prior_nominal_probability[indices],
        composite_weight=batch.composite_weight[indices],
    )


def _unsupported_mode_batch(
    supported_prior_variance: float,
    unsupported_prior_variance: float,
) -> GaugeAwareObservationBatch:
    count = 14
    first = np.linspace(-1.0, 1.0, count)
    second = np.cos(np.linspace(0.0, 2.0 * np.pi, count, endpoint=False))
    state = np.zeros((count, 3, 2), dtype=np.float64)
    state[:, 0, 0] = first
    state[:, 1, 1] = second
    query = np.zeros_like(state)
    query[:, 0, 0] = first
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = 0.009 * first
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.tile(np.eye(3) * 1e-6, (count, 1, 1)),
        state_jacobian=state,
        gauge_jacobian=_empty_design(count),
        shared_bias_jacobian=_empty_design(count),
        view_bias_jacobian=_empty_design(count),
        query_state_jacobian=query,
        gauge_prior_covariance=np.zeros((0, 0)),
        correlation_group_ids=("window",) * count,
        prior_reliability=np.ones(count),
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=np.diag(
            [supported_prior_variance, unsupported_prior_variance]
        ),
    )


def _confounded_result() -> GaugeAwareBeliefResult:
    count = 10
    mode = np.ones(count)
    state = np.zeros((count, 3, 1), dtype=np.float64)
    state[:, 0, 0] = mode
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = 0.02
    batch = GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.tile(np.eye(3) * 1e-6, (count, 1, 1)),
        state_jacobian=state,
        gauge_jacobian=state.copy(),
        shared_bias_jacobian=_empty_design(count),
        view_bias_jacobian=_empty_design(count),
        query_state_jacobian=state,
        gauge_prior_covariance=np.asarray([[0.01]]),
        correlation_group_ids=("window",) * count,
        prior_reliability=np.ones(count),
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=np.asarray([[0.01]]),
    )
    return update_gauge_aware_belief(batch)


@given(permutation=st.permutations(tuple(range(8))))
def test_gauge_aware_update_is_invariant_to_row_permutation(
    permutation: tuple[int, ...],
) -> None:
    batch = _permutation_batch()
    config = GaugeAwareBeliefConfig(effective_samples_per_correlation_group=2.0)
    original = update_gauge_aware_belief(batch, config=config)
    permuted = update_gauge_aware_belief(
        _permute_rows(batch, permutation),
        config=config,
    )

    assert original.inference_admissible and permuted.inference_admissible
    np.testing.assert_allclose(permuted.state_coefficients, original.state_coefficients)
    np.testing.assert_allclose(permuted.gauge_delta, original.gauge_delta)
    np.testing.assert_allclose(
        permuted.posterior_covariance, original.posterior_covariance
    )
    np.testing.assert_allclose(
        decode_gauge_aware_query(permuted, batch.query_state_jacobian),
        decode_gauge_aware_query(original, batch.query_state_jacobian),
    )
    restored_weights = np.empty_like(permuted.robust_weights)
    restored_weights[np.asarray(permutation)] = permuted.robust_weights
    np.testing.assert_allclose(restored_weights, original.robust_weights)


@given(labels=st.permutations(("renamed-a", "renamed-b", "renamed-c", "renamed-d")))
def test_gauge_aware_update_is_invariant_to_group_labels(
    labels: tuple[str, ...],
) -> None:
    batch = _permutation_batch()
    mapping = dict(zip(("a", "b", "c", "d"), labels, strict=True))
    renamed = replace(
        batch,
        correlation_group_ids=tuple(
            mapping[group] for group in batch.correlation_group_ids
        ),
    )
    config = GaugeAwareBeliefConfig(effective_samples_per_correlation_group=2.0)

    original = update_gauge_aware_belief(batch, config=config)
    relabelled = update_gauge_aware_belief(renamed, config=config)

    np.testing.assert_allclose(
        relabelled.state_coefficients, original.state_coefficients
    )
    np.testing.assert_allclose(relabelled.gauge_delta, original.gauge_delta)
    np.testing.assert_allclose(
        relabelled.posterior_covariance, original.posterior_covariance
    )
    np.testing.assert_allclose(relabelled.robust_weights, original.robust_weights)


@given(
    supported_prior_variance=st.floats(
        min_value=1e-4,
        max_value=0.05,
        allow_nan=False,
        allow_infinity=False,
    ),
    unsupported_prior_variance=st.floats(
        min_value=1e-4,
        max_value=0.05,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_query_irrelevant_mode_preserves_prior_variance(
    supported_prior_variance: float,
    unsupported_prior_variance: float,
) -> None:
    batch = _unsupported_mode_batch(
        supported_prior_variance,
        unsupported_prior_variance,
    )
    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=float(len(batch.innovation_m))
        ),
    )

    assert result.inference_admissible
    assert result.state_coefficients[1] == pytest.approx(0.0, abs=1e-12)
    assert result.posterior_covariance[1, 1] == pytest.approx(
        unsupported_prior_variance,
        rel=1e-10,
        abs=1e-12,
    )


@given(
    dtype_name=st.sampled_from(("float32", "float64")),
    values=st.lists(
        st.floats(
            min_value=-1_000.0,
            max_value=1_000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=16,
    ),
)
def test_rejected_update_preserves_exact_baseline_bytes(
    dtype_name: str,
    values: list[float],
) -> None:
    dtype = np.dtype(dtype_name)
    baseline = np.asarray(values, dtype=dtype)
    candidate = baseline + np.asarray(1.0, dtype=dtype)
    result = _confounded_result()

    selection = select_gauge_aware_candidate(baseline, candidate, result)

    assert not selection.candidate_accepted
    assert selection.selected_value.dtype == baseline.dtype
    assert selection.selected_value.shape == baseline.shape
    assert selection.selected_value.tobytes() == baseline.tobytes()
