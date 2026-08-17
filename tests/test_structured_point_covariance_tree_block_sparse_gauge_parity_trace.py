from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import bayesian_phystwin.tree_block_sparse_gauge_parity_trace as trace_module
from bayesian_phystwin._gauge_aware_contracts import GaugeAwareObservationBatch
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.sparse_prior_aware_gauge_belief import TreeSparseGaugeDesignV1
from bayesian_phystwin.tree_block_sparse_gauge_belief import (
    update_tree_block_sparse_prior_aware_gauge_belief,
)
from bayesian_phystwin.tree_block_sparse_gauge_parity_trace import (
    TreeBlockSparseGaugeParityTraceV1,
    update_tree_block_sparse_prior_aware_gauge_belief_with_parity_trace,
)
from bayesian_phystwin.tree_separator_gaussian_parity import (
    TreeSeparatorGaussianParityError,
)


def _config() -> PriorAwareGaugeConfigV1:
    return replace(
        PriorAwareGaugeConfigV1(),
        minimum_conditional_information_fraction=0.0,
        minimum_identifiable_fraction=1.0e-8,
        minimum_query_sensitivity_fraction=0.0,
        maximum_state_update_m=1.0,
        maximum_update_to_physical_response_ratio=100.0,
        maximum_iterations=20,
        convergence_tolerance=1.0e-11,
    )


def _problem(
    *,
    state_prior: np.ndarray | None = None,
) -> tuple[GaugeAwareObservationBatch, TreeSparseGaugeDesignV1]:
    gauge_count = 3
    block_size = 2
    observation_count = 9
    state_count = 2
    gauge_indices = np.repeat(np.arange(gauge_count), 3)

    state = np.zeros((observation_count, 3, state_count), dtype=np.float64)
    state[:, 0, 0] = np.linspace(0.5, 1.3, observation_count)
    state[:, 1, 1] = np.linspace(1.2, 0.4, observation_count)
    local = np.zeros((observation_count, 3, block_size), dtype=np.float64)
    local[:, 0, 0] = 1.0
    local[:, 1, 1] = 1.0
    injected_state = np.asarray([0.004, -0.003])
    injected_gauge = np.asarray(
        [[0.001, -0.002], [0.002, 0.001], [-0.001, 0.0015]],
        dtype=np.float64,
    )
    innovation = np.einsum(
        "mcs,s->mc",
        state,
        injected_state,
        optimize=True,
    )
    innovation += np.einsum(
        "mcg,mg->mc",
        local,
        injected_gauge[gauge_indices],
        optimize=True,
    )
    innovation[1, 0] += 0.0004
    innovation[7, 1] -= 0.0003

    query = np.zeros((2, 3, state_count), dtype=np.float64)
    query[0, 0, 0] = 1.0
    query[1, 1, 1] = 1.0
    groups = tuple(f"group-{index // 3}" for index in range(observation_count))
    batch = GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.repeat(
            (np.eye(3, dtype=np.float64) * 2.5e-5)[None],
            observation_count,
            axis=0,
        ),
        state_jacobian=state,
        gauge_jacobian=np.zeros((observation_count, 3, 0)),
        shared_bias_jacobian=np.zeros((observation_count, 3, 0)),
        view_bias_jacobian=np.zeros((observation_count, 3, 0)),
        query_state_jacobian=query,
        gauge_prior_covariance=np.zeros((0, 0)),
        correlation_group_ids=groups,
        prior_reliability=np.asarray(
            [0.95, 0.9, 0.85, 0.92, 0.88, 0.84, 0.9, 0.86, 0.82]
        ),
        association_probability=np.asarray(
            [0.98, 0.95, 0.9, 0.97, 0.93, 0.89, 0.96, 0.92, 0.88]
        ),
        prior_nominal_probability=np.repeat(
            np.asarray([0.96, 0.93, 0.9]),
            3,
        ),
        composite_weight=np.repeat(
            np.asarray([0.8, 0.75, 0.7]),
            3,
        ),
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=state_prior,
        metadata={"fixture": "tree-block-parity-trace"},
    )

    parents = np.asarray([-1, 0, 1], dtype=np.int64)
    transitions = np.zeros(
        (gauge_count, block_size, block_size),
        dtype=np.float64,
    )
    transitions[1] = np.eye(block_size) * 0.25
    transitions[2] = np.asarray([[0.2, 0.03], [-0.01, 0.18]])
    scales = np.zeros_like(transitions)
    scales[0] = np.asarray([[0.04, 0.0], [0.01, 0.035]])
    scales[1] = np.asarray([[0.025, 0.0], [0.004, 0.022]])
    scales[2] = np.asarray([[0.02, 0.0], [0.003, 0.018]])
    gauge = TreeSparseGaugeDesignV1(
        local_gauge_jacobian=local,
        gauge_indices=gauge_indices,
        parent_indices=parents,
        transition_matrices=transitions,
        innovation_scale_tril=scales,
        gauge_ids=("window-0", "window-1", "window-2"),
        prior_id="f" * 64,
    )
    return batch, gauge


def _trace() -> TreeBlockSparseGaugeParityTraceV1:
    batch, gauge = _problem()
    return update_tree_block_sparse_prior_aware_gauge_belief_with_parity_trace(
        batch,
        gauge,
        config=_config(),
    )


def test_trace_preserves_the_exact_production_result() -> None:
    batch, gauge = _problem()
    config = _config()
    historical = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        gauge,
        config=config,
    )
    traced = update_tree_block_sparse_prior_aware_gauge_belief_with_parity_trace(
        batch,
        gauge,
        config=config,
    )

    assert historical.inference_admissible
    assert traced.result.result_id == historical.result_id
    assert traced.result.descriptor() == historical.descriptor()
    assert traced.result.diagnostics == historical.diagnostics
    assert traced.result.covariance.descriptor() == historical.covariance.descriptor()
    expected_iterations = int(historical.diagnostics["iterations"])
    assert traced.observed_iteration_count == expected_iterations
    assert len(traced.steps) == 2 * expected_iterations
    assert [step.phase for step in traced.steps] == [
        phase
        for _ in range(expected_iterations)
        for phase in ("irls-solve", "irls-final")
    ]
    assert all(step.parity.passed for step in traced.steps)
    assert traced.dense_precision_avoided_bytes > 0
    assert len(traced.trace_id) == 64
    assert traced.to_dict()["trace_id"] == traced.trace_id

    repeated = update_tree_block_sparse_prior_aware_gauge_belief_with_parity_trace(
        batch,
        gauge,
        config=config,
    )
    assert repeated.trace_id == traced.trace_id


def test_no_identifiable_state_retains_the_exact_fallback_and_empty_trace() -> None:
    batch, gauge = _problem(state_prior=np.zeros((2, 2), dtype=np.float64))
    historical = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        gauge,
        config=_config(),
    )
    traced = update_tree_block_sparse_prior_aware_gauge_belief_with_parity_trace(
        batch,
        gauge,
        config=_config(),
    )

    assert not historical.inference_admissible
    assert historical.reason == "no-identifiable-query-state"
    assert traced.result.result_id == historical.result_id
    assert traced.steps == ()
    assert traced.observed_iteration_count == 0
    assert traced.dense_precision_avoided_bytes == 0


def test_parity_failure_propagates_without_returning_a_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, gauge = _problem()

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise TreeSeparatorGaussianParityError("synthetic parity failure")

    monkeypatch.setattr(
        trace_module,
        "require_tree_separator_gaussian_parity",
        fail,
    )
    with pytest.raises(TreeSeparatorGaussianParityError, match="synthetic"):
        update_tree_block_sparse_prior_aware_gauge_belief_with_parity_trace(
            batch,
            gauge,
            config=_config(),
        )


def test_step_and_trace_validation_fail_closed() -> None:
    trace = _trace()
    first = trace.steps[0]
    report = first.parity
    with pytest.raises(ValueError, match="phase"):
        replace(first, phase="other")
    with pytest.raises(ValueError, match="positive"):
        replace(first, iteration=0)
    with pytest.raises(TypeError, match="TreeSeparatorGaussianParityV1"):
        replace(first, parity=object())  # type: ignore[arg-type]
    failed_metrics = dict(report.metrics)
    failed_metrics["mean_maximum_scaled_error"] = 2.0
    failed_report = replace(report, metrics=failed_metrics, passed=False)
    with pytest.raises(ValueError, match="failed report"):
        replace(first, parity=failed_report)

    with pytest.raises(TypeError, match="result"):
        replace(trace, result=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple"):
        replace(trace, steps=list(trace.steps))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"steps\[0\]"):
        replace(trace, steps=(object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        replace(trace, maximum_condition_number=0.0)
    with pytest.raises(TypeError, match="real number"):
        replace(trace, relative_tolerance=True)
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(trace, relative_tolerance=-1.0)
    with pytest.raises(ValueError, match="must follow an irls-final"):
        replace(
            trace,
            steps=(replace(trace.steps[0], iteration=2),),
        )
    with pytest.raises(ValueError, match="contiguous"):
        replace(
            trace,
            steps=(replace(trace.steps[0], iteration=3),),
        )
    with pytest.raises(ValueError, match="duplicated or out of order"):
        replace(trace, steps=(trace.steps[0], trace.steps[0]))
    with pytest.raises(ValueError, match="must follow irls-solve"):
        replace(trace, steps=(trace.steps[1],))
    with pytest.raises(ValueError, match="condition limit"):
        replace(trace, maximum_condition_number=trace.maximum_condition_number * 2.0)
    with pytest.raises(ValueError, match="relative tolerance"):
        replace(trace, relative_tolerance=trace.relative_tolerance * 2.0)
    with pytest.raises(ValueError, match="absolute tolerance"):
        replace(trace, absolute_tolerance=trace.absolute_tolerance * 2.0)
    with pytest.raises(ValueError, match="complete final"):
        replace(trace, steps=(trace.steps[0],))


def test_trace_identity_changes_with_phase_and_result() -> None:
    trace = _trace()
    changed_step = replace(trace.steps[0], phase="irls-final")
    assert changed_step.step_id != trace.steps[0].step_id
    fallback_batch, gauge = _problem(state_prior=np.zeros((2, 2), dtype=np.float64))
    fallback = update_tree_block_sparse_prior_aware_gauge_belief_with_parity_trace(
        fallback_batch,
        gauge,
        config=_config(),
    )
    assert fallback.trace_id != trace.trace_id
