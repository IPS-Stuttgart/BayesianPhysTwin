from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.tree_block_sparse_prob4d as claim_module
from bayesian_phystwin._gauge_aware_contracts import GaugeAwareObservationBatch
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
    TreeSparseGaugeDesignV1,
    update_sparse_prior_aware_gauge_belief_structured,
)
from bayesian_phystwin.tree_block_gaussian import TreeBlockNormalSystemV1
from bayesian_phystwin.tree_block_sparse_gauge_belief import (
    TreeBlockGaugeAwareBeliefResultV1,
    update_tree_block_sparse_prior_aware_gauge_belief,
)
from bayesian_phystwin.tree_block_sparse_prob4d import (
    ClaimBearingTreeBlockProb4DUpdateV1,
    update_claim_bearing_tree_block_prob4d_from_artifacts,
    update_claim_bearing_tree_block_prob4d_from_path,
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
        metadata={
            "observation_artifact_id": "a" * 64,
            "linearization_artifact_id": "b" * 64,
            "prob4d_claim_bearing_provider_manifest_id": "c" * 64,
            "prob4d_claim_bearing_calibration_artifact_ids": {
                "gauge_artifact_id": "d" * 64,
                "point_artifact_id": "e" * 64,
            },
            "prob4d_claim_bearing_runtime_revision_source": "source_checkout",
            "prob4d_claim_bearing_runtime_revision_independently_verified": True,
        },
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


def _dense_matrix(system: TreeBlockNormalSystemV1) -> np.ndarray:
    global_count = system.global_size
    block_size = system.block_size
    result = np.zeros((system.dimension, system.dimension), dtype=np.float64)
    result[:global_count, :global_count] = system.global_precision
    for index in range(system.node_count):
        node = slice(
            global_count + index * block_size,
            global_count + (index + 1) * block_size,
        )
        result[node, node] = system.node_precision[index]
        result[node, :global_count] = system.global_coupling[index]
        result[:global_count, node] = system.global_coupling[index].T
        parent = int(system.parent_indices[index])
        if parent < 0:
            continue
        parent_node = slice(
            global_count + parent * block_size,
            global_count + (parent + 1) * block_size,
        )
        result[node, parent_node] = system.parent_coupling[index]
        result[parent_node, node] = system.parent_coupling[index].T
    return result


def test_tree_block_factorization_matches_dense_solve_and_covariance() -> None:
    rng = np.random.default_rng(8)
    node_count = 5
    block_size = 3
    global_size = 4
    parents = np.asarray([-1, 0, 0, 1, 1], dtype=np.int64)
    node_precision = np.zeros(
        (node_count, block_size, block_size),
        dtype=np.float64,
    )
    parent_coupling = np.zeros_like(node_precision)
    global_coupling = rng.normal(
        scale=0.04,
        size=(node_count, block_size, global_size),
    )
    for index in range(node_count):
        sample = rng.normal(size=(block_size, block_size))
        node_precision[index] = sample.T @ sample + np.eye(block_size) * 5.0
        if index:
            parent_coupling[index] = rng.normal(
                scale=0.03,
                size=(block_size, block_size),
            )
    global_precision = np.eye(global_size) * 8.0
    node_right = rng.normal(size=(node_count, block_size))
    global_right = rng.normal(size=global_size)
    system = TreeBlockNormalSystemV1(
        parent_indices=parents,
        node_precision=node_precision,
        parent_coupling=parent_coupling,
        global_coupling=global_coupling,
        global_precision=global_precision,
        node_right=node_right,
        global_right=global_right,
    )
    factorization = system.eliminate_nodes(maximum_condition_number=1e12).factor_global(
        maximum_condition_number=1e12
    )
    global_solution, node_solution = factorization.solve(
        global_right,
        node_right,
    )
    dense = _dense_matrix(system)
    dense_right = np.concatenate((global_right, node_right.reshape(-1)))
    dense_solution = np.linalg.solve(dense, dense_right)

    np.testing.assert_allclose(
        np.concatenate((global_solution, node_solution.reshape(-1))),
        dense_solution,
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        factorization.materialize_covariance(),
        np.linalg.inv(dense),
        atol=2e-12,
        rtol=2e-12,
    )
    global_residual, node_residual = system.residual(
        global_solution,
        node_solution,
    )
    assert np.linalg.norm(global_residual) < 1e-11
    assert np.linalg.norm(node_residual) < 1e-11


def test_tree_block_solver_matches_existing_dense_tree_solver() -> None:
    batch, gauge = _problem()
    block_result = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        gauge,
        config=_config(),
    )
    dense_result = update_sparse_prior_aware_gauge_belief_structured(
        batch,
        gauge,
        config=_config(),
    )

    assert block_result.inference_admissible
    assert dense_result.inference_admissible
    for name in (
        "state_coefficients",
        "gauge_delta",
        "robust_weights",
    ):
        np.testing.assert_allclose(
            getattr(block_result, name),
            getattr(dense_result, name),
            atol=2e-9,
            rtol=2e-8,
        )
    np.testing.assert_allclose(
        block_result.materialize_posterior_covariance(),
        dense_result.materialize_posterior_covariance(),
        atol=3e-9,
        rtol=3e-8,
    )
    np.testing.assert_allclose(
        block_result.covariance.state_marginal_covariance(),
        dense_result.materialize_posterior_covariance()[:2, :2],
        atol=3e-9,
        rtol=3e-8,
    )
    assert (
        block_result.diagnostics["tree_prior_information_matrix_materialized"] is False
    )
    assert (
        block_result.diagnostics["dense_nuisance_normal_matrix_materialized"] is False
    )
    assert block_result.diagnostics["dense_joint_normal_matrix_materialized"] is False
    assert block_result.diagnostics["tree_block_global_schur_dimension"] <= 2


def test_tree_block_rejection_stays_factorized_and_budgeted() -> None:
    batch, gauge = _problem(state_prior=np.zeros((2, 2), dtype=np.float64))
    result = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        gauge,
        config=_config(),
    )

    assert not result.inference_admissible
    assert result.reason == "no-identifiable-query-state"
    assert result.dense_covariance_materialized is False
    assert result.covariance.stored_nbytes < (
        result.covariance.estimated_dense_covariance_bytes
    )
    with pytest.raises(MemoryError, match="exceeding"):
        result.materialize_posterior_covariance(
            maximum_bytes=result.covariance.estimated_peak_materialization_bytes - 1
        )
    covariance = result.materialize_posterior_covariance()
    assert covariance.shape == (8, 8)
    np.testing.assert_array_equal(covariance[:2, :2], np.zeros((2, 2)))


def test_tree_block_storage_is_linear_in_tree_size() -> None:
    def storage(node_count: int) -> tuple[int, int]:
        block_size = 7
        global_size = 3
        parents = np.asarray([-1] + list(range(node_count - 1)), dtype=np.int64)
        system = TreeBlockNormalSystemV1(
            parent_indices=parents,
            node_precision=np.repeat(
                (np.eye(block_size) * 3.0)[None],
                node_count,
                axis=0,
            ),
            parent_coupling=np.zeros(
                (node_count, block_size, block_size),
                dtype=np.float64,
            ),
            global_coupling=np.zeros(
                (node_count, block_size, global_size),
                dtype=np.float64,
            ),
            global_precision=np.eye(global_size),
            node_right=np.zeros((node_count, block_size)),
            global_right=np.zeros(global_size),
        )
        return system.stored_nbytes, system.estimated_dense_precision_bytes

    small_stored, small_dense = storage(16)
    large_stored, large_dense = storage(64)
    assert 3.8 < large_stored / small_stored < 4.2
    assert large_dense / small_dense > 13.0


def _adapted(batch: GaugeAwareObservationBatch, gauge: TreeSparseGaugeDesignV1):
    return SimpleNamespace(
        batch=batch,
        tree_gauge_design=gauge,
        observation_artifact_id="a" * 64,
        linearization_artifact_id="b" * 64,
        provider_manifest_id="c" * 64,
        calibration_artifact_ids={
            "gauge_artifact_id": "d" * 64,
            "point_artifact_id": "e" * 64,
        },
        runtime_revision_source="source_checkout",
    )


def test_claim_bearing_tree_block_update_binds_factorized_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, gauge = _problem()
    monkeypatch.setattr(
        claim_module,
        "build_claim_bearing_tree_sparse_prob4d_batch",
        lambda *args, **kwargs: _adapted(batch, gauge),
    )
    update = update_claim_bearing_tree_block_prob4d_from_artifacts(
        object(),
        object(),
        physical_prediction_xyz_m=np.zeros((len(batch.innovation_m), 3)),
        config=_config(),
    )

    assert isinstance(update, ClaimBearingTreeBlockProb4DUpdateV1)
    assert update.inference_admissible
    assert update.dense_covariance_materialized is False
    assert len(update.admission_id) == 64
    assert len(update.tree_block_result_id) == 64
    assert len(update.update_id) == 64
    assert update.descriptor()["tree_block_result_id"] == (update.tree_block_result_id)
    legacy = update.to_legacy()
    np.testing.assert_allclose(
        legacy.result.posterior_covariance,
        update.result.materialize_posterior_covariance(),
    )


def test_claim_bearing_tree_block_path_uses_strict_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, gauge = _problem()
    sentinel = object()
    calls: list[Any] = []
    monkeypatch.setattr(
        claim_module,
        "load_claim_bearing_tree_sparse_prob4d",
        lambda path: calls.append(path) or sentinel,
    )
    monkeypatch.setattr(
        claim_module,
        "build_claim_bearing_tree_sparse_prob4d_batch",
        lambda validated, *args, **kwargs: (
            calls.append(validated) or _adapted(batch, gauge)
        ),
    )

    update = update_claim_bearing_tree_block_prob4d_from_path(
        "claim.json",
        object(),
        physical_prediction_xyz_m=np.zeros((len(batch.innovation_m), 3)),
        config=_config(),
    )
    assert update.inference_admissible
    assert calls == ["claim.json", sentinel]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("result", object(), "TreeBlockGaugeAwareBeliefResultV1"),
        ("runtime_revision_independently_verified", 1, "must be a bool"),
        ("runtime_revision_independently_verified", False, "must be True"),
    ],
)
def test_claim_wrapper_rejects_invalid_fields(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    batch, gauge = _problem()
    monkeypatch.setattr(
        claim_module,
        "build_claim_bearing_tree_sparse_prob4d_batch",
        lambda *args, **kwargs: _adapted(batch, gauge),
    )
    update = update_claim_bearing_tree_block_prob4d_from_artifacts(
        object(),
        object(),
        physical_prediction_xyz_m=np.zeros((len(batch.innovation_m), 3)),
        config=_config(),
    )
    with pytest.raises((TypeError, ValueError), match=message):
        replace(update, **{field: value})


def test_result_type_and_covariance_identity_are_immutable() -> None:
    batch, gauge = _problem()
    first = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        gauge,
        config=_config(),
    )
    second = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        gauge,
        config=_config(),
    )
    assert isinstance(first, TreeBlockGaugeAwareBeliefResultV1)
    assert first.result_id == second.result_id
    assert first.covariance.descriptor() == second.covariance.descriptor()
    with pytest.raises(ValueError):
        first.state_coefficients[0] = 1.0
