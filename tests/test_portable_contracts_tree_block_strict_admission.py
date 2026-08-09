from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import bayesian_phystwin.tree_block_sparse_gauge_belief_v2 as strict_module
import bayesian_phystwin.tree_block_sparse_prob4d as claim_module
from bayesian_phystwin._gauge_aware_contracts import GaugeAwareObservationBatch
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
    TreeSparseGaugeDesignV1,
)
from bayesian_phystwin.tree_block_sparse_gauge_belief import (
    TreeBlockGaugeAwareBeliefResultV1,
    update_tree_block_sparse_prior_aware_gauge_belief,
)
from bayesian_phystwin.tree_block_sparse_gauge_belief_v2 import (
    TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_IMPLEMENTATION,
    update_tree_block_sparse_prior_aware_gauge_belief_v2,
)


def _fixture(
    *,
    with_anchor: bool = False,
) -> tuple[GaugeAwareObservationBatch, TreeSparseGaugeDesignV1]:
    count = 12
    mode = np.linspace(-1.0, 1.0, count)
    innovation = np.zeros((count, 3), dtype=np.float64)
    innovation[:, 0] = 0.006 * mode
    state = np.zeros((count, 3, 1), dtype=np.float64)
    state[:, 0, 0] = mode
    local_gauge = np.zeros((count, 3, 1), dtype=np.float64)
    empty = np.zeros((count, 3, 0), dtype=np.float64)
    anchor_fields: dict[str, object] = {}
    if with_anchor:
        anchor_state = np.zeros((1, 3, 1), dtype=np.float64)
        anchor_state[0, 0, 0] = 1.0
        anchor_fields = {
            "anchor_innovation_m": np.asarray([[0.001, 0.0, 0.0]]),
            "anchor_covariance_m2": np.asarray([np.eye(3) * 0.01]),
            "anchor_state_jacobian": anchor_state,
            "anchor_correlation_group_ids": ("anchor-group",),
            "anchor_prior_reliability": np.ones(1),
            "anchor_prior_nominal_probability": np.full(1, 0.99),
            "anchor_composite_weight": np.ones(1),
        }
    batch = GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.repeat(
            (np.eye(3) * 0.01)[None],
            count,
            axis=0,
        ),
        state_jacobian=state,
        gauge_jacobian=empty,
        shared_bias_jacobian=empty,
        view_bias_jacobian=empty,
        query_state_jacobian=state.copy(),
        correlation_group_ids=("group-0",) * count,
        prior_reliability=np.ones(count),
        prior_nominal_probability=np.full(count, 0.99),
        composite_weight=np.ones(count),
        physical_response_scale_m=1.0,
        gauge_prior_covariance=np.zeros((0, 0)),
        state_prior_covariance_m2=np.asarray([[0.04]]),
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
        **anchor_fields,
    )
    tree = TreeSparseGaugeDesignV1(
        local_gauge_jacobian=local_gauge,
        gauge_indices=np.zeros(count, dtype=np.int64),
        parent_indices=np.asarray([-1], dtype=np.int64),
        transition_matrices=np.zeros((1, 1, 1), dtype=np.float64),
        innovation_scale_tril=np.asarray([[[0.3]]], dtype=np.float64),
        gauge_ids=("window-0",),
        prior_id="f" * 64,
    )
    return batch, tree


def _exhausted_config() -> PriorAwareGaugeConfigV1:
    return replace(
        PriorAwareGaugeConfigV1(),
        effective_samples_per_correlation_group=12.0,
        maximum_iterations=1,
        convergence_tolerance=1.0e-15,
        minimum_conditional_information_fraction=0.0,
        minimum_identifiable_fraction=1.0e-8,
        minimum_query_sensitivity_fraction=0.0,
        maximum_state_update_m=1.0,
        maximum_update_to_physical_response_ratio=100.0,
    )


def _converged_config() -> PriorAwareGaugeConfigV1:
    return replace(
        _exhausted_config(),
        maximum_iterations=100,
        convergence_tolerance=1.0e-12,
    )


def test_strict_v2_rejects_exhausted_tree_block_fixed_point() -> None:
    batch, tree = _fixture()
    config = _exhausted_config()

    historical = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        tree,
        config=config,
    )
    strict = update_tree_block_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=config,
    )

    assert historical.inference_admissible
    assert historical.diagnostics["mixture_fixed_point_converged"] is False
    assert not strict.inference_admissible
    assert strict.reason == "strict-v2-fixed-point-not-converged"
    assert strict.diagnostics["underlying_inference_admissible"] is True
    assert strict.diagnostics["strict_admission_passed"] is False
    assert strict.diagnostics["strict_admission_reason"] == strict.reason
    assert strict.diagnostics["implementation_id"] == (
        TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_IMPLEMENTATION
    )
    np.testing.assert_array_equal(strict.state_coefficients, 0.0)
    np.testing.assert_array_equal(strict.gauge_delta, 0.0)
    np.testing.assert_allclose(
        strict.materialize_posterior_covariance(),
        np.diag([0.04, 0.09]),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert not strict.dense_covariance_materialized


def test_strict_v2_admits_converged_tree_block_result() -> None:
    batch, tree = _fixture()
    result = update_tree_block_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_converged_config(),
    )

    assert result.inference_admissible
    assert result.reason == "inference-admissible"
    assert result.diagnostics["strict_admission_passed"] is True
    assert result.diagnostics["strict_admission_reason"] == ("strict-admission-passed")
    assert result.diagnostics["underlying_inference_admissible"] is True


def test_strict_v2_preserves_underlying_rejection() -> None:
    batch, tree = _fixture()
    config = replace(_converged_config(), maximum_state_update_m=1.0e-12)
    historical = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        tree,
        config=config,
    )
    result = update_tree_block_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=config,
    )

    assert not historical.inference_admissible
    assert not result.inference_admissible
    assert result.reason == historical.reason
    assert result.diagnostics["strict_admission_reason"] == (
        "underlying-inference-rejected"
    )
    assert result.diagnostics["underlying_inference_admissible"] is False
    np.testing.assert_array_equal(result.state_coefficients, 0.0)
    np.testing.assert_array_equal(result.gauge_delta, 0.0)


@pytest.mark.parametrize(
    ("name", "value", "expected_reason"),
    [
        (
            "robust_likelihood_objective",
            "precision-floored-group-mixture-approximation",
            "strict-v2-non-exact-mixture-objective",
        ),
        (
            "posterior_solver",
            "unexpected-solver",
            "strict-v2-invalid-admission-diagnostics",
        ),
        (
            "mixture_solution_delta",
            True,
            "strict-v2-invalid-admission-diagnostics",
        ),
        (
            "mixture_solution_delta",
            "not-a-number",
            "strict-v2-invalid-admission-diagnostics",
        ),
        (
            "mixture_solution_delta",
            -1.0,
            "strict-v2-invalid-admission-diagnostics",
        ),
        (
            "mixture_stationarity_norm",
            None,
            "strict-v2-invalid-admission-diagnostics",
        ),
        (
            "maximum_eliminated_node_condition_number",
            0.0,
            "strict-v2-invalid-admission-diagnostics",
        ),
    ],
)
def test_strict_v2_rejects_invalid_accepted_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: object,
    expected_reason: str,
) -> None:
    batch, tree = _fixture(with_anchor=True)
    accepted = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        tree,
        config=_converged_config(),
    )
    assert accepted.inference_admissible
    diagnostics = dict(accepted.diagnostics)
    diagnostics[name] = value
    malformed = replace(accepted, diagnostics=diagnostics)
    monkeypatch.setattr(
        strict_module,
        "update_tree_block_sparse_prior_aware_gauge_belief",
        lambda *_args, **_kwargs: malformed,
    )

    result = update_tree_block_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_converged_config(),
    )

    assert not result.inference_admissible
    assert result.reason == expected_reason
    assert result.diagnostics["strict_admission_passed"] is False
    assert result.diagnostics["underlying_inference_admissible"] is True
    np.testing.assert_array_equal(result.state_coefficients, 0.0)
    np.testing.assert_array_equal(result.gauge_delta, 0.0)
    assert result.materialize_posterior_covariance().shape == (2, 2)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("mixture_solution_delta", float("inf")),
        ("global_schur_condition_number", float("nan")),
    ],
)
def test_strict_failure_rejects_nonfinite_diagnostics(
    name: str,
    value: float,
) -> None:
    diagnostics: dict[str, object] = {
        "robust_likelihood_objective": "exact-group-mixture-gradient",
        "mixture_fixed_point_converged": True,
        "posterior_solver": "tree-block-leaf-schur-cholesky-v1",
        "mixture_solution_delta": 0.0,
        "mixture_stationarity_norm": 0.0,
        "maximum_eliminated_node_condition_number": 1.0,
        "global_schur_condition_number": 1.0,
    }
    diagnostics[name] = value

    assert strict_module._strict_failure(diagnostics) == (  # noqa: SLF001
        "strict-v2-invalid-admission-diagnostics"
    )


def test_claim_bearing_tree_block_adapter_binds_strict_v2() -> None:
    assert claim_module.update_tree_block_sparse_prior_aware_gauge_belief is (
        update_tree_block_sparse_prior_aware_gauge_belief_v2
    )


def test_strict_v2_argument_types_fail_closed() -> None:
    batch, tree = _fixture()
    with pytest.raises(TypeError, match="batch must"):
        update_tree_block_sparse_prior_aware_gauge_belief_v2(  # type: ignore[arg-type]
            object(),
            tree,
        )
    with pytest.raises(TypeError, match="gauge must"):
        update_tree_block_sparse_prior_aware_gauge_belief_v2(  # type: ignore[arg-type]
            batch,
            object(),
        )
    with pytest.raises(TypeError, match="config must"):
        update_tree_block_sparse_prior_aware_gauge_belief_v2(
            batch,
            tree,
            config=object(),  # type: ignore[arg-type]
        )


def test_strict_result_identity_binds_admission_diagnostics() -> None:
    batch, tree = _fixture()
    historical = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        tree,
        config=_converged_config(),
    )
    strict = update_tree_block_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_converged_config(),
    )

    assert strict.result_id != historical.result_id
    assert (
        strict.result_id
        == update_tree_block_sparse_prior_aware_gauge_belief_v2(
            batch,
            tree,
            config=_converged_config(),
        ).result_id
    )
    assert isinstance(strict, TreeBlockGaugeAwareBeliefResultV1)
