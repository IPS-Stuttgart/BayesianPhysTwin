from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

import bayesian_phystwin.tree_block_sparse_gauge_belief_v2 as strict_module
import bayesian_phystwin.tree_block_sparse_prob4d as claim_module
from bayesian_phystwin._gauge_aware_contracts import GaugeAwareObservationBatch
from bayesian_phystwin._prior_aware_gauge_math import PriorAwareGaugeConfigV1
from bayesian_phystwin.sparse_prior_aware_gauge_belief import (
    TreeSparseGaugeDesignV1,
)
from bayesian_phystwin.tree_block_claim_contract import (
    validate_tree_block_covariance,
    validate_tree_block_factorization,
    validate_tree_block_result,
)
from bayesian_phystwin.tree_block_sparse_gauge_belief import (
    TreeBlockGaugeAwareBeliefResultV1,
    update_tree_block_sparse_prior_aware_gauge_belief,
)
from bayesian_phystwin.tree_block_sparse_gauge_belief_v2 import (
    TREE_BLOCK_SPARSE_GAUGE_BELIEF_V2_IMPLEMENTATION,
    update_tree_block_sparse_prior_aware_gauge_belief_v2,
)
from bayesian_phystwin.tree_block_sparse_prob4d import (
    ClaimBearingTreeBlockProb4DUpdateV1,
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


def _unsafe_clone(value: object, **changes: object) -> object:
    clone = object.__new__(type(value))
    for field in fields(value):
        object.__setattr__(
            clone,
            field.name,
            changes.get(field.name, getattr(value, field.name)),
        )
    return clone


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


def test_strict_v2_reconstructs_prior_for_malformed_underlying_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, tree = _fixture()
    accepted = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        tree,
        config=_converged_config(),
    )
    malformed = replace(
        accepted,
        inference_admissible=False,
        reason="synthetic-underlying-rejection",
        state_coefficients=np.zeros_like(accepted.state_coefficients),
        gauge_delta=np.zeros_like(accepted.gauge_delta),
        shared_bias_coefficients=np.zeros_like(accepted.shared_bias_coefficients),
        view_bias_coefficients=np.zeros_like(accepted.view_bias_coefficients),
        anchor_bias_coefficients=np.zeros_like(accepted.anchor_bias_coefficients),
    )
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
    assert result.reason == "synthetic-underlying-rejection"
    assert result.covariance.descriptor() != malformed.covariance.descriptor()
    np.testing.assert_allclose(
        result.materialize_posterior_covariance(),
        np.diag([0.04, 0.09]),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_claim_contract_rejects_forged_tree_factors() -> None:
    batch, tree = _fixture()
    result = update_tree_block_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_converged_config(),
    )
    factorization = result.covariance.factorization

    node_factor = np.array(factorization.node_cholesky, copy=True)
    node_factor[0, 0, 0] = -node_factor[0, 0, 0]
    forged_factorization = replace(factorization, node_cholesky=node_factor)
    with pytest.raises(ValueError, match="positive diagonal"):
        validate_tree_block_factorization(forged_factorization)

    forged_factorization = replace(
        factorization,
        node_condition_numbers=factorization.node_condition_numbers * 2.0,
    )
    with pytest.raises(ValueError, match="do not match"):
        validate_tree_block_factorization(forged_factorization)

    forged_scalar = _unsafe_clone(
        factorization,
        global_condition_number=True,
    )
    with pytest.raises(TypeError, match="real number"):
        validate_tree_block_factorization(forged_scalar)  # type: ignore[arg-type]


def test_claim_contract_rejects_invalid_covariance_and_parameter_layout() -> None:
    batch, tree = _fixture()
    result = update_tree_block_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_converged_config(),
    )
    indefinite = replace(
        result.covariance,
        state_prior_covariance=np.asarray([[-0.04]], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_tree_block_result(replace(result, covariance=indefinite))

    ambiguous = replace(
        result,
        gauge_delta=np.zeros(0, dtype=np.float64),
        shared_bias_coefficients=np.zeros(1, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="gauge dimension"):
        validate_tree_block_result(ambiguous)

    invalid_bias = _unsafe_clone(result.covariance, bias_count=-1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        validate_tree_block_covariance(invalid_bias)  # type: ignore[arg-type]

    empty_state = _unsafe_clone(
        result.covariance,
        state_prior_covariance=np.zeros((0, 0), dtype=np.float64),
        state_mapping=np.zeros((0, 0), dtype=np.float64),
        bias_count=result.covariance.factorization.global_size,
    )
    assert validate_tree_block_covariance(empty_state) is empty_state


def test_claim_contract_defensive_types_fail_closed() -> None:
    with pytest.raises(TypeError, match="TreeBlockFactorizationV1"):
        validate_tree_block_factorization(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TreeBlockPosteriorCovarianceV1"):
        validate_tree_block_covariance(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TreeBlockGaugeAwareBeliefResultV1"):
        validate_tree_block_result(object())  # type: ignore[arg-type]

    batch, tree = _fixture()
    result = update_tree_block_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_converged_config(),
    )
    with pytest.raises(TypeError, match="must be a bool"):
        validate_tree_block_result(
            result,
            require_strict_admission=1,  # type: ignore[arg-type]
        )


def test_claim_wrapper_requires_strict_validated_tree_result() -> None:
    batch, tree = _fixture()
    raw = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        tree,
        config=_converged_config(),
    )
    with pytest.raises(ValueError, match="strict tree-block admission"):
        ClaimBearingTreeBlockProb4DUpdateV1(
            result=raw,
            observation_artifact_id="a" * 64,
            linearization_artifact_id="b" * 64,
            provider_manifest_id="c" * 64,
            calibration_artifact_ids={
                "gauge_artifact_id": "d" * 64,
                "point_artifact_id": "e" * 64,
            },
            runtime_revision_source="source_checkout",
            runtime_revision_independently_verified=True,
        )


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
