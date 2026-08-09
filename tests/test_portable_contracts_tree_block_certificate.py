from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np
import pytest

import bayesian_phystwin.tree_block_sparse_gauge_belief_v2 as strict_module
from bayesian_phystwin.tree_block_sparse_gauge_belief import (
    TreeBlockGaugeAwareBeliefResultV1,
    update_tree_block_sparse_prior_aware_gauge_belief,
)
from bayesian_phystwin.tree_block_sparse_gauge_belief_v2 import (
    TreeBlockGaugeAwareBeliefResultV2,
    update_tree_block_sparse_prior_aware_gauge_belief_v2,
)
from tests.test_portable_contracts_tree_block_strict_admission import (
    _converged_config,
    _fixture,
)


def _accepted_result() -> TreeBlockGaugeAwareBeliefResultV2:
    batch, tree = _fixture()
    result = update_tree_block_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_converged_config(),
    )
    assert result.inference_admissible
    return result


def _certificate(
    result: TreeBlockGaugeAwareBeliefResultV2,
) -> dict[str, object]:
    value = result.diagnostics["strict_admission_certificate"]
    assert isinstance(value, Mapping)
    return dict(value)


def _replace_certificate(
    result: TreeBlockGaugeAwareBeliefResultV2,
    certificate: dict[str, object],
) -> TreeBlockGaugeAwareBeliefResultV2:
    diagnostics = dict(result.diagnostics)
    diagnostics["strict_admission_certificate"] = certificate
    return replace(result, diagnostics=diagnostics)


def test_every_underlying_rejection_reconstructs_exact_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, tree = _fixture()
    historical = update_tree_block_sparse_prior_aware_gauge_belief(
        batch,
        tree,
        config=_converged_config(),
    )
    assert historical.inference_admissible

    object.__setattr__(historical, "inference_admissible", False)
    object.__setattr__(historical, "reason", "forged-underlying-rejection")
    object.__setattr__(
        historical,
        "state_coefficients",
        np.asarray([1000.0], dtype=np.float64),
    )
    object.__setattr__(
        historical,
        "gauge_delta",
        np.asarray([2000.0], dtype=np.float64),
    )
    object.__setattr__(
        historical,
        "shared_bias_coefficients",
        np.asarray([3000.0], dtype=np.float64),
    )
    monkeypatch.setattr(
        strict_module,
        "update_tree_block_sparse_prior_aware_gauge_belief",
        lambda *_args, **_kwargs: historical,
    )

    result = update_tree_block_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_converged_config(),
    )

    assert isinstance(result, TreeBlockGaugeAwareBeliefResultV2)
    assert not result.inference_admissible
    assert result.reason == "forged-underlying-rejection"
    np.testing.assert_array_equal(result.state_coefficients, 0.0)
    np.testing.assert_array_equal(result.gauge_delta, 0.0)
    np.testing.assert_array_equal(result.shared_bias_coefficients, 0.0)
    np.testing.assert_array_equal(result.view_bias_coefficients, 0.0)
    np.testing.assert_array_equal(result.anchor_bias_coefficients, 0.0)
    np.testing.assert_allclose(
        result.materialize_posterior_covariance(),
        np.diag([0.04, 0.09]),
        rtol=0.0,
        atol=1.0e-14,
    )
    certificate = _certificate(result)
    assert certificate["underlying_inference_admissible"] is False
    assert certificate["passed"] is False
    assert certificate["reason"] == "underlying-inference-rejected"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda certificate: certificate.pop("mixture_stationarity_norm"),
            "fields changed",
        ),
        (
            lambda certificate: certificate.__setitem__("unexpected", True),
            "fields changed",
        ),
        (
            lambda certificate: certificate.__setitem__("schema_version", True),
            "schema_version",
        ),
        (
            lambda certificate: certificate.__setitem__(
                "underlying_inference_reason",
                "",
            ),
            "underlying reason",
        ),
    ],
)
def test_certificate_requires_closed_literal_contract(
    mutation: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    result = _accepted_result()
    certificate = _certificate(result)
    mutation(certificate)

    with pytest.raises(ValueError, match=match):
        _replace_certificate(result, certificate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mixture_solution_delta", None),
        ("mixture_solution_delta", -1.0),
        ("mixture_stationarity_norm", None),
        ("maximum_eliminated_node_condition_number", 0.0),
        ("global_schur_condition_number", float("nan")),
        ("diagnostics_valid", False),
        ("exact_tree_block_solver", False),
        ("passed", False),
        ("reason", "strict-v2-invalid-admission-diagnostics"),
    ],
)
def test_passing_certificate_rejects_forged_numeric_or_decision_fields(
    field: str,
    value: object,
) -> None:
    result = _accepted_result()
    certificate = _certificate(result)
    certificate[field] = value

    with pytest.raises(ValueError):
        _replace_certificate(result, certificate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strict_admission_passed", False),
        ("strict_admission_reason", "forged-reason"),
        ("underlying_inference_admissible", False),
        ("underlying_inference_reason", "forged-underlying-reason"),
        ("mixture_solution_delta", 123.0),
        ("global_schur_condition_number", 123.0),
    ],
)
def test_certificate_is_bound_to_top_level_and_raw_solver_diagnostics(
    field: str,
    value: object,
) -> None:
    result = _accepted_result()
    diagnostics = dict(result.diagnostics)
    diagnostics[field] = value

    with pytest.raises(ValueError):
        replace(result, diagnostics=diagnostics)


def test_result_reason_and_admissibility_are_certificate_bound() -> None:
    result = _accepted_result()

    with pytest.raises(ValueError, match="result reason"):
        replace(result, reason="forged-result-reason")
    with pytest.raises(ValueError, match="admissibility"):
        replace(result, inference_admissible=False)


def test_accepted_and_rejected_results_use_v2_identity() -> None:
    batch, tree = _fixture()
    accepted = update_tree_block_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_converged_config(),
    )
    rejected = update_tree_block_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=replace(_converged_config(), maximum_state_update_m=1.0e-12),
    )

    assert isinstance(accepted, TreeBlockGaugeAwareBeliefResultV2)
    assert isinstance(rejected, TreeBlockGaugeAwareBeliefResultV2)
    assert isinstance(accepted, TreeBlockGaugeAwareBeliefResultV1)
    assert isinstance(rejected, TreeBlockGaugeAwareBeliefResultV1)
    assert accepted.result_id != rejected.result_id
