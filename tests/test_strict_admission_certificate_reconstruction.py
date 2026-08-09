from __future__ import annotations

from dataclasses import replace

import pytest

from bayesian_phystwin.prior_aware_gauge_belief_v2 import (
    update_sparse_prior_aware_gauge_belief_v2,
)
from tests.test_claim_bearing_strict_admission import (
    _certificate,
    _converged_config,
    _tree_fixture,
)


def _accepted_result():
    batch, tree = _tree_fixture()
    result = update_sparse_prior_aware_gauge_belief_v2(
        batch,
        tree,
        config=_converged_config(),
    )
    assert result.inference_admissible
    return result


def _replace_certificate(result, certificate: dict[str, object]):
    diagnostics = dict(result.diagnostics)
    diagnostics["strict_admission_certificate"] = certificate
    return replace(result, diagnostics=diagnostics)


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        (lambda certificate: certificate.pop("mixture_stationarity_norm"), "fields"),
        (lambda certificate: certificate.__setitem__("unexpected", True), "fields"),
        (
            lambda certificate: certificate.__setitem__("schema_version", True),
            "schema_version",
        ),
    ],
)
def test_certificate_requires_a_closed_literal_field_contract(
    mutation,
    expected_fragment: str,
) -> None:
    result = _accepted_result()
    certificate = _certificate(result)
    mutation(certificate)

    with pytest.raises(ValueError, match=expected_fragment):
        _replace_certificate(result, certificate)


@pytest.mark.parametrize(
    "updates",
    [
        {"mixture_solution_delta": None},
        {"mixture_stationarity_norm": None},
        {"exact_hessian_minimum_eigenvalue": None},
        {"exact_hessian_maximum_eigenvalue": None},
        {"exact_hessian_minimum_eigenvalue": 0.0},
        {"exact_hessian_condition_number": None},
        {"exact_hessian_condition_number": 1.0e30},
        {"maximum_exact_hessian_condition_number": 0.5},
    ],
)
def test_passing_certificate_requires_consistent_numeric_evidence(
    updates: dict[str, object],
) -> None:
    result = _accepted_result()
    certificate = _certificate(result)
    certificate.update(updates)

    with pytest.raises(ValueError):
        _replace_certificate(result, certificate)


def test_certificate_condition_number_is_derived_from_hessian_spectrum() -> None:
    result = _accepted_result()
    certificate = _certificate(result)
    minimum = certificate["exact_hessian_minimum_eigenvalue"]
    maximum = certificate["exact_hessian_maximum_eigenvalue"]
    assert isinstance(minimum, float)
    assert isinstance(maximum, float)
    certificate["exact_hessian_condition_number"] = maximum / minimum + 1.0

    with pytest.raises(ValueError, match="condition"):
        _replace_certificate(result, certificate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("underlying_inference_reason", "forged-underlying-reason"),
        ("strict_exact_hessian_condition_number", 123456.0),
        ("maximum_exact_hessian_condition_number", 2.0),
    ],
)
def test_top_level_admission_diagnostics_must_match_certificate(
    field: str,
    value: object,
) -> None:
    result = _accepted_result()
    diagnostics = dict(result.diagnostics)
    diagnostics[field] = value

    with pytest.raises(ValueError):
        replace(result, diagnostics=diagnostics)


def test_result_reason_must_match_the_selected_certificate_path() -> None:
    result = _accepted_result()

    with pytest.raises(ValueError, match="reason"):
        replace(result, reason="forged-result-reason")
