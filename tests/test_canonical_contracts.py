from __future__ import annotations

import importlib

import numpy as np
import pytest

import bayesian_phystwin.prior_aware_gauge_belief_v2 as strict_v2
from bayesian_phystwin._canonical_contracts import (
    canonical_relative_posix_path,
    literal_lower_hex,
)

_strict_v2_cases = importlib.import_module("test_prior_aware_gauge_belief_v2")
test_strict_v2_accepts_dense_and_sparse = (
    _strict_v2_cases.test_dense_and_sparse_v2_admit_converged_positive_curvature
)
test_strict_v2_rejects_nonconvergence = (
    _strict_v2_cases.test_v2_fails_closed_when_fixed_point_does_not_converge
)
test_strict_v2_preserves_rejection = (
    _strict_v2_cases.test_v2_preserves_underlying_rejection_reason
)
test_strict_v2_rejects_ill_conditioning = (
    _strict_v2_cases.test_v2_rejects_ill_conditioned_exact_curvature
)
test_strict_v2_rejects_inconsistent_diagnostics = (
    _strict_v2_cases.test_v2_rejects_inconsistent_curvature_diagnostics
)
test_strict_v2_rejects_nonpositive_curvature = (
    _strict_v2_cases.test_v2_rejects_nonpositive_exact_curvature
)
test_strict_v2_rejects_approximate_objective = (
    _strict_v2_cases.test_v2_rejects_precision_floored_approximate_objective
)


class _StringSubclass(str):
    pass


def test_literal_lower_hex_accepts_only_literal_lowercase_strings() -> None:
    assert literal_lower_hex("a" * 40, name="revision", lengths={40, 64}) == ("a" * 40)
    assert literal_lower_hex("1" * 64, name="digest", lengths={64}) == "1" * 64

    rejected = [
        int("1" * 40),
        b"1" * 40,
        _StringSubclass("1" * 40),
        "A" * 40,
        "1" * 39,
        "1" * 40 + " ",
    ]
    for value in rejected:
        with pytest.raises(ValueError):
            literal_lower_hex(value, name="revision", lengths={40, 64})


def test_literal_lower_hex_rejects_invalid_length_contract() -> None:
    for lengths in (set(), {0}, {-1}, {True}):
        with pytest.raises(ValueError, match="positive integers"):
            literal_lower_hex("a", name="digest", lengths=lengths)


def test_canonical_relative_posix_path_is_portable_and_non_normalizing() -> None:
    value = "raw/object-1/tactile.npy"
    assert canonical_relative_posix_path(value, name="artifact path") == value

    rejected = [
        "",
        b"raw/object",
        "/absolute/path",
        "//server/share",
        "C:/windows/path",
        "raw\\windows",
        "raw/../escape",
        "./raw/object",
        "raw/./object",
        "raw//object",
        "raw/object/",
        "raw/\x00object",
    ]
    for value in rejected:
        with pytest.raises(ValueError, match="POSIX|literal"):
            canonical_relative_posix_path(value, name="artifact path")


def test_strict_v2_rejects_nonfinite_diagnostic_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert strict_v2._finite_diagnostic({"value": np.inf}, "value") is None

    dense_batch, _, _ = _strict_v2_cases._batches()
    underlying = _strict_v2_cases._synthetic_result(
        dense_batch,
        gauge_count=1,
        diagnostics=_strict_v2_cases._diagnostics(
            minimum_eigenvalue=1.0e-308,
            maximum_eigenvalue=1.0e308,
        ),
    )
    monkeypatch.setattr(
        strict_v2,
        "update_prior_aware_gauge_belief",
        lambda *_args, **_kwargs: underlying,
    )

    with np.errstate(over="ignore"):
        result = strict_v2.update_prior_aware_gauge_belief_v2(
            dense_batch,
            config=_strict_v2_cases._config(),
        )

    assert not result.inference_admissible
    assert result.reason == "strict-v2-invalid-admission-diagnostics"


def test_strict_v2_rejects_invalid_dense_argument_types() -> None:
    dense_batch, _, _ = _strict_v2_cases._batches()

    with pytest.raises(TypeError, match="batch"):
        strict_v2.update_prior_aware_gauge_belief_v2(object())
    with pytest.raises(TypeError, match="config"):
        strict_v2.update_prior_aware_gauge_belief_v2(
            dense_batch,
            config=object(),
        )
    with pytest.raises(TypeError, match="admission_config"):
        strict_v2.update_prior_aware_gauge_belief_v2(
            dense_batch,
            admission_config=object(),
        )


def test_strict_v2_rejects_invalid_sparse_argument_types() -> None:
    _, sparse_batch, sparse_design = _strict_v2_cases._batches()

    with pytest.raises(TypeError, match="batch"):
        strict_v2.update_sparse_prior_aware_gauge_belief_v2(
            object(),
            sparse_design,
        )
    with pytest.raises(TypeError, match="gauge"):
        strict_v2.update_sparse_prior_aware_gauge_belief_v2(
            sparse_batch,
            object(),
        )
    with pytest.raises(TypeError, match="config"):
        strict_v2.update_sparse_prior_aware_gauge_belief_v2(
            sparse_batch,
            sparse_design,
            config=object(),
        )
    with pytest.raises(TypeError, match="admission_config"):
        strict_v2.update_sparse_prior_aware_gauge_belief_v2(
            sparse_batch,
            sparse_design,
            admission_config=object(),
        )
