from __future__ import annotations

import warnings

import pytest

import bayesian_phystwin as bpt
import bayesian_phystwin.robust_likelihood as robust_likelihood

_SYMBOL = "RobustLikelihoodConfig"


def _reset_symbol() -> None:
    bpt.__dict__.pop(_SYMBOL, None)
    bpt._WARNED_ROOT_EXPORTS.discard(_SYMBOL)


def test_root_deprecation_version_gate_is_explicit() -> None:
    assert not bpt._root_deprecations_enabled("0.4.99")
    assert bpt._root_deprecations_enabled("0.5.0")
    assert bpt._root_deprecations_enabled("0.5.0rc1")
    assert bpt._root_deprecations_enabled("1.0.0")
    assert not bpt._root_deprecations_enabled("not-a-version")


def test_current_0_4_root_access_remains_warning_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_symbol()
    monkeypatch.setattr(bpt, "_project_version", lambda: "0.4.99")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = getattr(bpt, _SYMBOL)

    assert value is robust_likelihood.RobustLikelihoodConfig
    assert caught == []
    _reset_symbol()


def test_0_5_root_access_warns_with_exact_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_symbol()
    monkeypatch.setattr(bpt, "_project_version", lambda: "0.5.0")

    message = (
        "from bayesian_phystwin.robust_likelihood import "
        "RobustLikelihoodConfig"
    )
    with pytest.warns(DeprecationWarning, match=message):
        value = getattr(bpt, _SYMBOL)

    assert value is robust_likelihood.RobustLikelihoodConfig

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert getattr(bpt, _SYMBOL) is value
    assert caught == []
    _reset_symbol()


def test_unknown_root_attribute_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bpt, "_project_version", lambda: "0.5.0")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(bpt, "DefinitelyNotAPublicExport")

    assert caught == []
