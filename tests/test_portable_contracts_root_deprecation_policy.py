from __future__ import annotations

import importlib
import importlib.metadata
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

import bayesian_phystwin as bpt

_SYMBOL = "RobustLikelihoodConfig"


def _reset_root_symbol() -> None:
    bpt.__dict__.pop(_SYMBOL, None)
    bpt._WARNED_ROOT_EXPORTS.discard(_SYMBOL)


def test_lazy_root_rosters_support_complete_sequence_and_mapping_contracts() -> None:
    modules = bpt._ROOT_EXPORT_MODULES
    names = bpt.__all__

    modules._mapping = None  # type: ignore[attr-defined]
    names._names = None  # type: ignore[attr-defined]

    assert len(modules) > 0
    assert len(tuple(iter(modules))) == len(modules)
    assert modules[_SYMBOL] == "robust_likelihood"
    assert modules[_SYMBOL] == "robust_likelihood"

    assert len(names) > 1
    first = names[0]
    first_two = names[:2]
    assert isinstance(first, str)
    assert isinstance(first_two, tuple)
    assert first_two[0] == first
    assert tuple(iter(names))[:2] == first_two


def test_project_version_falls_back_when_distribution_metadata_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing_distribution)
    assert bpt._project_version() == "0.4.0"


def test_invalid_release_line_and_first_warning_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert bpt._release_line("not-a-version") is None

    _reset_root_symbol()
    monkeypatch.setattr(bpt, "_project_version", lambda: "0.5.0")
    with pytest.warns(DeprecationWarning, match="robust_likelihood"):
        bpt._warn_historical_root_export(_SYMBOL, "robust_likelihood")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bpt._warn_historical_root_export(_SYMBOL, "robust_likelihood")
    assert caught == []
    _reset_root_symbol()


def test_private_roster_valid_and_invalid_resolution_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = importlib.import_module("bayesian_phystwin._root_exports_v0_4")
    helper.__dict__.pop(_SYMBOL, None)
    sentinel = object()
    fake_owner = SimpleNamespace(**{_SYMBOL: sentinel})

    def fake_import_module(name: str, package: str | None = None) -> Any:
        assert name == ".robust_likelihood"
        assert package == helper.__name__
        return fake_owner

    monkeypatch.setattr(helper, "import_module", fake_import_module)
    assert helper.__getattr__(_SYMBOL) is sentinel
    assert helper.__dict__[_SYMBOL] is sentinel

    with pytest.raises(AttributeError, match="has no attribute"):
        helper.__getattr__("DefinitelyNotAPublicExport")
    assert _SYMBOL in helper.__dir__()
    helper.__dict__.pop(_SYMBOL, None)
