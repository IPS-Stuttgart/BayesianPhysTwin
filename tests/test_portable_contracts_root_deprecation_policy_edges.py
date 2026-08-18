from __future__ import annotations

import importlib.metadata
import warnings

import pytest

import bayesian_phystwin as bpt
import bayesian_phystwin._root_exports_v0_4 as root_exports
import bayesian_phystwin.robust_likelihood as robust_likelihood

_SYMBOL = "RobustLikelihoodConfig"


def _reset_symbol() -> None:
    bpt.__dict__.pop(_SYMBOL, None)
    bpt._WARNED_ROOT_EXPORTS.discard(_SYMBOL)
    root_exports.__dict__.pop(_SYMBOL, None)


def test_lazy_root_collections_cover_mapping_and_sequence_protocols() -> None:
    assert len(bpt._ROOT_EXPORT_MODULES) == len(tuple(bpt._ROOT_EXPORT_MODULES))
    assert bpt._ROOT_EXPORT_MODULES[_SYMBOL] == "robust_likelihood"
    assert len(bpt.__all__) > 0
    assert bpt.__all__[0] in bpt._ROOT_EXPORT_MODULES
    assert tuple(bpt.__all__[:2]) == tuple(bpt.__all__)[:2]


def test_project_version_falls_back_when_distribution_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(name: str) -> str:
        del name
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing_distribution)
    assert bpt._project_version() == "0.4.0"


def test_release_line_rejects_incomplete_and_nonnumeric_versions() -> None:
    assert bpt._release_line("1") is None
    assert bpt._release_line("1.x.0") is None
    assert bpt._release_line("x.1.0") is None
    assert bpt._release_line("1.2.3") == (1, 2)


def test_default_version_gate_and_duplicate_warning_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_symbol()
    monkeypatch.setattr(bpt, "_project_version", lambda: "0.5.0")
    with pytest.warns(DeprecationWarning):
        bpt._warn_historical_root_export(_SYMBOL, "robust_likelihood")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bpt._warn_historical_root_export(_SYMBOL, "robust_likelihood")
    assert caught == []
    _reset_symbol()


def test_root_getattr_unknown_attribute_and_dir_contract() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        bpt.__getattr__("DefinitelyNotAPublicExport")
    assert _SYMBOL in bpt.__dir__()


def test_generated_root_export_helper_resolves_and_caches_owner_identity() -> None:
    _reset_symbol()
    value = root_exports.__getattr__(_SYMBOL)
    assert value is robust_likelihood.RobustLikelihoodConfig
    assert root_exports.__dict__[_SYMBOL] is value
    assert _SYMBOL in root_exports.__dir__()
    _reset_symbol()


def test_generated_root_export_helper_rejects_unknown_attribute() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        root_exports.__getattr__("DefinitelyNotAPublicExport")
