"""Reliability-aware Bayesian utilities for PhysTwin-style experiments.

The historical package-root export surface is retained as a lazy compatibility
shim. New integrations should import from :mod:`bayesian_phystwin.v1`,
:mod:`bayesian_phystwin.inference.v1`, or the owning module recorded in
``api/root-export-migration-v1.json``.

The 0.4 compatibility line remains warning-free. When the installed distribution
moves to 0.5 or later, first access to a historical root export emits a targeted
:class:`DeprecationWarning` containing its exact replacement import. No root
export is removed by this policy, and removal is not scheduled before 0.6.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from ._root_exports_v0_4 import *  # noqa: F403

_ROOT_DEPRECATION_START: Final[tuple[int, int]] = (0, 5)
_ROOT_REMOVAL_NOT_BEFORE: Final[str] = "0.6"
_SOURCE_FALLBACK_VERSION: Final[str] = "0.4.0"
_WARNED_ROOT_EXPORTS: set[str] = set()


class _LazyRootExportNames:
    """Sequence-like historical export roster loaded only when inspected."""

    _names: tuple[str, ...] | None = None

    def _load(self) -> tuple[str, ...]:
        names = self._names
        if names is None:
            helper = import_module("._root_exports_v0_4", __name__)
            names = tuple(helper.__all__)
            self._names = names
        return names

    def __iter__(self):
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())

    def __getitem__(self, index: int) -> str:
        return self._load()[index]


__all__ = _LazyRootExportNames()


def _project_version() -> str:
    """Return the installed distribution version without adding a dependency."""

    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("bayesian-phystwin")
    except PackageNotFoundError:
        return _SOURCE_FALLBACK_VERSION


def _release_line(version: str) -> tuple[int, int] | None:
    """Parse the leading major/minor line of a canonical release version."""

    pieces = version.split(".", maxsplit=2)
    if len(pieces) < 2 or not pieces[0].isdigit() or not pieces[1].isdigit():
        return None
    return int(pieces[0]), int(pieces[1])


def _root_deprecations_enabled(version: str | None = None) -> bool:
    """Return whether historical root access should emit deprecation warnings."""

    release_line = _release_line(_project_version() if version is None else version)
    return release_line is not None and release_line >= _ROOT_DEPRECATION_START


def _warn_historical_root_export(name: str, module_name: str) -> None:
    if not _root_deprecations_enabled() or name in _WARNED_ROOT_EXPORTS:
        return
    warnings.warn(
        (
            f"bayesian_phystwin.{name} is a historical package-root export "
            "deprecated from BayesianPhysTwin 0.5; use "
            f"`from bayesian_phystwin.{module_name} import {name}` instead. "
            "The compatibility root will not be removed before "
            f"{_ROOT_REMOVAL_NOT_BEFORE}."
        ),
        DeprecationWarning,
        stacklevel=3,
    )
    _WARNED_ROOT_EXPORTS.add(name)


def __getattr__(name: str) -> Any:
    """Resolve one historical package-root export on first use."""

    helper = import_module("._root_exports_v0_4", __name__)
    module_name = helper._ROOT_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    _warn_historical_root_export(name, module_name)
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose the complete compatibility surface without importing owners."""

    return sorted(set(globals()) | set(__all__))
