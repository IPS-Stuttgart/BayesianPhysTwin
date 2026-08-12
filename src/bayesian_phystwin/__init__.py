"""Reliability-aware Bayesian utilities for PhysTwin-style experiments.

The historical package-root API remains available for compatibility, but each
recorded export is resolved from its defining module only when first accessed.
New ecosystem integrations should prefer :mod:`bayesian_phystwin.v1`.
"""

from __future__ import annotations

import ast as _ast
from importlib import import_module as _import_module
from importlib.resources import files as _package_files
from types import MappingProxyType as _MappingProxyType
from typing import Any as _Any

_PACKAGE = __package__ or __name__


def _load_legacy_api() -> tuple[list[str], _MappingProxyType[str, tuple[str, str]]]:
    source = _package_files(_PACKAGE).joinpath("_legacy_root_eager.py").read_text(
        encoding="utf-8"
    )
    tree = _ast.parse(source, filename="_legacy_root_eager.py")
    exports: dict[str, tuple[str, str]] = {}
    ordered_names: list[str] | None = None
    for node in tree.body:
        if isinstance(node, _ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                public_name = alias.asname or alias.name
                exports[public_name] = (node.module, alias.name)
            continue
        if not isinstance(node, _ast.Assign):
            continue
        if not any(
            isinstance(target, _ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        value = _ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(type(name) is str for name in value):
            raise RuntimeError("historical package-root __all__ is not a string list")
        ordered_names = value

    if ordered_names is None:  # pragma: no cover - package corruption
        raise RuntimeError("historical package-root __all__ is missing")
    if len(ordered_names) != len(set(ordered_names)):
        raise RuntimeError("historical package-root __all__ contains duplicates")
    public_exports = {name: exports[name] for name in ordered_names if name in exports}
    if set(public_exports) != set(ordered_names):  # pragma: no cover - corruption
        missing = sorted(set(ordered_names) - set(public_exports))
        raise RuntimeError(
            f"historical package-root export registry drift: {missing!r}"
        )
    return ordered_names, _MappingProxyType(public_exports)


__all__, _LEGACY_EXPORTS = _load_legacy_api()


def __getattr__(name: str) -> _Any:
    """Resolve and cache one historical package-root export on first access."""

    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(_import_module(f".{module_name}", _PACKAGE), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes plus the frozen compatibility surface."""

    return sorted(set(globals()) | set(__all__))
