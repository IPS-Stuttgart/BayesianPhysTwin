"""Lazy dispatch for grouped and legacy-compatible commands."""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable
from typing import Any, Final

from .command_registry import CommandSpec

_OPTIONAL_IMPORTS: Final[dict[str, frozenset[str]]] = {
    "data": frozenset({"remotezip"}),
    "graph": frozenset({"scipy"}),
    "pyrecest": frozenset({"pyrecest"}),
    "vision": frozenset({"cv2"}),
}


def _is_declared_optional_import(
    command: CommandSpec, error: ModuleNotFoundError
) -> bool:
    missing_root = (error.name or "").partition(".")[0]
    declared_modules = {
        module
        for extra in command.optional_dependencies
        for module in _OPTIONAL_IMPORTS.get(extra, ())
    }
    return missing_root in declared_modules


def _accepts_argv(function: Callable[..., Any]) -> tuple[bool, bool]:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False, False
    positional = any(
        parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        }
        for parameter in parameters
    )
    keyword = any(
        parameter.name == "argv" and parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters
    )
    return positional, keyword


def invoke(command: CommandSpec, arguments: list[str]) -> int:
    """Import one target on demand and forward only its command arguments."""

    try:
        module = importlib.import_module(command.module)
    except ModuleNotFoundError as exc:
        if not _is_declared_optional_import(command, exc):
            raise
        extras = ",".join(command.optional_dependencies)
        print(
            f"{command.command_id} requires optional dependencies; "
            f"install bayesian-phystwin[{extras}] ({exc})",
            file=sys.stderr,
        )
        return 1

    function = getattr(module, command.function)
    positional, keyword = _accepts_argv(function)
    if positional:
        result = function(arguments)
    elif keyword:
        result = function(argv=arguments)
    else:
        previous_argv = sys.argv
        sys.argv = [command.grouped_command, *arguments]
        try:
            result = function()
        finally:
            sys.argv = previous_argv
    return 0 if result is None else int(result)
