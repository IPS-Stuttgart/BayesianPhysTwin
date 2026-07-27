"""Lazy dispatch for grouped Bayesian-PhysTwin commands."""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

from .command_registry import CommandSpec

_OPTIONAL_IMPORTS: Final[dict[str, frozenset[str]]] = {
    "data": frozenset({"remotezip"}),
    "graph": frozenset({"scipy"}),
    "pyrecest": frozenset({"pyrecest"}),
    "vision": frozenset({"cv2"}),
}
InvocationMode = Literal["positional", "keyword", "sys_argv"]


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


def _invocation_mode(function: Callable[..., Any]) -> InvocationMode:
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError) as exc:
        raise TypeError("cannot inspect registered command target") from exc
    if not parameters:
        return "sys_argv"
    if len(parameters) != 1:
        raise TypeError("registered command target must accept zero or one parameter")
    parameter = parameters[0]
    if parameter.kind in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        return "positional"
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.name == "argv":
        return "keyword"
    raise TypeError(
        "registered command target must expose main(), main(argv), or main(*, argv=...)"
    )


def invoke(command: CommandSpec, arguments: Sequence[str]) -> int:
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
    if not callable(function):
        raise TypeError(f"registered command target is not callable: {command.target}")

    mode = _invocation_mode(function)
    if mode == "positional":
        result = function(list(arguments))
    elif mode == "keyword":
        result = function(argv=list(arguments))
    else:
        previous_argv = sys.argv
        sys.argv = [command.canonical_command, *arguments]
        try:
            result = function()
        finally:
            sys.argv = previous_argv
    return 0 if result is None else int(result)
