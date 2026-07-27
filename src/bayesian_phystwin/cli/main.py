"""Grouped, lazily imported command surface for Bayesian-PhysTwin."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Final

from ._command_catalog import (
    CATALOGS,
    HELP_FLAGS,
    NAMESPACE_DESCRIPTIONS,
    catalog_main,
    commands_main,
)
from ._command_dispatch import invoke
from .command_registry import CommandSpec, CommandStatus, iter_commands

Route = tuple[str, ...]
_STABLE_COMMANDS: Final = iter_commands(status=CommandStatus.STABLE)
_STABLE_ROUTES: Final[dict[Route, CommandSpec]] = {
    command.route: command for command in _STABLE_COMMANDS
}


def _children(prefix: Route) -> list[str]:
    position = len(prefix)
    return sorted(
        {
            route[position]
            for route in _STABLE_ROUTES
            if route[:position] == prefix and len(route) > position
        }
    )


def _root_help() -> str:
    stable = {route[0] for route in _STABLE_ROUTES}
    children = sorted(stable | set(NAMESPACE_DESCRIPTIONS))
    lines = [
        "usage: bpt <command> [arguments]",
        "",
        "Grouped access to stable interfaces and registered research commands.",
        "",
        "commands:",
    ]
    for child in children:
        description = NAMESPACE_DESCRIPTIONS.get(child, f"{child} commands")
        lines.append(f"  {child:<14} {description}")
    lines.extend(
        [
            "",
            "Exactly one executable is installed: bpt.",
            "Use `bpt commands migrate LEGACY_ALIAS` for removed bpt-* names.",
        ]
    )
    return "\n".join(lines) + "\n"


def _stable_help(prefix: Route) -> str:
    lines = [
        f"usage: bpt {' '.join(prefix)} <command> [arguments]",
        "",
        "commands:",
    ]
    for child in _children(prefix):
        route = (*prefix, child)
        command = _STABLE_ROUTES.get(route)
        description = (
            command.description if command is not None else f"{child} commands"
        )
        lines.append(f"  {child:<14} {description}")
    return "\n".join(lines) + "\n"


def _resolve(arguments: Sequence[str]) -> tuple[CommandSpec, list[str]] | None:
    for length in range(1, len(arguments) + 1):
        route = tuple(arguments[:length])
        command = _STABLE_ROUTES.get(route)
        if command is not None:
            return command, list(arguments[length:])
        if not _children(route):
            return None
    return None


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in HELP_FLAGS:
        print(_root_help(), end="")
        return 0

    namespace = arguments[0]
    if namespace == "commands":
        return int(commands_main(arguments[1:]))
    if namespace in CATALOGS:
        return int(catalog_main(namespace, arguments[1:]))

    if arguments[-1] in HELP_FLAGS and _children(tuple(arguments[:-1])):
        print(_stable_help(tuple(arguments[:-1])), end="")
        return 0
    if _children(tuple(arguments)):
        print(_stable_help(tuple(arguments)), end="")
        return 0

    resolved = _resolve(arguments)
    if resolved is not None:
        return int(invoke(*resolved))

    prefix: list[str] = []
    for token in arguments:
        candidate = (*prefix, token)
        if not _children(candidate):
            break
        prefix.append(token)
    help_text = _stable_help(tuple(prefix)) if prefix else _root_help()
    print(help_text, file=sys.stderr, end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
