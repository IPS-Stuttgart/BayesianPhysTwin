"""Registry-backed grouped command surface for Bayesian-PhysTwin."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Iterable, Sequence
from typing import Final, cast

from .command_registry import (
    ROUTES,
    STATUS_ORDER,
    VISIBLE_STATUSES,
    CommandSpec,
    CommandStatus,
    Route,
    iter_commands,
)

_LIFECYCLE_NAMESPACES: Final[frozenset[str]] = frozenset(
    {"experiment", "diagnostic", "archived"}
)


def _children(prefix: Route, *, include_hidden: bool = False) -> list[str]:
    position = len(prefix)
    children: set[str] = set()
    for route, command in ROUTES.items():
        if route[:position] != prefix or len(route) <= position:
            continue
        if include_hidden or command.status in VISIBLE_STATUSES:
            children.add(route[position])
    return sorted(children)


def _namespace_help(prefix: Route) -> str | None:
    if len(prefix) == 1 and prefix[0] in _LIFECYCLE_NAMESPACES:
        namespace = prefix[0]
        return "\n".join(
            [
                f"usage: bpt {namespace} <list|run> [arguments]",
                "",
                "commands:",
                "  list           list registered command IDs",
                "  run <id>       dispatch a registered command",
                "",
            ]
        )
    if (
        len(prefix) == 2
        and prefix[0] in _LIFECYCLE_NAMESPACES
        and prefix[1] == "run"
    ):
        namespace = prefix[0]
        return "\n".join(
            [
                f"usage: bpt {namespace} run <id> [arguments]",
                "",
                f"Use `bpt {namespace} list` to inspect registered IDs.",
                "",
            ]
        )
    return None


def _render_help(prefix: Route = ()) -> str:
    namespace_help = _namespace_help(prefix)
    if namespace_help is not None:
        return namespace_help

    command = "bpt" + (" " + " ".join(prefix) if prefix else "")
    include_hidden = bool(prefix and prefix[0] in _LIFECYCLE_NAMESPACES)
    children = _children(prefix, include_hidden=include_hidden)
    lines = [f"usage: {command} <command> [arguments]", ""]
    if not prefix:
        lines.extend(
            [
                "Grouped access to stable and current Bayesian-PhysTwin commands.",
                "",
            ]
        )
    lines.append("commands:")
    width = max((len(child) for child in children), default=0)
    for child in children:
        candidate = (*prefix, child)
        registered = ROUTES.get(candidate)
        description = (
            registered.description
            if registered is not None
            else f"{child} commands"
        )
        lines.append(f"  {child:<{width}}  {description}")
    lines.extend(
        [
            "",
            "Use `bpt commands` for lifecycle, dependency, and alias metadata.",
            "Frozen bpt-* aliases remain available only for compatibility.",
        ]
    )
    return "\n".join(lines) + "\n"


def _resolve(tokens: Sequence[str]) -> tuple[CommandSpec, list[str]] | None:
    prefix: list[str] = []
    for index, token in enumerate(tokens):
        if token.startswith("-"):
            break
        prefix.append(token)
        command = ROUTES.get(tuple(prefix))
        if command is not None:
            return command, list(tokens[index + 1 :])
        if not _children(tuple(prefix), include_hidden=True):
            return None
    return None


def _status_selection(
    requested: Sequence[CommandStatus] | None,
    *,
    include_all: bool,
) -> frozenset[CommandStatus]:
    if requested:
        return frozenset(requested)
    if include_all:
        return frozenset(STATUS_ORDER)
    return VISIBLE_STATUSES


def _command_payload(command: CommandSpec) -> dict[str, object]:
    return {
        "id": command.command_id,
        "route": list(command.route),
        "status": command.status,
        "legacy_alias": command.legacy_alias,
        "target": command.target,
        "extras": list(command.extras),
        "milestone": command.milestone,
        "description": command.description,
    }


def _render_command_table(commands: Iterable[CommandSpec]) -> str:
    lines: list[str] = []
    for command in commands:
        alias = command.legacy_alias or "-"
        extras = ",".join(command.extras) or "-"
        lines.extend(
            [
                f"{command.status:<10} {command.command_id}",
                f"  route={' '.join(command.route)}  alias={alias}",
                f"  extras={extras}  milestone={command.milestone}",
                f"  {command.description}",
            ]
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _commands_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bpt commands",
        description="List grouped commands and compatibility metadata.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="include diagnostic and archived commands",
    )
    parser.add_argument(
        "--status",
        action="append",
        choices=STATUS_ORDER,
        help="include one lifecycle status; repeat to select several",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format",
    )
    namespace = parser.parse_args(argv)
    statuses = _status_selection(namespace.status, include_all=namespace.all)
    commands = list(iter_commands(statuses=statuses))
    if namespace.format == "json":
        print(
            json.dumps(
                [_command_payload(command) for command in commands],
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_render_command_table(commands), end="")
    return 0


def _namespace_list_main(
    status: CommandStatus,
    argv: Sequence[str],
) -> int:
    parser = argparse.ArgumentParser(prog=f"bpt {status} list")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format",
    )
    namespace = parser.parse_args(list(argv))
    commands = [
        command
        for command in iter_commands(statuses={status})
        if command.route[:2] == (status, "run")
    ]
    if namespace.format == "json":
        print(
            json.dumps(
                [_command_payload(command) for command in commands],
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for command in commands:
            extras = f" extras={','.join(command.extras)}" if command.extras else ""
            print(f"{command.command_id}{extras}  {command.description}")
    return 0


def _is_help_namespace(prefix: Route) -> bool:
    return bool(
        prefix == ()
        or _children(prefix, include_hidden=True)
        or _namespace_help(prefix) is not None
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_render_help(), end="")
        return 0

    if (
        arguments[0] in _LIFECYCLE_NAMESPACES
        and len(arguments) >= 2
        and arguments[1] == "list"
    ):
        return _namespace_list_main(
            cast(CommandStatus, arguments[0]),
            arguments[2:],
        )

    for length in range(1, len(arguments) + 1):
        prefix = tuple(arguments[:length])
        if arguments[length - 1] in {"-h", "--help"}:
            help_namespace = prefix[:-1]
            if _is_help_namespace(help_namespace):
                print(_render_help(help_namespace), end="")
                return 0
            break

    resolved = _resolve(arguments)
    if resolved is None:
        matched_namespace: list[str] = []
        for token in arguments:
            candidate = (*matched_namespace, token)
            if _is_help_namespace(candidate):
                matched_namespace.append(token)
            else:
                break
        matched = tuple(matched_namespace)
        if matched and len(matched_namespace) == len(arguments):
            print(_render_help(matched), end="")
            return 0
        print(_render_help(matched), file=sys.stderr, end="")
        return 2

    command, remaining = resolved
    function = getattr(
        importlib.import_module(command.module),
        command.function,
    )
    return int(function(remaining))


if __name__ == "__main__":
    raise SystemExit(main())
