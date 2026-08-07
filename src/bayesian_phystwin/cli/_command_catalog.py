"""Listing, inspection, migration, and lifecycle runners for commands."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Final

from ._command_dispatch import invoke
from .command_registry import (
    CommandSpec,
    CommandStatus,
    find_command_metadata,
    find_runnable_command,
    iter_commands,
)

HELP_FLAGS: Final = frozenset({"-h", "--help"})
CATALOGS: Final[dict[str, CommandStatus]] = {
    "experiment": CommandStatus.EXPERIMENT,
    "diagnostic": CommandStatus.DIAGNOSTIC,
    "archive": CommandStatus.ARCHIVED,
}
NAMESPACE_DESCRIPTIONS: Final[dict[str, str]] = {
    "commands": "inspect the canonical command registry",
    "experiment": "run current research protocols",
    "diagnostic": "run non-promotable audits and analyses",
    "archive": "run frozen historical commands",
}


def registry_help() -> str:
    return (
        "usage: bpt commands <list|describe|migrate> [arguments]\n\n"
        "commands:\n"
        "  list                    list entries [--status STATUS] [--json]\n"
        "  describe SELECTOR       show metadata by id, route, or legacy alias\n"
        "  migrate SELECTOR        print the current grouped invocation\n\n"
        "statuses: stable, experiment, diagnostic, archived\n"
    )


def catalog_help(namespace: str) -> str:
    status = CATALOGS[namespace].value
    return (
        f"usage: bpt {namespace} <list|describe|run> [arguments]\n\n"
        f"Manage commands classified as {status}.\n\n"
        "commands:\n"
        "  list              list command ids [--json]\n"
        "  describe ID       show command metadata [--json]\n"
        "  run ID ...        run a command with the remaining arguments\n"
    )


def _json_flag(arguments: Sequence[str]) -> tuple[list[str], bool] | None:
    remaining: list[str] = []
    json_output = False
    for argument in arguments:
        if argument == "--json":
            json_output = True
        elif argument.startswith("-"):
            return None
        else:
            remaining.append(argument)
    return remaining, json_output


def _print_list(commands: Sequence[CommandSpec], *, json_output: bool) -> None:
    if json_output:
        print(
            json.dumps(
                [item.to_dict() for item in commands],
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not commands:
        print("No commands are registered for this selection.")
        return
    mixed = len({item.status for item in commands}) > 1
    id_width = max(len("ID"), *(len(item.command_id) for item in commands))
    owner_width = max(len("OWNER"), *(len(item.owner) for item in commands))
    prefix = "STATUS      " if mixed else ""
    print(f"{prefix}{'ID':<{id_width}}  {'OWNER':<{owner_width}}  EXTRAS")
    for item in commands:
        status = f"{item.status.value:<10}  " if mixed else ""
        extras = ",".join(item.optional_dependencies) or "-"
        print(
            f"{status}{item.command_id:<{id_width}}  "
            f"{item.owner:<{owner_width}}  {extras}"
        )


def _print_command(command: CommandSpec, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(command.to_dict(), indent=2, sort_keys=True))
        return
    print(f"id: {command.command_id}")
    print(f"status: {command.status.value}")
    print(f"command: {command.canonical_command}")
    print(f"legacy alias: {command.legacy_alias or 'none'}")
    previous = ", ".join(command.previous_grouped_commands) or "none"
    print(f"previous grouped commands: {previous}")
    print(f"owner: {command.owner}")
    print(f"documentation: {command.documentation or 'none'}")
    print(
        "optional dependencies: " + (", ".join(command.optional_dependencies) or "none")
    )
    print(f"target: {command.target}")
    print(f"description: {command.description}")


def _migration_source(command: CommandSpec, selector: str) -> str | None:
    if command.legacy_alias == selector:
        return "legacy_alias"
    route = tuple(selector.removeprefix("bpt ").split())
    if route in command.previous_routes:
        return "previous_grouped_route"
    return None


def commands_main(arguments: Sequence[str]) -> int:
    if not arguments or arguments[0] in HELP_FLAGS:
        print(registry_help(), end="")
        return 0
    action = arguments[0]
    if action == "list":
        status: CommandStatus | None = None
        json_output = False
        index = 1
        while index < len(arguments):
            argument = arguments[index]
            if argument == "--json":
                json_output = True
                index += 1
            elif argument == "--status" and index + 1 < len(arguments):
                try:
                    status = CommandStatus(arguments[index + 1])
                except ValueError:
                    print("invalid command status", file=sys.stderr)
                    return 2
                index += 2
            else:
                print(registry_help(), file=sys.stderr, end="")
                return 2
        _print_list(iter_commands(status=status), json_output=json_output)
        return 0
    if action == "describe":
        parsed = _json_flag(arguments[1:])
        if parsed is None or not parsed[0]:
            print(registry_help(), file=sys.stderr, end="")
            return 2
        selectors, json_output = parsed
        command = find_command_metadata(" ".join(selectors))
        if command is None:
            print(f"unknown command: {' '.join(selectors)}", file=sys.stderr)
            return 2
        _print_command(command, json_output=json_output)
        return 0
    if action == "migrate":
        parsed = _json_flag(arguments[1:])
        if parsed is None or not parsed[0]:
            print(registry_help(), file=sys.stderr, end="")
            return 2
        selector = " ".join(parsed[0])
        command = find_command_metadata(selector)
        source_kind = None if command is None else _migration_source(command, selector)
        if command is None or source_kind is None:
            print(f"unknown previous command selector: {selector}", file=sys.stderr)
            return 2
        if parsed[1]:
            print(
                json.dumps(
                    {
                        "source_selector": selector,
                        "source_kind": source_kind,
                        "canonical_command": command.canonical_command,
                        "command_id": command.command_id,
                        "status": command.status.value,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(command.canonical_command)
        return 0
    print(registry_help(), file=sys.stderr, end="")
    return 2


def catalog_main(namespace: str, arguments: Sequence[str]) -> int:
    status = CATALOGS[namespace]
    if not arguments or arguments[0] in HELP_FLAGS:
        print(catalog_help(namespace), end="")
        return 0
    action = arguments[0]
    if action == "list":
        parsed = _json_flag(arguments[1:])
        if parsed is None or parsed[0]:
            print(catalog_help(namespace), file=sys.stderr, end="")
            return 2
        _print_list(iter_commands(status=status), json_output=parsed[1])
        return 0
    if action == "describe":
        parsed = _json_flag(arguments[1:])
        if parsed is None or len(parsed[0]) != 1:
            print(catalog_help(namespace), file=sys.stderr, end="")
            return 2
        command = find_runnable_command(parsed[0][0], status=status)
        if command is None:
            print(
                f"unknown {status.value} command: {parsed[0][0]}",
                file=sys.stderr,
            )
            return 2
        _print_command(command, json_output=parsed[1])
        return 0
    if action == "run" and len(arguments) >= 2:
        command = find_runnable_command(arguments[1], status=status)
        if command is None:
            print(
                f"unknown {status.value} command: {arguments[1]}",
                file=sys.stderr,
            )
            return 2
        return invoke(command, arguments[2:])
    print(catalog_help(namespace), file=sys.stderr, end="")
    return 2
