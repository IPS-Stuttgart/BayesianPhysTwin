"""Canonical metadata for the grouped ``bpt`` command interface."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Final

from ._command_inventory import (
    STABLE_ROUTES,
    description,
    optional_dependencies,
    owner,
    status_name,
)
from ._entry_points import load_frozen_legacy_entry_points


class CommandStatus(str, Enum):
    """Lifecycle classification for a registered command."""

    STABLE = "stable"
    EXPERIMENT = "experiment"
    DIAGNOSTIC = "diagnostic"
    ARCHIVED = "archived"


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One lazily imported command and its compatibility metadata."""

    command_id: str
    route: tuple[str, ...]
    module: str
    function: str
    description: str
    legacy_alias: str | None
    status: CommandStatus
    optional_dependencies: tuple[str, ...]
    owner: str

    @property
    def target(self) -> str:
        return f"{self.module}:{self.function}"

    @property
    def grouped_command(self) -> str:
        return "bpt " + " ".join(self.route)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible registry metadata."""

        payload = asdict(self)
        payload["route"] = list(self.route)
        payload["status"] = self.status.value
        payload["optional_dependencies"] = list(self.optional_dependencies)
        payload["target"] = self.target
        payload["grouped_command"] = self.grouped_command
        return payload


_STATUS_NAMESPACE: Final = {
    CommandStatus.EXPERIMENT: "experiment",
    CommandStatus.DIAGNOSTIC: "diagnostic",
    CommandStatus.ARCHIVED: "archive",
}


def _build_legacy_commands() -> tuple[CommandSpec, ...]:
    commands: list[CommandSpec] = []
    for alias, target in sorted(load_frozen_legacy_entry_points().items()):
        command_id = alias.removeprefix("bpt-")
        module, separator, function = target.partition(":")
        if not separator or not module or not function:
            raise RuntimeError(f"invalid command target for {alias}: {target}")
        status = CommandStatus(status_name(command_id))
        route = (
            STABLE_ROUTES[command_id]
            if status is CommandStatus.STABLE
            else (_STATUS_NAMESPACE[status], "run", command_id)
        )
        commands.append(
            CommandSpec(
                command_id=command_id,
                route=route,
                module=module,
                function=function,
                description=description(command_id),
                legacy_alias=alias,
                status=status,
                optional_dependencies=optional_dependencies(command_id),
                owner=owner(command_id),
            )
        )
    return tuple(commands)


# New commands are registered here without adding another console script.
_GROUPED_ONLY_COMMANDS: Final[tuple[CommandSpec, ...]] = (
    CommandSpec(
        command_id="decisive-evidence",
        route=("evidence", "summarize"),
        module="bayesian_phystwin.cli.decisive_evidence",
        function="main",
        description="summarize matched guarded prospective evidence",
        legacy_alias=None,
        status=CommandStatus.STABLE,
        optional_dependencies=(),
        owner="bayesian-phystwin-decisive-evidence-v1",
    ),
)
COMMANDS: Final[tuple[CommandSpec, ...]] = (
    *_build_legacy_commands(),
    *_GROUPED_ONLY_COMMANDS,
)


def validate_registry(commands: Iterable[CommandSpec] = COMMANDS) -> None:
    """Reject ambiguous or malformed registry entries."""

    command_ids: set[str] = set()
    routes: set[tuple[str, ...]] = set()
    aliases: set[str] = set()
    for command in commands:
        if not command.command_id or command.command_id.startswith("-"):
            raise ValueError(f"invalid command id: {command.command_id!r}")
        if command.command_id in command_ids:
            raise ValueError(f"duplicate command id: {command.command_id}")
        if command.route in routes:
            raise ValueError(f"duplicate grouped route: {' '.join(command.route)}")
        if not command.owner:
            raise ValueError(f"missing owner for {command.command_id}")
        if command.legacy_alias is not None:
            if command.legacy_alias in aliases:
                raise ValueError(f"duplicate legacy alias: {command.legacy_alias}")
            aliases.add(command.legacy_alias)
        command_ids.add(command.command_id)
        routes.add(command.route)


validate_registry()
COMMANDS_BY_ID: Final = {command.command_id: command for command in COMMANDS}
COMMANDS_BY_ROUTE: Final = {command.route: command for command in COMMANDS}
COMMANDS_BY_ALIAS: Final = {
    command.legacy_alias: command
    for command in COMMANDS
    if command.legacy_alias is not None
}


def iter_commands(
    *, status: CommandStatus | None = None
) -> tuple[CommandSpec, ...]:
    """Return registry entries in deterministic command-id order."""

    selected = (
        command for command in COMMANDS if status is None or command.status is status
    )
    return tuple(sorted(selected, key=lambda command: command.command_id))


def find_command(
    selector: str, *, status: CommandStatus | None = None
) -> CommandSpec | None:
    """Resolve a command id, legacy alias, or grouped route spelling."""

    normalized = selector.strip()
    command = COMMANDS_BY_ID.get(normalized) or COMMANDS_BY_ALIAS.get(normalized)
    if command is None:
        route = tuple(normalized.removeprefix("bpt ").split())
        command = COMMANDS_BY_ROUTE.get(route)
    if command is not None and (status is None or command.status is status):
        return command
    return None


def legacy_entry_points() -> dict[str, str]:
    """Return the frozen ``bpt-*`` compatibility entry-point mapping."""

    return {
        command.legacy_alias: command.target
        for command in COMMANDS
        if command.legacy_alias is not None
    }
