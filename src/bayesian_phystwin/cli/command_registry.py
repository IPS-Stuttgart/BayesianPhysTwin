"""Canonical metadata for the grouped ``bpt`` command interface."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Final

from ._command_inventory import description, optional_dependencies, owner, status_name
from .experiments import EXPERIMENTS


class CommandStatus(str, Enum):
    """Lifecycle classification for a registered command."""

    STABLE = "stable"
    EXPERIMENT = "experiment"
    DIAGNOSTIC = "diagnostic"
    ARCHIVED = "archived"


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One lazily imported command and its lifecycle metadata."""

    command_id: str
    route: tuple[str, ...]
    previous_routes: tuple[tuple[str, ...], ...]
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

    @property
    def canonical_command(self) -> str:
        return self.grouped_command

    @property
    def previous_grouped_commands(self) -> tuple[str, ...]:
        """Return grouped invocations that were canonical in an earlier release."""

        return tuple("bpt " + " ".join(route) for route in self.previous_routes)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible registry metadata."""

        payload = asdict(self)
        payload["route"] = list(self.route)
        payload["previous_routes"] = [list(route) for route in self.previous_routes]
        payload["status"] = self.status.value
        payload["optional_dependencies"] = list(self.optional_dependencies)
        payload["target"] = self.target
        payload["grouped_command"] = self.grouped_command
        payload["previous_grouped_commands"] = list(self.previous_grouped_commands)
        return payload


_STABLE_COMMANDS: Final[tuple[CommandSpec, ...]] = (
    CommandSpec(
        command_id="provider-manifest",
        route=("provider", "manifest"),
        previous_routes=(),
        module="bayesian_phystwin.cli.provider_manifest",
        function="main",
        description="print the Causal4D provider capability manifest",
        legacy_alias="bpt-provider-manifest",
        status=CommandStatus.STABLE,
        optional_dependencies=(),
        owner="causal4d-provider-v1",
    ),
    CommandSpec(
        command_id="validate-observation-belief",
        route=("observation", "validate"),
        previous_routes=(),
        module="bayesian_phystwin.cli.observation_belief",
        function="main",
        description="validate or summarize an ObservationBeliefV1 artifact",
        legacy_alias="bpt-validate-observation-belief",
        status=CommandStatus.STABLE,
        optional_dependencies=(),
        owner="observation-belief-v1",
    ),
    CommandSpec(
        command_id="merge-deform360-exclusions",
        route=("cohort", "merge-exclusions"),
        previous_routes=(),
        module="bayesian_phystwin.cli.deform360_object_exclusion",
        function="main",
        description="merge hash-only Deform360 object-exclusion manifests",
        legacy_alias=None,
        status=CommandStatus.STABLE,
        optional_dependencies=(),
        owner="deform360-fresh-object-exclusion-v1",
    ),
    CommandSpec(
        command_id="replay-residuals",
        route=("residual", "replay"),
        previous_routes=(),
        module="bayesian_phystwin.cli.residual_replay",
        function="main",
        description="replay exported residuals through the robust likelihood",
        legacy_alias="bpt-replay-residuals",
        status=CommandStatus.STABLE,
        optional_dependencies=(),
        owner="residual-replay-v1",
    ),
    CommandSpec(
        command_id="synthetic-benchmark",
        route=("benchmark", "synthetic"),
        previous_routes=(),
        module="bayesian_phystwin.cli.synthetic_benchmark",
        function="main",
        description="run the controlled synthetic benchmark",
        legacy_alias="bpt-synthetic-benchmark",
        status=CommandStatus.STABLE,
        optional_dependencies=(),
        owner="synthetic-benchmark-v3",
    ),
    CommandSpec(
        command_id="decisive-evidence",
        route=("evidence", "summarize"),
        previous_routes=(),
        module="bayesian_phystwin.cli.decisive_evidence",
        function="main",
        description="summarize matched guarded prospective evidence",
        legacy_alias=None,
        status=CommandStatus.STABLE,
        optional_dependencies=(),
        owner="bayesian-phystwin-decisive-evidence-v1",
    ),
    CommandSpec(
        command_id="run-manifest",
        route=("run", "manifest"),
        previous_routes=(),
        module="bayesian_phystwin.cli.run_manifest",
        function="main",
        description="create or validate a content-addressed run manifest",
        legacy_alias="bpt-run-manifest",
        status=CommandStatus.STABLE,
        optional_dependencies=(),
        owner="run-manifest-v2",
    ),
)

_STATUS_NAMESPACE: Final = {
    CommandStatus.EXPERIMENT: "experiment",
    CommandStatus.DIAGNOSTIC: "diagnostic",
    CommandStatus.ARCHIVED: "archive",
}


def _build_research_commands() -> tuple[CommandSpec, ...]:
    commands: list[CommandSpec] = []
    for command_id, experiment in sorted(EXPERIMENTS.items()):
        status = CommandStatus(status_name(command_id))
        previous_routes = (
            ()
            if status is CommandStatus.EXPERIMENT
            else (("experiment", "run", command_id),)
        )
        commands.append(
            CommandSpec(
                command_id=command_id,
                route=(_STATUS_NAMESPACE[status], "run", command_id),
                previous_routes=previous_routes,
                module=experiment.module,
                function=experiment.function_name,
                description=description(command_id),
                legacy_alias=f"bpt-{command_id}",
                status=status,
                optional_dependencies=optional_dependencies(command_id),
                owner=owner(command_id),
            )
        )
    return tuple(commands)


COMMANDS: Final[tuple[CommandSpec, ...]] = (
    *_STABLE_COMMANDS,
    *_build_research_commands(),
)


def validate_registry(commands: Iterable[CommandSpec] = COMMANDS) -> None:
    """Reject ambiguous or malformed registry entries."""

    command_ids: set[str] = set()
    routes: set[tuple[str, ...]] = set()
    previous_routes: set[tuple[str, ...]] = set()
    aliases: set[str] = set()
    supported_dependencies = {"data", "graph", "pyrecest", "vision"}
    for command in commands:
        if not command.command_id or command.command_id.startswith("-"):
            raise ValueError(f"invalid command id: {command.command_id!r}")
        if command.command_id in command_ids:
            raise ValueError(f"duplicate command id: {command.command_id}")
        if not command.route or command.route in routes:
            raise ValueError(
                "duplicate or empty grouped route: " + " ".join(command.route)
            )
        for previous_route in command.previous_routes:
            if not previous_route or previous_route in previous_routes:
                raise ValueError(
                    "duplicate or empty previous grouped route: "
                    + " ".join(previous_route)
                )
            previous_routes.add(previous_route)
        if not command.module.startswith("bayesian_phystwin.cli."):
            raise ValueError(f"invalid command module: {command.module}")
        if not command.owner:
            raise ValueError(f"missing owner for {command.command_id}")
        unsupported = set(command.optional_dependencies) - supported_dependencies
        if unsupported:
            raise ValueError(
                f"unsupported optional dependencies for {command.command_id}: "
                f"{sorted(unsupported)}"
            )
        if command.legacy_alias is not None:
            if command.legacy_alias in aliases:
                raise ValueError(f"duplicate legacy alias: {command.legacy_alias}")
            aliases.add(command.legacy_alias)
        command_ids.add(command.command_id)
        routes.add(command.route)
    collisions = routes & previous_routes
    if collisions:
        rendered = sorted(" ".join(route) for route in collisions)
        raise ValueError(
            "previous grouped route collides with current route: " + str(rendered)
        )


validate_registry()
COMMANDS_BY_ID: Final = {command.command_id: command for command in COMMANDS}
COMMANDS_BY_ROUTE: Final = {command.route: command for command in COMMANDS}
COMMANDS_BY_PREVIOUS_ROUTE: Final = {
    route: command for command in COMMANDS for route in command.previous_routes
}
COMMANDS_BY_LEGACY_ALIAS: Final = {
    command.legacy_alias: command
    for command in COMMANDS
    if command.legacy_alias is not None
}


def iter_commands(*, status: CommandStatus | None = None) -> tuple[CommandSpec, ...]:
    """Return registry entries in deterministic command-id order."""

    selected = (
        command for command in COMMANDS if status is None or command.status is status
    )
    return tuple(sorted(selected, key=lambda command: command.command_id))


def find_command(
    selector: str, *, status: CommandStatus | None = None
) -> CommandSpec | None:
    """Resolve metadata broadly, but execution selectors only within a status."""

    normalized = selector.strip()
    command = COMMANDS_BY_ID.get(normalized)
    if command is None and status is None:
        command = COMMANDS_BY_LEGACY_ALIAS.get(normalized)
    route = tuple(normalized.removeprefix("bpt ").split())
    if command is None:
        command = COMMANDS_BY_ROUTE.get(route)
    if command is None and status is None:
        command = COMMANDS_BY_PREVIOUS_ROUTE.get(route)
    if command is not None and (status is None or command.status is status):
        return command
    return None


def find_command_metadata(selector: str) -> CommandSpec | None:
    """Resolve an id, current/previous grouped route, or removed alias."""

    return find_command(selector)


def find_runnable_command(
    command_id: str, *, status: CommandStatus
) -> CommandSpec | None:
    """Resolve an exact command id for execution in one lifecycle catalog."""

    command = COMMANDS_BY_ID.get(command_id)
    if command is not None and command.status is status:
        return command
    return None
