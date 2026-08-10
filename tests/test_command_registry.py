from __future__ import annotations

from dataclasses import replace
from importlib import util

import pytest

from bayesian_phystwin.cli._command_inventory import STABLE_ROUTES
from bayesian_phystwin.cli.command_registry import (
    COMMANDS,
    COMMANDS_BY_LEGACY_ALIAS,
    COMMANDS_BY_PREVIOUS_ROUTE,
    CommandStatus,
    find_command_metadata,
    find_runnable_command,
    iter_commands,
    validate_registry,
)


def test_registry_is_complete_and_unambiguous() -> None:
    validate_registry()
    assert len(COMMANDS) == 87
    assert len(COMMANDS) == len({command.command_id for command in COMMANDS})
    assert len(COMMANDS) == len({command.route for command in COMMANDS})
    assert len(COMMANDS_BY_LEGACY_ALIAS) == 85
    assert len(COMMANDS_BY_PREVIOUS_ROUTE) == 46
    assert set(STABLE_ROUTES) == {
        command.command_id
        for command in COMMANDS
        if command.status is CommandStatus.STABLE
    }


def test_registry_covers_all_lifecycle_states() -> None:
    counts = {status: len(iter_commands(status=status)) for status in CommandStatus}
    assert counts == {
        CommandStatus.STABLE: 7,
        CommandStatus.EXPERIMENT: 34,
        CommandStatus.DIAGNOSTIC: 21,
        CommandStatus.ARCHIVED: 25,
    }
    assert len(iter_commands()) == len(COMMANDS)


def test_provider_failure_decomposition_is_a_registered_diagnostic() -> None:
    command = find_command_metadata("diagnose-provider-failures")
    assert command is not None
    assert command.status is CommandStatus.DIAGNOSTIC
    assert command.canonical_command == (
        "bpt diagnostic run diagnose-provider-failures"
    )
    assert command.owner == "provider-failure-decomposition-v1"
    assert command.optional_dependencies == ()


def test_discrepancy_tournament_is_a_registered_diagnostic() -> None:
    command = find_command_metadata("select-discrepancy-candidate")
    assert command is not None
    assert command.status is CommandStatus.DIAGNOSTIC
    assert command.canonical_command == (
        "bpt diagnostic run select-discrepancy-candidate"
    )
    assert command.owner == "discrepancy-candidate-tournament-v1"
    assert command.optional_dependencies == ()


def test_probabilistic_scoring_is_a_registered_diagnostic() -> None:
    command = find_command_metadata("score-probabilistic-predictions")
    assert command is not None
    assert command.status is CommandStatus.DIAGNOSTIC
    assert command.canonical_command == (
        "bpt diagnostic run score-probabilistic-predictions"
    )
    assert command.owner == "probabilistic-prediction-scoring-v1"
    assert command.optional_dependencies == ()


def test_removed_aliases_are_metadata_not_runnable_selectors() -> None:
    assert (
        find_runnable_command("bpt-phystwin-refit", status=CommandStatus.EXPERIMENT)
        is None
    )
    metadata = find_command_metadata("bpt-phystwin-refit")
    assert metadata is not None
    assert metadata.command_id == "phystwin-refit"
    assert find_command_metadata("bpt experiment run phystwin-refit") == metadata
    assert find_command_metadata("does-not-exist") is None


def test_previous_grouped_routes_resolve_to_new_lifecycle_namespaces() -> None:
    diagnostic = find_command_metadata("bpt experiment run audit-phystwin-calibration")
    assert diagnostic is not None
    assert diagnostic.status is CommandStatus.DIAGNOSTIC
    assert diagnostic.canonical_command == (
        "bpt diagnostic run audit-phystwin-calibration"
    )
    assert diagnostic.previous_grouped_commands == (
        "bpt experiment run audit-phystwin-calibration",
    )

    archived = find_command_metadata(
        "bpt experiment run evaluate-phystwin-state-injection"
    )
    assert archived is not None
    assert archived.status is CommandStatus.ARCHIVED
    assert archived.canonical_command == (
        "bpt archive run evaluate-phystwin-state-injection"
    )


def test_nonstable_routes_match_lifecycle_namespace() -> None:
    namespaces = {
        CommandStatus.EXPERIMENT: "experiment",
        CommandStatus.DIAGNOSTIC: "diagnostic",
        CommandStatus.ARCHIVED: "archive",
    }
    for command in COMMANDS:
        if command.status is CommandStatus.STABLE:
            assert command.previous_routes == ()
            continue
        assert command.route == (
            namespaces[command.status],
            "run",
            command.command_id,
        )
        expected_previous = (
            ()
            if command.status is CommandStatus.EXPERIMENT
            else (("experiment", "run", command.command_id),)
        )
        assert command.previous_routes == expected_previous


def test_every_registered_target_module_exists() -> None:
    for command in COMMANDS:
        assert util.find_spec(command.module) is not None


@pytest.mark.parametrize("command_id", ["", "-invalid"])
def test_registry_rejects_invalid_command_ids(command_id: str) -> None:
    command = replace(
        COMMANDS[0],
        command_id=command_id,
        route=("invalid", command_id or "empty"),
        legacy_alias=None,
    )
    with pytest.raises(ValueError, match="invalid command id"):
        validate_registry((command,))


def test_registry_rejects_duplicate_command_ids() -> None:
    first = COMMANDS[0]
    second = replace(
        first,
        route=("other", "route"),
        legacy_alias="bpt-other-route",
    )
    with pytest.raises(ValueError, match="duplicate command id"):
        validate_registry((first, second))


@pytest.mark.parametrize("route", [(), ("provider", "manifest")])
def test_registry_rejects_empty_or_duplicate_routes(route: tuple[str, ...]) -> None:
    first = COMMANDS[0]
    second = replace(
        first,
        command_id="other-command",
        route=route,
        legacy_alias="bpt-other-command",
    )
    commands = (second,) if not route else (first, second)
    with pytest.raises(ValueError, match="duplicate or empty grouped route"):
        validate_registry(commands)


def test_registry_rejects_invalid_module_owner_and_dependency() -> None:
    base = replace(
        COMMANDS[0],
        command_id="other-command",
        route=("other", "command"),
        legacy_alias="bpt-other-command",
    )
    with pytest.raises(ValueError, match="invalid command module"):
        validate_registry((replace(base, module="elsewhere.command"),))
    with pytest.raises(ValueError, match="missing owner"):
        validate_registry((replace(base, owner=""),))
    with pytest.raises(ValueError, match="unsupported optional dependencies"):
        validate_registry((replace(base, optional_dependencies=("unknown",)),))


def test_registry_rejects_duplicate_legacy_aliases() -> None:
    first = COMMANDS[0]
    second = replace(
        first,
        command_id="other-command",
        route=("other", "command"),
    )
    with pytest.raises(ValueError, match="duplicate legacy alias"):
        validate_registry((first, second))


def test_registry_rejects_invalid_previous_routes() -> None:
    first = COMMANDS[0]
    empty_previous = replace(first, previous_routes=((),))
    with pytest.raises(ValueError, match="duplicate or empty previous grouped route"):
        validate_registry((empty_previous,))

    duplicate_previous = replace(
        COMMANDS[1],
        previous_routes=(("experiment", "run", "old"),),
    )
    first_previous = replace(
        first,
        previous_routes=(("experiment", "run", "old"),),
    )
    with pytest.raises(ValueError, match="duplicate or empty previous grouped route"):
        validate_registry((first_previous, duplicate_previous))


def test_registry_rejects_previous_route_collision_with_current_route() -> None:
    first = COMMANDS[0]
    colliding = replace(
        COMMANDS[1],
        previous_routes=(first.route,),
    )
    with pytest.raises(ValueError, match="collides with current route"):
        validate_registry((first, colliding))
