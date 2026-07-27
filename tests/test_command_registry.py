from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

from bayesian_phystwin.cli.command_registry import (
    COMMANDS,
    FROZEN_LEGACY_ALIAS_SHA256,
    LEGACY_ENTRY_POINTS,
    ROUTES,
    CommandSpec,
    legacy_alias_fingerprint,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_registry_routes_and_command_identities_are_unique() -> None:
    assert len(ROUTES) == len(COMMANDS)
    assert len({(command.status, command.command_id) for command in COMMANDS}) == len(
        COMMANDS
    )
    assert set(ROUTES.values()) == set(COMMANDS)


def test_frozen_legacy_aliases_match_packaging_exactly() -> None:
    payload = tomllib.loads((_REPOSITORY_ROOT / "pyproject.toml").read_text())
    scripts = payload["project"]["scripts"]
    assert scripts.pop("bpt") == "bayesian_phystwin.cli.main:main"
    assert scripts == dict(LEGACY_ENTRY_POINTS)
    assert len(LEGACY_ENTRY_POINTS) == 79


def test_frozen_legacy_alias_fingerprint_is_unchanged() -> None:
    assert legacy_alias_fingerprint() == FROZEN_LEGACY_ALIAS_SHA256


def test_every_legacy_alias_has_one_registry_entry() -> None:
    registered = {
        command.legacy_alias: command.target
        for command in COMMANDS
        if command.legacy_alias is not None
    }
    assert registered == dict(LEGACY_ENTRY_POINTS)


def test_grouped_only_commands_do_not_create_console_scripts() -> None:
    grouped_only = {
        command.route
        for command in COMMANDS
        if command.legacy_alias is None
    }
    assert grouped_only == {("commands",), ("evidence", "summarize")}


def test_lifecycle_commands_use_list_run_namespace() -> None:
    for command in COMMANDS:
        if command.status in {"experiment", "diagnostic", "archived"}:
            assert command.route == (
                command.status,
                "run",
                command.command_id,
            )


def test_registry_metadata_is_complete_and_normalized() -> None:
    allowed_extras = {"data", "graph", "vision"}
    for command in COMMANDS:
        assert command.command_id
        assert command.description.strip()
        assert command.milestone.strip()
        assert set(command.extras) <= allowed_extras
        assert len(command.extras) == len(set(command.extras))
        assert all(token and not token.startswith("-") for token in command.route)


@pytest.mark.parametrize(
    "command",
    COMMANDS,
    ids=lambda command: " ".join(command.route),
)
def test_registered_implementation_module_exists(command: CommandSpec) -> None:
    assert importlib.util.find_spec(command.module) is not None


def test_registry_contains_all_lifecycle_statuses() -> None:
    assert {command.status for command in COMMANDS} == {
        "stable",
        "experiment",
        "diagnostic",
        "archived",
    }
