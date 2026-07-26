from __future__ import annotations

import ast
from pathlib import Path

from bayesian_phystwin.cli.command_registry import (
    COMMANDS,
    CommandStatus,
    legacy_entry_points,
    validate_registry,
)


def _project_scripts() -> dict[str, str]:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts: dict[str, str] = {}
    in_scripts = False
    for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[project.scripts]":
            in_scripts = True
            continue
        if in_scripts and line.startswith("["):
            break
        if in_scripts and line and not line.startswith("#"):
            name, value = line.split("=", maxsplit=1)
            parsed = ast.literal_eval(value.strip())
            if not isinstance(parsed, str):
                raise AssertionError(f"non-string script target for {name.strip()}")
            scripts[name.strip()] = parsed
    return scripts


def test_registry_is_unambiguous() -> None:
    validate_registry()
    assert len(COMMANDS) == len({command.command_id for command in COMMANDS})
    assert len(COMMANDS) == len({command.route for command in COMMANDS})


def test_registry_covers_all_lifecycle_states() -> None:
    assert {command.status for command in COMMANDS} == set(CommandStatus)


def test_installed_legacy_surface_is_frozen_in_registry() -> None:
    expected = {
        "bpt": "bayesian_phystwin.cli.main:main",
        **legacy_entry_points(),
    }
    assert _project_scripts() == expected


def test_nonstable_commands_use_registry_runner_routes() -> None:
    namespace = {
        CommandStatus.EXPERIMENT: "experiment",
        CommandStatus.DIAGNOSTIC: "diagnostic",
        CommandStatus.ARCHIVED: "archive",
    }
    for command in COMMANDS:
        if command.status is CommandStatus.STABLE:
            continue
        assert command.route == (
            namespace[command.status],
            "run",
            command.command_id,
        )


def test_registry_metadata_is_complete() -> None:
    for command in COMMANDS:
        assert command.description
        assert command.owner
        assert command.module.startswith("bayesian_phystwin.cli.")
        assert command.function == "main"
        supported = {"data", "graph", "pyrecest", "vision"}
        assert all(
            dependency in supported for dependency in command.optional_dependencies
        )
