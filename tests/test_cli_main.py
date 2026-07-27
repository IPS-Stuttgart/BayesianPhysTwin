from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bayesian_phystwin.cli.command_registry import COMMANDS, ROUTES, CommandSpec
from bayesian_phystwin.cli.main import main


def test_root_help_lists_stable_and_current_namespaces_only(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    for namespace in (
        "provider",
        "observation",
        "benchmark",
        "evidence",
        "run",
        "experiment",
        "commands",
    ):
        assert namespace in output
    assert "\n  diagnostic" not in output
    assert "\n  archived" not in output


def test_stable_namespace_help_lists_nested_command(capsys) -> None:
    assert main(["provider"]) == 0
    assert "manifest" in capsys.readouterr().out


@pytest.mark.parametrize("status", ["experiment", "diagnostic", "archived"])
def test_lifecycle_namespace_help_exposes_list_and_run(status: str, capsys) -> None:
    assert main([status]) == 0
    output = capsys.readouterr().out
    assert "list" in output
    assert "run <id>" in output


def test_experiment_list_excludes_diagnostics(capsys) -> None:
    assert main(["experiment", "list"]) == 0
    output = capsys.readouterr().out
    assert "evaluate-phystwin-official" in output
    assert "analyze-phystwin-horizon" not in output


def test_hidden_namespace_is_directly_listable(capsys) -> None:
    assert main(["diagnostic", "list"]) == 0
    output = capsys.readouterr().out
    assert "analyze-phystwin-horizon" in output
    assert "evaluate-phystwin-official" not in output


def test_namespace_list_can_emit_json(capsys) -> None:
    assert main(["archived", "list", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload
    assert {entry["status"] for entry in payload} == {"archived"}
    assert all(entry["route"][:2] == ["archived", "run"] for entry in payload)


def test_command_listing_defaults_to_stable_and_current(capsys) -> None:
    assert main(["commands"]) == 0
    output = capsys.readouterr().out
    assert "stable" in output
    assert "experiment" in output
    assert "diagnostic" not in output
    assert "archived" not in output
    assert "milestone=run-manifest-v2" in output


def test_command_listing_can_emit_complete_json(capsys) -> None:
    assert main(["commands", "--all", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == len(COMMANDS)
    assert {entry["status"] for entry in payload} == {
        "stable",
        "experiment",
        "diagnostic",
        "archived",
    }
    evidence = next(entry for entry in payload if entry["id"] == "decisive-evidence")
    assert evidence["legacy_alias"] is None
    assert evidence["route"] == ["evidence", "summarize"]


def test_command_listing_can_filter_repeated_statuses(capsys) -> None:
    assert (
        main(
            [
                "commands",
                "--status",
                "stable",
                "--status",
                "diagnostic",
                "--format",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert {entry["status"] for entry in payload} == {"stable", "diagnostic"}


_STABLE_COMMANDS = tuple(
    command
    for command in COMMANDS
    if command.status == "stable" and command.route != ("commands",)
)


@pytest.mark.parametrize(
    "command",
    _STABLE_COMMANDS,
    ids=lambda command: " ".join(command.route),
)
def test_stable_routes_dispatch_from_registry(
    command: CommandSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_main(arguments: list[str]) -> int:
        calls.append(arguments)
        return 0

    def fake_import_module(module_name: str) -> SimpleNamespace:
        assert module_name == command.module
        return SimpleNamespace(**{command.function: fake_main})

    monkeypatch.setattr(
        "bayesian_phystwin.cli.main.importlib.import_module",
        fake_import_module,
    )
    assert main([*command.route, "--smoke"]) == 0
    assert calls == [["--smoke"]]


def test_experiment_run_dispatches_registered_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ROUTES[("experiment", "run", "evaluate-phystwin-official")]
    calls: list[list[str]] = []

    def fake_main(arguments: list[str]) -> int:
        calls.append(arguments)
        return 7

    monkeypatch.setattr(
        "bayesian_phystwin.cli.main.importlib.import_module",
        lambda module_name: SimpleNamespace(**{command.function: fake_main}),
    )
    assert main([*command.route, "--case", "01"]) == 7
    assert calls == [["--case", "01"]]


def test_run_namespace_without_id_prints_help(capsys) -> None:
    assert main(["experiment", "run"]) == 0
    assert "<id>" in capsys.readouterr().out


def test_unknown_lifecycle_id_is_rejected(capsys) -> None:
    assert main(["experiment", "run", "unknown"]) == 2
    assert "usage: bpt experiment run" in capsys.readouterr().err


def test_unknown_root_command_is_rejected(capsys) -> None:
    assert main(["unknown"]) == 2
    assert "usage: bpt" in capsys.readouterr().err
