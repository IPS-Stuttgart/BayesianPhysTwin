from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from bayesian_phystwin.cli import _command_dispatch as dispatch
from bayesian_phystwin.cli import main as cli_main


def test_grouped_cli_lists_stable_and_registry_namespaces(capsys) -> None:
    assert cli_main.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "provider" in output
    assert "observation" in output
    assert "benchmark" in output
    assert "run" in output
    assert "commands" in output
    assert "experiment" in output
    assert "diagnostic" in output
    assert "archive" in output


def test_grouped_cli_lists_nested_stable_commands(capsys) -> None:
    assert cli_main.main(["provider"]) == 0
    output = capsys.readouterr().out
    assert "manifest" in output


def test_experiment_list_hides_archived_commands(capsys) -> None:
    assert cli_main.main(["experiment", "list"]) == 0
    output = capsys.readouterr().out
    assert "confirm-phystwin-bayesian-anchor" in output
    assert "evaluate-phystwin-state-injection" not in output


def test_archive_list_exposes_frozen_historical_commands(capsys) -> None:
    assert cli_main.main(["archive", "list"]) == 0
    output = capsys.readouterr().out
    assert "evaluate-phystwin-state-injection" in output


def test_registry_json_contains_compatibility_metadata(capsys) -> None:
    assert cli_main.main(["commands", "list", "--status", "stable", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    provider = next(
        item for item in payload if item["command_id"] == "provider-manifest"
    )
    assert provider["route"] == ["provider", "manifest"]
    assert provider["legacy_alias"] == "bpt-provider-manifest"
    assert provider["owner"] == "causal4d-provider-v1"
    assert provider["optional_dependencies"] == []


def test_registry_describes_command_by_legacy_alias(capsys) -> None:
    assert (
        cli_main.main(
            ["commands", "describe", "bpt-confirm-phystwin-bayesian-anchor"]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "status: experiment" in output
    assert "bpt experiment run confirm-phystwin-bayesian-anchor" in output
    assert "owner: phystwin-full22-v1" in output
    assert "optional dependencies: graph" in output


def test_grouped_cli_dispatches_argv_aware_command(monkeypatch) -> None:
    captured: list[str] = []

    def fake_main(argv: list[str]) -> int:
        captured.extend(argv)
        return 7

    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda _: SimpleNamespace(main=fake_main),
    )
    assert cli_main.main(["provider", "manifest", "--output", "manifest.json"]) == 7
    assert captured == ["--output", "manifest.json"]


def test_grouped_cli_dispatches_legacy_no_arg_command(monkeypatch) -> None:
    captured: list[str] = []
    original_argv = list(sys.argv)

    def fake_main() -> None:
        captured.extend(sys.argv[1:])

    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda _: SimpleNamespace(main=fake_main),
    )
    assert (
        cli_main.main(
            [
                "experiment",
                "run",
                "confirm-phystwin-bayesian-anchor",
                "data",
                "output",
                "--force",
            ]
        )
        == 0
    )
    assert captured == ["data", "output", "--force"]
    assert sys.argv == original_argv


def test_grouped_cli_reports_declared_missing_optional_dependency(
    monkeypatch, capsys
) -> None:
    def missing_module(_: str):
        raise ModuleNotFoundError("No module named 'scipy'", name="scipy")

    monkeypatch.setattr(dispatch.importlib, "import_module", missing_module)
    assert (
        cli_main.main(
            ["experiment", "run", "confirm-phystwin-bayesian-anchor"]
        )
        == 1
    )
    assert "install bayesian-phystwin[graph]" in capsys.readouterr().err


def test_grouped_cli_does_not_mask_internal_import_errors(monkeypatch) -> None:
    def missing_module(_: str):
        raise ModuleNotFoundError(
            "No module named 'bayesian_phystwin.missing'",
            name="bayesian_phystwin.missing",
        )

    monkeypatch.setattr(dispatch.importlib, "import_module", missing_module)
    with pytest.raises(ModuleNotFoundError, match="bayesian_phystwin.missing"):
        cli_main.main(
            ["experiment", "run", "confirm-phystwin-bayesian-anchor"]
        )


def test_grouped_cli_rejects_unknown_command(capsys) -> None:
    assert cli_main.main(["unknown"]) == 2
    assert "usage: bpt" in capsys.readouterr().err


def test_catalog_rejects_command_from_wrong_status(capsys) -> None:
    assert (
        cli_main.main(
            ["diagnostic", "run", "confirm-phystwin-bayesian-anchor"]
        )
        == 2
    )
    assert "unknown diagnostic command" in capsys.readouterr().err
