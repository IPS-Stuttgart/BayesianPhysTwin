from __future__ import annotations

import json
import sys
from dataclasses import replace
from importlib import metadata
from types import SimpleNamespace

import pytest

from bayesian_phystwin.cli import _command_catalog as catalog
from bayesian_phystwin.cli import _command_dispatch as dispatch
from bayesian_phystwin.cli import main as main_module
from bayesian_phystwin.cli import run_manifest as run_manifest_cli
from bayesian_phystwin.cli.command_registry import (
    COMMANDS_BY_ID,
    validate_registry,
)
from bayesian_phystwin.repository_provenance import RepositoryState

main = main_module.main


def test_grouped_cli_lists_stable_and_registry_namespaces(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    for name in (
        "provider",
        "observation",
        "residual",
        "benchmark",
        "evidence",
        "run",
        "commands",
        "experiment",
        "diagnostic",
        "archive",
    ):
        assert name in output
    assert "Exactly one executable is installed: bpt." in output


def test_grouped_cli_lists_nested_stable_commands(capsys) -> None:
    assert main(["evidence"]) == 0
    assert "summarize" in capsys.readouterr().out


def test_catalogs_are_lifecycle_specific(capsys) -> None:
    assert main(["experiment", "list"]) == 0
    experiments = capsys.readouterr().out
    assert "confirm-phystwin-bayesian-anchor" in experiments
    assert "audit-phystwin-calibration" not in experiments
    assert "evaluate-phystwin-state-injection" not in experiments

    assert main(["diagnostic", "list"]) == 0
    diagnostics = capsys.readouterr().out
    assert "audit-phystwin-calibration" in diagnostics

    assert main(["archive", "list"]) == 0
    archived = capsys.readouterr().out
    assert "evaluate-phystwin-state-injection" in archived


def test_registry_json_contains_ownership_and_migration_metadata(capsys) -> None:
    assert main(["commands", "list", "--status", "stable", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    provider = next(
        item for item in payload if item["command_id"] == "provider-manifest"
    )
    assert provider["route"] == ["provider", "manifest"]
    assert provider["legacy_alias"] == "bpt-provider-manifest"
    assert provider["owner"] == "causal4d-provider-v1"
    assert provider["optional_dependencies"] == []

    evidence = next(
        item for item in payload if item["command_id"] == "decisive-evidence"
    )
    assert evidence["route"] == ["evidence", "summarize"]
    assert evidence["legacy_alias"] is None


def test_registry_inspects_and_migrates_removed_alias_without_running_it(
    capsys,
) -> None:
    assert main(["commands", "describe", "bpt-phystwin-refit"]) == 0
    described = capsys.readouterr().out
    assert "status: experiment" in described
    assert "command: bpt experiment run phystwin-refit" in described

    assert main(["commands", "migrate", "bpt-phystwin-refit"]) == 0
    assert capsys.readouterr().out.strip() == "bpt experiment run phystwin-refit"

    assert main(["experiment", "run", "bpt-phystwin-refit"]) == 2
    assert "unknown experiment command" in capsys.readouterr().err


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
    assert main(["provider", "manifest", "--output", "manifest.json"]) == 7
    assert captured == ["--output", "manifest.json"]


def test_grouped_cli_dispatches_no_arg_command_with_canonical_argv(monkeypatch) -> None:
    captured: list[str] = []
    original_argv = sys.argv

    def fake_main() -> None:
        captured.extend(sys.argv)

    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda _: SimpleNamespace(main=fake_main),
    )
    assert (
        main(
            [
                "experiment",
                "run",
                "confirm-phystwin-bayesian-anchor",
                "--force",
            ]
        )
        == 0
    )
    assert captured == [
        "bpt experiment run confirm-phystwin-bayesian-anchor",
        "--force",
    ]
    assert sys.argv is original_argv


def test_grouped_cli_reports_only_declared_missing_optional_dependency(
    monkeypatch, capsys
) -> None:
    def missing_scipy(_: str):
        raise ModuleNotFoundError("No module named 'scipy'", name="scipy")

    monkeypatch.setattr(dispatch.importlib, "import_module", missing_scipy)
    assert main(["experiment", "run", "confirm-phystwin-bayesian-anchor"]) == 1
    assert "install bayesian-phystwin[graph]" in capsys.readouterr().err

    def missing_internal(_: str):
        raise ModuleNotFoundError(
            "No module named 'bayesian_phystwin.missing'",
            name="bayesian_phystwin.missing",
        )

    monkeypatch.setattr(dispatch.importlib, "import_module", missing_internal)
    with pytest.raises(ModuleNotFoundError, match="bayesian_phystwin.missing"):
        main(["experiment", "run", "confirm-phystwin-bayesian-anchor"])


def test_distribution_installs_only_grouped_console_script() -> None:
    distribution = metadata.distribution("bayesian-phystwin")
    scripts = {
        entry_point.name: entry_point.value
        for entry_point in distribution.entry_points
        if entry_point.group == "console_scripts"
    }
    assert scripts == {"bpt": "bayesian_phystwin.cli.main:main"}


def test_run_manifest_discovers_clean_primary_repository(monkeypatch, tmp_path) -> None:
    expected = RepositoryState(
        repository="FlorianPfaff/Bayesian-PhysTwin",
        revision="a" * 40,
        dirty=False,
        role="primary",
    )
    monkeypatch.setattr(
        run_manifest_cli,
        "discover_git_repository_state",
        lambda root, repository: expected,
    )
    arguments = SimpleNamespace(
        revision=None,
        dirty=False,
        repository_root=tmp_path,
        repository=None,
        allow_dirty=False,
    )
    assert run_manifest_cli._primary_repository_state(arguments) == expected


def test_catalog_help_json_and_error_paths(capsys) -> None:
    assert main(["commands"]) == 0
    assert "migrate" in capsys.readouterr().out
    assert main(["diagnostic"]) == 0
    assert "classified as diagnostic" in capsys.readouterr().out

    assert main(["commands", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)
    assert main(["commands", "describe", "provider-manifest", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["command_id"] == "provider-manifest"
    assert main(["commands", "migrate", "bpt-provider-manifest", "--json"]) == 0
    assert (
        json.loads(capsys.readouterr().out)["canonical_command"]
        == "bpt provider manifest"
    )

    error_cases = (
        ["commands", "list", "--status", "invalid"],
        ["commands", "list", "--bad"],
        ["commands", "describe"],
        ["commands", "describe", "missing"],
        ["commands", "describe", "--bad"],
        ["commands", "migrate"],
        ["commands", "migrate", "missing"],
        ["commands", "unknown"],
        ["experiment", "list", "unexpected"],
        ["experiment", "describe"],
        ["experiment", "describe", "missing"],
        ["experiment", "describe", "phystwin-refit", "--bad"],
        ["experiment", "run", "missing"],
        ["experiment", "unknown"],
    )
    for arguments in error_cases:
        assert main(arguments) == 2
        assert capsys.readouterr().err

    catalog._print_list((), json_output=False)
    assert "No commands" in capsys.readouterr().out


def test_main_group_help_and_unknown_paths(capsys) -> None:
    assert main(["provider", "--help"]) == 0
    assert "manifest" in capsys.readouterr().out
    assert main(["provider", "unknown"]) == 2
    assert "usage: bpt provider" in capsys.readouterr().err
    assert main(["unknown"]) == 2
    assert "usage: bpt" in capsys.readouterr().err


def test_dispatch_keyword_signature_and_failure_modes(monkeypatch) -> None:
    captured: list[str] = []

    def keyword_main(*, argv: list[str]) -> int:
        captured.extend(argv)
        return 3

    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda _: SimpleNamespace(main=keyword_main),
    )
    assert main(["provider", "manifest", "one"]) == 3
    assert captured == ["one"]

    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda _: SimpleNamespace(main=None),
    )
    with pytest.raises(TypeError, match="not callable"):
        main(["provider", "manifest"])

    def too_many(first: str, second: str) -> int:
        return 0

    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda _: SimpleNamespace(main=too_many),
    )
    with pytest.raises(TypeError, match="zero or one"):
        main(["provider", "manifest"])

    def variadic(*args: str) -> int:
        return len(args)

    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda _: SimpleNamespace(main=variadic),
    )
    with pytest.raises(TypeError, match="must expose"):
        main(["provider", "manifest"])

    monkeypatch.setattr(
        dispatch.inspect,
        "signature",
        lambda _: (_ for _ in ()).throw(ValueError("opaque")),
    )
    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda _: SimpleNamespace(main=lambda argv: 0),
    )
    with pytest.raises(TypeError, match="cannot inspect"):
        main(["provider", "manifest"])


def test_dispatch_restores_sys_argv_when_legacy_main_fails(monkeypatch) -> None:
    original_argv = sys.argv

    def failing_main() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda _: SimpleNamespace(main=failing_main),
    )
    with pytest.raises(RuntimeError, match="boom"):
        main(["archive", "run", "evaluate-phystwin-state-injection"])
    assert sys.argv is original_argv


def test_catalog_run_returns_none_as_success(monkeypatch) -> None:
    def no_result(argv: list[str]) -> None:
        assert argv == ["x"]

    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda _: SimpleNamespace(main=no_result),
    )
    assert main(["experiment", "run", "phystwin-refit", "x"]) == 0


def test_registry_route_describe(capsys) -> None:
    assert main(["commands", "describe", "bpt", "provider", "manifest"]) == 0
    assert "provider-manifest" in capsys.readouterr().out
    assert COMMANDS_BY_ID["provider-manifest"].canonical_command == (
        "bpt provider manifest"
    )


def test_registry_validation_fails_closed_on_invalid_metadata() -> None:
    valid = COMMANDS_BY_ID["provider-manifest"]
    invalid = replace(valid, command_id="-invalid")
    with pytest.raises(ValueError, match="invalid command id"):
        validate_registry((invalid,))

    duplicate = replace(
        valid,
        command_id="duplicate-provider",
        legacy_alias="bpt-duplicate-provider",
    )
    with pytest.raises(ValueError, match="duplicate or empty grouped route"):
        validate_registry((valid, duplicate))
