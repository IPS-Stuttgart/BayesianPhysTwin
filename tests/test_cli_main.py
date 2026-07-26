from __future__ import annotations

import sys
from importlib import metadata, util
from types import SimpleNamespace

from bayesian_phystwin.cli import experiments
from bayesian_phystwin.cli.main import main


def test_grouped_cli_lists_stable_namespaces(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "provider" in output
    assert "observation" in output
    assert "benchmark" in output
    assert "evidence" in output
    assert "run" in output
    assert "experiment" in output
    assert "No legacy `bpt-*` executables are installed." in output


def test_grouped_cli_lists_nested_commands(capsys) -> None:
    assert main(["provider"]) == 0
    output = capsys.readouterr().out
    assert "manifest" in output


def test_grouped_cli_rejects_unknown_command(capsys) -> None:
    assert main(["unknown"]) == 2
    assert "usage: bpt" in capsys.readouterr().err


def test_experiment_registry_lists_and_describes_commands(capsys) -> None:
    assert main(["experiment", "list"]) == 0
    listed = capsys.readouterr().out
    assert "build-phystwin-cues" in listed
    assert "phystwin-refit" in listed
    assert "synthetic-benchmark" not in listed

    assert main(["experiment", "describe", "phystwin-refit"]) == 0
    described = capsys.readouterr().out
    assert "stability: experimental" in described
    assert "bayesian_phystwin.cli.phystwin_refit:main" in described
    assert "bpt experiment run phystwin-refit" in described


def test_experiment_registry_rejects_removed_executable_name(capsys) -> None:
    assert main(["experiment", "describe", "bpt-phystwin-refit"]) == 2
    error = capsys.readouterr().err
    assert "unknown experiment: bpt-phystwin-refit" in error


def test_experiment_registry_covers_every_nonstable_command_module() -> None:
    assert len(experiments.EXPERIMENTS) == 74
    assert experiments.experiment_ids() == tuple(sorted(experiments.EXPERIMENTS))
    for spec in experiments.EXPERIMENTS.values():
        assert not spec.experiment_id.startswith("bpt-")
        assert util.find_spec(spec.module) is not None


def test_experiment_registry_forwards_to_argv_aware_main(monkeypatch) -> None:
    received: list[list[str]] = []

    def entrypoint(argv: list[str]) -> int:
        received.append(argv)
        return 7

    module = SimpleNamespace(main=entrypoint)
    monkeypatch.setattr(experiments.importlib, "import_module", lambda _: module)

    assert experiments.main(["run", "build-phystwin-cues", "--flag", "value"]) == 7
    assert received == [["--flag", "value"]]


def test_experiment_registry_adapts_sys_argv_main(monkeypatch) -> None:
    received: list[tuple[str, ...]] = []
    original_argv = sys.argv

    def entrypoint() -> None:
        received.append(tuple(sys.argv))

    module = SimpleNamespace(main=entrypoint)
    monkeypatch.setattr(experiments.importlib, "import_module", lambda _: module)

    assert experiments.main(["run", "phystwin-refit", "--epochs", "1"]) == 0
    assert received == [("bpt experiment run phystwin-refit", "--epochs", "1")]
    assert sys.argv is original_argv


def test_distribution_installs_only_grouped_console_script() -> None:
    distribution = metadata.distribution("bayesian-phystwin")
    scripts = {
        entry_point.name: entry_point.value
        for entry_point in distribution.entry_points
        if entry_point.group == "console_scripts"
    }
    assert scripts == {"bpt": "bayesian_phystwin.cli.main:main"}
