from __future__ import annotations

import json

from bayesian_phystwin.cli.experiment import describe_main, list_main, run_main
from bayesian_phystwin.cli.main import main


def test_grouped_help_discovers_experiment_namespace(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "experiment" in output

    assert main(["experiment", "--help"]) == 0
    output = capsys.readouterr().out
    assert "describe" in output
    assert "list" in output
    assert "run" in output


def test_experiment_list_and_describe_use_installed_metadata(capsys) -> None:
    assert list_main(["--category", "phystwin", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload
    assert all(item["category"] == "phystwin" for item in payload)
    assert any(item["experiment_id"] == "build-phystwin-cues" for item in payload)

    assert describe_main(["build-phystwin-cues", "--json"]) == 0
    description = json.loads(capsys.readouterr().out)
    assert description["console_script"] == "bpt-build-phystwin-cues"
    assert description["target"] == "bayesian_phystwin.cli.phystwin_cues:main"


def test_grouped_dispatch_calls_experiment_cli(monkeypatch) -> None:
    received: list[list[str]] = []

    def fake_run(arguments):
        received.append(list(arguments))
        return 11

    monkeypatch.setattr("bayesian_phystwin.cli.experiment.run_main", fake_run)
    assert main(["experiment", "run", "some-id", "--", "--seed", "5"]) == 11
    assert received == [["some-id", "--", "--seed", "5"]]


def test_run_cli_strips_separator_before_forwarding(monkeypatch) -> None:
    received: list[tuple[str, list[str]]] = []

    def fake_run(name, arguments):
        received.append((name, list(arguments)))
        return 3

    monkeypatch.setattr("bayesian_phystwin.cli.experiment.run_experiment", fake_run)
    assert run_main(["some-id", "--", "--alpha", "2"]) == 3
    assert received == [("some-id", ["--alpha", "2"])]
