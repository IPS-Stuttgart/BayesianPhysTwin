from __future__ import annotations

from bayesian_phystwin.cli.main import main


def test_grouped_cli_lists_stable_namespaces(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "provider" in output
    assert "observation" in output
    assert "benchmark" in output
    assert "run" in output


def test_grouped_cli_lists_nested_commands(capsys) -> None:
    assert main(["provider"]) == 0
    output = capsys.readouterr().out
    assert "manifest" in output


def test_grouped_cli_rejects_unknown_command(capsys) -> None:
    assert main(["unknown"]) == 2
    assert "usage: bpt" in capsys.readouterr().err
