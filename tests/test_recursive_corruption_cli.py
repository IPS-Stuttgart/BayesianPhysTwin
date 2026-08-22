from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from bayesian_phystwin.cli.recursive_corruption_benchmark import main


def test_cli_writes_json_and_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"
    assert (
        main(
            [
                "--seeds",
                "0:2",
                "--conditions",
                "clean,outlier_burst",
                "--steps",
                "96",
                "--corruption-start",
                "30",
                "--corruption-length",
                "16",
                "--recovery-window",
                "24",
                "--output-json",
                str(json_path),
                "--output-csv",
                str(csv_path),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["guarded_exact_fallback_violation_count"] == 0
    assert json_path.is_file()
    assert csv_path.is_file()


def test_cli_parsers_reject_malformed_values() -> None:
    from bayesian_phystwin.cli.recursive_corruption_benchmark import (
        _parse_conditions,
        _parse_seeds,
    )

    assert _parse_seeds("1,3") == [1, 3]
    with pytest.raises(argparse.ArgumentTypeError, match="seed range"):
        _parse_seeds("0:1:2:3")
    with pytest.raises(argparse.ArgumentTypeError, match="seeds must"):
        _parse_seeds("zero")
    with pytest.raises(argparse.ArgumentTypeError, match="unknown conditions"):
        _parse_conditions("clean,unknown")


def test_cli_runs_without_output_files(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "--seeds",
                "0",
                "--conditions",
                "clean",
                "--steps",
                "96",
                "--corruption-start",
                "30",
                "--corruption-length",
                "16",
                "--recovery-window",
                "24",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["corrupted_sequence_count_per_method"] == 0
