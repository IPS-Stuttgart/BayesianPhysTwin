from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_horizon_conditioned_discrepancy_tournament import _payload

from bayesian_phystwin.cli.discrepancy_candidate_tournament import main


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_cli_writes_positive_report_and_rejects_replacement(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "input.json"
    output = tmp_path / "report.json"
    _write(source, _payload())

    assert main([str(source), str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text())
    assert summary["selected_candidate"] == "structured"
    assert report["source_gate_passed"] is True
    assert report["input_artifact"]["sha256"]
    assert len(report["status_sha256"]) == 64

    with pytest.raises(FileExistsError):
        main([str(source), str(output)])
    assert main([str(source), str(output), "--overwrite"]) == 0
    capsys.readouterr()


def test_cli_returns_three_for_valid_reference_retention(
    tmp_path: Path, capsys
) -> None:
    payload = _payload()
    records = payload["records"]
    assert isinstance(records, list)
    for record in records:
        if record["candidate_id"] in {"dynamic", "structured"}:
            record["point_loss"] = 9.0
            record["deployed_point_loss"] = 9.0
            record["proper_score"] = 4.5
            record["deployed_proper_score"] = 4.5
    source = tmp_path / "input.json"
    output = tmp_path / "report.json"
    _write(source, payload)

    assert main([str(source), str(output)]) == 3
    summary = json.loads(capsys.readouterr().out)
    assert summary["decision"] == "retain-reference-candidate"
    assert json.loads(output.read_text())["source_gate_passed"] is False


def test_cli_rejects_duplicate_keys_and_byte_budget(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    output = tmp_path / "report.json"
    source.write_text('{"contract":"x","contract":"y"}\n')
    with pytest.raises(ValueError, match="duplicate key"):
        main([str(source), str(output)])


def test_cli_module_entrypoint(tmp_path: Path, monkeypatch) -> None:
    import runpy
    import sys

    from bayesian_phystwin.cli import discrepancy_candidate_tournament as cli

    source = tmp_path / "input.json"
    output = tmp_path / "report.json"
    _write(source, _payload())
    monkeypatch.setattr(sys, "argv", ["bpt-tournament", str(source), str(output)])

    with pytest.raises(SystemExit) as error:
        assert cli.__file__ is not None
        runpy.run_path(cli.__file__, run_name="__main__")

    assert error.value.code == 0
    assert json.loads(output.read_text())["source_gate_passed"] is True
