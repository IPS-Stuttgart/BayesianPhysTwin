from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import bayesian_phystwin.cli.probabilistic_scoring as cli_module
from bayesian_phystwin.cli.probabilistic_scoring import main
from bayesian_phystwin.decisive_evidence import parse_decisive_evidence
from bayesian_phystwin.probabilistic_scoring import (
    ENERGY_SCORE,
    PROBABILISTIC_SCORE_INPUT_CONTRACT,
)


def _payload() -> dict[str, object]:
    def arm(method: str, offset: float, accepted: bool) -> dict[str, object]:
        return {
            "method": method,
            "accepted": accepted,
            "risk_score": abs(offset),
            "samples": [[offset], [offset + 0.1]],
        }

    return {
        "contract": PROBABILISTIC_SCORE_INPUT_CONTRACT,
        "schema_version": 1,
        "protocol_id": "cli-score-test-v1",
        "statistical_unit": "session",
        "claim_boundary": "test fixture only",
        "fallback_method": "physical_fallback",
        "reference_method": "last_residual",
        "score_configuration": {
            "score_names": [ENERGY_SCORE],
            "energy_beta": 1.0,
        },
        "units": [
            {
                "unit_id": "session-1",
                "group_id": "session-1",
                "horizon": 1,
                "observation": [0.0],
                "predictions": [
                    arm("bayesian_full_guarded", 0.0, False),
                    arm("last_residual", 0.1, True),
                    arm("physical_fallback", 1.0, True),
                ],
            }
        ],
    }


def test_cli_writes_verified_report_and_decisive_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    evidence = tmp_path / "evidence.json"
    assert main([str(source), str(report), "--evidence-json", str(evidence)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "written"
    assert summary["score_names"] == [ENERGY_SCORE]

    report_value = json.loads(report.read_text(encoding="utf-8"))
    assert (
        report_value["input_artifact"]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert len(report_value["report_id"]) == 64
    assert len(report_value["status_sha256"]) == 64

    evidence_value = json.loads(evidence.read_text(encoding="utf-8"))
    parsed = parse_decisive_evidence(evidence_value)
    assert len(parsed.records) == 3
    rejected = next(
        record for record in parsed.records if record.method == "bayesian_full_guarded"
    )
    assert rejected.deployed_loss == rejected.fallback_loss
    assert evidence_value["source_score_report_id"] == report_value["report_id"]

    with pytest.raises(FileExistsError):
        main([str(source), str(report), "--evidence-json", str(evidence)])


def test_cli_rejects_duplicate_keys_and_input_over_budget(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n')
    with pytest.raises(ValueError, match="duplicate key"):
        main([str(duplicate), str(tmp_path / "report.json")])

    source = tmp_path / "input.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(ValueError, match="byte budget"):
        main(
            [
                str(source),
                str(tmp_path / "bounded-report.json"),
                "--maximum-input-bytes",
                "1",
            ]
        )


def test_cli_rejects_symlinked_input(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    linked = tmp_path / "linked.json"
    try:
        linked.symlink_to(source)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="must not contain symlinks"):
        main([str(linked), str(tmp_path / "report.json")])


def test_cli_rejects_nonfinite_invalid_and_nonobject_json(tmp_path: Path) -> None:
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite constant"):
        main([str(nonfinite), str(tmp_path / "nonfinite-report.json")])

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        main([str(invalid), str(tmp_path / "invalid-report.json")])

    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a JSON object"):
        main([str(sequence), str(tmp_path / "sequence-report.json")])

    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        main([str(binary), str(tmp_path / "binary-report.json")])


def test_cli_rejects_missing_directory_and_invalid_byte_limit(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="unreadable"):
        main([str(missing), str(tmp_path / "missing-report.json")])

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="ordinary file"):
        main([str(directory), str(tmp_path / "directory-report.json")])

    source = tmp_path / "input.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(ValueError, match="must be positive"):
        main(
            [
                str(source),
                str(tmp_path / "zero-report.json"),
                "--maximum-input-bytes",
                "0",
            ]
        )


def test_cli_overwrite_replaces_complete_outputs(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    report = tmp_path / "report.json"
    evidence = tmp_path / "evidence.json"
    assert main([str(source), str(report), "--evidence-json", str(evidence)]) == 0
    first_report = report.read_bytes()
    assert (
        main(
            [
                str(source),
                str(report),
                "--evidence-json",
                str(evidence),
                "--overwrite",
            ]
        )
        == 0
    )
    assert report.read_bytes() == first_report


def test_strict_loader_rejects_nonnumeric_byte_budget(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(TypeError, match="genuine integer"):
        cli_module._load_input(source, maximum_bytes=True)


def test_strict_loader_detects_open_and_file_identity_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")

    original_open = cli_module.os.open
    monkeypatch.setattr(
        cli_module.os,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("blocked")),
    )
    with pytest.raises(ValueError, match="unreadable"):
        cli_module._load_input(source, maximum_bytes=1_000_000)
    monkeypatch.setattr(cli_module.os, "open", original_open)

    original_isreg = cli_module.stat.S_ISREG
    calls = 0

    def changing_isreg(mode: int) -> bool:
        nonlocal calls
        calls += 1
        return calls == 1 and original_isreg(mode)

    monkeypatch.setattr(cli_module.stat, "S_ISREG", changing_isreg)
    with pytest.raises(ValueError, match="ordinary file"):
        cli_module._load_input(source, maximum_bytes=1_000_000)
    monkeypatch.setattr(cli_module.stat, "S_ISREG", original_isreg)

    original_fstat = cli_module.os.fstat

    def changed_inode(descriptor: int) -> object:
        metadata = original_fstat(descriptor)
        return SimpleNamespace(
            st_mode=metadata.st_mode,
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino + 1,
        )

    monkeypatch.setattr(cli_module.os, "fstat", changed_inode)
    with pytest.raises(ValueError, match="changed while it was opened"):
        cli_module._load_input(source, maximum_bytes=1_000_000)
    monkeypatch.setattr(cli_module.os, "fstat", original_fstat)


def test_strict_loader_detects_post_read_path_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    original_lstat = cli_module.os.lstat
    calls = 0

    def disappearing_lstat(path: object) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("gone")
        return original_lstat(path)

    monkeypatch.setattr(cli_module.os, "lstat", disappearing_lstat)
    with pytest.raises(ValueError, match="changed while it was read"):
        cli_module._load_input(source, maximum_bytes=1_000_000)
    monkeypatch.setattr(cli_module.os, "lstat", original_lstat)

    original_fstat = cli_module.os.fstat
    fstat_calls = 0

    def changed_timestamp(descriptor: int) -> object:
        nonlocal fstat_calls
        fstat_calls += 1
        metadata = original_fstat(descriptor)
        if fstat_calls == 1:
            return metadata
        return SimpleNamespace(
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_size=metadata.st_size,
            st_mtime_ns=metadata.st_mtime_ns + 1,
            st_ctime_ns=metadata.st_ctime_ns,
        )

    monkeypatch.setattr(cli_module.os, "fstat", changed_timestamp)
    with pytest.raises(ValueError, match="changed while it was read"):
        cli_module._load_input(source, maximum_bytes=1_000_000)


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("aggregate", [], "aggregate changed type"),
        ("score_configuration", [], "configuration changed type"),
        ("score_names", "energy_score", "names changed type"),
        ("methods", "method", "methods changed type"),
        ("unit_score_rows", "rows", "rows changed type"),
    ],
)
def test_cli_rejects_internal_report_type_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    source = tmp_path / f"{field}-input.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    report_path = tmp_path / f"{field}-report.json"
    report: dict[str, object] = {
        "report_id": "a" * 64,
        "protocol_id": "test",
        "aggregate": {},
        "score_configuration": {"score_names": [ENERGY_SCORE]},
        "methods": ["physical_fallback"],
        "unit_score_rows": [],
    }
    if field == "score_names":
        report["score_configuration"] = {"score_names": invalid_value}
    else:
        report[field] = invalid_value
    monkeypatch.setattr(
        cli_module,
        "score_probabilistic_bundle",
        lambda payload: report,
    )
    with pytest.raises(AssertionError, match=message):
        cli_module.main([str(source), str(report_path)])
