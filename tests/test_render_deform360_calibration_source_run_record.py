"""Receipts consume only strictly validated Deform360 terminal records."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin.deform360_calibration_source_run_record import (
    _canonical_sha256,
)
from test_deform360_calibration_source_run_record import _build_chain, _record

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "render_deform360_calibration_source_run_record.py"
SPEC = importlib.util.spec_from_file_location("deform360_run_receipt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_record(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    record = _record(_build_chain(tmp_path))
    path = tmp_path / "execution-manifest.json"
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, record


def _args(path: Path, *, source_revision: str = "1" * 40) -> list[str]:
    return [
        "--manifest",
        str(path),
        "--source-revision",
        source_revision,
        "--workflow-run-id",
        "123",
        "--workflow-run-attempt",
        "2",
    ]


def test_valid_issue_receipt_and_summary_share_the_strict_record(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, record = _write_record(tmp_path)
    output = tmp_path / "receipt.json"

    assert MODULE.main(["issue", *_args(path), "--output", str(output)]) == 0
    body = json.loads(output.read_text(encoding="utf-8"))["body"]
    assert f"execution-record digest: `{record['record_sha256']}`" in body
    assert "prepared objects: `10`" in body
    assert "confirmation boundary verified: `True`" in body
    assert str(tmp_path) not in body
    assert "cal-sheet-0" not in body

    assert MODULE.main(["summary", *_args(path)]) == 0
    summary = capsys.readouterr().out
    assert f"Execution-record digest: `{record['record_sha256']}`" in summary
    assert "Prepared objects: `10`" not in summary
    assert "prepared objects: `10`" in summary


def test_redigested_extra_field_downgrades_to_unavailable(tmp_path: Path) -> None:
    path, record = _write_record(tmp_path)
    forged = deepcopy(record)
    forged["private_path"] = "/sensitive/object-name"
    forged["record_sha256"] = _canonical_sha256(forged)
    path.write_text(json.dumps(forged), encoding="utf-8")
    output = tmp_path / "receipt.json"

    assert MODULE.main(["issue", *_args(path), "--output", str(output)]) == 0
    body = json.loads(output.read_text(encoding="utf-8"))["body"]
    assert "execution-record-unavailable" in body
    assert "/sensitive/object-name" not in body


def test_recomputed_status_or_source_mismatch_is_not_rendered(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, record = _write_record(tmp_path)
    forged = deepcopy(record)
    forged["status"] = "failed"
    forged["record_sha256"] = _canonical_sha256(forged)
    path.write_text(json.dumps(forged), encoding="utf-8")

    assert MODULE.main(["summary", *_args(path)]) == 0
    assert "unavailable or invalid" in capsys.readouterr().out

    path, _ = _write_record(tmp_path)
    output = tmp_path / "receipt.json"
    assert MODULE.main(
        [
            "issue",
            *_args(path, source_revision="9" * 40),
            "--output",
            str(output),
        ]
    ) == 0
    body = json.loads(output.read_text(encoding="utf-8"))["body"]
    assert "execution-record-unavailable" in body


def test_duplicate_json_keys_are_not_rendered(tmp_path: Path) -> None:
    path, record = _write_record(tmp_path)
    encoded = json.dumps(record, sort_keys=True)
    path.write_text('{"schema":"forged",' + encoded[1:], encoding="utf-8")
    output = tmp_path / "receipt.json"

    assert MODULE.main(["issue", *_args(path), "--output", str(output)]) == 0
    body = json.loads(output.read_text(encoding="utf-8"))["body"]
    assert "execution-record-unavailable" in body
