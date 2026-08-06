from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

from bayesian_phystwin.tapnextpp_depth_completion import (
    TAPNextPPDepthCompletionConfig,
)


def _load_runner() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "remote"
        / "run_phystwin_tapnextpp_depth_completion.py"
    )
    spec = importlib.util.spec_from_file_location("tapnextpp_completion_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_transfer_protocol_requires_frozen_completion_config(tmp_path: Path) -> None:
    runner = _load_runner()
    protocol = {
        "protocol_id": runner.TRACKER_PROTOCOL_ID,
        "status": "locked-before-tapnextpp-prediction",
        "source_panel_protocol_id": runner.TRANSFER_PANEL_PROTOCOL_ID,
        "depth_completion_config": asdict(TAPNextPPDepthCompletionConfig()),
        "depth_completion_gates": {"minimum_supported_fraction": 0.85},
    }
    path = tmp_path / "protocol.json"
    _write_json(path, protocol)
    loaded = runner._load_protocol(path)
    assert loaded == protocol
    assert runner._depth_completion_gates(loaded) == protocol[
        "depth_completion_gates"
    ]


def test_transfer_strict_prediction_must_match_its_seal(tmp_path: Path) -> None:
    runner = _load_runner()
    strict_path = tmp_path / "tapnextpp_prediction.npz"
    strict_path.write_bytes(b"strict-carrier")
    report = {
        "protocol_id": runner.TRACKER_PROTOCOL_ID,
        "case": "transfer_case",
    }
    report["result_sha256"] = runner._canonical_sha256(report)
    report_path = tmp_path / "tapnextpp_prediction_report.json"
    _write_json(report_path, report)
    seal = {
        "protocol_id": runner.TRACKER_PROTOCOL_ID,
        "case": "transfer_case",
        "prediction_archive_sha256": runner._file_sha256(strict_path),
        "prediction_report_sha256": runner._file_sha256(report_path),
    }
    seal["result_sha256"] = runner._canonical_sha256(seal)
    seal_path = tmp_path / "tapnextpp_prediction_seal.json"
    _write_json(seal_path, seal)
    protocol = {
        "protocol_id": runner.TRACKER_PROTOCOL_ID,
        "case": "transfer_case",
    }

    provenance = runner._validate_strict_prediction_seal(
        strict_path,
        protocol,
    )
    assert provenance["strict_prediction_seal_sha256"] == runner._file_sha256(
        seal_path
    )

    strict_path.write_bytes(b"mutated")
    try:
        runner._validate_strict_prediction_seal(strict_path, protocol)
    except ValueError as error:
        assert "differs from its seal" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("mutated carrier passed its prediction seal")
