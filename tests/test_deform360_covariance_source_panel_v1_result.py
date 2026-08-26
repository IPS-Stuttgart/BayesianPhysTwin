from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/sota/diagnostics/deform360_covariance_source_panel_v1"


def _load(name: str) -> dict[str, object]:
    value = json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(name: str) -> str:
    return hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()


def test_covariance_source_result_is_terminal_and_content_bound() -> None:
    summary = _load("summary.json")
    result_id = summary.pop("result_id")
    assert result_id == content_id(summary)
    assert summary["status"] == "source-technical-negative"

    decision = summary["decision"]
    assert isinstance(decision, dict)
    assert decision == {
        "confirmation_authorized": False,
        "confirmation_run": False,
        "metric_source_gate_evaluated": False,
        "retry_authorized": False,
        "source_panel_complete": False,
        "study_terminal_under_frozen_protocol": True,
    }

    failure = summary["failure"]
    assert isinstance(failure, dict)
    assert failure["prediction_record_count"] == 0
    assert failure["required_prediction_record_count"] == 100
    assert (
        failure["exception_message"]
        == "metric gauge lacks eight independent causal clusters"
    )


def test_covariance_source_result_rehashes_preserved_receipts() -> None:
    summary = _load("summary.json")
    producer = summary["producer"]
    assert isinstance(producer, dict)
    assert producer["receipt_file_sha256"] == _sha256("technical-receipt.json")
    assert producer["attempt_file_sha256"] == _sha256("attempt.json")
    assert producer["log_file_sha256"] == _sha256("producer.log")

    receipt = _load("technical-receipt.json")
    receipt_id = receipt.pop("receipt_id")
    assert receipt_id == content_id(receipt)
    assert receipt["complete_barrier"] is False
    assert receipt["prediction_record_count"] == 0
    assert receipt["source_suffix_scoring_authorized"] is False
    assert receipt["confirmation_prediction_authorized"] is False
    assert receipt["confirmation_outcome_opening_authorized"] is False

    attempt = _load("attempt.json")
    assert attempt == {
        "implementation_revision": ("d772b8ba84e52b99beb22e1aab2a37d766abab77"),
        "run_id": "33012751418",
        "run_attempt": "1",
        "command": "/bpt-produce-covariance-source-v1",
        "consumed": True,
    }

    expected_sums = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    }
    assert expected_sums == {
        name: _sha256(name)
        for name in (
            "attempt.json",
            "producer.log",
            "summary.json",
            "technical-receipt.json",
        )
    }
