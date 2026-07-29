from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_query import (
    QUERY_REPORT_FILENAME,
    validate_causal_response_query_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results"
    / "sota"
    / "diagnostics"
    / "deform360_causal_response_query_feasibility_v12_source"
)


def _canonical_sha256(payload: dict) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_v12_query_feasibility_result_is_closed_and_target_free() -> None:
    result = json.loads((RESULT_ROOT / "summary.json").read_text(encoding="utf-8"))

    assert result["result_sha256"] == _canonical_sha256(result)
    assert result["result_sha256"] == (
        "03ae0a299f24a84fcc8e4ae5d808d9bbc935e4f14d9df30017f1bf463c990bfa"
    )
    assert result["status"] == "failed"
    assert result["admitted_case_count"] == 2
    assert result["technical_failure_count"] == 4
    assert result["gate"]["minimum_admitted_case_count"] == 6
    assert result["gate"]["maximum_technical_failure_count"] == 0
    assert result["gate"]["passed"] is False
    assert all(
        value is False
        for key, value in result["information_boundary"].items()
        if key != "maximum_object_observation_frame"
    )
    assert result["information_boundary"]["maximum_object_observation_frame"] == 0


def test_v12_query_case_artifacts_remain_checksummed() -> None:
    result = json.loads((RESULT_ROOT / "summary.json").read_text(encoding="utf-8"))

    for row in result["cases"]:
        case_dir = RESULT_ROOT / row["case"]
        if row["status"] == "technical_failure":
            assert not (case_dir / QUERY_REPORT_FILENAME).exists()
            continue
        report, arrays = validate_causal_response_query_artifacts(case_dir)
        assert report["result_sha256"] == row["result_sha256"]
        assert len(arrays["entity_ids"]) == row["selected_entity_count"]
