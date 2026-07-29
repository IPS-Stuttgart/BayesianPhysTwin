from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    INFLATED_FALLBACK_ARM,
    STRICT_ARM,
    validate_adaptive_causal_response_query_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results"
    / "sota"
    / "diagnostics"
    / "deform360_causal_response_adaptive_query_v13_source"
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


def test_v13_adaptive_source_result_passes_only_its_carrier_gate() -> None:
    summary_path = RESULT_ROOT / "summary.json"
    result = json.loads(summary_path.read_text(encoding="utf-8"))

    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == (
        "851a034984e9661a1e18097f514bc780967d5254afbd6ce1cf20fc1b086ad1b0"
    )
    assert result["result_sha256"] == _canonical_sha256(result)
    assert result["result_sha256"] == (
        "63a9ffde07de4e0378605b8574a8d4a8acfe6a382c65df2be185c47f6276239c"
    )
    assert result["status"] == "passed"
    assert result["case_count"] == 8
    assert result["admitted_case_count"] == 6
    assert result["strict_admitted_case_count"] == 2
    assert result["fallback_admitted_case_count"] == 4
    assert result["technical_failure_count"] == 0
    assert result["gate"]["passed"] is True
    assert all(
        value is False
        for key, value in result["information_boundary"].items()
        if key != "maximum_object_observation_frame"
    )
    assert result["information_boundary"]["maximum_object_observation_frame"] == 0


def test_v13_adaptive_case_artifacts_remain_checksummed() -> None:
    result = json.loads((RESULT_ROOT / "summary.json").read_text(encoding="utf-8"))
    strict = fallback = admitted = 0

    for row in result["cases"]:
        report, arrays = validate_adaptive_causal_response_query_artifacts(
            RESULT_ROOT / row["case"]
        )
        assert report["result_sha256"] == row["result_sha256"]
        assert len(arrays["entity_ids"]) == row["selected_entity_count"]
        assert report["information_boundary"]["future_metric_read"] is False
        assert report["information_boundary"]["state_update_constructed"] is False
        if row["admitted"]:
            admitted += 1
            strict += row["arm"] == STRICT_ARM
            fallback += row["arm"] == INFLATED_FALLBACK_ARM

    assert admitted == 6
    assert strict == 2
    assert fallback == 4
