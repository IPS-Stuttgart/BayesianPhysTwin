from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin.deform360_active_query_feasibility import (
    validate_active_query_feasibility_artifacts,
)
from bayesian_phystwin.observation_belief import file_sha256


def test_v10_source_gate_is_hash_bound_and_failed_without_tracker_access() -> None:
    root = Path(__file__).resolve().parents[1]
    result_root = (
        root
        / "results/sota/diagnostics/"
        "deform360_active_query_feasibility_source_v10"
    )
    summary_path = result_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    unsigned = dict(summary)
    expected_digest = unsigned.pop("summary_sha256")
    observed_digest = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    assert observed_digest == expected_digest
    assert summary["status"] == "source_gate_failed"
    assert summary["decision"] == "do_not_run_tracker_stage"
    assert summary["gate"]["admitted_case_count"] == 4
    assert summary["gate"]["minimum_admitted_case_count"] == 6
    assert summary["gate"]["passed"] is False
    assert summary["information_boundary"]["tracker_output_read"] is False
    assert (
        summary["information_boundary"]["candidate_state_update_constructed"]
        is False
    )
    assert summary["information_boundary"]["future_identity_or_metric_read"] is False
    assert summary["information_boundary"]["held_v8_read"] is False

    for record in summary["cases"]:
        case_dir = result_root / record["case"]
        report, _ = validate_active_query_feasibility_artifacts(case_dir)
        assert report["status"] == record["status"]
        assert report["result_sha256"] == record["result_sha256"]
        assert (
            file_sha256(case_dir / "active_query_feasibility.json")
            == record["report_file_sha256"]
        )
