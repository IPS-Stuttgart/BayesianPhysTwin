"""Compatibility guard for the superseded geometric-v4 pipeline fixture.

The claim-bearing pipeline is covered by the exact 313-supported/11-excluded
end-to-end fixture, the adversarial roster suite, and the causal-window coverage
suite.  This file remains only to preserve historical test-suite references.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
END_TO_END = ROOT / "tests/test_deform360_joint_sparse_geometric_v4_end_to_end.py"
ROSTER = ROOT / "tests/test_deform360_joint_sparse_geometric_v4_roster.py"
CAUSAL = ROOT / "tests/test_deform360_joint_sparse_geometric_v4_causal_coverage.py"


def test_pipeline_contract_is_owned_by_exact_fail_closed_suites() -> None:
    end_to_end = END_TO_END.read_text(encoding="utf-8")
    roster = ROSTER.read_text(encoding="utf-8")
    causal = CAUSAL.read_text(encoding="utf-8")

    assert "SUPPORTED_STREAM_COUNTS" in end_to_end
    assert '"supported_stream_count": 313' in end_to_end
    assert '"support_negative_stream_count": 11' in end_to_end
    assert "test_source_chain_and_atomic_materializer_end_to_end" in end_to_end
    assert "test_duplicate_included_job_fails_closed" in roster
    assert "test_missing_supported_stream_cannot_hide_behind_aggregate_counts" in roster
    assert "test_changed_support_negative_reason_fails_closed" in roster
    assert "test_included_camera_cannot_reappear_as_excluded" in roster
    assert "test_prediction_windows_cover_every_registered_causal_frame" in causal
