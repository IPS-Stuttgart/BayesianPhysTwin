from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "sota"
    / "diagnostics"
    / "deform360_direct_depth_sentinel_v8"
    / "059-shoe-ep0000"
    / "hidden_source_audit.json"
)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_frozen_v8_hidden_source_result_is_closed() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))

    assert payload["result_sha256"] == _canonical_sha256(payload)
    assert payload["advancement_gate_passed"] is False
    assert payload["decision"] == "close-direct-depth-sentinel-v8"
    assert payload["query_identity_count"] == 12
    assert payload["hidden_identity_count"] == 440
    assert payload["information_boundary"] == {
        "already_open_source_target_read": True,
        "fresh_target_read": False,
        "held_v8_artifact_read": False,
        "query_identities_excluded_from_all_scores": True,
        "v1_sealed_target_cohort_read": False,
    }
    comparison = payload["comparison"]["persistence"]
    assert comparison["joint_improvement"] is False
    assert comparison["relative_identity_rmse_change"] == pytest.approx(
        13.977389699147581
    )
    assert comparison["relative_symmetric_chamfer_change"] == pytest.approx(
        30.761504208724727
    )
