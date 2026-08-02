from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results/sota/diagnostics/deform360_causal_support_guard_source_v1/result.json"
)


def _canonical_sha256(payload: dict[str, object]) -> str:
    stripped = dict(payload)
    stripped.pop("artifact_sha256", None)
    blob = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def test_registered_causal_support_source_v1_result() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["artifact_sha256"] == _canonical_sha256(payload)
    assert payload["artifact_sha256"] == (
        "f2b61906f7d0f9cee0cac0cc4a3a91311e15dcba8b0c92837e27b0843c0410d5"
    )
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "0c8a0fbfe70f466f4a9559fe31be19e74972822a412b51220fd5e5dd6556e7da"
    )

    cross_fitted = payload["cross_fitted"]
    combined = cross_fitted["combined"]
    open27 = cross_fitted["open27"]
    stress12 = cross_fitted["stress12"]
    assert combined["accepted_beneficial_update_count"] == 11
    assert combined["accepted_regressive_update_count"] == 0
    assert combined["admitted_object_count"] == 4
    assert cross_fitted["joint_case_wins"] == 6
    assert open27["metrics"]["identity"]["guarded_relative_percent"] == pytest.approx(
        -5.920446005945335
    )
    assert open27["metrics"]["chamfer"]["guarded_relative_percent"] == pytest.approx(
        -4.882565235037317
    )
    assert stress12["accepted_update_count"] == 0
    assert payload["all_development_advancement_checks_passed"] is False
    assert (
        payload["development_advancement_checks"][
            "at_least_five_objects_with_admission"
        ]
        is False
    )
