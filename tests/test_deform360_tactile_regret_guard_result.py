from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/deform360_tactile_regret_guard_source_v1.json"
RESULT_ROOT = (
    ROOT / "results/sota/diagnostics/deform360_tactile_regret_guard_source_v1"
)
RESULT = RESULT_ROOT / "result.json"


def _canonical_sha256(payload: dict, key: str) -> str:
    stripped = dict(payload)
    stripped.pop(key)
    blob = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tactile_source_protocol_and_inputs_are_bound() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["protocol_sha256"] == _canonical_sha256(
        protocol,
        "protocol_sha256",
    )
    assert protocol["information_boundary"]["future_tactile_read"] is False
    assert (
        protocol["information_boundary"][
            "episode_wide_tactile_normalization_used"
        ]
        is False
    )
    assert protocol["inputs"]["open27_tactile_features_sha256"] == _file_sha256(
        RESULT_ROOT / "open27_tactile_features.json"
    )
    assert protocol["inputs"]["stress12_tactile_features_sha256"] == _file_sha256(
        RESULT_ROOT / "stress12_tactile_features.json"
    )


def test_tactile_source_result_passes_only_development_gates() -> None:
    result = json.loads(RESULT.read_text())
    assert result["artifact_sha256"] == _canonical_sha256(
        result,
        "artifact_sha256",
    )
    assert result["all_advancement_gates_passed"] is True
    assert "Post-open" in result["claim_boundary"]
    assert "state of the art" in result["claim_boundary"]
    assert result["information_boundary"]["held_v8_read"] is False

    open27 = result["cross_fitted"]["open27"]
    stress = result["cross_fitted"]["stress12"]
    assert open27["object_count"] == 5
    assert open27["case_count"] == 27
    assert open27["accepted_update_count"] == 6
    assert open27["accepted_regressive_update_count"] == 0
    assert open27["metrics"]["identity"]["guarded_relative_percent"] == pytest.approx(
        -4.057769705957337
    )
    assert open27["metrics"]["chamfer"]["guarded_relative_percent"] == pytest.approx(
        -2.9386300517530106
    )
    assert stress["object_count"] == 12
    assert stress["accepted_update_count"] == 0
    assert stress["accepted_regressive_update_count"] == 0
    assert stress["metrics"]["identity"]["guarded_relative_percent"] == 0.0
    assert stress["metrics"]["chamfer"]["guarded_relative_percent"] == 0.0
