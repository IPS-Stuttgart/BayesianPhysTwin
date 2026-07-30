from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT / "results" / "sota" / "cloth_sim2real_online_belief_v1"
)
TARGET_RESULT = RESULT_ROOT / "target_result.json"
TARGET_TRANSFER = RESULT_ROOT / "target_transfer"
TARGET_RESULT_SHA256 = (
    "2b013b3e3214b4c8a1a6838b31fe304b6ea91df6b2e2610403ca37948feaa49a"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cloth_sim2real_target_result_is_hash_bound() -> None:
    result = json.loads(TARGET_RESULT.read_text(encoding="utf-8"))

    assert _sha256(TARGET_RESULT) == TARGET_RESULT_SHA256
    assert result["artifact_kind"] == "ClothSim2RealTargetResult"
    assert result["formal_90_split_conformal_claim"] is False
    assert result["dynamic_primary"][
        "object_balanced_symmetric_relative_improvement"
    ] == pytest.approx(0.07470992054685653)
    assert result["dynamic_primary"]["symmetric_win_count"] == 3
    assert result["quasi_static_secondary"][
        "object_balanced_symmetric_relative_improvement"
    ] == pytest.approx(-0.08133730594997833)


def test_cloth_sim2real_target_case_seals_match_aggregate() -> None:
    aggregate = json.loads(TARGET_RESULT.read_text(encoding="utf-8"))

    assert len(aggregate["result_sha256s"]) == 6
    assert len(aggregate["prediction_seal_sha256s"]) == 6
    for case_dir in TARGET_TRANSFER.iterdir():
        result_path = case_dir / "result.json"
        seal_path = case_dir / "prediction_seal.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))

        assert aggregate["result_sha256s"][case_dir.name] == _sha256(
            result_path
        )
        assert aggregate["prediction_seal_sha256s"][
            result["case_id"]
        ] == _sha256(seal_path)
        assert result["authorized_split"] == "target"
        assert result["future_outcomes_read_only_after_prediction_seal"] is True
