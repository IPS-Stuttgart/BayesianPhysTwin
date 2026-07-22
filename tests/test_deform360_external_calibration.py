from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causal4d_public.deform360_external_calibration import (
    EXTERNAL_CALIBRATION_CASE_NAME,
    EXTERNAL_CALIBRATION_EPISODE_ID,
    EXTERNAL_CALIBRATION_OBJECT_ID,
    authorize_external_held_calibration,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_lock(root: Path) -> Path:
    root.mkdir()
    lock = root / "calibration-lock.json"
    artifact = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldOnlineBeliefLock",
        "protocol_id": "deform360-held-online-belief-v8",
        "held_root": str(root),
        "stage": "calibration",
        "confirmation_access_authorized": False,
        "calibration_case_whitelist": [EXTERNAL_CALIBRATION_CASE_NAME],
        "information_boundary": {
            "target_reconstruction_before_barrier_one_permitted": False,
            "future_target_read_before_barrier_two_permitted": False,
            "confirmation_before_calibration_go_permitted": False,
        },
        "freshness_and_reuse": {
            "all_predictions_must_be_fresh_v8_outputs": True,
            "v7_prediction_artifacts_reused": False,
        },
    }
    artifact["artifact_sha256"] = hashlib.sha256(
        _canonical_bytes(artifact)
    ).hexdigest()
    lock.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lock.chmod(0o400)
    return lock


def test_exact_external_case_is_authorized_without_target_access(
    tmp_path: Path,
) -> None:
    lock = _write_lock(tmp_path / "held-v8")
    authorization = authorize_external_held_calibration(
        lock,
        object_id=EXTERNAL_CALIBRATION_OBJECT_ID,
        episode_id=EXTERNAL_CALIBRATION_EPISODE_ID,
    )
    assert authorization["case_name"] == EXTERNAL_CALIBRATION_CASE_NAME
    assert authorization["phase"] == "calibration"
    assert authorization["target_access"] is False


def test_external_authorization_rejects_other_cases_and_unsealed_locks(
    tmp_path: Path,
) -> None:
    lock = _write_lock(tmp_path / "held-v8")
    with pytest.raises(ValueError, match="not the frozen external calibration case"):
        authorize_external_held_calibration(
            lock,
            object_id=EXTERNAL_CALIBRATION_OBJECT_ID,
            episode_id=4,
        )
    lock.chmod(0o600)
    with pytest.raises(ValueError, match="not sealed mode 0400"):
        authorize_external_held_calibration(
            lock,
            object_id=EXTERNAL_CALIBRATION_OBJECT_ID,
            episode_id=EXTERNAL_CALIBRATION_EPISODE_ID,
        )
