"""Narrow authorization for a held calibration case outside the dense panel.

The reusable-twin builder was originally restricted to the five-object dense
panel.  A later held protocol adds one calibration-only object while retaining
the same numerical method.  This module lets that one case reuse the builder
without weakening the default panel authorization or granting target access.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


HELD_PROTOCOL_ID = "deform360-held-online-belief-v8"
HELD_LOCK_KIND = "Deform360HeldOnlineBeliefLock"
EXTERNAL_CALIBRATION_CASE_NAME = "072-cotton-clohesline-ep0003"
EXTERNAL_CALIBRATION_OBJECT_ID = "072-cotton-clohesline"
EXTERNAL_CALIBRATION_EPISODE_ID = 3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authorize_external_held_calibration(
    lock_path: str | Path,
    *,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    """Authorize only the frozen v8 replacement-source calibration case."""

    path = Path(os.path.abspath(os.fspath(lock_path)))
    observed = os.lstat(path)
    _require(not stat.S_ISLNK(observed.st_mode), "held calibration lock is linked")
    _require(stat.S_ISREG(observed.st_mode), "held calibration lock is not a file")
    _require(path.resolve() == path, "held calibration lock has a linked ancestor")
    _require(
        stat.S_IMODE(observed.st_mode) == 0o400,
        "held calibration lock is not sealed mode 0400",
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(artifact, dict), "held calibration lock is not an object")
    _require(
        artifact.get("schema_version") == 1
        and artifact.get("artifact_kind") == HELD_LOCK_KIND
        and artifact.get("protocol_id") == HELD_PROTOCOL_ID
        and artifact.get("stage") == "calibration"
        and artifact.get("confirmation_access_authorized") is False,
        "unsupported held calibration lock",
    )
    _require(
        artifact.get("artifact_sha256") == _artifact_sha256(artifact),
        "held calibration lock checksum changed",
    )
    held_root = Path(str(artifact.get("held_root", "")))
    _require(
        held_root.is_absolute()
        and path == held_root / "calibration-lock.json",
        "held calibration lock is outside its bound root",
    )
    case_name = f"{object_id}-ep{int(episode_id):04d}"
    _require(
        object_id == EXTERNAL_CALIBRATION_OBJECT_ID
        and int(episode_id) == EXTERNAL_CALIBRATION_EPISODE_ID
        and case_name == EXTERNAL_CALIBRATION_CASE_NAME,
        "case is not the frozen external calibration case",
    )
    whitelist = artifact.get("calibration_case_whitelist")
    _require(
        isinstance(whitelist, list)
        and whitelist.count(EXTERNAL_CALIBRATION_CASE_NAME) == 1,
        "external calibration case is not uniquely locked",
    )
    boundary = artifact.get("information_boundary", {})
    freshness = artifact.get("freshness_and_reuse", {})
    _require(
        boundary.get("target_reconstruction_before_barrier_one_permitted") is False
        and boundary.get("future_target_read_before_barrier_two_permitted") is False
        and boundary.get("confirmation_before_calibration_go_permitted") is False,
        "held calibration information boundary changed",
    )
    _require(
        freshness.get("all_predictions_must_be_fresh_v8_outputs") is True
        and freshness.get("v7_prediction_artifacts_reused") is False,
        "held calibration freshness contract changed",
    )
    return {
        "authorization_kind": "Deform360ExternalHeldCalibrationAuthorization",
        "held_protocol_id": HELD_PROTOCOL_ID,
        "held_lock_path": str(path),
        "held_lock_file_sha256": _file_sha256(path),
        "held_lock_artifact_sha256": artifact["artifact_sha256"],
        "case_name": case_name,
        "object_id": object_id,
        "episode_id": int(episode_id),
        "phase": "calibration",
        "target_access": False,
    }


__all__ = [
    "EXTERNAL_CALIBRATION_CASE_NAME",
    "EXTERNAL_CALIBRATION_EPISODE_ID",
    "EXTERNAL_CALIBRATION_OBJECT_ID",
    "HELD_LOCK_KIND",
    "HELD_PROTOCOL_ID",
    "authorize_external_held_calibration",
]
