"""Contracts and common validation for Deform360 calibration-source opening."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_calibration_execution import (
    load_deform360_stage0_selection,
)

PROTOCOL_SCHEMA = (
    "bayesian-phystwin/deform360-official-hub-calibration-source-v1"
)
PROTOCOL_VERSION = 1
PLAN_SCHEMA = "bayesian-phystwin/deform360-calibration-source-plan-v1"
DOWNLOAD_SCHEMA = "bayesian-phystwin/deform360-calibration-download-v1"
RESULT_SCHEMA = "bayesian-phystwin/deform360-calibration-source-result-v1"
PROTOCOL_ID = "deform360-official-hub-calibration-source-v1"
PARENT_PROTOCOL_ID = "deform360-official-hub-visuotactile-v1"
DATASET_REPOSITORY = "brownu/deform360"
DATASET_REVISION = "f804696d7a133908c7497ffdab43819d879b5cbc"
PROCESSING_REPOSITORY = "lhy0807/deform360"
PROCESSING_REVISION = "d8522a4403b766aeb387510c04e89032a56fdf35"
VISUAL_PROVIDER_LOCK_ID = (
    "b04341bf8c5e9f5250b87e35f1428bd21d5b79507e4e0c27ec24226e244befaf"
)
MINIMUM_SUPPORTED_OBJECTS = 8
MINIMUM_SUPPORTED_PER_STRATUM = 4
MINIMUM_CAMERA_STREAMS = 8
CAMERA_RE = re.compile(r"^brics-odroid-\d+_cam\d+$")
TACTILE_RE = re.compile(r"^brics-odroid_tactile[^/]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, order=True)
class CalibrationUnit:
    object_id: str
    episode_id: int
    stratum: str
    metadata_path: str
    metadata_sha256: str


@dataclass(frozen=True)
class RepositoryFile:
    path: str
    size: int | None
    blob_id: str | None
    lfs_sha256: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size": self.size,
            "blob_id": self.blob_id,
            "lfs_sha256": self.lfs_sha256,
        }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def canonical_sha256(value: Mapping[str, Any], *, digest_key: str) -> str:
    payload = dict(value)
    payload.pop(digest_key, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_revision(repository: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_revision(repository: Path, expected: str) -> None:
    revision = git_revision(repository)
    require(revision == expected, f"repository revision changed: {repository}")
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(not status.strip(), f"repository is dirty: {repository}")


def load_protocol(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require(value.get("schema") == PROTOCOL_SCHEMA, "protocol schema changed")
    require(
        value.get("schema_version") == PROTOCOL_VERSION,
        "protocol version changed",
    )
    require(value.get("protocol_id") == PROTOCOL_ID, "protocol_id changed")
    require(
        value.get("parent_protocol_id") == PARENT_PROTOCOL_ID,
        "parent protocol changed",
    )
    expected = value.get("protocol_sha256")
    require(
        isinstance(expected, str) and SHA256_RE.fullmatch(expected) is not None,
        "bad protocol digest",
    )
    require(
        expected == canonical_sha256(value, digest_key="protocol_sha256"),
        "protocol digest changed",
    )
    return value


def load_units(
    selection_path: Path,
) -> tuple[tuple[CalibrationUnit, ...], tuple[str, ...]]:
    stage0 = load_deform360_stage0_selection(selection_path)
    require(stage0.protocol_id == PARENT_PROTOCOL_ID, "Stage-0 protocol changed")
    require(stage0.dataset_revision == DATASET_REVISION, "dataset revision changed")
    require(
        stage0.processing_revision == PROCESSING_REVISION,
        "processing revision changed",
    )
    units = tuple(
        CalibrationUnit(
            object_id=unit.object_id,
            episode_id=unit.episode_id,
            stratum=unit.stratum,
            metadata_path=unit.metadata_path,
            metadata_sha256=unit.metadata_sha256,
        )
        for unit in stage0.calibration_units
    )
    confirmations = tuple(unit.object_id for unit in stage0.confirmation_units)
    require(len(units) == 10, "calibration cohort must contain ten objects")
    require(
        len(confirmations) == 12,
        "confirmation cohort must contain twelve objects",
    )
    require(
        not {unit.object_id for unit in units} & set(confirmations),
        "calibration and confirmation cohorts overlap",
    )
    return units, confirmations


def validate_provider_lock(path: Path) -> dict[str, Any]:
    value = load_json(path)
    require(
        value.get("artifact_id") == VISUAL_PROVIDER_LOCK_ID,
        "provider lock changed",
    )
    require(
        value.get("protocol_id") == PARENT_PROTOCOL_ID,
        "provider protocol changed",
    )
    require(
        value.get("selected_raw_payloads_opened") is False,
        "provider opened payloads",
    )
    require(
        value.get("target_outcomes_used") is False,
        "provider used target outcomes",
    )
    return value


def summary_gate(
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    status: str,
) -> dict[str, Any]:
    supported = [row for row in rows if row.get("status") == status]
    by_stratum = {
        stratum: sum(row.get("stratum") == stratum for row in supported)
        for stratum in ("sheet", "volumetric")
    }
    passed = (
        len(supported) >= MINIMUM_SUPPORTED_OBJECTS
        and all(
            count >= MINIMUM_SUPPORTED_PER_STRATUM
            for count in by_stratum.values()
        )
    )
    return {
        "supported_object_count": len(supported),
        "supported_by_stratum": by_stratum,
        "minimum_supported_objects": MINIMUM_SUPPORTED_OBJECTS,
        "minimum_supported_per_stratum": MINIMUM_SUPPORTED_PER_STRATUM,
        "support_passed": passed,
    }
