"""Tests for runner-resident Deform360 calibration cache staging."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path("scripts/ci/stage_deform360_local_calibration_cache.py")
SPEC = importlib.util.spec_from_file_location("_stage_deform360_local_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_plan(path: Path, records: list[dict[str, object]]) -> Path:
    payload: dict[str, object] = {
        "schema": MODULE.PLAN_SCHEMA,
        "schema_version": 1,
        "protocol_id": "fixture",
        "dataset_revision": "f804696d7a133908c7497ffdab43819d879b5cbc",
        "objects": [
            {
                "object_id": "001-fixture",
                "status": "planned",
                "selected_files": records,
            }
        ],
        "information_boundary": {
            "calibration_payloads_opened": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
        },
    }
    payload["plan_sha256"] = MODULE._canonical_sha256(
        payload,
        digest_key="plan_sha256",
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_stages_verified_files_by_hardlink_and_leaves_missing_for_downloader(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official"
    destination = tmp_path / "calibration"
    available = source / "raw" / "001-fixture" / "metadata.json"
    available.parent.mkdir(parents=True)
    available.write_bytes(b"fixture-metadata")
    missing = source / "raw" / "001-fixture" / "camera.mp4"
    plan = _write_plan(
        tmp_path / "plan.json",
        [
            {
                "path": available.relative_to(source).as_posix(),
                "size": available.stat().st_size,
                "lfs_sha256": _sha256(available),
            },
            {
                "path": missing.relative_to(source).as_posix(),
                "size": 123,
                "lfs_sha256": "0" * 64,
            },
        ],
    )

    result = MODULE.stage_local_calibration_cache(
        plan_path=plan,
        source_root=source,
        destination_root=destination,
    )

    staged = destination / available.relative_to(source)
    assert staged.read_bytes() == available.read_bytes()
    assert staged.stat().st_ino == available.stat().st_ino
    assert result["planned_file_count"] == 2
    assert result["hardlinked_file_count"] == 1
    assert result["missing_file_count"] == 1
    assert result["missing_paths"] == [missing.relative_to(source).as_posix()]
    assert result["information_boundary"]["adaptive_confirmation_root_accessed"] is False


def test_reuses_existing_verified_destination(tmp_path: Path) -> None:
    source = tmp_path / "official"
    destination = tmp_path / "calibration"
    relative = Path("raw/001-fixture/metadata.json")
    source_file = source / relative
    destination_file = destination / relative
    source_file.parent.mkdir(parents=True)
    destination_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"same")
    destination_file.write_bytes(b"same")
    plan = _write_plan(
        tmp_path / "plan.json",
        [
            {
                "path": relative.as_posix(),
                "size": 4,
                "lfs_sha256": _sha256(source_file),
            }
        ],
    )

    result = MODULE.stage_local_calibration_cache(
        plan_path=plan,
        source_root=source,
        destination_root=destination,
    )

    assert result["reused_file_count"] == 1
    assert result["hardlinked_file_count"] == 0


def test_rejects_mismatched_local_bytes(tmp_path: Path) -> None:
    source = tmp_path / "official"
    destination = tmp_path / "calibration"
    relative = Path("raw/001-fixture/payload.bin")
    source_file = source / relative
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"wrong")
    plan = _write_plan(
        tmp_path / "plan.json",
        [
            {
                "path": relative.as_posix(),
                "size": source_file.stat().st_size,
                "lfs_sha256": hashlib.sha256(b"expected").hexdigest(),
            }
        ],
    )

    with pytest.raises(ValueError, match="LFS digest mismatch"):
        MODULE.stage_local_calibration_cache(
            plan_path=plan,
            source_root=source,
            destination_root=destination,
        )


def test_refuses_destination_inside_immutable_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "official"
    source.mkdir()
    plan = _write_plan(
        tmp_path / "plan.json",
        [{"path": "raw/001-fixture/metadata.json", "size": 1}],
    )

    with pytest.raises(ValueError, match="must not be inside"):
        MODULE.stage_local_calibration_cache(
            plan_path=plan,
            source_root=source,
            destination_root=source / "derived",
        )
