from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/development/build_rct_real_decision_source_plan_v1.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "rct_source_plan_builder", BUILDER_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_plan_builder_registers_every_custody_critical_path() -> None:
    builder = _module()

    assert set(builder.REGISTERED_PATHS) == {
        "runner",
        "method",
        "protocol_loader",
        "protocol",
        "clarification",
        "amendment_v2",
        "archive_lock",
    }
    assert builder.REGISTERED_PATHS["runner"].endswith(
        "run_rct_real_decision_source_v1.py"
    )


def test_source_plan_builder_accepts_only_content_bound_unopened_archive_lock(
    tmp_path: Path,
) -> None:
    builder = _module()
    identity = {
        "schema": builder.ARCHIVE_LOCK_SCHEMA,
        "schema_version": 1,
        "archive_integrity_verified": True,
        "force_metadata_content_opened": False,
        "confirmation_opened": False,
        "held_v8_accessed": False,
    }
    lock = {**identity, "lock_id": content_id(identity)}
    path = tmp_path / "archive-lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")

    assert builder._load_archive_lock(path) == lock

    opened_identity = {**identity, "force_metadata_content_opened": True}
    opened = {**opened_identity, "lock_id": content_id(opened_identity)}
    path.write_text(json.dumps(opened), encoding="utf-8")
    with pytest.raises(ValueError, match="opened"):
        builder._load_archive_lock(path)


def test_source_plan_builder_rejects_tampered_archive_lock_identity(
    tmp_path: Path,
) -> None:
    builder = _module()
    path = tmp_path / "archive-lock.json"
    path.write_text(
        json.dumps(
            {
                "schema": builder.ARCHIVE_LOCK_SCHEMA,
                "schema_version": 1,
                "archive_integrity_verified": True,
                "force_metadata_content_opened": False,
                "confirmation_opened": False,
                "held_v8_accessed": False,
                "lock_id": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity"):
        builder._load_archive_lock(path)
