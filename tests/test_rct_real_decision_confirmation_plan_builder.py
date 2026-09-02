from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from bayesian_phystwin._portable_contracts import content_id

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    ROOT / "scripts/development/build_rct_real_decision_confirmation_plan_v1.py"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "rct_confirmation_plan_builder",
        BUILDER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_source_result(
    path: Path, builder: ModuleType, *, passed: bool
) -> dict[str, object]:
    identity = {
        "schema": builder.SOURCE_RESULT_SCHEMA,
        "schema_version": 1,
        "source_gate": {"passed": passed},
        "source_test_opened": True,
        "confirmation_opened": False,
        "confirmation_force_fields_parsed": False,
        "held_v8_accessed": False,
        "attempt_count": 1,
        "replacement_or_retry_authorized": False,
        "target_authorized": False,
    }
    result = {**identity, "source_result_id": content_id(identity)}
    path.write_text(json.dumps(result), encoding="utf-8")
    return result


def test_confirmation_plan_builder_registers_every_custody_critical_path() -> None:
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
        "run_rct_real_decision_confirmation_v1.py"
    )


def test_confirmation_plan_builder_requires_passing_unopened_source_result(
    tmp_path: Path,
) -> None:
    builder = _module()
    path = tmp_path / "source-result.json"
    passing = _write_source_result(path, builder, passed=True)

    assert builder._load_source_result(path) == passing

    _write_source_result(path, builder, passed=False)
    with pytest.raises(ValueError, match="source gate failed"):
        builder._load_source_result(path)


def test_confirmation_plan_builder_rejects_tampered_source_identity(
    tmp_path: Path,
) -> None:
    builder = _module()
    path = tmp_path / "source-result.json"
    result = _write_source_result(path, builder, passed=True)
    result["source_result_id"] = "0" * 64
    path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        builder._load_source_result(path)


def test_confirmation_plan_source_hash_is_stable(tmp_path: Path) -> None:
    builder = _module()
    path = tmp_path / "source-result.json"
    _write_source_result(path, builder, passed=True)

    first = builder._sha256(path)
    second = hashlib.sha256(path.read_bytes()).hexdigest()

    assert first == second
