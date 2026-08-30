from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.controlled_query_competence_v1 import (
    experiment_protocol_v1,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUNNER = ROOT / "scripts/science/run_controlled_query_competence_source_v1.py"
CONFIRMATION_RUNNER = (
    ROOT / "scripts/science/run_controlled_query_competence_confirmation_v1.py"
)


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_plan(path: Path, schema: str) -> tuple[dict[str, object], str]:
    identity = {
        "schema": schema,
        "schema_version": 1,
        "protocol": experiment_protocol_v1(),
        "output_root": str(path.parent / "output"),
        "attempt_ledger_path": str(path.parent / "attempt.json"),
    }
    plan = {**identity, "plan_id": content_id(identity)}
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return plan, digest


def test_source_runner_cannot_import_or_name_confirmation_execution() -> None:
    source = SOURCE_RUNNER.read_text(encoding="utf-8")

    assert "run_source_stage_v1" in source
    assert "run_confirmation_stage_v1" not in source
    assert "CONFIRMATION_SEED_BASE" not in source
    assert '"confirmation_outcomes_opened": False' in source
    assert '"retry_authorized": False' in source


def test_runner_plan_identity_and_attempt_ledger_are_fail_closed(
    tmp_path: Path,
) -> None:
    source_module = _module(SOURCE_RUNNER, "controlled_source_runner")
    plan_path = tmp_path / "source-plan.json"
    plan, digest = _write_plan(plan_path, source_module.PLAN_SCHEMA)

    assert source_module._load_plan(plan_path, digest) == plan
    with pytest.raises(ValueError, match="SHA-256"):
        source_module._load_plan(plan_path, "0" * 64)

    attempt_path = tmp_path / "attempt.json"
    output_root = tmp_path / "source-output"
    source_module._consume_attempt(attempt_path, str(plan["plan_id"]), output_root)
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert attempt["attempt_consumed"] is True
    assert attempt["plan_id"] == plan["plan_id"]
    with pytest.raises(FileExistsError):
        source_module._consume_attempt(attempt_path, str(plan["plan_id"]), output_root)

    tampered = dict(plan)
    tampered["output_root"] = str(tmp_path / "different")
    plan_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tampered_digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="identity"):
        source_module._load_plan(plan_path, tampered_digest)


def test_confirmation_runner_requires_passing_unopened_source_result(
    tmp_path: Path,
) -> None:
    confirmation_module = _module(CONFIRMATION_RUNNER, "controlled_confirmation_runner")
    source_identity = {
        "schema": "bayesian-phystwin.controlled-query-competence-source-execution-result",
        "schema_version": 1,
        "source_gate_passed": True,
        "confirmation_authorized_by_source_gate": True,
        "confirmation_outcomes_opened": False,
        "source_result": {"source_result_id": "a" * 64},
    }
    execution = {**source_identity, "execution_result_id": content_id(source_identity)}
    result_path = tmp_path / "source-result.json"
    result_path.write_text(
        json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan = {
        "source_execution_result_path": str(result_path),
        "source_execution_result_sha256": hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest(),
        "source_execution_result_id": execution["execution_result_id"],
    }
    assert confirmation_module._load_source_result(plan) == execution["source_result"]

    blocked_identity = dict(source_identity)
    blocked_identity["source_gate_passed"] = False
    blocked = {
        **blocked_identity,
        "execution_result_id": content_id(blocked_identity),
    }
    result_path.write_text(
        json.dumps(blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan["source_execution_result_sha256"] = hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    plan["source_execution_result_id"] = blocked["execution_result_id"]
    with pytest.raises(ValueError, match="did not pass"):
        confirmation_module._load_source_result(plan)
