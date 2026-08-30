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
SOURCE_PLAN = (
    ROOT / "protocols/execution_requests/controlled_query_competence_source_v1.json"
)
SOURCE_RECEIPT = (
    ROOT / "evidence/controlled_query_competence_source_receipt_v1.json"
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


def test_registered_source_plan_binds_immutable_implementation() -> None:
    source_module = _module(SOURCE_RUNNER, "registered_controlled_source_runner")
    digest = hashlib.sha256(SOURCE_PLAN.read_bytes()).hexdigest()
    plan = source_module._load_plan(SOURCE_PLAN, digest)
    implementation = plan["implementation"]

    assert plan["plan_id"] == (
        "17cf7c8765bbb244de77b0ea41d6b4023622f832a0db671a9b57db4c8c9b8f13"
    )
    assert implementation["revision"] == ("f9f0d72d050efff44559e4e000d71a73170b95d8")
    assert (
        implementation["runner_sha256"]
        == hashlib.sha256(SOURCE_RUNNER.read_bytes()).hexdigest()
    )
    module_path = ROOT / implementation["module_relative_path"]
    assert (
        implementation["module_sha256"]
        == hashlib.sha256(module_path.read_bytes()).hexdigest()
    )
    assert plan["protocol"] == experiment_protocol_v1()
    assert plan["confirmation_outcomes_authorized"] is False
    assert plan["attempt_limit"] == 1
    assert plan["replacement_or_retry_authorized"] is False
    assert plan["protected_artifacts_authorized"] is False


def test_registered_source_receipt_binds_private_evidence() -> None:
    receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    identity = dict(receipt)
    receipt_id = identity.pop("receipt_id")

    assert receipt_id == content_id(identity)
    assert receipt["source_gate_passed"] is True
    assert receipt["controlled_confirmation_authorized"] is True
    assert receipt["controlled_confirmation_outcomes_opened"] is False
    assert receipt["physical_confirmation_authorized"] is False
    assert receipt["replacement_or_retry_authorized"] is False
    assert receipt["prob4d_used"] is False
    assert receipt["protected_artifacts_used"] is False
    assert receipt["plan_id"] == (
        "17cf7c8765bbb244de77b0ea41d6b4023622f832a0db671a9b57db4c8c9b8f13"
    )
    assert receipt["execution_result_id"] == (
        "1da59f6463f3b8e72badb3a177650b6a85cc2a78fc47089b94a80c16c5ccc921"
    )
    assert receipt["source_result_id"] == (
        "96bd9816dfc4ff7d5154ce5b73c8a7ecd8261eb68bad1ea21d658469fa350c97"
    )
    private = receipt["private_evidence"]
    assert private["repository"] == "FlorianPfaff/BayesianPhysTwin-Paper"
    assert private["commit"] == "834426c11bc5de17b657efb6f21b500f1f741e07"
    assert private["evidence_record_id"] == (
        "879acdf40ea00ef6677bffaf33814b7837474262a9a6ca3fa0eaa817a599a7e3"
    )


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
