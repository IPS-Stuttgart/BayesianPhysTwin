from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.rct_real_decision import discover_rct_material_ids
from bayesian_phystwin.rct_real_decision_protocol import CONFIRMATION_MATERIALS

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/science/run_rct_real_decision_confirmation_v1.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rct_confirmation_runner", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_plan(path: Path, runner: ModuleType) -> tuple[dict[str, object], str]:
    identity = {
        "schema": runner.PLAN_SCHEMA,
        "schema_version": runner.PLAN_VERSION,
        "output_root": str(path.parent / "confirmation-output"),
        "attempt_ledger_path": str(path.parent / "confirmation-attempt.json"),
        "attempt_limit": 1,
        "target_authorized": True,
        "source_gate_passed": True,
        "replacement_or_retry_authorized": False,
        "held_v8_access_authorized": False,
    }
    plan = {**identity, "plan_id": content_id(identity)}
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan, hashlib.sha256(path.read_bytes()).hexdigest()


def test_confirmation_plan_requires_every_authorization_boundary(tmp_path: Path) -> None:
    runner = _module()
    path = tmp_path / "plan.json"
    plan, digest = _write_plan(path, runner)

    assert runner._load_plan(path, digest) == plan
    for key, blocked_value in (
        ("target_authorized", False),
        ("source_gate_passed", False),
        ("replacement_or_retry_authorized", True),
        ("held_v8_access_authorized", True),
    ):
        identity = {name: value for name, value in plan.items() if name != "plan_id"}
        identity[key] = blocked_value
        blocked = {**identity, "plan_id": content_id(identity)}
        path.write_text(
            json.dumps(blocked, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        blocked_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ValueError):
            runner._load_plan(path, blocked_digest)


def test_confirmation_attempt_ledger_is_write_once(tmp_path: Path) -> None:
    runner = _module()
    attempt = tmp_path / "attempt.json"
    output = tmp_path / "confirmation-output"

    runner._consume_attempt(attempt, "plan-id", output)

    assert json.loads(attempt.read_text(encoding="utf-8"))["attempt_consumed"] is True
    with pytest.raises(FileExistsError):
        runner._consume_attempt(attempt, "plan-id", output)


def test_confirmation_filter_admits_only_registered_held_materials(
    tmp_path: Path,
) -> None:
    runner = _module()
    lines = ["material_id,position,sensor,z_frame,raw_fz\n"]
    for material_id in CONFIRMATION_MATERIALS:
        lines.append(f"material_{material_id},3,3,1.2,-0.2\n")
    lines.append("material_999999,SOURCE_SECRET,SOURCE_SECRET,SOURCE_SECRET,SOURCE_SECRET\n")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("force_metadata.csv", "".join(lines))
    filtered = tmp_path / "confirmation-only.csv"

    custody = runner._write_confirmation_only_force_csv(
        archive,
        "force_metadata.csv",
        filtered,
    )

    filtered_text = filtered.read_text(encoding="utf-8")
    assert "SOURCE_SECRET" not in filtered_text
    assert discover_rct_material_ids(filtered) == tuple(sorted(CONFIRMATION_MATERIALS))
    assert custody["admitted_confirmation_material_count"] == 20
    assert custody["discarded_source_material_count"] == 1


def test_confirmation_runner_requires_content_bound_passing_source_result(
    tmp_path: Path,
) -> None:
    runner = _module()
    identity = {
        "schema": runner.SOURCE_RESULT_SCHEMA,
        "schema_version": 1,
        "source_gate": {"passed": True},
        "source_test_opened": True,
        "confirmation_opened": False,
        "confirmation_force_fields_parsed": False,
        "held_v8_accessed": False,
        "attempt_count": 1,
        "replacement_or_retry_authorized": False,
        "target_authorized": False,
    }
    result = {**identity, "source_result_id": content_id(identity)}
    path = tmp_path / "source-result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan = {
        "source_result_path": str(path),
        "source_result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_result_id": result["source_result_id"],
    }

    assert runner._load_source_result(plan) == result

    blocked_identity = {**identity, "source_gate": {"passed": False}}
    blocked = {
        **blocked_identity,
        "source_result_id": content_id(blocked_identity),
    }
    path.write_text(
        json.dumps(blocked, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan["source_result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan["source_result_id"] = blocked["source_result_id"]
    with pytest.raises(ValueError, match="source gate failed"):
        runner._load_source_result(plan)


def test_confirmation_gate_freezes_positive_or_negative_result() -> None:
    runner = _module()
    passing = {
        "paired_mean_auc_difference": -0.01,
        "one_sided_exact_paired_sign_flip_p": 0.049,
        "decision_directed_simultaneous_force_coverage": 0.9,
        "decision_directed_false_safe_rate": 0.1,
        "decision_directed_unsafe_action_rate": 0.1,
        "system_identification_unsafe_action_rate": 0.05,
    }

    passed = runner._confirmation_gate(passing)
    failed = runner._confirmation_gate(
        {**passing, "one_sided_exact_paired_sign_flip_p": 0.051}
    )

    assert passed["passed"] is True
    assert passed["decision"] == "rct-real-decision-confirmation-pass"
    assert passed["retry_authorized"] is False
    assert failed["passed"] is False
    assert failed["decision"] == "rct-real-decision-confirmation-fail"
    assert failed["method_change_authorized"] is False
