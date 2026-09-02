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
RUNNER_PATH = ROOT / "scripts/science/run_rct_real_decision_source_v1.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rct_source_runner", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_plan(path: Path, runner: ModuleType) -> tuple[dict[str, object], str]:
    identity = {
        "schema": runner.PLAN_SCHEMA,
        "schema_version": runner.PLAN_VERSION,
        "output_root": str(path.parent / "source-output"),
        "attempt_ledger_path": str(path.parent / "source-attempt.json"),
        "attempt_limit": 1,
        "confirmation_outcomes_authorized": False,
        "replacement_or_retry_authorized": False,
        "held_v8_access_authorized": False,
    }
    plan = {**identity, "plan_id": content_id(identity)}
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan, hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_runner_has_no_confirmation_evaluation_or_authorization_path() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert '"confirmation_outcomes_authorized") is False' in source
    assert '"target_authorized": False' in source
    assert '"confirmation_opened": False' in source
    assert '"retry_authorized": False' in source
    assert "require_confirmation_count=True" not in source
    assert "run_rct_real_decision_confirmation" not in source


def test_source_plan_and_attempt_ledger_are_fail_closed(tmp_path: Path) -> None:
    runner = _module()
    plan_path = tmp_path / "plan.json"
    plan, digest = _write_plan(plan_path, runner)

    assert runner._load_plan(plan_path, digest) == plan
    with pytest.raises(ValueError, match="SHA-256"):
        runner._load_plan(plan_path, "0" * 64)

    attempt = tmp_path / "attempt.json"
    output = tmp_path / "source-output"
    runner._consume_attempt(attempt, str(plan["plan_id"]), output)
    attempt_payload = json.loads(attempt.read_text(encoding="utf-8"))
    assert attempt_payload["attempt_consumed"] is True
    assert attempt_payload["plan_id"] == plan["plan_id"]
    with pytest.raises(FileExistsError):
        runner._consume_attempt(attempt, str(plan["plan_id"]), output)


def test_raw_custody_filter_discards_confirmation_before_csv_force_parsing(
    tmp_path: Path,
) -> None:
    runner = _module()
    member_name = "force_metadata.csv"
    lines = ["material_id,position,sensor,z_frame,raw_fz\n"]
    for material_id in CONFIRMATION_MATERIALS:
        lines.append(
            f"material_{material_id},SECRET_POSITION,SECRET_SENSOR,"
            "SECRET_Z,SECRET_FORCE\n"
        )
    lines.extend(
        (
            "material_999999,1,1,1.2,-0.2\n",
            "material_999999,1,1,0.8,-1.2\n",
        )
    )
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(member_name, "".join(lines))
    filtered = tmp_path / "source-only.csv"

    custody = runner._write_source_only_force_csv(archive, member_name, filtered)

    filtered_text = filtered.read_text(encoding="utf-8")
    assert "SECRET_FORCE" not in filtered_text
    assert "999999" in filtered_text
    assert discover_rct_material_ids(filtered) == ("999999",)
    assert custody["skipped_confirmation_material_count"] == 20
    assert custody["skipped_confirmation_row_count"] == 20
    assert custody["confirmation_force_fields_parsed"] is False

    with pytest.raises(ValueError, match="header SHA-256"):
        runner._write_source_only_force_csv(
            archive,
            member_name,
            tmp_path / "wrong-header.csv",
            expected_header_sha256="0" * 64,
        )


def test_raw_custody_filter_rejects_nonfirst_material_column(tmp_path: Path) -> None:
    runner = _module()
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "force_metadata.csv",
            "position,material_id,sensor,z_frame,raw_fz\n",
        )

    with pytest.raises(ValueError, match="first CSV column"):
        runner._write_source_only_force_csv(
            archive,
            "force_metadata.csv",
            tmp_path / "source-only.csv",
        )
