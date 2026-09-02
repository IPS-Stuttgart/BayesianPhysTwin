#!/usr/bin/env python3
"""Run the exactly-once source stage for the public RCT real-decision study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Final

import numpy as np

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.rct_real_decision import (
    RCTDecisionMethod,
    discover_rct_material_ids,
    evaluate_material,
    load_rct_force_responses,
    source_promotion_gate,
    summarize_evaluation,
)
from bayesian_phystwin.rct_real_decision_protocol import (
    CALIBRATION_MATERIALS,
    CONFIRMATION_MATERIALS,
    SOURCE_TEST_MATERIALS,
    cohort_from_protocol,
    load_rct_preoutcome_clarification,
    load_rct_real_decision_protocol,
    protocol_config_sha256,
    protocol_file_sha256,
)

PLAN_SCHEMA: Final = "bayesian-phystwin.rct-real-decision-source-execution"
PLAN_VERSION: Final = 1
ARCHIVE_LOCK_SCHEMA: Final = "bayesian-phystwin.rct-real-decision-archive-lock"
ARCHIVE_LOCK_VERSION: Final = 1
SOURCE_RESULT_SCHEMA: Final = "bayesian-phystwin.rct-real-decision-source-result"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_content_bound_json(
    path: Path,
    expected_sha256: str,
    *,
    schema: str,
    schema_version: int,
    identity_field: str,
) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{schema} path is invalid")
    _require(_sha256(path) == expected_sha256, f"{schema} SHA-256 changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{schema} must be an object")
    _require(payload.get("schema") == schema, f"{schema} schema changed")
    _require(
        payload.get("schema_version") == schema_version,
        f"{schema} version changed",
    )
    identity = dict(payload)
    declared = identity.pop(identity_field, None)
    _require(declared == content_id(identity), f"{schema} identity changed")
    return payload


def _load_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    plan = _load_content_bound_json(
        path,
        expected_sha256,
        schema=PLAN_SCHEMA,
        schema_version=PLAN_VERSION,
        identity_field="plan_id",
    )
    _require(plan.get("attempt_limit") == 1, "source attempt limit changed")
    _require(
        plan.get("confirmation_outcomes_authorized") is False,
        "source plan authorized confirmation outcomes",
    )
    _require(
        plan.get("replacement_or_retry_authorized") is False,
        "source plan authorized a retry",
    )
    _require(
        plan.get("held_v8_access_authorized") is False,
        "source plan authorized held-v8 access",
    )
    return plan


def _consume_attempt(path: Path, plan_id: str, output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "schema": "bayesian-phystwin.rct-real-decision-source-attempt",
                "schema_version": 1,
                "plan_id": plan_id,
                "output_root": str(output_root),
                "attempt_consumed": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_implementation(plan: dict[str, Any]) -> dict[str, Any]:
    implementation = plan.get("implementation")
    _require(isinstance(implementation, dict), "implementation lock is missing")
    repository = Path(implementation["repository_path"]).resolve(strict=True)
    _require(not _git(repository, "status", "--porcelain"), "checkout is dirty")
    revision = str(implementation["revision"])
    _git(repository, "cat-file", "-e", f"{revision}^{{commit}}")
    registered_paths = implementation.get("paths")
    _require(isinstance(registered_paths, dict), "implementation paths are missing")
    verified_paths: dict[str, dict[str, str]] = {}
    for label, record in registered_paths.items():
        _require(isinstance(record, dict), f"implementation path is invalid: {label}")
        relative = Path(str(record["relative_path"]))
        current = (repository / relative).resolve(strict=True)
        if label == "runner":
            _require(current == Path(__file__).resolve(strict=True), "runner path changed")
        expected = str(record["sha256"])
        _require(_sha256(current) == expected, f"{label} SHA-256 changed")
        committed = subprocess.run(
            ("git", "show", f"{revision}:{relative.as_posix()}"),
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        _require(
            hashlib.sha256(committed).hexdigest() == expected,
            f"{label} revision binding changed",
        )
        verified_paths[str(label)] = {
            "relative_path": relative.as_posix(),
            "sha256": expected,
        }
    return {
        "repository": str(repository),
        "revision": revision,
        "head": _git(repository, "rev-parse", "HEAD"),
        "paths": verified_paths,
    }


def _load_archive_lock(plan: dict[str, Any]) -> dict[str, Any]:
    lock = _load_content_bound_json(
        Path(plan["archive_lock_path"]).resolve(strict=True),
        str(plan["archive_lock_sha256"]),
        schema=ARCHIVE_LOCK_SCHEMA,
        schema_version=ARCHIVE_LOCK_VERSION,
        identity_field="lock_id",
    )
    _require(lock.get("protocol_file_sha256") == plan["protocol_file_sha256"], "protocol lock changed")
    _require(
        lock.get("clarification_file_sha256")
        == plan["clarification_file_sha256"],
        "clarification lock changed",
    )
    _require(lock.get("confirmation_opened") is False, "archive lock opened confirmation")
    _require(lock.get("held_v8_accessed") is False, "archive lock accessed held-v8")
    return lock


def _canonical_material_token(token: bytes) -> str:
    value = token.decode("ascii").strip()
    if value.startswith("material_"):
        value = value.removeprefix("material_")
    _require(bool(value), "material token is empty")
    _require(value.isdigit(), "material token is not a numeric RCT identifier")
    return value


def _write_source_only_force_csv(
    archive: Path,
    member_name: str,
    output_path: Path,
) -> dict[str, Any]:
    """Discard registered confirmation lines before CSV force fields are parsed."""

    forbidden = frozenset(CONFIRMATION_MATERIALS)
    admitted_materials: set[str] = set()
    skipped_materials: set[str] = set()
    admitted_rows = 0
    skipped_rows = 0
    with zipfile.ZipFile(archive) as bundle:
        member = bundle.getinfo(member_name)
        _require(member.flag_bits & 0x1 == 0, "force metadata member is encrypted")
        with bundle.open(member, "r") as source, output_path.open("xb") as target:
            header = source.readline()
            _require(header.rstrip(b"\r\n").split(b",")[0] == b"material_id", "material_id is not the first CSV column")
            _require(b'"' not in header, "quoted force metadata header is unsupported")
            target.write(header)
            for line in source:
                _require(b'"' not in line, "quoted force metadata row is unsupported")
                material_id = _canonical_material_token(line.split(b",", 1)[0])
                if material_id in forbidden:
                    skipped_materials.add(material_id)
                    skipped_rows += 1
                    continue
                admitted_materials.add(material_id)
                admitted_rows += 1
                target.write(line)
            target.flush()
            os.fsync(target.fileno())
    _require(skipped_materials == forbidden, "confirmation roster was not fully discarded")
    _require(
        admitted_materials.isdisjoint(forbidden),
        "confirmation material entered source CSV",
    )
    return {
        "member_name": member_name,
        "member_crc32": f"{member.CRC:08x}",
        "member_uncompressed_size": member.file_size,
        "admitted_material_count": len(admitted_materials),
        "admitted_row_count": admitted_rows,
        "skipped_confirmation_material_count": len(skipped_materials),
        "skipped_confirmation_row_count": skipped_rows,
        "source_only_csv_sha256": _sha256(output_path),
        "confirmation_force_fields_parsed": False,
    }


def _run(plan: dict[str, Any], output_root: Path) -> dict[str, Any]:
    implementation = _verify_implementation(plan)
    protocol_path = Path(plan["protocol_path"]).resolve(strict=True)
    clarification_path = Path(plan["clarification_path"]).resolve(strict=True)
    _require(
        protocol_file_sha256(protocol_path) == plan["protocol_file_sha256"],
        "protocol file changed",
    )
    protocol = load_rct_real_decision_protocol(protocol_path)
    _require(
        protocol_config_sha256(protocol) == plan["protocol_config_sha256"],
        "protocol configuration changed",
    )
    _require(
        protocol_file_sha256(clarification_path)
        == plan["clarification_file_sha256"],
        "clarification file changed",
    )
    load_rct_preoutcome_clarification(clarification_path)
    cohort = cohort_from_protocol(protocol)
    archive_lock = _load_archive_lock(plan)
    archive = Path(plan["archive_path"]).resolve(strict=True)
    _require(archive.stat().st_size == archive_lock["archive_size_bytes"], "archive size changed")
    _require(_sha256(archive) == archive_lock["archive_sha256"], "archive SHA-256 changed")

    source_csv = output_root / "force_metadata_source_only.csv"
    custody = _write_source_only_force_csv(
        archive,
        str(archive_lock["force_metadata_member"]),
        source_csv,
    )
    source_material_ids = discover_rct_material_ids(source_csv)
    _require(len(source_material_ids) == 102, "source material count changed")
    _require(
        set(source_material_ids).isdisjoint(CONFIRMATION_MATERIALS),
        "confirmation material entered source roster",
    )
    fit_materials = tuple(
        material_id
        for material_id in source_material_ids
        if material_id not in set(CALIBRATION_MATERIALS + SOURCE_TEST_MATERIALS)
    )
    _require(len(fit_materials) == cohort.expected_fit_count, "fit roster changed")
    responses = load_rct_force_responses(
        source_csv,
        allowed_material_ids=source_material_ids,
        forbidden_material_ids=CONFIRMATION_MATERIALS,
    )
    by_material = {response.material_id: response for response in responses}
    _require(len(by_material) == 102, "source response roster changed")
    method = RCTDecisionMethod.fit(
        [by_material[material_id] for material_id in fit_materials],
        [by_material[material_id] for material_id in CALIBRATION_MATERIALS],
    )
    method_payload = {
        "schema": "bayesian-phystwin.rct-real-decision-method-seal",
        "schema_version": 1,
        "protocol_file_sha256": plan["protocol_file_sha256"],
        "protocol_config_sha256": plan["protocol_config_sha256"],
        "clarification_file_sha256": plan["clarification_file_sha256"],
        "archive_lock_id": archive_lock["lock_id"],
        "implementation_revision": implementation["revision"],
        "fit_material_count": len(fit_materials),
        "calibration_material_count": len(CALIBRATION_MATERIALS),
        "source_test_material_count": len(SOURCE_TEST_MATERIALS),
        "method": method.as_dict(),
        "confirmation_opened": False,
        "held_v8_accessed": False,
    }
    method_payload["method_seal_id"] = content_id(method_payload)
    method_path = output_root / "method_seal.json"
    _write_json(method_path, method_payload)

    started = time.monotonic()
    material_results = [
        evaluate_material(method, by_material[material_id])
        for material_id in SOURCE_TEST_MATERIALS
    ]
    summary = summarize_evaluation(
        material_results,
        require_confirmation_count=False,
    )
    gate = source_promotion_gate(summary)
    elapsed = time.monotonic() - started
    _write_json(output_root / "source_test_material_results.json", material_results)
    result = {
        "schema": SOURCE_RESULT_SCHEMA,
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "implementation": implementation,
        "archive_lock_id": archive_lock["lock_id"],
        "custody": custody,
        "method_seal_id": method_payload["method_seal_id"],
        "method_seal_sha256": _sha256(method_path),
        "source_test_summary": summary,
        "source_gate": gate,
        "runtime": {
            "elapsed_source_test_seconds": elapsed,
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "source_test_opened": True,
        "confirmation_opened": False,
        "confirmation_force_fields_parsed": False,
        "held_v8_accessed": False,
        "attempt_count": 1,
        "replacement_or_retry_authorized": False,
        "target_authorized": False,
    }
    result["source_result_id"] = content_id(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    arguments = parser.parse_args()
    plan = _load_plan(arguments.plan.resolve(), arguments.expected_plan_sha256)
    output_root = Path(plan["output_root"]).resolve()
    attempt_path = Path(plan["attempt_ledger_path"]).resolve()
    _require(not output_root.exists(), "source output root already exists")
    _require(not attempt_path.exists(), "source attempt was already consumed")
    _consume_attempt(attempt_path, str(plan["plan_id"]), output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        result = _run(plan, output_root)
        result_path = output_root / "source_result.json"
        _write_json(result_path, result)
        members = (
            "force_metadata_source_only.csv",
            "method_seal.json",
            "source_test_material_results.json",
            "source_result.json",
        )
        manifest = {
            "schema": "bayesian-phystwin.rct-real-decision-source-manifest",
            "schema_version": 1,
            "members": [
                {"path": name, "sha256": _sha256(output_root / name)}
                for name in members
            ],
            "confirmation_opened": False,
            "held_v8_accessed": False,
        }
        _write_json(output_root / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "source_gate_passed": result["source_gate"]["passed"],
                    "target_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        failure = {
            "schema": "bayesian-phystwin.rct-real-decision-source-failure",
            "schema_version": 1,
            "plan_id": plan["plan_id"],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "confirmation_opened": False,
            "held_v8_accessed": False,
            "retry_authorized": False,
        }
        _write_json(output_root / "failure.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
