#!/usr/bin/env python3
"""Run the one authorized RCT held-material confirmation evaluation."""

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
    summarize_evaluation,
)
from bayesian_phystwin.rct_real_decision_protocol import (
    CONFIRMATION_MATERIALS,
    load_rct_preoutcome_amendment_v2,
    load_rct_preoutcome_clarification,
    load_rct_real_decision_protocol,
    protocol_config_sha256,
    protocol_file_sha256,
)

PLAN_SCHEMA: Final = "bayesian-phystwin.rct-real-decision-confirmation-execution"
PLAN_VERSION: Final = 1
ARCHIVE_LOCK_SCHEMA: Final = "bayesian-phystwin.rct-real-decision-archive-lock"
ARCHIVE_LOCK_VERSION: Final = 1
SOURCE_RESULT_SCHEMA: Final = "bayesian-phystwin.rct-real-decision-source-result"
CONFIRMATION_RESULT_SCHEMA: Final = (
    "bayesian-phystwin.rct-real-decision-confirmation-result"
)


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
    _require(plan.get("attempt_limit") == 1, "confirmation attempt limit changed")
    _require(
        plan.get("target_authorized") is True,
        "confirmation target is not authorized",
    )
    _require(
        plan.get("source_gate_passed") is True,
        "confirmation source gate is not passed",
    )
    _require(
        plan.get("replacement_or_retry_authorized") is False,
        "confirmation plan authorized a retry",
    )
    _require(
        plan.get("held_v8_access_authorized") is False,
        "confirmation plan authorized held-v8 access",
    )
    return plan


def _consume_attempt(path: Path, plan_id: str, output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "schema": "bayesian-phystwin.rct-real-decision-confirmation-attempt",
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
            _require(
                current == Path(__file__).resolve(strict=True), "runner path changed"
            )
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


def _load_source_result(plan: dict[str, Any]) -> dict[str, Any]:
    result = _load_content_bound_json(
        Path(plan["source_result_path"]).resolve(strict=True),
        str(plan["source_result_sha256"]),
        schema=SOURCE_RESULT_SCHEMA,
        schema_version=1,
        identity_field="source_result_id",
    )
    _require(result.get("source_gate", {}).get("passed") is True, "source gate failed")
    _require(result.get("source_test_opened") is True, "source test was not completed")
    _require(
        result.get("confirmation_opened") is False, "confirmation was opened early"
    )
    _require(
        result.get("confirmation_force_fields_parsed") is False,
        "confirmation force fields were parsed early",
    )
    _require(result.get("held_v8_accessed") is False, "held-v8 was accessed")
    _require(result.get("attempt_count") == 1, "source attempt count changed")
    _require(
        result.get("replacement_or_retry_authorized") is False,
        "source result authorized a retry",
    )
    _require(
        result.get("target_authorized") is False, "source runner authorized target"
    )
    _require(
        result.get("source_result_id") == plan["source_result_id"],
        "source result identity changed",
    )
    return result


def _load_method(
    plan: dict[str, Any], source_result: dict[str, Any]
) -> RCTDecisionMethod:
    path = Path(plan["method_seal_path"]).resolve(strict=True)
    _require(_sha256(path) == plan["method_seal_sha256"], "method seal SHA-256 changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "method seal must be an object")
    identity = dict(payload)
    declared = identity.pop("method_seal_id", None)
    _require(declared == content_id(identity), "method seal identity changed")
    _require(declared == plan["method_seal_id"], "planned method seal changed")
    _require(declared == source_result["method_seal_id"], "source method seal changed")
    _require(
        plan["method_seal_sha256"] == source_result["method_seal_sha256"],
        "source method SHA-256 changed",
    )
    _require(payload.get("confirmation_opened") is False, "method seal opened target")
    _require(payload.get("held_v8_accessed") is False, "method seal accessed held-v8")
    _require(
        payload.get("amendment_v2_file_sha256") == plan["amendment_v2_file_sha256"],
        "method amendment-v2 lock changed",
    )
    method = payload.get("method")
    _require(isinstance(method, dict), "sealed method is missing")
    return RCTDecisionMethod.from_dict(method)


def _load_archive_lock(
    plan: dict[str, Any], source_result: dict[str, Any]
) -> dict[str, Any]:
    lock = _load_content_bound_json(
        Path(plan["archive_lock_path"]).resolve(strict=True),
        str(plan["archive_lock_sha256"]),
        schema=ARCHIVE_LOCK_SCHEMA,
        schema_version=ARCHIVE_LOCK_VERSION,
        identity_field="lock_id",
    )
    _require(
        lock["lock_id"] == source_result["archive_lock_id"], "archive lock changed"
    )
    _require(lock.get("confirmation_opened") is False, "archive lock opened target")
    _require(lock.get("held_v8_accessed") is False, "archive lock accessed held-v8")
    _require(
        lock.get("amendment_v2_file_sha256") == plan["amendment_v2_file_sha256"],
        "archive amendment-v2 lock changed",
    )
    return lock


def _canonical_material_token(token: bytes) -> str:
    value = token.decode("ascii").strip()
    if value.startswith("material_"):
        value = value.removeprefix("material_")
    _require(bool(value), "material token is empty")
    _require(value.isdigit(), "material token is not a numeric RCT identifier")
    return value


def _write_confirmation_only_force_csv(
    archive: Path,
    member_name: str,
    output_path: Path,
    *,
    expected_header_sha256: str | None = None,
    expected_header_columns: list[str] | None = None,
) -> dict[str, Any]:
    allowed = frozenset(CONFIRMATION_MATERIALS)
    admitted_materials: set[str] = set()
    discarded_materials: set[str] = set()
    admitted_rows = 0
    discarded_rows = 0
    with zipfile.ZipFile(archive) as bundle:
        member = bundle.getinfo(member_name)
        _require(member.flag_bits & 0x1 == 0, "force metadata member is encrypted")
        with bundle.open(member, "r") as source, output_path.open("xb") as target:
            header = source.readline()
            if expected_header_sha256 is not None:
                _require(
                    hashlib.sha256(header).hexdigest() == expected_header_sha256,
                    "force metadata header SHA-256 changed",
                )
            if expected_header_columns is not None:
                columns = header.rstrip(b"\r\n").decode("ascii").split(",")
                _require(
                    columns == expected_header_columns, "force metadata header changed"
                )
            _require(
                header.rstrip(b"\r\n").split(b",")[0] == b"material_id",
                "material_id is not the first CSV column",
            )
            _require(b'"' not in header, "quoted force metadata header is unsupported")
            target.write(header)
            for line in source:
                _require(b'"' not in line, "quoted force metadata row is unsupported")
                material_id = _canonical_material_token(line.split(b",", 1)[0])
                if material_id not in allowed:
                    discarded_materials.add(material_id)
                    discarded_rows += 1
                    continue
                admitted_materials.add(material_id)
                admitted_rows += 1
                target.write(line)
            target.flush()
            os.fsync(target.fileno())
    _require(
        admitted_materials == allowed, "confirmation roster was not fully admitted"
    )
    _require(
        admitted_materials.isdisjoint(discarded_materials),
        "confirmation and source rosters overlap",
    )
    return {
        "member_name": member_name,
        "member_crc32": f"{member.CRC:08x}",
        "member_uncompressed_size": member.file_size,
        "admitted_confirmation_material_count": len(admitted_materials),
        "admitted_confirmation_row_count": admitted_rows,
        "discarded_source_material_count": len(discarded_materials),
        "discarded_source_row_count": discarded_rows,
        "confirmation_only_csv_sha256": _sha256(output_path),
    }


def _confirmation_gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "decision_auc_below_system_identification": (
            float(summary["paired_mean_auc_difference"]) < 0.0
        ),
        "minimum_relative_auc_improvement": (
            float(summary["relative_auc_improvement"]) >= 0.05
        ),
        "one_sided_exact_paired_sign_flip": (
            float(summary["one_sided_exact_paired_sign_flip_p"]) <= 0.05
        ),
        "simultaneous_force_coverage": (
            float(summary["decision_directed_simultaneous_force_coverage"]) >= 0.9
        ),
        "false_safe_rate": (float(summary["decision_directed_false_safe_rate"]) <= 0.1),
        "unsafe_rate_noninferiority": (
            float(summary["decision_directed_unsafe_action_rate"])
            <= float(summary["system_identification_unsafe_action_rate"]) + 0.05
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "decision": (
            "rct-real-decision-confirmation-pass"
            if all(checks.values())
            else "rct-real-decision-confirmation-fail"
        ),
        "retry_authorized": False,
        "method_change_authorized": False,
    }


def _run(plan: dict[str, Any], output_root: Path) -> dict[str, Any]:
    implementation = _verify_implementation(plan)
    protocol_path = Path(plan["protocol_path"]).resolve(strict=True)
    clarification_path = Path(plan["clarification_path"]).resolve(strict=True)
    amendment_v2_path = Path(plan["amendment_v2_path"]).resolve(strict=True)
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
        protocol_file_sha256(clarification_path) == plan["clarification_file_sha256"],
        "clarification file changed",
    )
    load_rct_preoutcome_clarification(clarification_path)
    _require(
        protocol_file_sha256(amendment_v2_path) == plan["amendment_v2_file_sha256"],
        "amendment-v2 file changed",
    )
    load_rct_preoutcome_amendment_v2(amendment_v2_path)
    source_result = _load_source_result(plan)
    method = _load_method(plan, source_result)
    archive_lock = _load_archive_lock(plan, source_result)
    archive = Path(plan["archive_path"]).resolve(strict=True)
    _require(
        archive.stat().st_size == archive_lock["archive_size_bytes"],
        "archive size changed",
    )
    _require(
        _sha256(archive) == archive_lock["archive_sha256"], "archive SHA-256 changed"
    )

    confirmation_csv = output_root / "force_metadata_confirmation_only.csv"
    custody = _write_confirmation_only_force_csv(
        archive,
        str(archive_lock["force_metadata_member"]),
        confirmation_csv,
        expected_header_sha256=str(archive_lock["force_metadata_header_sha256"]),
        expected_header_columns=list(archive_lock["force_metadata_header_columns"]),
    )
    _require(
        discover_rct_material_ids(confirmation_csv)
        == tuple(sorted(CONFIRMATION_MATERIALS)),
        "confirmation material roster changed",
    )
    responses = load_rct_force_responses(
        confirmation_csv,
        allowed_material_ids=CONFIRMATION_MATERIALS,
    )
    by_material = {response.material_id: response for response in responses}
    _require(len(by_material) == 20, "confirmation response count changed")
    started = time.monotonic()
    material_results = [
        evaluate_material(method, by_material[material_id])
        for material_id in CONFIRMATION_MATERIALS
    ]
    summary = summarize_evaluation(material_results, require_confirmation_count=True)
    gate = _confirmation_gate(summary)
    elapsed = time.monotonic() - started
    _write_json(output_root / "confirmation_material_results.json", material_results)
    result = {
        "schema": CONFIRMATION_RESULT_SCHEMA,
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "implementation": implementation,
        "archive_lock_id": archive_lock["lock_id"],
        "source_result_id": source_result["source_result_id"],
        "method_seal_id": plan["method_seal_id"],
        "custody": custody,
        "confirmation_summary": summary,
        "confirmation_gate": gate,
        "runtime": {
            "elapsed_confirmation_seconds": elapsed,
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "confirmation_opened": True,
        "held_v8_accessed": False,
        "attempt_count": 1,
        "attempt_limit": 1,
        "replacement_or_retry_authorized": False,
        "method_change_authorized": False,
    }
    result["confirmation_result_id"] = content_id(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    arguments = parser.parse_args()
    plan = _load_plan(arguments.plan.resolve(), arguments.expected_plan_sha256)
    output_root = Path(plan["output_root"]).resolve()
    attempt_path = Path(plan["attempt_ledger_path"]).resolve()
    _require(not output_root.exists(), "confirmation output root already exists")
    _require(not attempt_path.exists(), "confirmation attempt was already consumed")
    _consume_attempt(attempt_path, str(plan["plan_id"]), output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        result = _run(plan, output_root)
        result_path = output_root / "confirmation_result.json"
        _write_json(result_path, result)
        members = (
            "force_metadata_confirmation_only.csv",
            "confirmation_material_results.json",
            "confirmation_result.json",
        )
        manifest = {
            "schema": "bayesian-phystwin.rct-real-decision-confirmation-manifest",
            "schema_version": 1,
            "members": [
                {"path": name, "sha256": _sha256(output_root / name)}
                for name in members
            ],
            "attempt_count": 1,
            "retry_authorized": False,
            "held_v8_accessed": False,
        }
        _write_json(output_root / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "decision": result["confirmation_gate"]["decision"],
                    "retry_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        failure = {
            "schema": "bayesian-phystwin.rct-real-decision-confirmation-failure",
            "schema_version": 1,
            "plan_id": plan["plan_id"],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "confirmation_opened": True,
            "held_v8_accessed": False,
            "retry_authorized": False,
            "method_change_authorized": False,
        }
        _write_json(output_root / "failure.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
