#!/usr/bin/env python3
"""Build a one-shot RCT confirmation plan from a passing sealed source result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.rct_real_decision_protocol import (
    load_rct_preoutcome_clarification,
    load_rct_real_decision_protocol,
    protocol_config_sha256,
    protocol_file_sha256,
)

PLAN_SCHEMA = "bayesian-phystwin.rct-real-decision-confirmation-execution"
PLAN_VERSION = 1
SOURCE_RESULT_SCHEMA = "bayesian-phystwin.rct-real-decision-source-result"
ARCHIVE_LOCK_SCHEMA = "bayesian-phystwin.rct-real-decision-archive-lock"
REGISTERED_PATHS = {
    "runner": "scripts/science/run_rct_real_decision_confirmation_v1.py",
    "method": "src/bayesian_phystwin/rct_real_decision.py",
    "protocol_loader": "src/bayesian_phystwin/rct_real_decision_protocol.py",
    "protocol": "protocols/rct_real_decision_probe_v1.json",
    "clarification": (
        "protocols/rct_real_decision_probe_v1_preoutcome_clarification.json"
    ),
    "archive_lock": "protocols/rct_real_decision_archive_lock_v1.json",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_content_bound(path: Path, *, schema: str, identity_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{schema} must be an object")
    _require(payload.get("schema") == schema, f"{schema} schema changed")
    identity = dict(payload)
    declared = identity.pop(identity_field, None)
    _require(declared == content_id(identity), f"{schema} identity changed")
    return payload


def _load_source_result(path: Path) -> dict[str, Any]:
    result = _load_content_bound(
        path,
        schema=SOURCE_RESULT_SCHEMA,
        identity_field="source_result_id",
    )
    _require(result.get("source_gate", {}).get("passed") is True, "source gate failed")
    _require(result.get("source_test_opened") is True, "source test is incomplete")
    _require(result.get("confirmation_opened") is False, "confirmation opened early")
    _require(
        result.get("confirmation_force_fields_parsed") is False,
        "confirmation force fields were parsed early",
    )
    _require(result.get("held_v8_accessed") is False, "held-v8 was accessed")
    _require(result.get("attempt_count") == 1, "source attempt count changed")
    _require(
        result.get("replacement_or_retry_authorized") is False,
        "source result authorized retry",
    )
    _require(result.get("target_authorized") is False, "source runner opened target")
    return result


def _load_method_seal(path: Path, source_result: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "method seal must be an object")
    _require(
        payload.get("schema") == "bayesian-phystwin.rct-real-decision-method-seal",
        "method seal schema changed",
    )
    identity = dict(payload)
    declared = identity.pop("method_seal_id", None)
    _require(declared == content_id(identity), "method seal identity changed")
    _require(declared == source_result["method_seal_id"], "source method seal changed")
    _require(
        _sha256(path) == source_result["method_seal_sha256"],
        "source method seal SHA-256 changed",
    )
    _require(payload.get("confirmation_opened") is False, "method seal opened target")
    _require(payload.get("held_v8_accessed") is False, "method seal accessed held-v8")
    return payload


def _load_archive_lock(path: Path, source_result: dict[str, Any]) -> dict[str, Any]:
    lock = _load_content_bound(
        path,
        schema=ARCHIVE_LOCK_SCHEMA,
        identity_field="lock_id",
    )
    _require(lock["lock_id"] == source_result["archive_lock_id"], "archive lock changed")
    _require(lock.get("confirmation_opened") is False, "archive lock opened target")
    _require(lock.get("held_v8_accessed") is False, "archive lock accessed held-v8")
    return lock


def _implementation_lock(repository: Path, revision: str) -> dict[str, Any]:
    paths: dict[str, dict[str, str]] = {}
    for label, relative_name in REGISTERED_PATHS.items():
        relative = Path(relative_name)
        current = repository / relative
        _require(current.is_file() and not current.is_symlink(), f"missing path: {label}")
        expected = _sha256(current)
        committed = subprocess.run(
            ("git", "show", f"{revision}:{relative.as_posix()}"),
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        _require(
            hashlib.sha256(committed).hexdigest() == expected,
            f"registered path is not committed at revision: {label}",
        )
        paths[label] = {"relative_path": relative.as_posix(), "sha256": expected}
    return {
        "repository_path": str(repository),
        "revision": revision,
        "paths": paths,
    }


def _build_plan(
    repository: Path,
    archive: Path,
    source_result_path: Path,
    method_seal_path: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    _require(not _git(repository, "status", "--porcelain"), "repository is dirty")
    revision = _git(repository, "rev-parse", "HEAD")
    implementation = _implementation_lock(repository, revision)
    protocol_path = repository / REGISTERED_PATHS["protocol"]
    clarification_path = repository / REGISTERED_PATHS["clarification"]
    archive_lock_path = repository / REGISTERED_PATHS["archive_lock"]
    protocol = load_rct_real_decision_protocol(protocol_path)
    load_rct_preoutcome_clarification(clarification_path)
    source_result_path = source_result_path.resolve(strict=True)
    source_result = _load_source_result(source_result_path)
    method_seal_path = method_seal_path.resolve(strict=True)
    method_seal = _load_method_seal(method_seal_path, source_result)
    archive_lock = _load_archive_lock(archive_lock_path, source_result)
    archive = archive.resolve(strict=True)
    _require(archive.stat().st_size == archive_lock["archive_size_bytes"], "archive size changed")
    _require(_sha256(archive) == archive_lock["archive_sha256"], "archive SHA-256 changed")
    suffix = f"{revision[:8]}-{source_result['source_result_id'][:8]}"
    output_root = runtime_root / f"confirmation-run-{suffix}"
    attempt_path = runtime_root / "attempts" / f"confirmation-attempt-{suffix}.json"
    _require(not output_root.exists(), "confirmation output root already exists")
    _require(not attempt_path.exists(), "confirmation attempt already exists")
    identity = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_VERSION,
        "implementation": implementation,
        "archive_path": str(archive),
        "archive_lock_path": str(archive_lock_path.resolve(strict=True)),
        "archive_lock_sha256": _sha256(archive_lock_path),
        "protocol_path": str(protocol_path.resolve(strict=True)),
        "protocol_file_sha256": protocol_file_sha256(protocol_path),
        "protocol_config_sha256": protocol_config_sha256(protocol),
        "clarification_path": str(clarification_path.resolve(strict=True)),
        "clarification_file_sha256": protocol_file_sha256(clarification_path),
        "source_result_path": str(source_result_path),
        "source_result_sha256": _sha256(source_result_path),
        "source_result_id": source_result["source_result_id"],
        "method_seal_path": str(method_seal_path),
        "method_seal_sha256": _sha256(method_seal_path),
        "method_seal_id": method_seal["method_seal_id"],
        "output_root": str(output_root.resolve()),
        "attempt_ledger_path": str(attempt_path.resolve()),
        "attempt_limit": 1,
        "source_gate_passed": True,
        "target_authorized": True,
        "replacement_or_retry_authorized": False,
        "held_v8_access_authorized": False,
    }
    return {**identity, "plan_id": content_id(identity)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--method-seal", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    _require(not output.exists(), "confirmation plan output already exists")
    plan = _build_plan(
        arguments.repository,
        arguments.archive,
        arguments.source_result,
        arguments.method_seal,
        arguments.runtime_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "plan_id": plan["plan_id"],
                "target_authorized": True,
                "attempt_limit": 1,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
