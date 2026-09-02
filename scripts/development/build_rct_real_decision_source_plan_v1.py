#!/usr/bin/env python3
"""Build the exact source execution plan after the RCT archive lock is committed."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.rct_real_decision_protocol import (
    load_rct_preoutcome_amendment_v2,
    load_rct_preoutcome_clarification,
    load_rct_real_decision_protocol,
    protocol_config_sha256,
    protocol_file_sha256,
)

PLAN_SCHEMA = "bayesian-phystwin.rct-real-decision-source-execution"
PLAN_VERSION = 1
ARCHIVE_LOCK_SCHEMA = "bayesian-phystwin.rct-real-decision-archive-lock"
REGISTERED_PATHS = {
    "runner": "scripts/science/run_rct_real_decision_source_v1.py",
    "method": "src/bayesian_phystwin/rct_real_decision.py",
    "protocol_loader": "src/bayesian_phystwin/rct_real_decision_protocol.py",
    "protocol": "protocols/rct_real_decision_probe_v1.json",
    "clarification": (
        "protocols/rct_real_decision_probe_v1_preoutcome_clarification.json"
    ),
    "amendment_v2": (
        "protocols/rct_real_decision_probe_v1_preoutcome_amendment_v2.json"
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


def _load_archive_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "archive lock must be an object")
    _require(
        payload.get("schema") == ARCHIVE_LOCK_SCHEMA, "archive lock schema changed"
    )
    identity = dict(payload)
    declared = identity.pop("lock_id", None)
    _require(declared == content_id(identity), "archive lock identity changed")
    _require(
        payload.get("archive_integrity_verified") is True, "archive is not verified"
    )
    _require(
        payload.get("force_metadata_content_opened") is False,
        "force metadata was opened before source planning",
    )
    _require(payload.get("confirmation_opened") is False, "confirmation opened early")
    _require(payload.get("held_v8_accessed") is False, "held-v8 was accessed")
    return payload


def _implementation_lock(repository: Path, revision: str) -> dict[str, Any]:
    paths: dict[str, dict[str, str]] = {}
    for label, relative_name in REGISTERED_PATHS.items():
        relative = Path(relative_name)
        current = repository / relative
        _require(
            current.is_file() and not current.is_symlink(), f"missing path: {label}"
        )
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
    runtime_root: Path,
) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    _require(not _git(repository, "status", "--porcelain"), "repository is dirty")
    revision = _git(repository, "rev-parse", "HEAD")
    implementation = _implementation_lock(repository, revision)
    protocol_path = repository / REGISTERED_PATHS["protocol"]
    clarification_path = repository / REGISTERED_PATHS["clarification"]
    amendment_v2_path = repository / REGISTERED_PATHS["amendment_v2"]
    archive_lock_path = repository / REGISTERED_PATHS["archive_lock"]
    protocol = load_rct_real_decision_protocol(protocol_path)
    load_rct_preoutcome_clarification(clarification_path)
    load_rct_preoutcome_amendment_v2(amendment_v2_path)
    archive_lock = _load_archive_lock(archive_lock_path)
    _require(archive.is_file() and not archive.is_symlink(), "archive path is invalid")
    _require(
        archive.stat().st_size == archive_lock["archive_size_bytes"],
        "archive size changed",
    )
    _require(
        _sha256(archive) == archive_lock["archive_sha256"], "archive SHA-256 changed"
    )
    suffix = revision[:8]
    output_root = runtime_root / f"source-run-{suffix}"
    attempt_path = runtime_root / "attempts" / f"source-attempt-{suffix}.json"
    _require(not output_root.exists(), "source output root already exists")
    _require(not attempt_path.exists(), "source attempt already exists")
    identity = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_VERSION,
        "implementation": implementation,
        "archive_path": str(archive.resolve(strict=True)),
        "archive_lock_path": str(archive_lock_path.resolve(strict=True)),
        "archive_lock_sha256": _sha256(archive_lock_path),
        "protocol_path": str(protocol_path.resolve(strict=True)),
        "protocol_file_sha256": protocol_file_sha256(protocol_path),
        "protocol_config_sha256": protocol_config_sha256(protocol),
        "clarification_path": str(clarification_path.resolve(strict=True)),
        "clarification_file_sha256": protocol_file_sha256(clarification_path),
        "amendment_v2_path": str(amendment_v2_path.resolve(strict=True)),
        "amendment_v2_file_sha256": protocol_file_sha256(amendment_v2_path),
        "output_root": str(output_root.resolve()),
        "attempt_ledger_path": str(attempt_path.resolve()),
        "attempt_limit": 1,
        "confirmation_outcomes_authorized": False,
        "replacement_or_retry_authorized": False,
        "held_v8_access_authorized": False,
    }
    return {**identity, "plan_id": content_id(identity)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    _require(not output.exists(), "source plan output already exists")
    plan = _build_plan(
        arguments.repository,
        arguments.archive,
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
                "target_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
