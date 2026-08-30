#!/usr/bin/env python3
"""Run one registered source stage for controlled query competence."""

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
from pathlib import Path
from typing import Any, Final

import numpy as np

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.controlled_query_competence_v1 import (
    SOURCE_BUNDLE_SHA256,
    experiment_protocol_v1,
    run_source_stage_v1,
)

PLAN_SCHEMA: Final = "bayesian-phystwin.controlled-query-competence-source-execution"
PLAN_VERSION: Final = 1


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


def _load_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "source plan is invalid")
    _require(_sha256(path) == expected_sha256, "source plan SHA-256 changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "source plan must be an object")
    _require(value.get("schema") == PLAN_SCHEMA, "source plan schema changed")
    _require(value.get("schema_version") == PLAN_VERSION, "source plan version changed")
    identity = dict(value)
    declared = identity.pop("plan_id", None)
    _require(declared == content_id(identity), "source plan identity changed")
    return value


def _consume_attempt(path: Path, plan_id: str, output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "schema": "bayesian-phystwin.controlled-query-source-attempt",
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


def _verify_implementation(plan: dict[str, Any]) -> dict[str, str]:
    implementation = plan["implementation"]
    _require(isinstance(implementation, dict), "implementation must be an object")
    repository = Path(implementation["repository_path"]).resolve(strict=True)
    _require(
        not _git(repository, "status", "--porcelain"),
        "implementation checkout is dirty",
    )
    revision = str(implementation["revision"])
    _git(repository, "cat-file", "-e", f"{revision}^{{commit}}")
    paths = {
        "runner": Path(__file__).resolve(strict=True),
        "module": (repository / implementation["module_relative_path"]).resolve(
            strict=True
        ),
    }
    expected_paths = {
        "runner": (repository / implementation["runner_relative_path"]).resolve(
            strict=True
        ),
        "module": (repository / implementation["module_relative_path"]).resolve(
            strict=True
        ),
    }
    _require(paths == expected_paths, "implementation path changed")
    for label, path in paths.items():
        expected = str(implementation[f"{label}_sha256"])
        _require(_sha256(path) == expected, f"{label} SHA-256 changed")
        committed = subprocess.run(
            ("git", "show", f"{revision}:{path.relative_to(repository)}"),
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        _require(
            hashlib.sha256(committed).hexdigest() == expected,
            f"{label} revision binding changed",
        )
    return {
        "repository": str(repository),
        "revision": revision,
        "head": _git(repository, "rev-parse", "HEAD"),
    }


def _run(plan: dict[str, Any], output_root: Path) -> dict[str, object]:
    implementation = _verify_implementation(plan)
    source_bundle = Path(plan["source_bundle_path"]).resolve(strict=True)
    _require(_sha256(source_bundle) == SOURCE_BUNDLE_SHA256, "source bundle changed")
    _require(
        plan["source_bundle_sha256"] == SOURCE_BUNDLE_SHA256,
        "source bundle binding changed",
    )
    _require(
        plan["protocol"] == experiment_protocol_v1(), "experiment protocol changed"
    )
    _require(
        plan["confirmation_outcomes_authorized"] is False,
        "source stage cannot authorize confirmation outcomes",
    )

    started = time.monotonic()
    source_result = run_source_stage_v1()
    elapsed = time.monotonic() - started
    result = {
        "schema": "bayesian-phystwin.controlled-query-competence-source-execution-result",
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "implementation": implementation,
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "python_executable": str(Path(sys.executable).resolve()),
        },
        "elapsed_seconds": elapsed,
        "source_result": source_result,
        "source_result_id": source_result["source_result_id"],
        "source_gate_passed": source_result["source_gate_passed"],
        "confirmation_authorized_by_source_gate": source_result[
            "confirmation_authorized"
        ],
        "confirmation_outcomes_opened": False,
        "attempt_count": 1,
    }
    return {**result, "execution_result_id": content_id(result)}


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
        _write_json(output_root / "source_execution_result.json", result)
        manifest = {
            "schema": "bayesian-phystwin.controlled-query-source-manifest",
            "schema_version": 1,
            "members": [
                {
                    "path": "source_execution_result.json",
                    "sha256": _sha256(output_root / "source_execution_result.json"),
                }
            ],
        }
        _write_json(output_root / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "source_gate_passed": result["source_gate_passed"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        failure = {
            "schema": "bayesian-phystwin.controlled-query-source-failure",
            "schema_version": 1,
            "plan_id": plan["plan_id"],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "confirmation_outcomes_opened": False,
            "retry_authorized": False,
        }
        _write_json(output_root / "failure.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
