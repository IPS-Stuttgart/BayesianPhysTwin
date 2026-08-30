#!/usr/bin/env python3
"""Run one authorized controlled query-competence confirmation."""

from __future__ import annotations

import argparse
import csv
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
    experiment_protocol_v1,
    outcome_records_v1,
    run_confirmation_stage_v1,
)

PLAN_SCHEMA: Final = (
    "bayesian-phystwin.controlled-query-competence-confirmation-execution"
)
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
    _require(path.is_file() and not path.is_symlink(), "confirmation plan is invalid")
    _require(_sha256(path) == expected_sha256, "confirmation plan SHA-256 changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "confirmation plan must be an object")
    _require(value.get("schema") == PLAN_SCHEMA, "confirmation plan schema changed")
    _require(
        value.get("schema_version") == PLAN_VERSION, "confirmation plan version changed"
    )
    identity = dict(value)
    declared = identity.pop("plan_id", None)
    _require(declared == content_id(identity), "confirmation plan identity changed")
    return value


def _consume_attempt(path: Path, plan_id: str, output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "schema": "bayesian-phystwin.controlled-query-confirmation-attempt",
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
        ("git", *arguments), cwd=repository, check=True, capture_output=True, text=True
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


def _load_source_result(plan: dict[str, Any]) -> dict[str, Any]:
    path = Path(plan["source_execution_result_path"]).resolve(strict=True)
    _require(
        _sha256(path) == plan["source_execution_result_sha256"],
        "source execution result changed",
    )
    execution = json.loads(path.read_text(encoding="utf-8"))
    _require(
        execution["execution_result_id"] == plan["source_execution_result_id"],
        "source execution identity changed",
    )
    identity = dict(execution)
    declared = identity.pop("execution_result_id")
    _require(declared == content_id(identity), "source execution content changed")
    _require(execution["source_gate_passed"] is True, "source gate did not pass")
    _require(
        execution["confirmation_authorized_by_source_gate"] is True,
        "source gate did not authorize confirmation",
    )
    _require(
        execution["confirmation_outcomes_opened"] is False,
        "source execution opened confirmation outcomes",
    )
    return dict(execution["source_result"])


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError("confirmation records are empty")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _run(plan: dict[str, Any], output_root: Path) -> dict[str, object]:
    implementation = _verify_implementation(plan)
    _require(
        plan["protocol"] == experiment_protocol_v1(), "experiment protocol changed"
    )
    source_result = _load_source_result(plan)
    started = time.monotonic()
    confirmation_result, outcomes = run_confirmation_stage_v1(source_result)
    elapsed = time.monotonic() - started
    _write_csv(
        output_root / "episode_metrics.csv", outcome_records_v1(outcomes, source_result)
    )
    result = {
        "schema": "bayesian-phystwin.controlled-query-competence-confirmation-execution-result",
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "implementation": implementation,
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "python_executable": str(Path(sys.executable).resolve()),
        },
        "elapsed_seconds": elapsed,
        "confirmation_result": confirmation_result,
        "confirmation_result_id": confirmation_result["confirmation_result_id"],
        "attempt_count": 1,
        "reselection_or_retry": False,
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
    _require(not output_root.exists(), "confirmation output root already exists")
    _require(not attempt_path.exists(), "confirmation attempt was already consumed")
    _consume_attempt(attempt_path, str(plan["plan_id"]), output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        result = _run(plan, output_root)
        _write_json(output_root / "confirmation_execution_result.json", result)
        members = ("confirmation_execution_result.json", "episode_metrics.csv")
        manifest = {
            "schema": "bayesian-phystwin.controlled-query-confirmation-manifest",
            "schema_version": 1,
            "members": [
                {"path": name, "sha256": _sha256(output_root / name)}
                for name in members
            ],
        }
        _write_json(output_root / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "decision": result["confirmation_result"]["decision"],
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        failure = {
            "schema": "bayesian-phystwin.controlled-query-confirmation-failure",
            "schema_version": 1,
            "plan_id": plan["plan_id"],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "retry_authorized": False,
        }
        _write_json(output_root / "failure.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
