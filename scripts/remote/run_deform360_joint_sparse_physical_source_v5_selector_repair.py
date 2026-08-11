#!/usr/bin/env python3
"""Run the frozen Deform360 prefix stage with its corrected selector identity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import file_sha256
from bayesian_phystwin.deform360_joint_sparse_physical_source_v5 import (
    activate_joint_sparse_physical_runtime_v5,
    patch_joint_sparse_physical_stage_v5,
    validate_joint_sparse_physical_execution_v5,
)

REPAIR_RELATIVE_PATH = (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_stage_selector_binding_repair.json"
)
REPAIR_ID = "001910b84ded7b3f860aa208b87fedf51605fb977af8aab8df3b7e1fa45eeb67"
STAGE_RELATIVE_PATH = "scripts/remote/stage_deform360_bias_aware_prediction_prefix.py"
STAGE_GIT_BLOB_SHA1 = "188e39f28099f8862c1d0cad66761bcf5d5fb955"
PREVIOUS_SELECTOR_SHA256 = (
    "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
)
CORRECTED_SELECTOR_SHA256 = (
    "c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
)
CORRECTED_SELECTOR_BYTE_COUNT = 17310


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_id(payload: dict[str, Any], *, digest_key: str) -> str:
    body = {key: value for key, value in payload.items() if key != digest_key}
    canonical = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_blob_sha1(repository: Path, path: Path) -> str:
    return subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _argument_value(arguments: list[str], option: str) -> str:
    matches = [index for index, value in enumerate(arguments) if value == option]
    _require(len(matches) == 1, f"expected one {option} argument")
    index = matches[0]
    _require(index + 1 < len(arguments), f"{option} has no value")
    return arguments[index + 1]


def _load_stage(path: Path) -> ModuleType:
    name = "_deform360_v6_stage_prefix_selector_identity_repair"
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, "cannot load prefix stage")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--execution-repo", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage-prefix",), required=True)
    return parser.parse_known_args()


def main() -> int:
    args, stage_arguments = _parse_args()
    repository = args.execution_repo.resolve(strict=True)
    execution_lock = args.execution_lock.resolve(strict=True)

    # This validates the original checksum-bound stage files before any
    # process-local compatibility change is made.
    validate_joint_sparse_physical_execution_v5(
        execution_lock,
        repository=repository,
    )

    repair_path = (repository / REPAIR_RELATIVE_PATH).resolve(strict=True)
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    _require(isinstance(repair, dict), "selector repair must be a JSON object")
    _require(repair.get("repair_id") == REPAIR_ID, "selector repair ID changed")
    _require(
        _canonical_id(repair, digest_key="repair_id") == REPAIR_ID,
        "selector repair content changed",
    )
    root_cause = repair.get("root_cause")
    _require(isinstance(root_cause, dict), "selector repair root cause changed")
    _require(
        root_cause.get("stage_registered_selector_sha256") == PREVIOUS_SELECTOR_SHA256
        and root_cause.get("corrected_selector_sha256") == CORRECTED_SELECTOR_SHA256,
        "selector repair digest binding changed",
    )

    protocol_path = Path(_argument_value(stage_arguments, "--protocol")).resolve()
    _require(protocol_path == execution_lock, "stage protocol must be the v5 lock")
    selector_path = Path(
        _argument_value(stage_arguments, "--generic-selector-source")
    ).resolve(strict=True)
    _require(
        selector_path.is_file() and not selector_path.is_symlink(), "selector missing"
    )
    _require(
        selector_path.stat().st_size == CORRECTED_SELECTOR_BYTE_COUNT
        and file_sha256(selector_path) == CORRECTED_SELECTOR_SHA256,
        "corrected selector bytes changed",
    )

    stage_path = (repository / STAGE_RELATIVE_PATH).resolve(strict=True)
    _require(
        _git_blob_sha1(repository, stage_path) == STAGE_GIT_BLOB_SHA1,
        "checksum-bound prefix stage changed",
    )
    module = _load_stage(stage_path)

    with activate_joint_sparse_physical_runtime_v5():
        patch_joint_sparse_physical_stage_v5(
            module,
            stage="stage-prefix",
            repository=repository,
            execution_lock=execution_lock,
        )
        _require(
            getattr(module, "GENERIC_SELECTOR_SHA256", None)
            == PREVIOUS_SELECTOR_SHA256,
            "prefix stage no longer carries the superseded selector identity",
        )
        module.GENERIC_SELECTOR_SHA256 = CORRECTED_SELECTOR_SHA256
        previous = sys.argv
        sys.argv = [str(stage_path), *stage_arguments]
        try:
            return int(module.main())
        finally:
            sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
