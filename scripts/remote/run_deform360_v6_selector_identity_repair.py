#!/usr/bin/env python3
"""Run the locked v5 prefix stage under the reviewed v6 selector repair."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_joint_sparse_physical_source_v5 import (
    activate_joint_sparse_physical_runtime_v5,
    patch_joint_sparse_physical_stage_v5,
    validate_joint_sparse_physical_execution_v5,
)

REPAIR_SCHEMA = (
    "bayesian-phystwin.deform360-v6-source-runtime-selector-identity-repair"
)
REPAIR_ID = "41f3580de5ca7e09bcd4c2623569c293e29ed796634c60c84ededdbd945af042"
EXECUTION_AMENDMENT_ID = (
    "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
)
CAUSAL4D_REVISION = "50e3682a5dbf976b20cc9115b6e7a975d0144ea5"
SELECTOR_RELATIVE_PATH = Path("src/causal4d_public/deform360_object_sam2.py")
PREVIOUS_SELECTOR_SHA256 = (
    "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
)
CORRECTED_SELECTOR_SHA256 = (
    "c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
)
CORRECTED_SELECTOR_BYTE_COUNT = 17_310
STAGE_SCRIPT = "stage_deform360_bias_aware_prediction_prefix.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
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


def load_selector_identity_repair(path: str | Path) -> Mapping[str, Any]:
    """Validate the exact target-closed selector byte-identity repair."""

    repair_path = Path(path).resolve(strict=True)
    payload = json.loads(repair_path.read_text(encoding="utf-8"))
    repair = _mapping(payload, name="selector identity repair")
    identity = {key: value for key, value in repair.items() if key != "repair_id"}
    _require(
        repair.get("schema") == REPAIR_SCHEMA
        and repair.get("schema_version") == 1
        and repair.get("repair_id") == content_id(identity) == REPAIR_ID,
        "selector identity repair changed",
    )
    _require(
        repair.get("superseded_execution_amendment_id")
        == EXECUTION_AMENDMENT_ID,
        "selector repair uses another execution amendment",
    )
    correction = _mapping(repair.get("correction"), name="selector correction")
    _require(
        correction.get("repository") == "IPS-Stuttgart/Causal4D"
        and correction.get("repository_revision") == CAUSAL4D_REVISION
        and correction.get("path") == SELECTOR_RELATIVE_PATH.as_posix()
        and correction.get("previous_sha256") == PREVIOUS_SELECTOR_SHA256
        and correction.get("corrected_sha256") == CORRECTED_SELECTOR_SHA256
        and correction.get("corrected_byte_count")
        == CORRECTED_SELECTOR_BYTE_COUNT
        and correction.get("historical_registered_digest_found_in_repository_history")
        is False
        and correction.get("selector_class")
        == "DeformableObjectSam2VideoPredictor"
        and correction.get("model_id_prefix")
        == "causal4d_public/deformable-object-sam2.1-small-automatic-v1@",
        "selector correction changed",
    )
    failed = _mapping(
        repair.get("failed_execution_evidence"),
        name="failed execution evidence",
    )
    _require(
        failed.get("workflow_run_id") == 31458096956
        and failed.get("workflow_run_attempt") == 1
        and failed.get("source_revision")
        == "67daacdaafe98b63b8aa0357dccdcd11b9a81d51"
        and failed.get("execution_receipt_id")
        == "cfcfeab74ee9cc88002e398afa2655ccc1a56752787fe6b44a961061fb7cd040"
        and failed.get("terminal_stage") == "locate-frozen-generic-selector"
        and failed.get("exit_code") == 3
        and failed.get("physical_manifest_count") == 0
        and failed.get("source_prediction_seal_count") == 0,
        "selector repair failure evidence changed",
    )
    scope = _mapping(repair.get("repair_scope"), name="repair scope")
    _require(
        scope
        == {
            "camera_panel_changed": False,
            "candidate_roster_changed": False,
            "claim_authorized": False,
            "loss_or_gate_changed": False,
            "model_family_changed": False,
            "model_size_changed": False,
            "replacement_allowed": False,
            "repository_revision_changed": False,
            "runtime_byte_identity_only": True,
            "selector_algorithm_changed": False,
            "source_cohort_changed": False,
        },
        "selector repair scope changed",
    )
    boundary = _mapping(repair.get("information_boundary"), name="repair boundary")
    _require(
        boundary
        == {
            "development_suffix_opened": False,
            "future_object_observations_used_for_prediction": False,
            "v5_confirmation_outcomes_used": False,
            "v5_confirmation_payloads_opened": False,
            "v6_fresh_target_selected": False,
            "v6_target_outcomes_used": False,
            "v6_target_payloads_opened": False,
        },
        "selector repair information boundary changed",
    )
    authorization = _mapping(
        repair.get("execution_authorization"),
        name="repair execution authorization",
    )
    _require(
        authorization.get("event")
        == "push-to-protected-main-after-reviewed-merge"
        and authorization.get("runner_name") == "workstation2"
        and authorization.get("runner_labels")
        == ["self-hosted", "Linux", "X64", "nvidia-smi"]
        and authorization.get("source_prediction_batch_required_before_suffix_access")
        is True
        and authorization.get("fresh_target_selection_authorized") is False
        and authorization.get("fresh_target_payload_access_authorized") is False,
        "selector repair execution authorization changed",
    )
    return repair


def _load_stage(path: Path) -> ModuleType:
    name = "_deform360_v6_selector_identity_repair_stage_prefix"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load stage-prefix script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--execution-repo", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--runtime-repair", type=Path, required=True)
    parser.add_argument("--selector-repository", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage-prefix",), required=True)
    return parser.parse_known_args()


def main() -> int:
    args, stage_arguments = _parse_args()
    repository = args.execution_repo.resolve(strict=True)
    execution_lock = args.execution_lock.resolve(strict=True)
    load_selector_identity_repair(args.runtime_repair)
    validate_joint_sparse_physical_execution_v5(
        execution_lock,
        repository=repository,
    )
    protocol_path = Path(_argument_value(stage_arguments, "--protocol")).resolve()
    _require(protocol_path == execution_lock, "stage protocol must be the v5 lock")

    selector_repository = args.selector_repository.resolve(strict=True)
    _require(
        selector_repository.is_dir() and not selector_repository.is_symlink(),
        "selector repository is invalid",
    )
    _require(
        _git_output(selector_repository, "rev-parse", "HEAD") == CAUSAL4D_REVISION,
        "Causal4D selector revision changed",
    )
    _require(
        not _git_output(selector_repository, "status", "--porcelain"),
        "Causal4D selector repository is dirty",
    )
    selector = (selector_repository / SELECTOR_RELATIVE_PATH).resolve(strict=True)
    _require(
        selector.is_file()
        and not selector.is_symlink()
        and selector.parent.parent.parent == selector_repository,
        "selector source path changed",
    )
    requested_selector = Path(
        _argument_value(stage_arguments, "--generic-selector-source")
    ).resolve(strict=True)
    _require(requested_selector == selector, "stage received another selector source")
    _require(
        selector.stat().st_size == CORRECTED_SELECTOR_BYTE_COUNT
        and _file_sha256(selector) == CORRECTED_SELECTOR_SHA256,
        "selector source bytes changed",
    )

    script = repository / "scripts" / "remote" / STAGE_SCRIPT
    module = _load_stage(script)
    _require(
        getattr(module, "GENERIC_SELECTOR_SHA256", None)
        == PREVIOUS_SELECTOR_SHA256,
        "locked stage no longer carries the superseded selector digest",
    )
    _require(
        getattr(module, "SAM2_REPOSITORY_REVISION", None)
        == "2b90b9f5ceec907a1c18123530e92e794ad901a4",
        "locked stage changed the SAM2 source revision",
    )
    dynamic_module = cast(Any, module)
    with activate_joint_sparse_physical_runtime_v5():
        patch_joint_sparse_physical_stage_v5(
            module,
            stage="stage-prefix",
            repository=repository,
            execution_lock=execution_lock,
        )
        dynamic_module.GENERIC_SELECTOR_SHA256 = CORRECTED_SELECTOR_SHA256
        previous = sys.argv
        sys.argv = [str(script), *stage_arguments]
        try:
            return int(dynamic_module.main())
        finally:
            sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
