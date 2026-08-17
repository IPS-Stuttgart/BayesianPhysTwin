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

REPAIR_SCHEMA = "bayesian-phystwin.deform360-v6-stage-selector-consumer-identity-repair"
REPAIR_ID = "aea2506a8c648fcbaad460ae6eb0311801466015268271c5492bac9a6e1d2bae"
API_REPAIR_SCHEMA = "bayesian-phystwin.deform360-v6-selector-api-compatibility-repair"
API_REPAIR_ID = "5502830e01585cb1bb208d2d49e05d1f5e1d164dd707c5ff291038949dd0917c"
EXECUTION_AMENDMENT_ID = (
    "f8ed525480a6a96265af3cd58e62a96bf1ed748294d0af02aa6386763b993b7f"
)
PREDECESSOR_REPAIR_ID = (
    "d7e516ced90469589c3e4c3c12672a503fe8bbdb3a6f3316d852c266fd0f3d90"
)
CAUSAL4D_REVISION = "50e3682a5dbf976b20cc9115b6e7a975d0144ea5"
SELECTOR_RELATIVE_PATH = Path("src/causal4d_public/deform360_object_sam2.py")
PREVIOUS_SELECTOR_SHA256 = (
    "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
)
CORRECTED_SELECTOR_SHA256 = (
    "c10391578c73dde47fbce160312559a7e638007e9053ec89373fe575cc64d7e5"
)
SELECTOR_BYTE_COUNT = 17_310
PHYSICAL_WRAPPER_SHA256 = (
    "061fea23aeb83cbaeada9335417d99795de886c8ee6c6ae1013bddfe79bddb37"
)
STAGE_SCRIPT = "stage_deform360_bias_aware_prediction_prefix.py"
STAGE_SCRIPT_SHA256 = "a90578e8a83e5a72388b86f25c6b7b9dee872b75e2919c352e3a3a3ea431e5d6"
SAM2_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SELECTOR_CLASS_NAME = "DeformableObjectSam2VideoPredictor"
SELECTOR_EXISTING_METHOD = "select_initial_mask"
SELECTOR_REQUIRED_METHOD = "select_initial_mask_from_rgb"


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


def load_stage_selector_identity_repair(path: str | Path) -> Mapping[str, Any]:
    """Validate the exact target-closed stage-consumer identity repair."""

    payload = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    repair = _mapping(payload, name="stage selector identity repair")
    identity = {key: value for key, value in repair.items() if key != "repair_id"}
    _require(
        repair.get("schema") == REPAIR_SCHEMA
        and repair.get("schema_version") == 1
        and repair.get("repair_id") == content_id(identity) == REPAIR_ID,
        "stage selector identity repair changed",
    )
    _require(
        repair.get("superseded_execution_amendment_id") == EXECUTION_AMENDMENT_ID
        and repair.get("predecessor_selector_repair_id") == PREDECESSOR_REPAIR_ID,
        "stage selector repair lineage changed",
    )
    correction = _mapping(repair.get("correction"), name="selector correction")
    _require(
        correction
        == {
            "application": "process-local-loaded-module-constant-only",
            "consumer_field": "GENERIC_SELECTOR_SHA256",
            "consumer_file_sha256": STAGE_SCRIPT_SHA256,
            "consumer_path": f"scripts/remote/{STAGE_SCRIPT}",
            "corrected_expected_sha256": CORRECTED_SELECTOR_SHA256,
            "previous_expected_sha256": PREVIOUS_SELECTOR_SHA256,
            "selector_byte_count": SELECTOR_BYTE_COUNT,
            "selector_path": SELECTOR_RELATIVE_PATH.as_posix(),
            "selector_repository": "IPS-Stuttgart/Causal4D",
            "selector_repository_revision": CAUSAL4D_REVISION,
            "selector_semantics": "deform360-object-sam2-generic-selector",
        },
        "stage selector correction changed",
    )
    failed = _mapping(repair.get("failed_execution_evidence"), name="failure")
    _require(
        failed.get("workflow_run_id") == 31513816637
        and failed.get("workflow_run_attempt") == 1
        and failed.get("source_revision") == "dba748cafc1979dd697f99fb8aa70dc1cfaf9b81"
        and failed.get("artifact_id") == 9110649986
        and failed.get("artifact_sha256")
        == "6db988a14351b9fa8744c5e42b42f6d87f06f1cdaf3ff0607e773d3748bdc4b1"
        and failed.get("execution_receipt_id")
        == "741968a414984dca9c8c2dab2efbe716a151877d7ac7946830240bd292a47eee"
        and failed.get("terminal_stage") == "stage-prefix:026-sock-cloth-ep0007"
        and failed.get("exit_code") == 1
        and failed.get("physical_manifest_count") == 0
        and failed.get("source_prediction_seal_count") == 0
        and failed.get("error_type") == "ValueError"
        and failed.get("error_message") == "selector changed",
        "stage selector failure evidence changed",
    )
    expected_scope = {
        "camera_panel_changed": False,
        "candidate_roster_changed": False,
        "claim_authorized": False,
        "loss_or_gate_changed": False,
        "model_family_changed": False,
        "model_size_changed": False,
        "replacement_allowed": False,
        "runtime_expected_identity_only": True,
        "selector_algorithm_changed": False,
        "selector_bytes_changed": False,
        "selector_repository_revision_changed": False,
        "source_cohort_changed": False,
    }
    _require(repair.get("repair_scope") == expected_scope, "repair scope changed")
    boundary = _mapping(repair.get("information_boundary"), name="boundary")
    _require(
        bool(boundary) and all(value is False for value in boundary.values()),
        "information boundary changed",
    )
    authorization = _mapping(
        repair.get("execution_authorization"), name="authorization"
    )
    _require(
        authorization.get("event") == "push-to-protected-main-after-reviewed-merge"
        and authorization.get("runner_name") == "workstation2"
        and authorization.get("runner_labels")
        == ["self-hosted", "Linux", "X64", "nvidia-smi"]
        and authorization.get("source_prediction_batch_required_before_suffix_access")
        is True
        and authorization.get("fresh_target_selection_authorized") is False
        and authorization.get("fresh_target_payload_access_authorized") is False,
        "execution authorization changed",
    )
    return repair


def load_selector_api_compatibility_repair(path: str | Path) -> Mapping[str, Any]:
    """Validate the exact target-closed selector API compatibility repair."""

    payload = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    repair = _mapping(payload, name="selector API compatibility repair")
    identity = {key: value for key, value in repair.items() if key != "repair_id"}
    _require(
        repair.get("schema") == API_REPAIR_SCHEMA
        and repair.get("schema_version") == 1
        and repair.get("repair_id") == content_id(identity) == API_REPAIR_ID,
        "selector API compatibility repair changed",
    )
    _require(
        repair.get("execution_amendment_id") == EXECUTION_AMENDMENT_ID
        and repair.get("predecessor_stage_selector_repair_id") == REPAIR_ID,
        "selector API repair lineage changed",
    )
    correction = _mapping(repair.get("correction"), name="API correction")
    _require(
        correction
        == {
            "application": "process-local-selector-class-adapter",
            "consumer_file_sha256": STAGE_SCRIPT_SHA256,
            "consumer_path": f"scripts/remote/{STAGE_SCRIPT}",
            "consumer_required_method": SELECTOR_REQUIRED_METHOD,
            "delegation": (
                "existing-select-initial-mask-with-exact-rgb-reader-override"
            ),
            "producer_class": SELECTOR_CLASS_NAME,
            "producer_existing_method": SELECTOR_EXISTING_METHOD,
            "selector_byte_count": SELECTOR_BYTE_COUNT,
            "selector_file_sha256": CORRECTED_SELECTOR_SHA256,
            "selector_path": SELECTOR_RELATIVE_PATH.as_posix(),
            "selector_repository": "IPS-Stuttgart/Causal4D",
            "selector_repository_revision": CAUSAL4D_REVISION,
        },
        "selector API correction changed",
    )
    failed = _mapping(repair.get("failed_execution_evidence"), name="failure")
    _require(
        failed.get("workflow_run_id") == 31522573008
        and failed.get("workflow_run_attempt") == 1
        and failed.get("source_revision") == "e7c303bbac8af462c1437dfc9fb57deaa5537d8f"
        and failed.get("artifact_id") == 9113689469
        and failed.get("artifact_digest")
        == "sha256:e082d5bccf2d8ac6476396aad9ca0eac9d906521a3069e4867d8457d5ebef417"
        and failed.get("execution_receipt_id")
        == "d4012d47004669d4a220e9f57fbac19a3b514da97ca91280ff64a4d9b8922acf"
        and failed.get("terminal_stage") == "stage-prefix:026-sock-cloth-ep0007"
        and failed.get("exit_code") == 1
        and failed.get("physical_manifest_count") == 0
        and failed.get("source_prediction_seal_count") == 0
        and failed.get("error_type") == "AttributeError"
        and failed.get("error_message")
        == (
            "'DeformableObjectSam2VideoPredictor' object has no attribute "
            "'select_initial_mask_from_rgb'"
        ),
        "selector API failure evidence changed",
    )
    expected_scope = {
        "camera_panel_changed": False,
        "candidate_roster_changed": False,
        "claim_authorized": False,
        "consumer_bytes_changed": False,
        "loss_or_gate_changed": False,
        "model_family_changed": False,
        "model_size_changed": False,
        "replacement_allowed": False,
        "runtime_api_adapter_only": True,
        "selector_algorithm_changed": False,
        "selector_bytes_changed": False,
        "selector_repository_revision_changed": False,
        "source_cohort_changed": False,
    }
    _require(repair.get("repair_scope") == expected_scope, "API repair scope changed")
    boundary = _mapping(repair.get("information_boundary"), name="boundary")
    _require(
        bool(boundary) and all(value is False for value in boundary.values()),
        "information boundary changed",
    )
    authorization = _mapping(
        repair.get("execution_authorization"), name="authorization"
    )
    _require(
        authorization.get("event") == "push-to-protected-main-after-reviewed-merge"
        and authorization.get("runner_name") == "workstation2"
        and authorization.get("runner_labels")
        == ["self-hosted", "Linux", "X64", "nvidia-smi"]
        and authorization.get("source_prediction_batch_required_before_suffix_access")
        is True
        and authorization.get("fresh_target_selection_authorized") is False
        and authorization.get("fresh_target_payload_access_authorized") is False,
        "execution authorization changed",
    )
    return repair


def _load_stage(path: Path) -> ModuleType:
    name = "_deform360_v6_stage_selector_identity_repair"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load stage-prefix script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_prefix_rgb_selector_adapter(module: ModuleType) -> None:
    dynamic_module = cast(Any, module)
    candidate_loader = getattr(dynamic_module, "_load_selector_class", None)
    _require(callable(candidate_loader), "stage selector loader changed")
    original_loader = cast(Any, candidate_loader)

    def load_selector_class(source: Path) -> Any:
        selector_class = original_loader(source)
        _require(
            getattr(selector_class, "__name__", None) == SELECTOR_CLASS_NAME,
            "selector class changed",
        )
        _require(
            not hasattr(selector_class, SELECTOR_REQUIRED_METHOD),
            "selector now supplies the required RGB method",
        )
        _require(
            callable(getattr(selector_class, SELECTOR_EXISTING_METHOD, None))
            and callable(getattr(selector_class, "_first_frame_rgb", None)),
            "selector delegation surface changed",
        )

        def select_initial_mask_from_rgb(
            self: Any,
            rgb: Any,
            *,
            camera: str,
            video_name: str,
        ) -> tuple[Any, Any]:
            _require(bool(camera) and Path(camera).name == camera, "camera changed")
            _require(
                bool(video_name) and Path(video_name).name == video_name,
                "video name changed",
            )
            synthetic_path = Path(camera) / video_name
            _require(hasattr(self, "__dict__"), "selector instance state changed")
            instance_state = vars(self)
            had_instance_reader = "_first_frame_rgb" in instance_state
            previous_instance_reader = instance_state.get("_first_frame_rgb")

            def read_exact_rgb(requested_path: Path) -> Any:
                _require(
                    Path(requested_path) == synthetic_path,
                    "selector requested another frame",
                )
                return rgb

            instance_state["_first_frame_rgb"] = read_exact_rgb
            try:
                return cast(
                    tuple[Any, Any],
                    getattr(self, SELECTOR_EXISTING_METHOD)(synthetic_path),
                )
            finally:
                if had_instance_reader:
                    instance_state["_first_frame_rgb"] = previous_instance_reader
                else:
                    instance_state.pop("_first_frame_rgb", None)

        setattr(selector_class, SELECTOR_REQUIRED_METHOD, select_initial_mask_from_rgb)
        return selector_class

    dynamic_module._load_selector_class = load_selector_class


def _write_activation_marker(path: Path) -> None:
    marker = {
        "application": "process-local-loaded-module-constant-only",
        "api_compatibility": {
            "application": "process-local-selector-class-adapter",
            "consumer_required_method": SELECTOR_REQUIRED_METHOD,
            "delegated_method": SELECTOR_EXISTING_METHOD,
            "repair_id": API_REPAIR_ID,
        },
        "consumer_file_sha256": STAGE_SCRIPT_SHA256,
        "corrected_expected_sha256": CORRECTED_SELECTOR_SHA256,
        "previous_expected_sha256": PREVIOUS_SELECTOR_SHA256,
        "repair_id": REPAIR_ID,
        "selector_file_sha256": CORRECTED_SELECTOR_SHA256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(marker, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--execution-repo", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--runtime-repair", type=Path, required=True)
    parser.add_argument("--api-repair", type=Path, required=True)
    parser.add_argument("--activation-marker", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage-prefix",), required=True)
    return parser.parse_known_args()


def main() -> int:
    args, stage_arguments = _parse_args()
    repository = args.execution_repo.resolve(strict=True)
    execution_lock = args.execution_lock.resolve(strict=True)
    load_stage_selector_identity_repair(args.runtime_repair)
    load_selector_api_compatibility_repair(args.api_repair)
    validate_joint_sparse_physical_execution_v5(
        execution_lock,
        repository=repository,
    )
    physical_wrapper = (
        repository
        / "scripts"
        / "remote"
        / "run_deform360_joint_sparse_physical_source_v5.py"
    )
    _require(
        _file_sha256(physical_wrapper) == PHYSICAL_WRAPPER_SHA256,
        "physical wrapper changed",
    )
    protocol_path = Path(_argument_value(stage_arguments, "--protocol")).resolve()
    _require(protocol_path == execution_lock, "stage protocol must be the v5 lock")

    selector = Path(
        _argument_value(stage_arguments, "--generic-selector-source")
    ).resolve(strict=True)
    _require(
        selector.is_file() and not selector.is_symlink(),
        "selector source path is invalid",
    )
    selector_repository = selector.parent.parent.parent
    _require(
        selector.relative_to(selector_repository) == SELECTOR_RELATIVE_PATH,
        "selector source path changed",
    )
    _require(
        (selector_repository / ".git").exists()
        and _git_output(selector_repository, "rev-parse", "HEAD") == CAUSAL4D_REVISION
        and not _git_output(selector_repository, "status", "--porcelain"),
        "selector repository identity changed",
    )
    _require(
        selector.stat().st_size == SELECTOR_BYTE_COUNT
        and _file_sha256(selector) == CORRECTED_SELECTOR_SHA256,
        "selector source bytes changed",
    )

    script = repository / "scripts" / "remote" / STAGE_SCRIPT
    _require(_file_sha256(script) == STAGE_SCRIPT_SHA256, "stage script changed")
    module = _load_stage(script)
    _require(
        getattr(module, "GENERIC_SELECTOR_SHA256", None) == PREVIOUS_SELECTOR_SHA256,
        "locked stage no longer carries the superseded selector digest",
    )
    _require(
        getattr(module, "SAM2_REPOSITORY_REVISION", None) == SAM2_REVISION,
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
        _install_prefix_rgb_selector_adapter(module)
        _write_activation_marker(args.activation_marker.resolve())
        previous = sys.argv
        sys.argv = [str(script), *stage_arguments]
        try:
            return int(dynamic_module.main())
        finally:
            sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
