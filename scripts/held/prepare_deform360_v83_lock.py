#!/usr/bin/env python3
"""Create the fresh Deform360 v8.3 process-isolated calibration lock."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from types import ModuleType
from typing import Any, Mapping


EXPECTED_HOST = "workstation2"
HELD_BASE = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
HELD_ROOT = HELD_BASE / "held-v83"
LOCK_PATH = HELD_ROOT / "calibration-lock.json"
QUALIFICATION_BASE = Path("/mnt/corsair/florianpfaff")
QUALIFICATION_ROOT_PREFIX = "bpt-process-isolation-qualification-"
QUALIFICATION_EVIDENCE_NAME = "process-isolation-qualification.json"
V82_TECHNICAL_FAILURE_ARCHIVE = (
    HELD_BASE / "held-v82-attempt-1-technical-failure"
)
V82_TECHNICAL_FAILURE_REPORT = (
    V82_TECHNICAL_FAILURE_ARCHIVE / "execution-technical-failure-attempt1.json"
)
V82_TECHNICAL_FAILURE_POINTER = (
    HELD_BASE / "held-v82-attempt-1-technical-failure-pointer.json"
)
V82_TECHNICAL_FAILURE_COMPLETION = (
    HELD_BASE / "held-v82-attempt-1-technical-failure-completion.json"
)
SOURCE_RELATIVE = Path("scripts/held/prepare_deform360_v83_lock.py")
SUPPORT_RELATIVE = Path("scripts/held/prepare_deform360_v8_lock.py")
ATTEMPT5_RESULT_RELATIVE = Path(
    "results/sota/deform360_held_v81_attempt5_admission_inconclusive.json"
)
SKIPPED_ATTEMPT5_EXTERNAL_BINDINGS = frozenset(
    {
        "v8_external_admission_metadata_only_replay",
        "v8_external_admission_replay_code_binding",
    }
)
PROCESS_SOURCE_BINDINGS: Mapping[str, str] = {
    "held_v83_lock_preparer_source": SOURCE_RELATIVE.as_posix(),
    "process_isolation_qualification_operator_source": (
        "scripts/development/qualify_deform360_process_isolation.py"
    ),
    "process_isolation_qualification_sealer_source": (
        "scripts/held/seal_deform360_process_isolation_qualification.py"
    ),
    "held_v83_process_isolation_source": (
        "src/bayesian_phystwin/deform360_case_process_isolation.py"
    ),
    "held_v83_process_isolation_worker_source": (
        "scripts/held/run_deform360_isolated_reconstruction.py"
    ),
    "held_v8_outcome_driver_source": (
        "src/bayesian_phystwin/deform360_held_v8_outcome_driver.py"
    ),
    "held_v83_gsplat_runtime_adapter_source": (
        "src/bayesian_phystwin/deform360_held_v83_gsplat_runtime.py"
    ),
    "held_v83_viser_process_churn_guard_source": (
        "src/bayesian_phystwin/deform360_held_v83_viser_guard.py"
    ),
    "held_v82_technical_failure_integrity_source": (
        "src/bayesian_phystwin/deform360_held_v82_technical_failure.py"
    ),
    "held_v82_technical_failure_sealer_source": (
        "scripts/held/seal_deform360_v82_technical_failure.py"
    ),
    "held_official_reconstruction_numerical_source": (
        "src/bayesian_phystwin/deform360_held_outcome_reconstruction.py"
    ),
    "v81_attempt5_admission_inconclusive_result": (
        ATTEMPT5_RESULT_RELATIVE.as_posix()
    ),
}
QUALIFIED_SOURCE_MAP: Mapping[str, str] = {
    "qualification_source_sha256": (
        "process_isolation_qualification_operator_source"
    ),
    "numerical_adapter_source_sha256": (
        "held_official_reconstruction_numerical_source"
    ),
    "isolation_source_sha256": "held_v83_process_isolation_source",
    "worker_source_sha256": "held_v83_process_isolation_worker_source",
    "worker_runtime_source_sha256": "held_v83_gsplat_runtime_adapter_source",
    "viser_guard_source_sha256": (
        "held_v83_viser_process_churn_guard_source"
    ),
    "outcome_driver_source_sha256": "held_v8_outcome_driver_source",
    "sealer_source_sha256": "process_isolation_qualification_sealer_source",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _support(code: Path) -> ModuleType:
    return _load_module(code / SUPPORT_RELATIVE, "_deform360_v83_lock_support")


def _import_runtime_modules(code: Path) -> tuple[Any, Any, Any]:
    source = code / "src"
    sys.path.insert(0, os.fspath(source))
    try:
        from bayesian_phystwin import deform360_held_v8_builders as builders
        from bayesian_phystwin import deform360_held_v8_protocol as protocol
        from bayesian_phystwin import (
            deform360_held_v8_replacement_source as replacement,
        )
        from bayesian_phystwin import (
            deform360_process_isolation_qualification as qualification,
        )
    finally:
        sys.path.pop(0)
    for module, label in (
        (builders, "builders"),
        (protocol, "protocol"),
        (replacement, "replacement"),
        (qualification, "process qualification"),
    ):
        path = Path(module.__file__).resolve()
        _require(
            path.is_relative_to(source),
            f"{label} imported outside the exact source tree",
        )
    return protocol, replacement, qualification


def qualification_paths(head: str) -> tuple[Path, Path, Path]:
    _require(
        len(head) in {40, 64}
        and all(character in "0123456789abcdef" for character in head),
        "qualification source revision is invalid",
    )
    root = QUALIFICATION_BASE / f"{QUALIFICATION_ROOT_PREFIX}{head}"
    return (
        root,
        root / QUALIFICATION_EVIDENCE_NAME,
        Path(f"{root}-integrity-completion.json"),
    )


def _external_bindings(support: ModuleType) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, (
        path,
        expected_sha256,
        required_mode,
    ) in support._EXPECTED_EXTERNAL_FILES.items():
        if name in SKIPPED_ATTEMPT5_EXTERNAL_BINDINGS:
            continue
        _require(
            support._valid_sha256(expected_sha256),
            f"{name} expected SHA-256 is absent",
        )
        observed = support._sha256_file(
            path,
            role=name.replace("_", " "),
            required_mode=required_mode,
        )
        _require(observed == expected_sha256, f"{name} SHA-256 changed")
        result[name] = observed
        expected_artifact = support._EXPECTED_EXTERNAL_ARTIFACT_SHA256.get(name)
        if expected_artifact is None:
            continue
        _require(
            support._valid_sha256(expected_artifact),
            f"{name} artifact SHA-256 is absent",
        )
        _, payload, _ = support._read_file(
            path,
            role=f"{name.replace('_', ' ')} artifact",
            required_mode=required_mode,
        )
        artifact = json.loads(payload.decode("utf-8"))
        _require(
            isinstance(artifact, dict)
            and artifact.get("artifact_sha256") == expected_artifact,
            f"{name} artifact identity changed",
        )
        unsigned = dict(artifact)
        unsigned.pop("artifact_sha256")
        _require(
            hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
            == expected_artifact,
            f"{name} canonical artifact digest changed",
        )
        result[f"{name}_artifact"] = expected_artifact
    result["pinned_python_executable_target"] = support._validate_pinned_python()
    return result


def _local_bindings(code: Path, support: ModuleType) -> dict[str, str]:
    bindings = support._local_file_bindings(code)
    for name, relative in PROCESS_SOURCE_BINDINGS.items():
        bindings[name] = support._sha256_file(
            code / relative,
            role=name.replace("_", " "),
        )
    return bindings


def _validate_qualification(
    *,
    code: Path,
    head: str,
    qualification: Any,
    local_bindings: Mapping[str, str],
) -> tuple[dict[str, Any], Path, Path, Path]:
    root, evidence, completion = qualification_paths(head)
    lineage = qualification.validate_process_isolation_qualification_lineage(
        evidence_path=evidence,
        completion_path=completion,
        expected_source_head=head,
        verify_content_inventory=True,
    )
    integrity = lineage["process_isolation_qualification_integrity"]
    _require(
        integrity["terminal_outcome"] == "qualified"
        and integrity["admission_eligible"] is True,
        "process-isolation qualification did not pass",
    )
    for integrity_name, binding_name in QUALIFIED_SOURCE_MAP.items():
        _require(
            integrity[integrity_name] == local_bindings[binding_name],
            f"{binding_name} differs from the qualified source",
        )
    _require(
        Path(__file__).resolve() == code / SOURCE_RELATIVE,
        "v8.3 lock preparer is outside the exact source tree",
    )
    return lineage, root, evidence, completion


def prospective_bindings(
    source_code: str | Path,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any], tuple[Path, Path, Path]]:
    code_hint = Path(os.path.abspath(os.fspath(source_code)))
    support = _support(code_hint)
    provenance = support._validate_repository(code_hint)
    code = provenance["root"]
    protocol, replacement, qualification = _import_runtime_modules(code)
    _require(
        protocol.PROTOCOL_ID == "deform360-held-online-belief-v8.3"
        and protocol.EXECUTION_ATTEMPT == 1,
        "source protocol is not fresh v8.3 attempt 1",
    )

    inherited = support._inherited_v7_bindings()
    bindings = dict(inherited)
    bindings["v7_inherited_immutable_bindings_contract"] = hashlib.sha256(
        _canonical_bytes(inherited)
    ).hexdigest()
    bindings.update(_external_bindings(support))
    local = _local_bindings(code, support)
    support._validate_attempt2_operator_source_lineage(local)
    support._validate_attempt3_archive_lineage(local)
    bindings.update(local)
    attempt4 = support._validate_attempt4_archive_lineage(local, protocol)
    v82_failure = protocol.validate_v82_technical_failure_lineage(
        archive_path=V82_TECHNICAL_FAILURE_ARCHIVE,
        report_path=V82_TECHNICAL_FAILURE_REPORT,
        pointer_path=V82_TECHNICAL_FAILURE_POINTER,
        completion_path=V82_TECHNICAL_FAILURE_COMPLETION,
        verify_content_inventory=True,
    )
    v82_report, _v82_report_record = protocol.v82_technical_failure.load_signed(
        V82_TECHNICAL_FAILURE_REPORT,
        role="v8.2 technical-failure report",
    )
    _require(
        v82_report["executed_operator_source"]["sha256"]
        == local["held_v82_technical_failure_sealer_source"],
        "v8.2 technical-failure operator differs from the current source",
    )
    lineage, qualification_root, evidence, completion = _validate_qualification(
        code=code,
        head=provenance["head"],
        qualification=qualification,
        local_bindings=local,
    )
    integrity = lineage["process_isolation_qualification_integrity"]
    processing_revision, processing_tree = support._processing_revision()
    _require(
        processing_revision == replacement.PROCESSING_CODE_REVISION,
        "Deform360 processing revision changed",
    )

    bindings.update(
        {
            "method_deployed_snapshot_tree": provenance["tree_sha256"],
            "method_head_text_sha256": provenance["head_text_sha256"],
            "method_deployed_commit_text_sha256": provenance["head_text_sha256"],
            "replacement_source_inventory_contract": protocol.held_contract_sha256(
                replacement.REPLACEMENT_SOURCE_INVENTORY_CONTRACT
            ),
            "held_v8_confirmation_source_contract": (
                protocol.confirmation_source.confirmation_source_contract_sha256()
            ),
            "replacement_automatic_twin_admission_contract": (
                protocol.REPLACEMENT_AUTOMATIC_TWIN_ADMISSION_CONTRACT_SHA256
            ),
            "frame_zero_exact_eight_subset_bounded_audit_contract": (
                protocol.frame_zero_assets.EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT_SHA256
            ),
            "frozen_query_field_contract": protocol.held_contract_sha256(
                protocol.FROZEN_FIELD_CONTRACT
            ),
            "center_exclusion_contract": (
                protocol.query_artifacts.CENTER_EXCLUSION_CONTRACT_SHA256
            ),
            "primary_method_contract": protocol.held_contract_sha256(
                protocol.PRIMARY_METHOD
            ),
            "process_isolation_policy_contract": protocol.held_contract_sha256(
                protocol.PROCESS_ISOLATION_POLICY_CONTRACT
            ),
            "post_case_resource_boundary_contract": protocol.held_contract_sha256(
                protocol.POST_CASE_RESOURCE_BOUNDARY_CONTRACT
            ),
            "v8_attempt3_postseal_noncode_inventory": (
                support._V8_ATTEMPT3_ARCHIVE_INVENTORY_SHA256
            ),
            "v8_attempt3_postseal_noncode_inventory_contract": hashlib.sha256(
                _canonical_bytes(support._attempt3_archive_inventory_contract())
            ).hexdigest(),
            "v8_attempt4_postseal_noncode_inventory": (
                support._V8_ATTEMPT4_ARCHIVE_INVENTORY_SHA256
            ),
            "v8_attempt4_withdrawal_lineage_contract": hashlib.sha256(
                _canonical_bytes(attempt4)
            ).hexdigest(),
            "v82_technical_failure_report": v82_failure[
                "v82_technical_failure_report"
            ]["sha256"],
            "v82_technical_failure_report_artifact": v82_failure[
                "v82_technical_failure_report"
            ]["artifact_sha256"],
            "v82_technical_failure_pointer": v82_failure[
                "v82_technical_failure_pointer"
            ]["sha256"],
            "v82_technical_failure_pointer_artifact": v82_failure[
                "v82_technical_failure_pointer"
            ]["artifact_sha256"],
            "v82_technical_failure_integrity_completion": v82_failure[
                "v82_technical_failure_integrity_completion"
            ]["sha256"],
            "v82_technical_failure_integrity_completion_artifact": v82_failure[
                "v82_technical_failure_integrity_completion"
            ]["artifact_sha256"],
            "v82_technical_failure_archive_inventory": v82_failure[
                "v82_technical_failure_archive_integrity"
            ]["inventory_sha256"],
            "v82_technical_failure_lineage_contract": hashlib.sha256(
                _canonical_bytes(v82_failure)
            ).hexdigest(),
            "process_isolation_qualification_attempt": lineage[
                "process_isolation_qualification_attempt"
            ]["sha256"],
            "process_isolation_qualification_attempt_artifact": lineage[
                "process_isolation_qualification_attempt"
            ]["artifact_sha256"],
            "process_isolation_qualification_evidence": lineage[
                "process_isolation_qualification_evidence"
            ]["sha256"],
            "process_isolation_qualification_evidence_artifact": lineage[
                "process_isolation_qualification_evidence"
            ]["artifact_sha256"],
            "process_isolation_qualification_integrity_completion": lineage[
                "process_isolation_qualification_integrity_completion"
            ]["sha256"],
            "process_isolation_qualification_integrity_completion_artifact": lineage[
                "process_isolation_qualification_integrity_completion"
            ]["artifact_sha256"],
            "process_isolation_qualification_inventory": integrity[
                "inventory_sha256"
            ],
            "process_isolation_qualification_metadata_inventory": integrity[
                "metadata_inventory_sha256"
            ],
            "deform360_processing_head_text_sha256": support._sha256_text(
                processing_revision
            ),
            "deform360_processing_tree_text_sha256": support._sha256_text(
                processing_tree
            ),
            "hf_dataset_revision_text_sha256": support._sha256_text(
                replacement.HF_DATASET_REVISION
            ),
        }
    )
    _require(
        all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in bindings.values()
        ),
        "prospective v8.3 binding is not SHA-256",
    )
    return (
        dict(sorted(bindings.items())),
        provenance,
        lineage,
        (qualification_root, evidence, completion),
    )


def create_lock_and_deployment(source_code: str | Path) -> dict[str, Any]:
    _require(socket.gethostname() == EXPECTED_HOST, "formal lock must run on workstation2")
    _require(not os.path.lexists(HELD_ROOT), "formal held-v83 root is not fresh")
    bindings, provenance, _lineage, qualification_paths_value = prospective_bindings(
        source_code
    )
    source = provenance["root"]
    head = provenance["head"]
    support = _support(source)
    protocol, _replacement, _qualification = _import_runtime_modules(source)
    stage = HELD_BASE / f".held-v83-code-stage-{head}"
    destination = HELD_ROOT / f"code-{head}"
    staged = support._clone_staged_deployment(source, head, stage)
    _require(
        staged["tree_sha256"] == provenance["tree_sha256"],
        "staged deployment tree differs from source",
    )
    capability = protocol.prepare_fresh_held_root(HELD_ROOT)
    deployment_moved = False
    try:
        lock = protocol.create_calibration_protocol_lock(
            LOCK_PATH,
            held_root=HELD_ROOT,
            fresh_root_capability=capability,
            immutable_bindings=bindings,
            v7_withdrawal_report_path=support._V7_WITHDRAWAL,
            development_decision_path=support._OPEN27_DECISION,
            attempt3_withdrawal_report_path=support._V8_ATTEMPT3_WITHDRAWAL_REPORT,
            attempt3_withdrawal_pointer_path=support._V8_ATTEMPT3_WITHDRAWAL_POINTER,
            attempt3_withdrawal_integrity_completion_path=(
                support._V8_ATTEMPT3_INTEGRITY_COMPLETION
            ),
            attempt4_withdrawal_report_path=support._V8_ATTEMPT4_WITHDRAWAL_REPORT,
            attempt4_withdrawal_pointer_path=support._V8_ATTEMPT4_WITHDRAWAL_POINTER,
            attempt4_withdrawal_integrity_completion_path=(
                support._V8_ATTEMPT4_INTEGRITY_COMPLETION
            ),
            v82_technical_failure_archive_path=V82_TECHNICAL_FAILURE_ARCHIVE,
            v82_technical_failure_report_path=V82_TECHNICAL_FAILURE_REPORT,
            v82_technical_failure_pointer_path=V82_TECHNICAL_FAILURE_POINTER,
            v82_technical_failure_completion_path=V82_TECHNICAL_FAILURE_COMPLETION,
            process_isolation_qualification_path=qualification_paths_value[1],
            process_isolation_qualification_completion_path=qualification_paths_value[
                2
            ],
        )
        _require(not os.path.lexists(destination), "deployment destination exists")
        os.chmod(stage, 0o755, follow_symlinks=False)
        os.rename(stage, destination)
        deployment_moved = True
        os.chmod(destination, 0o555, follow_symlinks=False)
        support._require_deployed_read_only(destination)
        deployed = support._validate_repository(destination)
        _require(
            deployed["head"] == head
            and deployed["tree_sha256"] == bindings["method_deployed_snapshot_tree"],
            "deployed repository differs after atomic move",
        )
        _require(
            protocol.validate_protocol_lock(LOCK_PATH) == lock,
            "calibration lock changed after deployment",
        )
        return {
            "operation": "created_held_v83_calibration_lock_and_deployment",
            "protocol_id": lock["protocol_id"],
            "execution_attempt": lock["execution_attempt"],
            "lock_path": os.fspath(LOCK_PATH),
            "lock_file_sha256": support._sha256_file(
                LOCK_PATH,
                role="v8.3 calibration lock",
                required_mode=0o400,
            ),
            "lock_artifact_sha256": lock["artifact_sha256"],
            "deployed_code": os.fspath(destination),
            "deployed_head": head,
            "deployed_tree_sha256": deployed["tree_sha256"],
            "process_isolation_qualification_root": os.fspath(
                qualification_paths_value[0]
            ),
            "binding_count": len(bindings),
            "formal_root_was_absent": True,
        }
    except BaseException:
        if not deployment_moved and os.path.lexists(stage):
            for root, directories, files in os.walk(stage, topdown=False):
                for name in files:
                    os.chmod(Path(root) / name, 0o600, follow_symlinks=False)
                for name in directories:
                    os.chmod(Path(root) / name, 0o700, follow_symlinks=False)
            os.chmod(stage, 0o700, follow_symlinks=False)
            shutil.rmtree(stage)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-code", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--create", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    if arguments.preflight:
        bindings, provenance, lineage, paths = prospective_bindings(
            arguments.source_code
        )
        result = {
            "operation": "validated_held_v83_source_and_qualification",
            "source_head": provenance["head"],
            "source_tree_sha256": provenance["tree_sha256"],
            "binding_count": len(bindings),
            "process_isolation_qualification_root": os.fspath(paths[0]),
            "process_isolation_terminal_outcome": lineage[
                "process_isolation_qualification_integrity"
            ]["terminal_outcome"],
            "formal_root_exists": os.path.lexists(HELD_ROOT),
        }
    else:
        result = create_lock_and_deployment(arguments.source_code)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
