#!/usr/bin/env python3
"""Materialize source-only camera reuse for the sealed Deform360 v6 batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_joint_sparse_camera_recovery_v5_2 import (
    audit_deform360_joint_sparse_source_cameras_v5_2,
    validate_deform360_joint_sparse_camera_audit_v5_2,
)
from bayesian_phystwin.deform360_joint_sparse_source_evidence_v5 import (
    validate_deform360_joint_sparse_source_prediction_batch_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_runner_v5 import (
    validate_deform360_joint_sparse_source_prediction_plan_v5,
    validate_deform360_joint_sparse_source_prediction_receipt_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_runner_v5_2 import (
    build_deform360_joint_sparse_source_prediction_plan_v5_2,
)
from bayesian_phystwin.deform360_v6_source_camera_reuse import (
    EXECUTION_ARTIFACT_NAMES,
    TECHNICAL_FAILURE_RECEIPT_SCHEMA,
    build_deform360_v6_source_camera_reuse_execution_receipt,
    build_deform360_v6_source_camera_reuse_lineage,
    build_deform360_v6_source_camera_reuse_plan,
    build_deform360_v6_source_camera_reuse_preflight,
    build_deform360_v6_source_camera_reuse_technical_failure_receipt,
    validate_deform360_v6_source_camera_reuse_amendment,
    validate_deform360_v6_source_camera_reuse_amendment_bindings,
    validate_deform360_v6_source_camera_reuse_execution_receipt,
    validate_deform360_v6_source_camera_reuse_preflight,
    validate_deform360_v6_source_camera_reuse_receipt,
    validate_deform360_v6_source_camera_reuse_technical_failure_receipt,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, *, label: str) -> dict[str, Any]:
    return load_strict_json_object(path.resolve(strict=True), label=label)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    write_atomic_json(value, path, overwrite=False)


def _print(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False))


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--base-source-plan", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit-base")
    _common(audit)
    audit.add_argument("--input-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    rank = commands.add_parser("rank-reuse")
    _common(rank)
    rank.add_argument("--base-camera-audit", type=Path, required=True)
    rank.add_argument("--metric-prefix-plan", type=Path, required=True)
    rank.add_argument("--metric-batch-result", type=Path, required=True)
    rank.add_argument("--metric-files-root", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)

    combined = commands.add_parser("build-combined-plan")
    _common(combined)
    combined.add_argument("--base-camera-audit", type=Path, required=True)
    combined.add_argument("--preflight", type=Path, required=True)
    combined.add_argument("--metric-prefix-plan", type=Path, required=True)
    combined.add_argument("--results-root", type=Path, required=True)
    combined.add_argument("--prediction-root", type=Path, required=True)
    combined.add_argument("--metric-files-root", type=Path, required=True)
    combined.add_argument("--implementation-revision", required=True)
    combined.add_argument("--output-plan", type=Path, required=True)
    combined.add_argument("--output-receipt", type=Path, required=True)

    final_audit = commands.add_parser("audit-combined")
    final_audit.add_argument("--execution-lock", type=Path, required=True)
    final_audit.add_argument("--combined-plan", type=Path, required=True)
    final_audit.add_argument("--input-root", type=Path, required=True)
    final_audit.add_argument("--output", type=Path, required=True)

    lineage = commands.add_parser("build-lineage")
    _common(lineage)
    lineage.add_argument("--amendment", type=Path, required=True)
    lineage.add_argument("--base-prediction-batch", type=Path, required=True)
    lineage.add_argument("--base-prediction-receipt", type=Path, required=True)
    lineage.add_argument("--base-camera-audit", type=Path, required=True)
    lineage.add_argument("--preflight", type=Path, required=True)
    lineage.add_argument("--reuse-receipt", type=Path, required=True)
    lineage.add_argument("--combined-plan", type=Path, required=True)
    lineage.add_argument("--final-camera-audit", type=Path, required=True)
    lineage.add_argument("--metric-prefix-plan", type=Path, required=True)
    lineage.add_argument("--metric-batch-result", type=Path, required=True)
    lineage.add_argument("--output", type=Path, required=True)

    freeze = commands.add_parser("freeze-source-plan")
    freeze.add_argument("--execution-lock", type=Path, required=True)
    freeze.add_argument("--combined-plan", type=Path, required=True)
    freeze.add_argument("--final-camera-audit", type=Path, required=True)
    freeze.add_argument("--lineage", type=Path, required=True)
    freeze.add_argument("--implementation-revision", required=True)
    freeze.add_argument("--output", type=Path, required=True)

    seal = commands.add_parser("seal-execution")
    seal.add_argument("--amendment", type=Path, required=True)
    seal.add_argument("--execution-lock", type=Path, required=True)
    seal.add_argument("--source-revision", required=True)
    seal.add_argument("--runner-name", required=True)
    seal.add_argument("--workflow-run-id", type=int, required=True)
    seal.add_argument("--workflow-run-attempt", type=int, required=True)
    seal.add_argument("--base-run-id", type=int, required=True)
    seal.add_argument("--base-run-attempt", type=int, required=True)
    seal.add_argument("--base-head-sha", required=True)
    seal.add_argument("--base-artifact-id", type=int, required=True)
    seal.add_argument("--base-artifact-name", required=True)
    seal.add_argument("--base-artifact-digest-sha256", required=True)
    seal.add_argument("--artifact-root", type=Path, required=True)
    seal.add_argument("--source-plan", type=Path, required=True)
    seal.add_argument("--prediction-batch", type=Path, required=True)
    seal.add_argument("--prediction-receipt", type=Path, required=True)
    seal.add_argument("--source-seal-root", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)

    validate_execution = commands.add_parser("validate-execution")
    validate_execution.add_argument("--receipt", type=Path, required=True)

    retain_failure = commands.add_parser("retain-failure")
    retain_failure.add_argument("--amendment", type=Path, required=True)
    retain_failure.add_argument("--execution-lock", type=Path, required=True)
    retain_failure.add_argument("--source-revision", required=True)
    retain_failure.add_argument("--runner-name", required=True)
    retain_failure.add_argument("--workflow-run-id", type=int, required=True)
    retain_failure.add_argument("--workflow-run-attempt", type=int, required=True)
    retain_failure.add_argument("--base-run-id", type=int, required=True)
    retain_failure.add_argument("--base-run-attempt", type=int, required=True)
    retain_failure.add_argument("--base-head-sha", required=True)
    retain_failure.add_argument("--base-artifact-id", type=int, required=True)
    retain_failure.add_argument("--base-artifact-name", required=True)
    retain_failure.add_argument("--base-artifact-digest-sha256", required=True)
    retain_failure.add_argument("--terminal-stage", required=True)
    retain_failure.add_argument("--exit-code", type=int, required=True)
    retain_failure.add_argument("--artifact-root", type=Path, required=True)
    retain_failure.add_argument("--output", type=Path, required=True)

    validate_receipt = commands.add_parser("validate-receipt")
    validate_receipt.add_argument("--receipt", type=Path, required=True)

    validate = commands.add_parser("validate-amendment")
    validate.add_argument("--amendment", type=Path, required=True)
    return parser


def _base(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(
        arguments.execution_lock
    )
    source_path = arguments.base_source_plan.resolve(strict=True)
    source = validate_deform360_joint_sparse_source_prediction_plan_v5(
        _load(source_path, label="base source plan"), lock=lock
    )
    return lock, source, source_path


def _lineage(arguments: argparse.Namespace) -> dict[str, Any]:
    lock, base, base_path = _base(arguments)
    amendment_path = arguments.amendment.resolve(strict=True)
    amendment = validate_deform360_v6_source_camera_reuse_amendment(
        _load(amendment_path, label="camera reuse amendment")
    )
    batch_path = arguments.base_prediction_batch.resolve(strict=True)
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        _load(batch_path, label="base prediction batch"), lock
    )
    receipt_path = arguments.base_prediction_receipt.resolve(strict=True)
    receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5(
        _load(receipt_path, label="base prediction receipt"),
        lock=lock,
        plan=base,
        prediction_batch=batch,
        prediction_batch_file_sha256=_sha256(batch_path),
    )
    audit_path = arguments.base_camera_audit.resolve(strict=True)
    audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        _load(audit_path, label="base camera audit"), lock=lock
    )
    metric_plan_path = arguments.metric_prefix_plan.resolve(strict=True)
    metric_plan = _load(metric_plan_path, label="metric-prefix plan")
    metric_result_path = arguments.metric_batch_result.resolve(strict=True)
    metric_result = _load(metric_result_path, label="metric result")
    preflight_path = arguments.preflight.resolve(strict=True)
    preflight = validate_deform360_v6_source_camera_reuse_preflight(
        _load(preflight_path, label="camera reuse preflight"),
        lock=lock,
        base_source_plan=base,
        base_camera_audit=audit,
        metric_prefix_plan=metric_plan,
    )
    reuse_path = arguments.reuse_receipt.resolve(strict=True)
    reuse = validate_deform360_v6_source_camera_reuse_receipt(
        _load(reuse_path, label="camera reuse receipt")
    )
    combined_path = arguments.combined_plan.resolve(strict=True)
    combined = validate_deform360_joint_sparse_source_prediction_plan_v5(
        _load(combined_path, label="combined plan"), lock=lock
    )
    final_path = arguments.final_camera_audit.resolve(strict=True)
    final_audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        _load(final_path, label="final camera audit"), lock=lock
    )
    validate_deform360_v6_source_camera_reuse_amendment_bindings(
        amendment,
        execution_lock_id=lock["execution_lock_id"],
        execution_lock_file_sha256=_sha256(
            arguments.execution_lock.resolve(strict=True)
        ),
        base_source_plan_id=base["plan_id"],
        base_source_plan_file_sha256=_sha256(base_path),
        base_prediction_batch_id=batch["prediction_batch_id"],
        base_prediction_batch_file_sha256=_sha256(batch_path),
        base_prediction_receipt_id=receipt["receipt_id"],
        base_prediction_receipt_file_sha256=_sha256(receipt_path),
        metric_prefix_plan_id=metric_plan["plan_id"],
        metric_prefix_plan_file_sha256=_sha256(metric_plan_path),
        metric_batch_result_id=metric_result["result_id"],
        metric_batch_result_file_sha256=_sha256(metric_result_path),
        visual_production_result_id=metric_plan["visual_production_result_id"],
        visual_production_result_file_sha256=metric_result["source_artifacts"][
            "visual-production-result.json"
        ],
    )
    return build_deform360_v6_source_camera_reuse_lineage(
        lock=lock,
        execution_lock_file_sha256=_sha256(
            arguments.execution_lock.resolve(strict=True)
        ),
        amendment=amendment,
        amendment_file_sha256=_sha256(amendment_path),
        base_source_plan=base,
        base_source_plan_file_sha256=_sha256(base_path),
        base_prediction_batch=batch,
        base_prediction_batch_file_sha256=_sha256(batch_path),
        base_prediction_receipt=receipt,
        base_prediction_receipt_file_sha256=_sha256(receipt_path),
        base_camera_audit=audit,
        base_camera_audit_file_sha256=_sha256(audit_path),
        preflight=preflight,
        preflight_file_sha256=_sha256(preflight_path),
        reuse_receipt=reuse,
        reuse_receipt_file_sha256=_sha256(reuse_path),
        combined_plan=combined,
        combined_plan_file_sha256=_sha256(combined_path),
        final_camera_audit=final_audit,
        final_camera_audit_file_sha256=_sha256(final_path),
        metric_prefix_plan=metric_plan,
        metric_prefix_plan_file_sha256=_sha256(metric_plan_path),
        metric_batch_result=metric_result,
        metric_batch_result_file_sha256=_sha256(metric_result_path),
    )


def _seal_execution(arguments: argparse.Namespace) -> dict[str, Any]:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(
        arguments.execution_lock
    )
    amendment = validate_deform360_v6_source_camera_reuse_amendment(
        _load(arguments.amendment, label="camera reuse amendment")
    )
    artifact_root = arguments.artifact_root.resolve(strict=True)
    if (
        not artifact_root.is_dir()
        or artifact_root.is_symlink()
        or any(parent.is_symlink() for parent in artifact_root.parents)
    ):
        raise ValueError("artifact root must be an ordinary directory")
    artifacts: dict[str, str] = {}
    for name in sorted(EXECUTION_ARTIFACT_NAMES):
        path = artifact_root / f"{name.replace('_', '-')}.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"execution artifact is missing: {name}")
        artifacts[name] = _sha256(path)
    seal_root = arguments.source_seal_root.resolve(strict=True)
    if (
        not seal_root.is_dir()
        or seal_root.is_symlink()
        or any(parent.is_symlink() for parent in seal_root.parents)
    ):
        raise ValueError("source seal root must be an ordinary directory")
    seal_paths = sorted(seal_root.glob("*.json"))
    if len(seal_paths) != 100 or any(path.is_symlink() for path in seal_paths):
        raise ValueError("source seal root must contain exactly 100 ordinary seals")
    seal_digests = {path.name: _sha256(path) for path in seal_paths}
    source_plan = _load(arguments.source_plan, label="camera reuse source plan")
    prediction_batch = _load(arguments.prediction_batch, label="prediction batch")
    prediction_receipt = _load(arguments.prediction_receipt, label="prediction receipt")
    if artifacts["source_plan"] != _sha256(arguments.source_plan.resolve(strict=True)):
        raise ValueError("source plan is outside the execution artifact roster")
    if artifacts["source_prediction_batch"] != _sha256(
        arguments.prediction_batch.resolve(strict=True)
    ):
        raise ValueError("prediction batch is outside the execution artifact roster")
    if artifacts["source_prediction_receipt"] != _sha256(
        arguments.prediction_receipt.resolve(strict=True)
    ):
        raise ValueError("prediction receipt is outside the execution artifact roster")
    return build_deform360_v6_source_camera_reuse_execution_receipt(
        amendment=amendment,
        lock=lock,
        source_revision=arguments.source_revision,
        runner_name=arguments.runner_name,
        workflow_run_id=arguments.workflow_run_id,
        workflow_run_attempt=arguments.workflow_run_attempt,
        base_source_execution=_base_source_execution(arguments),
        artifact_file_sha256=artifacts,
        source_plan=source_plan,
        prediction_batch=prediction_batch,
        source_prediction_receipt=prediction_receipt,
        source_prediction_seal_file_sha256=seal_digests,
    )


def _base_source_execution(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_id": arguments.base_run_id,
        "run_attempt": arguments.base_run_attempt,
        "head_sha": arguments.base_head_sha,
        "artifact_id": arguments.base_artifact_id,
        "artifact_name": arguments.base_artifact_name,
        "artifact_digest_sha256": arguments.base_artifact_digest_sha256,
    }


def _retain_failure(arguments: argparse.Namespace) -> dict[str, Any]:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(
        arguments.execution_lock
    )
    amendment = validate_deform360_v6_source_camera_reuse_amendment(
        _load(arguments.amendment, label="camera reuse amendment")
    )
    root = arguments.artifact_root.resolve(strict=True)
    if (
        not root.is_dir()
        or root.is_symlink()
        or any(parent.is_symlink() for parent in root.parents)
    ):
        raise ValueError("failure artifact root must be an ordinary directory")
    output = arguments.output.absolute()
    retained: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError("retained failure artifacts must not contain symlinks")
        if not path.is_file():
            continue
        if path.absolute() == output or path.name == "SHA256SUMS":
            continue
        retained[path.relative_to(root).as_posix()] = _sha256(path)
    return build_deform360_v6_source_camera_reuse_technical_failure_receipt(
        amendment=amendment,
        lock=lock,
        source_revision=arguments.source_revision,
        runner_name=arguments.runner_name,
        workflow_run_id=arguments.workflow_run_id,
        workflow_run_attempt=arguments.workflow_run_attempt,
        base_source_execution=_base_source_execution(arguments),
        terminal_stage=arguments.terminal_stage,
        exit_code=arguments.exit_code,
        retained_artifact_file_sha256=retained,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate-amendment":
        amendment = validate_deform360_v6_source_camera_reuse_amendment(
            _load(arguments.amendment, label="camera reuse amendment")
        )
        _print({"amendment_id": amendment["amendment_id"]})
        return 0
    if arguments.command == "audit-base":
        lock, source, _ = _base(arguments)
        result = audit_deform360_joint_sparse_source_cameras_v5_2(
            lock=lock, source_plan=source, input_root=arguments.input_root
        )
        _write(arguments.output, result)
        _print({"audit_id": result["audit_id"]})
        return 0
    if arguments.command == "audit-combined":
        lock = load_deform360_joint_sparse_source_execution_lock_v5(
            arguments.execution_lock
        )
        combined = validate_deform360_joint_sparse_source_prediction_plan_v5(
            _load(arguments.combined_plan, label="combined plan"), lock=lock
        )
        result = audit_deform360_joint_sparse_source_cameras_v5_2(
            lock=lock, source_plan=combined, input_root=arguments.input_root
        )
        _write(arguments.output, result)
        _print({"audit_id": result["audit_id"]})
        return 0
    if arguments.command == "rank-reuse":
        lock, source, source_path = _base(arguments)
        audit_path = arguments.base_camera_audit.resolve(strict=True)
        audit = validate_deform360_joint_sparse_camera_audit_v5_2(
            _load(audit_path, label="base camera audit"), lock=lock
        )
        metric_plan_path = arguments.metric_prefix_plan.resolve(strict=True)
        metric_result_path = arguments.metric_batch_result.resolve(strict=True)
        result = build_deform360_v6_source_camera_reuse_preflight(
            lock=lock,
            base_source_plan=source,
            base_source_plan_file_sha256=_sha256(source_path),
            base_camera_audit=audit,
            base_camera_audit_file_sha256=_sha256(audit_path),
            metric_prefix_plan=_load(metric_plan_path, label="metric-prefix plan"),
            metric_prefix_plan_file_sha256=_sha256(metric_plan_path),
            metric_batch_result=_load(metric_result_path, label="metric result"),
            metric_batch_result_file_sha256=_sha256(metric_result_path),
            metric_files_root=arguments.metric_files_root,
        )
        _write(arguments.output, result)
        _print({"preflight_id": result["preflight_id"]})
        return 0
    if arguments.command == "build-combined-plan":
        lock, source, _ = _base(arguments)
        audit = validate_deform360_joint_sparse_camera_audit_v5_2(
            _load(arguments.base_camera_audit, label="base camera audit"), lock=lock
        )
        combined, receipt = build_deform360_v6_source_camera_reuse_plan(
            lock=lock,
            base_source_plan=source,
            base_camera_audit=audit,
            preflight=_load(arguments.preflight, label="camera reuse preflight"),
            metric_prefix_plan=_load(
                arguments.metric_prefix_plan, label="metric-prefix plan"
            ),
            results_root=arguments.results_root,
            prediction_root=arguments.prediction_root,
            metric_files_root=arguments.metric_files_root,
            implementation_revision=arguments.implementation_revision,
        )
        _write(arguments.output_plan, combined)
        _write(arguments.output_receipt, receipt)
        _print(
            {
                "combined_plan_id": combined["plan_id"],
                "reuse_receipt_id": receipt["receipt_id"],
            }
        )
        return 0
    if arguments.command == "build-lineage":
        lineage = _lineage(arguments)
        _write(arguments.output, lineage)
        _print({"artifact_count": len(lineage["camera_recovery"]["artifact_ids"])})
        return 0
    if arguments.command == "freeze-source-plan":
        lock = load_deform360_joint_sparse_source_execution_lock_v5(
            arguments.execution_lock
        )
        combined = validate_deform360_joint_sparse_source_prediction_plan_v5(
            _load(arguments.combined_plan, label="combined plan"), lock=lock
        )
        final_audit = validate_deform360_joint_sparse_camera_audit_v5_2(
            _load(arguments.final_camera_audit, label="final camera audit"), lock=lock
        )
        lineage = _load(arguments.lineage, label="camera reuse lineage")
        if set(lineage) != {"camera_recovery"}:
            raise ValueError("camera reuse lineage has unexpected fields")
        objects = cast(Sequence[Mapping[str, Any]], combined["objects"])
        result = build_deform360_joint_sparse_source_prediction_plan_v5_2(
            lock=lock,
            implementation_revision=arguments.implementation_revision,
            attempted_objects=objects,
            final_camera_audit=final_audit,
            camera_recovery=cast(Mapping[str, Any], lineage["camera_recovery"]),
        )
        _write(arguments.output, result)
        _print({"plan_id": result["plan_id"]})
        return 0
    if arguments.command == "seal-execution":
        result = _seal_execution(arguments)
        _write(arguments.output, result)
        _print({"receipt_id": result["receipt_id"], "status": result["status"]})
        return 0
    if arguments.command == "validate-execution":
        result = validate_deform360_v6_source_camera_reuse_execution_receipt(
            _load(arguments.receipt, label="camera reuse execution receipt")
        )
        _print({"receipt_id": result["receipt_id"], "status": result["status"]})
        return 0
    if arguments.command == "retain-failure":
        result = _retain_failure(arguments)
        _write(arguments.output, result)
        _print({"receipt_id": result["receipt_id"], "status": result["status"]})
        return 0
    if arguments.command == "validate-receipt":
        raw = _load(arguments.receipt, label="camera reuse receipt")
        if raw.get("schema") == TECHNICAL_FAILURE_RECEIPT_SCHEMA:
            result = (
                validate_deform360_v6_source_camera_reuse_technical_failure_receipt(raw)
            )
        else:
            result = validate_deform360_v6_source_camera_reuse_execution_receipt(raw)
        _print({"receipt_id": result["receipt_id"], "status": result["status"]})
        return 0
    raise AssertionError("unreachable camera reuse command")


if __name__ == "__main__":
    raise SystemExit(main())
