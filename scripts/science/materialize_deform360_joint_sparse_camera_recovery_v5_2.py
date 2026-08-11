#!/usr/bin/env python3
"""Materialize outcome-blind Deform360 v5.2 camera-recovery artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin._portable_contracts import load_strict_json_object
from bayesian_phystwin.deform360_calibration_visual_execution_admission import (
    validate_deform360_prepared_source_inventory,
)
from bayesian_phystwin.deform360_joint_sparse_camera_recovery_v5_2 import (
    RECOVERY_POLICY,
    audit_deform360_joint_sparse_source_cameras_v5_2,
    build_deform360_joint_sparse_camera_recovery_preflight_v5_2,
    build_deform360_joint_sparse_combined_camera_audit_plan_v5_2,
    build_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2,
    merge_deform360_joint_sparse_motioncrafter_recovery_runs_v5_2,
    save_deform360_joint_sparse_camera_recovery_artifact_v5_2,
    validate_deform360_joint_sparse_camera_audit_v5_2,
    validate_deform360_joint_sparse_camera_recovery_amendment_v5_2,
    validate_deform360_joint_sparse_camera_recovery_preflight_v5_2,
    validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2,
    validate_deform360_joint_sparse_motioncrafter_recovery_run_v5_2,
)
from bayesian_phystwin.deform360_joint_sparse_motioncrafter_source_v5 import (
    load_deform360_joint_sparse_motioncrafter_source_plan_v5,
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
    CAMERA_RECOVERY_ARTIFACT_NAMES,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-base")
    audit.add_argument("--execution-lock", type=Path, required=True)
    audit.add_argument("--source-plan", type=Path, required=True)
    audit.add_argument("--input-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser("rank-recovery")
    preflight.add_argument("--execution-lock", type=Path, required=True)
    preflight.add_argument("--base-provider-plan", type=Path, required=True)
    preflight.add_argument("--base-camera-audit", type=Path, required=True)
    preflight.add_argument("--metric-root", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    provider = subparsers.add_parser("build-provider-plan")
    provider.add_argument("--execution-lock", type=Path, required=True)
    provider.add_argument("--prepared-source-inventory", type=Path, required=True)
    provider.add_argument("--base-provider-plan", type=Path, required=True)
    provider.add_argument("--base-camera-audit", type=Path, required=True)
    provider.add_argument("--recovery-preflight", type=Path, required=True)
    provider.add_argument("--amendment", type=Path, required=True)
    provider.add_argument("--implementation-revision", required=True)
    provider.add_argument("--runner-source", type=Path, required=True)
    provider.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate-provider-plan")
    validate.add_argument("plan", type=Path)

    merge = subparsers.add_parser("merge-provider-runs")
    merge.add_argument("--recovery-provider-plan", type=Path, required=True)
    merge.add_argument("--shard-report", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)

    validate_run = subparsers.add_parser("validate-provider-run")
    validate_run.add_argument("--recovery-provider-plan", type=Path, required=True)
    validate_run.add_argument("--provider-run", type=Path, required=True)

    combined = subparsers.add_parser("build-combined-audit-plan")
    combined.add_argument("--execution-lock", type=Path, required=True)
    combined.add_argument("--base-source-plan", type=Path, required=True)
    combined.add_argument("--base-camera-audit", type=Path, required=True)
    combined.add_argument("--recovery-provider-plan", type=Path, required=True)
    combined.add_argument("--recovery-provider-run", type=Path, required=True)
    combined.add_argument("--input-root", type=Path, required=True)
    combined.add_argument("--recovery-decoded-root", type=Path, required=True)
    combined.add_argument("--recovery-metric-root", type=Path, required=True)
    combined.add_argument("--implementation-revision", required=True)
    combined.add_argument("--output", type=Path, required=True)

    lineage = subparsers.add_parser("build-recovery-lineage")
    lineage.add_argument("--execution-lock", type=Path, required=True)
    lineage.add_argument("--base-source-plan", type=Path, required=True)
    lineage.add_argument("--base-provider-plan", type=Path, required=True)
    lineage.add_argument("--amendment", type=Path, required=True)
    lineage.add_argument("--base-camera-audit", type=Path, required=True)
    lineage.add_argument("--base-prediction-batch", type=Path, required=True)
    lineage.add_argument("--base-prediction-receipt", type=Path, required=True)
    lineage.add_argument("--recovery-preflight", type=Path, required=True)
    lineage.add_argument("--recovery-provider-plan", type=Path, required=True)
    lineage.add_argument("--recovery-provider-run", type=Path, required=True)
    lineage.add_argument("--combined-camera-audit-plan", type=Path, required=True)
    lineage.add_argument("--final-camera-audit", type=Path, required=True)
    lineage.add_argument("--output", type=Path, required=True)
    return parser


def _print_result(value: dict[str, object]) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False))


def _load(path: Path, *, label: str) -> dict[str, object]:
    return load_strict_json_object(path.resolve(strict=True), label=label)


def _build_recovery_lineage(
    arguments: argparse.Namespace,
    *,
    lock: dict[str, object],
) -> dict[str, object]:
    base_source_path = arguments.base_source_plan.resolve(strict=True)
    base_source_plan = validate_deform360_joint_sparse_source_prediction_plan_v5(
        _load(base_source_path, label="base source prediction plan"),
        lock=lock,
    )
    base_provider_path = arguments.base_provider_plan.resolve(strict=True)
    base_provider = load_deform360_joint_sparse_motioncrafter_source_plan_v5(
        base_provider_path
    )
    amendment_path = arguments.amendment.resolve(strict=True)
    amendment = validate_deform360_joint_sparse_camera_recovery_amendment_v5_2(
        _load(amendment_path, label="camera recovery amendment")
    )
    base_audit_path = arguments.base_camera_audit.resolve(strict=True)
    base_audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        _load(base_audit_path, label="base camera audit"), lock=lock
    )
    if base_audit["base_source_plan_id"] != base_source_plan["plan_id"]:
        raise ValueError("base camera audit does not bind the base source plan")

    batch_path = arguments.base_prediction_batch.resolve(strict=True)
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        _load(batch_path, label="base prediction batch"), lock
    )
    receipt_path = arguments.base_prediction_receipt.resolve(strict=True)
    receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5(
        _load(receipt_path, label="base prediction receipt"),
        lock=lock,
        plan=base_source_plan,
        prediction_batch=batch,
        prediction_batch_file_sha256=_sha256(batch_path),
    )
    preflight_path = arguments.recovery_preflight.resolve(strict=True)
    preflight = validate_deform360_joint_sparse_camera_recovery_preflight_v5_2(
        _load(preflight_path, label="recovery preflight"),
        lock=lock,
        base_provider_plan=base_provider,
        base_camera_audit=base_audit,
    )
    provider_path = arguments.recovery_provider_plan.resolve(strict=True)
    provider = validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(
        _load(provider_path, label="recovery provider plan")
    )
    if provider["base_provider_plan"] != {
        "manifest_sha256": base_provider["manifest_sha256"],
        "file_sha256": _sha256(base_provider_path),
    }:
        raise ValueError("recovery provider does not bind the base provider plan")
    if provider["camera_recovery_preflight"] != {
        "preflight_id": preflight["preflight_id"],
        "file_sha256": _sha256(preflight_path),
    }:
        raise ValueError("recovery provider does not bind the recovery preflight")
    if provider["camera_recovery_amendment"] != {
        "amendment_id": amendment["amendment_id"],
        "file_sha256": _sha256(amendment_path),
    }:
        raise ValueError("recovery provider does not bind the amendment")
    provider_run_path = arguments.recovery_provider_run.resolve(strict=True)
    provider_run = validate_deform360_joint_sparse_motioncrafter_recovery_run_v5_2(
        _load(provider_run_path, label="recovery provider run"), plan=provider
    )
    combined_path = arguments.combined_camera_audit_plan.resolve(strict=True)
    combined = validate_deform360_joint_sparse_source_prediction_plan_v5(
        _load(combined_path, label="combined camera audit plan"), lock=lock
    )
    final_audit_path = arguments.final_camera_audit.resolve(strict=True)
    final_audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        _load(final_audit_path, label="final camera audit"), lock=lock
    )
    if final_audit["base_source_plan_id"] != combined["plan_id"]:
        raise ValueError("final camera audit does not bind the combined audit plan")

    paths = {
        "amendment": amendment_path,
        "base_camera_audit": base_audit_path,
        "base_prediction_batch": batch_path,
        "base_prediction_receipt": receipt_path,
        "combined_camera_audit_plan": combined_path,
        "final_camera_audit": final_audit_path,
        "recovery_preflight": preflight_path,
        "recovery_provider_plan": provider_path,
        "recovery_provider_run": provider_run_path,
    }
    identities = {
        "amendment": amendment["amendment_id"],
        "base_camera_audit": base_audit["audit_id"],
        "base_prediction_batch": batch["prediction_batch_id"],
        "base_prediction_receipt": receipt["receipt_id"],
        "combined_camera_audit_plan": combined["plan_id"],
        "final_camera_audit": final_audit["audit_id"],
        "recovery_preflight": preflight["preflight_id"],
        "recovery_provider_plan": provider["manifest_sha256"],
        "recovery_provider_run": provider_run["run_sha256"],
    }
    if set(paths) != CAMERA_RECOVERY_ARTIFACT_NAMES:
        raise AssertionError("recovery lineage artifact roster changed")
    return {
        "camera_recovery": {
            "artifact_ids": identities,
            "source_artifacts": {name: _sha256(path) for name, path in paths.items()},
            "policy": dict(RECOVERY_POLICY),
            "base_prediction_batch_preserved": True,
        }
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate-provider-plan":
        value = load_strict_json_object(arguments.plan, label="recovery provider plan")
        plan = validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(value)
        _print_result(
            {
                "manifest_sha256": plan["manifest_sha256"],
                "object_count": plan["object_count"],
                "job_count": plan["job_count"],
            }
        )
        return 0

    if arguments.command in {"merge-provider-runs", "validate-provider-run"}:
        provider_plan = (
            validate_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(
                load_strict_json_object(
                    arguments.recovery_provider_plan.resolve(strict=True),
                    label="recovery provider plan",
                )
            )
        )
        if arguments.command == "validate-provider-run":
            run = validate_deform360_joint_sparse_motioncrafter_recovery_run_v5_2(
                load_strict_json_object(
                    arguments.provider_run.resolve(strict=True),
                    label="recovery provider run",
                ),
                plan=provider_plan,
            )
            _print_result(
                {
                    "run_sha256": run["run_sha256"],
                    "completed_job_count": run["completed_job_count"],
                }
            )
            return 0
        shard_reports = [
            load_strict_json_object(path.resolve(strict=True), label="provider shard")
            for path in arguments.shard_report
        ]
        merged = merge_deform360_joint_sparse_motioncrafter_recovery_runs_v5_2(
            plan=provider_plan,
            shard_reports=shard_reports,
        )
        save_deform360_joint_sparse_camera_recovery_artifact_v5_2(
            arguments.output, merged
        )
        _print_result(
            {
                "run_sha256": merged["run_sha256"],
                "completed_job_count": merged["completed_job_count"],
            }
        )
        return 0

    lock_path = arguments.execution_lock.resolve(strict=True)
    lock = load_deform360_joint_sparse_source_execution_lock_v5(lock_path)
    if arguments.command == "build-recovery-lineage":
        result = _build_recovery_lineage(arguments, lock=lock)
        save_deform360_joint_sparse_camera_recovery_artifact_v5_2(
            arguments.output, result
        )
        _print_result(
            {
                "artifact_count": len(CAMERA_RECOVERY_ARTIFACT_NAMES),
                "output_sha256": _sha256(arguments.output.resolve(strict=True)),
            }
        )
        return 0
    if arguments.command == "build-combined-audit-plan":
        base_source_plan = load_strict_json_object(
            arguments.base_source_plan.resolve(strict=True),
            label="base source prediction plan",
        )
        base_audit = load_strict_json_object(
            arguments.base_camera_audit.resolve(strict=True),
            label="base camera audit",
        )
        recovery_provider_plan = load_strict_json_object(
            arguments.recovery_provider_plan.resolve(strict=True),
            label="recovery provider plan",
        )
        recovery_provider_run = load_strict_json_object(
            arguments.recovery_provider_run.resolve(strict=True),
            label="recovery provider run",
        )
        result = build_deform360_joint_sparse_combined_camera_audit_plan_v5_2(
            lock=lock,
            base_source_plan=base_source_plan,
            base_camera_audit=base_audit,
            recovery_provider_plan=recovery_provider_plan,
            recovery_provider_run=recovery_provider_run,
            input_root=arguments.input_root,
            recovery_decoded_root=arguments.recovery_decoded_root,
            recovery_metric_root=arguments.recovery_metric_root,
            implementation_revision=arguments.implementation_revision,
        )
        objects = result.get("objects")
        if isinstance(objects, (str, bytes)) or not isinstance(objects, Sequence):
            raise ValueError("combined camera audit objects must be a JSON array")
        save_deform360_joint_sparse_camera_recovery_artifact_v5_2(
            arguments.output, result
        )
        _print_result(
            {
                "plan_id": result["plan_id"],
                "object_count": len(objects),
            }
        )
        return 0
    if arguments.command == "audit-base":
        source_plan = load_strict_json_object(
            arguments.source_plan.resolve(strict=True), label="source prediction plan"
        )
        validate_deform360_joint_sparse_source_prediction_plan_v5(
            source_plan,
            lock=lock,
        )
        result = audit_deform360_joint_sparse_source_cameras_v5_2(
            lock=lock,
            source_plan=source_plan,
            input_root=arguments.input_root,
        )
        save_deform360_joint_sparse_camera_recovery_artifact_v5_2(
            arguments.output, result
        )
        _print_result({"audit_id": result["audit_id"]})
        return 0

    base_plan_path = arguments.base_provider_plan.resolve(strict=True)
    base_plan = load_deform360_joint_sparse_motioncrafter_source_plan_v5(base_plan_path)
    audit_path = arguments.base_camera_audit.resolve(strict=True)
    audit = validate_deform360_joint_sparse_camera_audit_v5_2(
        load_strict_json_object(audit_path, label="base camera audit"),
        lock=lock,
    )
    if arguments.command == "rank-recovery":
        result = build_deform360_joint_sparse_camera_recovery_preflight_v5_2(
            lock=lock,
            base_provider_plan=base_plan,
            base_provider_plan_file_sha256=_sha256(base_plan_path),
            base_camera_audit=audit,
            base_camera_audit_file_sha256=_sha256(audit_path),
            metric_root=arguments.metric_root,
        )
        save_deform360_joint_sparse_camera_recovery_artifact_v5_2(
            arguments.output, result
        )
        _print_result({"preflight_id": result["preflight_id"]})
        return 0

    preflight_path = arguments.recovery_preflight.resolve(strict=True)
    preflight = validate_deform360_joint_sparse_camera_recovery_preflight_v5_2(
        load_strict_json_object(preflight_path, label="recovery preflight"),
        lock=lock,
        base_provider_plan=base_plan,
        base_camera_audit=audit,
    )
    amendment_path = arguments.amendment.resolve(strict=True)
    amendment = validate_deform360_joint_sparse_camera_recovery_amendment_v5_2(
        load_strict_json_object(amendment_path, label="camera recovery amendment")
    )
    inventory_path = arguments.prepared_source_inventory.resolve(strict=True)
    inventory = validate_deform360_prepared_source_inventory(
        load_strict_json_object(inventory_path, label="prepared source inventory")
    )
    runner_path = arguments.runner_source.resolve(strict=True)
    result = build_deform360_joint_sparse_motioncrafter_recovery_plan_v5_2(
        lock=lock,
        execution_lock_file_sha256=_sha256(lock_path),
        inventory=inventory,
        base_provider_plan=base_plan,
        base_provider_plan_file_sha256=_sha256(base_plan_path),
        recovery_preflight=preflight,
        recovery_preflight_file_sha256=_sha256(preflight_path),
        amendment=amendment,
        amendment_file_sha256=_sha256(amendment_path),
        implementation_revision=arguments.implementation_revision,
        runner_source_sha256=_sha256(runner_path),
    )
    save_deform360_joint_sparse_camera_recovery_artifact_v5_2(arguments.output, result)
    _print_result(
        {
            "manifest_sha256": result["manifest_sha256"],
            "object_count": result["object_count"],
            "job_count": result["job_count"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
