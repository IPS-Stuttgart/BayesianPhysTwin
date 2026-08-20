#!/usr/bin/env python3
"""Bind the all-train method, dry run, environment, and source archive."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin_experiments.deform_dlo_robustness import (
    DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS,
    load_deform_dlo_robustness_v1_protocol,
    validate_deform_bayesian_audit_v1,
    validate_deform_dlo3_alltrain_compute_match_v1,
    verify_deform_dlo3_backend_artifacts_v1,
    verify_deform_dlo3_evaluator_bayesian_artifacts_v1,
    verify_deform_dlo3_evaluator_compute_matched_artifacts_v1,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--alltrain-result", type=Path, required=True)
    parser.add_argument("--dry-run-result", type=Path, required=True)
    parser.add_argument("--custody-deviation", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verified_file(value: object, *, label: str) -> Path:
    identity = _mapping(value, label=label)
    path = Path(str(identity.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(cast(Any, identity.get("size_bytes", -1)))
        or sha256_file(path) != identity.get("sha256")
    ):
        raise ValueError(f"{label} identity changed")
    return path


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    alltrain_path = args.alltrain_result.resolve()
    dry_run_path = args.dry_run_result.resolve()
    deviation_path = args.custody_deviation.resolve()
    archive_path = args.source_archive.resolve()
    protocol = load_deform_dlo_robustness_v1_protocol(protocol_path)
    alltrain = _read_json(alltrain_path)
    dry_run = _read_json(dry_run_path)
    deviation = _read_json(deviation_path)
    revision = str(args.source_revision)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("source revision must be a full lowercase Git commit")
    if (
        alltrain.get("contract") != "deform-dlo3-robustness-alltrain-result-v1"
        or alltrain.get("primary_eval_read") is not False
        or alltrain.get("target_authorized") is not False
        or alltrain.get("bayesian_audit_complete") is not True
        or int(cast(Any, alltrain.get("bayesian_distribution_count", -1)))
        != len(DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS)
        or dry_run.get("contract") != "deform-dlo3-robustness-evaluator-dry-run-v1"
        or dry_run.get("pipeline_passed") is not True
        or dry_run.get("primary_eval_read") is not False
        or dry_run.get("target_authorized") is not False
        or deviation.get("contract") != "deform-dlo3-count-only-custody-deviation-v1"
        or deviation.get("official_eval_read") is not False
        or not archive_path.is_file()
    ):
        raise ValueError("DLO3 readiness inputs differ")
    final_method_path = _verified_file(
        alltrain.get("final_method"), label="alltrain final method"
    )
    final_method = _read_json(final_method_path)
    if (
        final_method.get("contract")
        != "deform-dlo3-robustness-alltrain-final-method-v1"
        or final_method.get("source_bayesian_audit_complete") is not True
        or final_method.get("source_diagnostics_verified") is not True
        or tuple(
            str(value)
            for value in cast(
                list[object], final_method.get("bayesian_ablation_distributions", [])
            )
        )
        != DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        or final_method.get("distribution_selection") != "none"
        or final_method.get("primary_eval_read") is not False
    ):
        raise ValueError("DLO3 alltrain Bayesian method differs")
    physical_checkpoint_path = _verified_file(
        final_method.get("physical_checkpoint"), label="physical checkpoint"
    )
    compute_checkpoint_path = _verified_file(
        final_method.get("compute_matched_checkpoint"),
        label="compute-matched checkpoint",
    )
    compute_match_path = _verified_file(
        final_method.get("compute_match"), label="compute-matched record"
    )
    compute_match = _read_json(compute_match_path)
    compute_verification = validate_deform_dlo3_alltrain_compute_match_v1(
        compute_match, protocol
    )
    physical_identity = dict(
        _mapping(final_method.get("physical_checkpoint"), label="physical checkpoint")
    )
    compute_identity = dict(
        _mapping(
            final_method.get("compute_matched_checkpoint"),
            label="compute-matched checkpoint",
        )
    )
    checkpoint_identities = tuple(
        dict(_mapping(value, label="alltrain checkpoint"))
        for value in cast(list[object], alltrain.get("checkpoints", []))
    )
    if (
        int(cast(Any, physical_identity.get("update", -1)))
        != int(cast(Any, compute_verification["start_update"]))
        or int(cast(Any, compute_identity.get("update", -1)))
        != int(cast(Any, compute_verification["end_update"]))
        or physical_identity not in checkpoint_identities
        or compute_identity not in checkpoint_identities
        or dict(_mapping(alltrain.get("compute_match"), label="alltrain compute"))
        != dict(_mapping(final_method.get("compute_match"), label="method compute"))
        or alltrain.get("compute_match_verification") != compute_verification
        or final_method.get("compute_match_verification") != compute_verification
    ):
        raise ValueError("DLO3 alltrain compute-matched lineage differs")
    authorization = _mapping(
        alltrain.get("authorization"), label="alltrain authorization"
    )
    backend_result_path = _verified_file(
        authorization.get("backend_result"), label="backend source result"
    )
    backend_result = _read_json(backend_result_path)
    backend_artifacts = verify_deform_dlo3_backend_artifacts_v1(
        backend_result, protocol
    )
    final_backend = dict(
        _mapping(final_method.get("backend_target_arm"), label="final backend arm")
    )
    if final_backend != backend_artifacts:
        raise ValueError("DLO3 final backend target arm differs")
    dry_backend = _mapping(
        dry_run.get("backend_portability"), label="dry-run backend portability"
    )
    backend_authorized = bool(backend_artifacts["backend_target_arm_authorized"])
    expected_backend_status = "scored" if backend_authorized else "not-authorized"
    if (
        dry_backend.get("status") != expected_backend_status
        or dry_backend.get("source_gate_authorized") is not backend_authorized
        or dry_backend.get("selection_effect") != "none"
    ):
        raise ValueError("DLO3 backend dry-run status differs")
    bayesian_audit = validate_deform_bayesian_audit_v1(dry_run, context="evaluator")
    bayesian_artifacts = verify_deform_dlo3_evaluator_bayesian_artifacts_v1(
        dry_run, expected_mode="dry-run"
    )
    compute_artifacts = verify_deform_dlo3_evaluator_compute_matched_artifacts_v1(
        dry_run, expected_mode="dry-run"
    )
    dry_authorization_path = _verified_file(
        dry_run.get("authorization"), label="dry-run authorization"
    )
    dry_authorization = _read_json(dry_authorization_path)
    if (
        compute_artifacts.get("status") != "scored"
        or compute_artifacts.get("selection_effect") != "none"
        or dry_authorization.get("contract")
        != "deform-dlo3-robustness-evaluator-authorization-v1"
        or dry_authorization.get("mode") != "dry-run"
        or dry_authorization.get("protocol") != _identity(protocol_path)
        or dry_authorization.get("alltrain_result") != _identity(alltrain_path)
        or dry_authorization.get("final_method") != _identity(final_method_path)
        or dry_authorization.get("compute_matched_checkpoint")
        != _identity(compute_checkpoint_path)
        or dry_authorization.get("compute_matched_record")
        != _identity(compute_match_path)
        or dry_authorization.get("compute_match_verification") != compute_verification
        or dry_authorization.get("one_shot_execution_authorized") is not False
        or dry_authorization.get("target_selection") is not False
        or dry_authorization.get("target_calibration") is not False
        or dry_authorization.get("target_retries") is not False
        or dry_authorization.get("official_eval_read") is not False
    ):
        raise ValueError("DLO3 compute-matched dry-run lineage differs")
    alltrain_runtime = _mapping(alltrain.get("runtime"), label="alltrain runtime")
    dry_runtime = _mapping(dry_run.get("runtime"), label="dry-run runtime")
    if alltrain_runtime.get("torch") != dry_runtime.get(
        "torch"
    ) or alltrain_runtime.get("cuda") != dry_runtime.get("cuda"):
        raise ValueError("DLO3 readiness runtime differs")
    payload = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-readiness-v1",
        "protocol": _identity(protocol_path),
        "alltrain_result": _identity(alltrain_path),
        "dry_run_result": _identity(dry_run_path),
        "custody_deviation": _identity(deviation_path),
        "source_archive": _identity(archive_path),
        "source_revision": revision,
        "runtime": dict(alltrain_runtime),
        "dry_run_pipeline_passed": True,
        "bayesian_audit_complete": True,
        "bayesian_artifacts_verified": True,
        "bayesian_audit": bayesian_audit,
        "bayesian_artifacts": bayesian_artifacts,
        "physical_checkpoint": _identity(physical_checkpoint_path),
        "compute_matched_checkpoint": _identity(compute_checkpoint_path),
        "compute_matched_record": _identity(compute_match_path),
        "compute_match_verification": compute_verification,
        "compute_matched_dry_run": compute_artifacts,
        "compute_matched_control_verified": True,
        "backend_target_arm_authorized": backend_authorized,
        "backend_artifacts": backend_artifacts,
        "backend_dry_run_status": expected_backend_status,
        "count_only_custody_deviation_acknowledged": True,
        "target_authorized": True,
        "official_eval_read": False,
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "case_replacement": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked readiness output differs: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
