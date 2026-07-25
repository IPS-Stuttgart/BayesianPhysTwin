#!/usr/bin/env python3
"""Seal the complete v2 calibration disposition and evaluate support only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import file_sha256
from bayesian_phystwin.deform360_bias_aware_prospective_v2_protocol import (
    load_bias_aware_prospective_v2_protocol,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_runtime import (
    validate_v2_execution_lock,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_support import (
    build_v2_calibration_cohort_seal,
    build_v2_calibration_support_gate,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _clean_revision(repository: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository is dirty: {repository}")
    return revision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--base-prediction-root", type=Path, required=True)
    parser.add_argument("--base-backbone-root", type=Path, required=True)
    parser.add_argument("--base-cohort-seal", type=Path, required=True)
    parser.add_argument("--base-support-rejection", type=Path, required=True)
    parser.add_argument("--fresh-prediction-root", type=Path, required=True)
    parser.add_argument("--fresh-backbone-root", type=Path, required=True)
    parser.add_argument("--cohort-seal-output", type=Path, required=True)
    parser.add_argument("--support-gate-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repo.resolve()
    revision = _clean_revision(repository)
    lock_path = args.execution_lock.resolve()
    lock = validate_v2_execution_lock(lock_path, repository=repository)
    protocol_path = args.protocol.resolve()
    load_bias_aware_prospective_v2_protocol(protocol_path, root=repository)
    lock_record = {
        "path": str(lock_path),
        "file_sha256": file_sha256(lock_path),
        "config_sha256": lock["config_sha256"],
        "adapter_lock_commit": lock["adapter_lock_commit"],
        "execution_revision": revision,
    }
    cohort = build_v2_calibration_cohort_seal(
        protocol_path,
        base_protocol_path=args.base_protocol,
        base_prediction_root=args.base_prediction_root,
        base_backbone_root=args.base_backbone_root,
        base_cohort_seal_path=args.base_cohort_seal,
        base_support_rejection_path=args.base_support_rejection,
        fresh_prediction_root=args.fresh_prediction_root,
        fresh_backbone_root=args.fresh_backbone_root,
        execution_lock_record=lock_record,
        output_path=args.cohort_seal_output,
    )
    support = build_v2_calibration_support_gate(
        protocol_path,
        cohort_seal_path=args.cohort_seal_output,
        output_path=args.support_gate_output,
    )
    print(
        json.dumps(
            {
                "calibration_cohort_result_sha256": cohort["result_sha256"],
                "support_gate_result_sha256": support["result_sha256"],
                "automatic_twin_object_count": support["automatic_twin_object_count"],
                "automatic_twin_object_count_by_stratum": support[
                    "automatic_twin_object_count_by_stratum"
                ],
                "fresh_filament_automatic_twin_count": support[
                    "fresh_filament_automatic_twin_count"
                ],
                "finite_sample_coverage": support["finite_sample_coverage"],
                "failed_support_gates": support["failed_support_gates"],
                "support_passed": support["support_passed"],
                "calibration_future_access_authorized": support[
                    "calibration_future_access_authorized"
                ],
                "target_access_authorized": support["target_access_authorized"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
