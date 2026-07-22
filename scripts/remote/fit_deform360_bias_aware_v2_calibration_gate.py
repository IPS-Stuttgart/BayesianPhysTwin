#!/usr/bin/env python3
"""Fit the locked v2 calibration accuracy gate and authorize no target on failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    file_sha256,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    SOURCE_LOCK_SHA256,
    load_bias_aware_prospective_protocol,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_calibration import (
    AUTHORIZATION_ARTIFACT_KIND,
    fit_v2_calibration_accuracy_gate,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_protocol import (
    load_bias_aware_prospective_v2_protocol,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_runtime import (
    validate_v2_execution_lock,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_support import (
    validate_v2_calibration_cohort_seal,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object expected: {path}")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--cohort-seal", type=Path, required=True)
    parser.add_argument("--support-gate", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repo.resolve()
    validate_v2_execution_lock(
        args.execution_lock.resolve(),
        repository=repository,
    )
    protocol_path = args.protocol.resolve()
    load_bias_aware_prospective_v2_protocol(protocol_path, root=repository)
    base_protocol = load_bias_aware_prospective_protocol(args.base_protocol.resolve())
    cohort = _load_json(args.cohort_seal)
    validate_v2_calibration_cohort_seal(cohort, protocol_path=protocol_path)
    reports = []
    root = args.evaluation_root.resolve()
    for row in cohort["cases"]:
        if not row["automatic_twin"]:
            continue
        case_root = root / str(row["case"])
        report_path = case_root / "evaluation.json"
        sidecar_path = case_root / "v2_evaluation_authorization.json"
        report = _load_json(report_path)
        sidecar = _load_json(sidecar_path)
        _require(
            sidecar.get("artifact_kind") == AUTHORIZATION_ARTIFACT_KIND
            and sidecar.get("stage") == "evaluation"
            and sidecar.get("origin") == row["origin"]
            and sidecar.get("source_artifact_file_sha256") == file_sha256(report_path)
            and sidecar.get("source_artifact_result_sha256") == report["result_sha256"]
            and sidecar.get("result_sha256")
            == canonical_sha256(sidecar, digest_key="result_sha256"),
            f"evaluation authorization changed: {row['case']}",
        )
        reports.append(report)
    source_lock_path = args.source_lock.resolve()
    _require(file_sha256(source_lock_path) == SOURCE_LOCK_SHA256, "source lock changed")
    result = fit_v2_calibration_accuracy_gate(
        reports,
        protocol_path=protocol_path,
        base_protocol_config_sha256=str(base_protocol["config_sha256"]),
        cohort_seal_path=args.cohort_seal,
        support_gate_path=args.support_gate,
        source_lock=_load_json(source_lock_path),
    )
    destination = args.output.resolve()
    _require(not destination.exists(), "v2 calibration gate already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
