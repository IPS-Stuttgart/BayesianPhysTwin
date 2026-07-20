"""Open and score only prediction-authorized Deform360 outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    file_sha256,
)
from bayesian_phystwin.deform360_bias_aware_prospective_evaluation import (
    CASE_EVALUATION_FILENAME,
    aggregate_bias_aware_target_result,
    collect_prospective_case_evaluations,
    evaluate_bias_aware_prospective_case,
    fit_bias_aware_calibration_gate,
    record_prospective_outcome_failure,
    validate_bias_aware_calibration_gate,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    SOURCE_LOCK_SHA256,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object expected: {path}")
    return value


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"result already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate-case")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--role", choices=("calibration", "target"), required=True)
    evaluate.add_argument("--cohort-seal", type=Path, required=True)
    evaluate.add_argument("--prediction-root", type=Path, required=True)
    evaluate.add_argument("--outcome-root", type=Path, required=True)
    evaluate.add_argument("--object-id", required=True)
    evaluate.add_argument("--episode-id", type=int, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--calibration-gate", type=Path)

    calibration = subparsers.add_parser("calibration-gate")
    calibration.add_argument("--protocol", type=Path, required=True)
    calibration.add_argument("--cohort-seal", type=Path, required=True)
    calibration.add_argument("--artifact-root", type=Path, required=True)
    calibration.add_argument("--evaluation-root", type=Path, required=True)
    calibration.add_argument("--outcome-failure-root", type=Path, required=True)
    calibration.add_argument("--source-lock", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate-gate")
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--gate", type=Path, required=True)
    validate.add_argument("--require-passed", action="store_true")

    target = subparsers.add_parser("target-result")
    target.add_argument("--protocol", type=Path, required=True)
    target.add_argument("--cohort-seal", type=Path, required=True)
    target.add_argument("--artifact-root", type=Path, required=True)
    target.add_argument("--evaluation-root", type=Path, required=True)
    target.add_argument("--outcome-failure-root", type=Path, required=True)
    target.add_argument("--calibration-gate", type=Path, required=True)
    target.add_argument("--output", type=Path, required=True)

    failure = subparsers.add_parser("record-outcome-failure")
    failure.add_argument("--protocol", type=Path, required=True)
    failure.add_argument("--output-dir", type=Path, required=True)
    failure.add_argument("--object-id", required=True)
    failure.add_argument("--episode-id", type=int, required=True)
    failure.add_argument(
        "--stage",
        choices=("authorized-future", "authorized-outcome", "evaluation"),
        required=True,
    )
    failure.add_argument("--error-type", required=True)
    failure.add_argument("--error-message", required=True)
    failure.add_argument("--prediction-cohort-result-sha256", required=True)
    failure.add_argument("--prediction-result-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate-case":
        cohort = _load_json(args.cohort_seal.resolve())
        gate_result = None
        if args.role == "target":
            _require(args.calibration_gate is not None, "target needs calibration gate")
            gate = _load_json(args.calibration_gate.resolve())
            validate_bias_aware_calibration_gate(
                gate, protocol_path=args.protocol, require_passed=True
            )
            gate_result = str(gate["result_sha256"])
        else:
            _require(
                args.calibration_gate is None,
                "calibration evaluation must not consume target gate",
            )
        result = evaluate_bias_aware_prospective_case(
            args.protocol,
            cohort,
            args.prediction_root,
            args.outcome_root,
            role=args.role,
            object_id=args.object_id,
            episode_id=args.episode_id,
            calibration_gate_result_sha256=gate_result,
        )
        destination = (
            args.output_root.resolve() / str(result["case"]) / CASE_EVALUATION_FILENAME
        )
        _write_new_json(destination, result)
    elif args.command == "calibration-gate":
        cohort = _load_json(args.cohort_seal.resolve())
        reports, failures = collect_prospective_case_evaluations(
            args.protocol,
            cohort,
            args.artifact_root,
            args.evaluation_root,
            args.outcome_failure_root,
            role="calibration",
        )
        source_lock_path = args.source_lock.resolve()
        _require(
            file_sha256(source_lock_path) == SOURCE_LOCK_SHA256, "source lock changed"
        )
        source_lock = _load_json(source_lock_path)
        result = fit_bias_aware_calibration_gate(
            reports,
            protocol_path=args.protocol,
            source_lock=source_lock,
            calibration_cohort_result_sha256=str(cohort["result_sha256"]),
            quality_failures=failures,
        )
        _write_new_json(args.output.resolve(), result)
    elif args.command == "validate-gate":
        result = _load_json(args.gate.resolve())
        validate_bias_aware_calibration_gate(
            result,
            protocol_path=args.protocol,
            require_passed=args.require_passed,
        )
    elif args.command == "target-result":
        cohort = _load_json(args.cohort_seal.resolve())
        gate = _load_json(args.calibration_gate.resolve())
        validate_bias_aware_calibration_gate(
            gate, protocol_path=args.protocol, require_passed=True
        )
        reports, failures = collect_prospective_case_evaluations(
            args.protocol,
            cohort,
            args.artifact_root,
            args.evaluation_root,
            args.outcome_failure_root,
            role="target",
        )
        result = aggregate_bias_aware_target_result(
            reports,
            protocol_path=args.protocol,
            target_cohort_result_sha256=str(cohort["result_sha256"]),
            calibration_gate_result_sha256=str(gate["result_sha256"]),
            quality_failures=failures,
        )
        _write_new_json(args.output.resolve(), result)
    else:
        result = record_prospective_outcome_failure(
            args.protocol,
            args.output_dir,
            object_id=args.object_id,
            episode_id=args.episode_id,
            stage=args.stage,
            error_type=args.error_type,
            error_message=args.error_message,
            prediction_cohort_result_sha256=(args.prediction_cohort_result_sha256),
            prediction_result_sha256=args.prediction_result_sha256,
        )
    _require(
        result.get("result_sha256")
        == canonical_sha256(result, digest_key="result_sha256"),
        "result checksum changed",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
