#!/usr/bin/env python3
"""Run one calibration-only future, outcome, or evaluation under v2 support."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
)
from bayesian_phystwin.deform360_bias_aware_prospective_evaluation import (
    AUTHORIZED_FUTURE_MANIFEST_FILENAME,
    CASE_EVALUATION_FILENAME,
    OUTCOME_MANIFEST_FILENAME,
    evaluate_bias_aware_prospective_case,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_calibration import (
    activate_fresh_v2_evaluation_runtime,
    build_v2_calibration_authorization_sidecar,
    patch_fresh_v2_calibration_stage,
    validate_v2_calibration_access,
    validate_v2_calibration_execution_lock,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_protocol import (
    load_bias_aware_prospective_v2_protocol,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_runtime import (
    activate_v2_prediction_runtime,
)


STAGE_SCRIPTS = {
    "authorized-future": "stage_deform360_bias_aware_authorized_future.py",
    "authorized-outcome": "build_deform360_bias_aware_authorized_outcome.py",
}
STAGE_ARTIFACTS = {
    "authorized-future": AUTHORIZED_FUTURE_MANIFEST_FILENAME,
    "authorized-outcome": OUTCOME_MANIFEST_FILENAME,
    "evaluation": CASE_EVALUATION_FILENAME,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _argument_value(arguments: list[str], option: str) -> str:
    matches = [index for index, value in enumerate(arguments) if value == option]
    _require(len(matches) == 1, f"expected one {option} argument")
    index = matches[0]
    _require(index + 1 < len(arguments), f"{option} has no value")
    return arguments[index + 1]


def _load_stage(path: Path, stage: str):
    name = f"_deform360_bias_aware_v2_calibration_{stage.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, "cannot load stage")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    _require(not path.exists(), f"artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--execution-repo", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--v2-protocol", type=Path, required=True)
    parser.add_argument("--v2-cohort-seal", type=Path, required=True)
    parser.add_argument("--v2-support-gate", type=Path, required=True)
    parser.add_argument("--origin", choices=("inherited_v1", "fresh_v2"), required=True)
    parser.add_argument(
        "--stage",
        choices=("authorized-future", "authorized-outcome", "evaluation"),
        required=True,
    )
    return parser.parse_known_args()


def _evaluate(
    args: argparse.Namespace,
    stage_arguments: list[str],
    *,
    object_id: str,
    episode_id: int,
) -> Path:
    protocol = Path(_argument_value(stage_arguments, "--protocol")).resolve()
    cohort_path = Path(_argument_value(stage_arguments, "--cohort-seal")).resolve()
    prediction_root = Path(
        _argument_value(stage_arguments, "--prediction-root")
    ).resolve()
    outcome_root = Path(_argument_value(stage_arguments, "--outcome-root")).resolve()
    output_root = Path(_argument_value(stage_arguments, "--output-root")).resolve()
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if args.origin == "fresh_v2":
        with activate_fresh_v2_evaluation_runtime(
            protocol_path=args.v2_protocol,
            cohort_seal_path=args.v2_cohort_seal,
            support_gate_path=args.v2_support_gate,
            prediction_root=prediction_root,
        ):
            result = evaluate_bias_aware_prospective_case(
                protocol,
                cohort,
                prediction_root,
                outcome_root,
                role="calibration",
                object_id=object_id,
                episode_id=episode_id,
            )
    else:
        result = evaluate_bias_aware_prospective_case(
            protocol,
            cohort,
            prediction_root,
            outcome_root,
            role="calibration",
            object_id=object_id,
            episode_id=episode_id,
        )
    destination = output_root / str(result["case"]) / CASE_EVALUATION_FILENAME
    _write_new_json(destination, result)
    return destination


def main() -> int:
    args, stage_arguments = _parse_args()
    repository = args.execution_repo.resolve()
    validate_v2_calibration_execution_lock(
        args.execution_lock.resolve(),
        repository=repository,
    )
    v2_protocol = args.v2_protocol.resolve()
    load_bias_aware_prospective_v2_protocol(v2_protocol, root=repository)
    object_id = _argument_value(stage_arguments, "--object-id")
    episode_id = int(_argument_value(stage_arguments, "--episode-id"))
    role = _argument_value(stage_arguments, "--role")
    _require(role == "calibration", "v2 wrapper cannot authorize targets")
    record, _, _ = validate_v2_calibration_access(
        v2_protocol,
        cohort_seal_path=args.v2_cohort_seal,
        support_gate_path=args.v2_support_gate,
        object_id=object_id,
        episode_id=episode_id,
        expected_origin=args.origin,
    )
    if args.stage == "evaluation":
        artifact_path = _evaluate(
            args,
            stage_arguments,
            object_id=object_id,
            episode_id=episode_id,
        )
    else:
        script = repository / "scripts/remote" / STAGE_SCRIPTS[args.stage]
        module = _load_stage(script, args.stage)
        if args.origin == "fresh_v2":
            prediction_root = Path(
                _argument_value(stage_arguments, "--prediction-root")
            ).resolve()
            patch_fresh_v2_calibration_stage(
                module,
                protocol_path=v2_protocol,
                cohort_seal_path=args.v2_cohort_seal,
                support_gate_path=args.v2_support_gate,
                prediction_root=prediction_root,
            )
        previous = sys.argv
        sys.argv = [str(script), *stage_arguments]
        try:
            if args.origin == "fresh_v2":
                with activate_v2_prediction_runtime():
                    return_code = int(module.main())
            else:
                return_code = int(module.main())
        finally:
            sys.argv = previous
        _require(return_code == 0, f"{args.stage} failed")
        output_root = Path(_argument_value(stage_arguments, "--output-root")).resolve()
        artifact_path = output_root / str(record["case"]) / STAGE_ARTIFACTS[args.stage]
    sidecar = build_v2_calibration_authorization_sidecar(
        v2_protocol,
        cohort_seal_path=args.v2_cohort_seal,
        support_gate_path=args.v2_support_gate,
        object_id=object_id,
        episode_id=episode_id,
        origin=args.origin,
        stage=args.stage,
        stage_artifact_path=artifact_path,
    )
    sidecar_path = artifact_path.with_name(
        f"v2_{args.stage.replace('-', '_')}_authorization.json"
    )
    _write_new_json(sidecar_path, sidecar)
    _require(
        sidecar["result_sha256"]
        == canonical_sha256(sidecar, digest_key="result_sha256"),
        "authorization sidecar checksum changed",
    )
    print(
        json.dumps(
            {
                "case": record["case"],
                "origin": args.origin,
                "stage": args.stage,
                "stage_artifact": str(artifact_path),
                "stage_artifact_result_sha256": json.loads(
                    artifact_path.read_text(encoding="utf-8")
                )["result_sha256"],
                "authorization_result_sha256": sidecar["result_sha256"],
                "target_access_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
