"""Run target-free stages of the H2-locked adaptive-covariance confirmation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_adaptive_covariance_confirmation_failure import (
    materialize_and_seal_retained_confirmation_failure,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_external_runtime import (
    validate_confirmation_h2_production_entrypoint,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock import (
    load_confirmation_cohort_lock,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_evaluation import (
    PRODUCTION_EVALUATION_MODE,
    evaluate_adaptive_covariance_confirmation,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_measurement import (
    MANIFEST_FILENAME,
    MEASUREMENT_ARCHIVE_FILENAME,
    UNCERTAINTY_ARCHIVE_FILENAME,
    build_confirmation_nested_measurements,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_outcome_adapter import (
    build_confirmation_outcome_compatibility,
    validate_confirmation_outcome_compatibility,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_prediction import (
    RETAINED_FAILURE_CODES,
    assemble_and_seal_confirmation_prediction,
    seal_retained_confirmation_failure,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_scoring import (
    build_confirmation_case_target_loader,
    validate_confirmation_case_target_loader_attestation,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_seal import (
    create_confirmation_prediction_barrier,
    validate_confirmation_case_seal,
    validate_confirmation_prediction_barrier,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_source_custody import (
    build_confirmation_source_custody_seal,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
)
from bayesian_phystwin.deform360_raw_camera_uncertainty import (
    RawCameraUncertaintyConfig,
)

ENTRYPOINT_REPOSITORY_PATH = (
    "src/bayesian_phystwin/cli/deform360_adaptive_covariance_confirmation.py"
)
SOURCE_BOOTSTRAP_REPOSITORY_PATH = (
    "scripts/remote/run_deform360_adaptive_confirmation_cli.py"
)


def _common_case_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter-repo", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--h2-commit", required=True)
    parser.add_argument("--expected-h1", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--physical-archive", required=True)
    parser.add_argument("--output-dir", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    measurement = subparsers.add_parser(
        "measurement",
        help="Build causal nested four/eight-view measurements and uncertainty.",
    )
    _common_case_arguments(measurement)
    measurement.add_argument("--physical-manifest", required=True)
    measurement.add_argument("--physical-prediction-seal", required=True)
    measurement.add_argument("--source-custody-seal", required=True)
    measurement.add_argument("--processed-episode-dir", required=True)
    measurement.add_argument("--alltracker-source", required=True)
    measurement.add_argument("--checkpoint", required=True)
    measurement.add_argument("--device", default="cuda:0")

    source_custody = subparsers.add_parser(
        "source-custody",
        help=(
            "Create the write-once source-custodian seal before prediction "
            "measurement or the complete cohort barrier."
        ),
    )
    source_custody.add_argument("--adapter-repo", required=True)
    source_custody.add_argument("--lock", required=True)
    source_custody.add_argument("--h2-commit", required=True)
    source_custody.add_argument("--expected-h1", required=True)
    source_custody.add_argument("--case-id", required=True)
    source_custody.add_argument("--source-episode-dir", required=True)
    source_custody.add_argument("--staged-case-dir", required=True)
    source_custody.add_argument("--output", required=True)

    materialize_failure = subparsers.add_parser(
        "materialize-retained-failure",
        help=(
            "Materialize and seal a target-free persistence package after a "
            "physical or tracking failure."
        ),
    )
    materialize_failure.add_argument("--adapter-repo", required=True)
    materialize_failure.add_argument("--external-execution-repo", required=True)
    materialize_failure.add_argument("--lock", required=True)
    materialize_failure.add_argument("--h2-commit", required=True)
    materialize_failure.add_argument("--expected-h1", required=True)
    materialize_failure.add_argument("--case-id", required=True)
    materialize_failure.add_argument("--staged-case-dir", required=True)
    materialize_failure.add_argument("--processed-episode-dir", required=True)
    materialize_failure.add_argument("--source-custody-seal", required=True)
    materialize_failure.add_argument("--physical-work-dir", required=True)
    materialize_failure.add_argument("--backbone-dir", required=True)
    materialize_failure.add_argument("--measurement-output-dir", required=True)
    materialize_failure.add_argument("--case-output-dir", required=True)
    materialize_failure.add_argument(
        "--failure-code",
        choices=RETAINED_FAILURE_CODES,
        required=True,
    )

    for name, help_text in (
        ("predict", "Assemble and seal all frozen prediction arms."),
        (
            "retain-failure",
            "Seal a declared target-free technical failure without dropping a case.",
        ),
    ):
        command = subparsers.add_parser(name, help=help_text)
        _common_case_arguments(command)
        command.add_argument("--measurement-dir", required=True)
        if name == "retain-failure":
            command.add_argument(
                "--failure-code",
                choices=RETAINED_FAILURE_CODES,
                required=True,
            )

    validate_case = subparsers.add_parser(
        "validate-case",
        help="Replay one case seal and every content hash.",
    )
    validate_case.add_argument("--adapter-repo", required=True)
    validate_case.add_argument("--lock", required=True)
    validate_case.add_argument("--h2-commit", required=True)
    validate_case.add_argument("--expected-h1", required=True)
    validate_case.add_argument("--case-id", required=True)
    validate_case.add_argument("--case-dir", required=True)

    barrier = subparsers.add_parser(
        "barrier",
        help="Create the write-once 34-case target-free completeness barrier.",
    )
    barrier.add_argument("--adapter-repo", required=True)
    barrier.add_argument("--lock", required=True)
    barrier.add_argument("--h2-commit", required=True)
    barrier.add_argument("--expected-h1", required=True)
    barrier.add_argument("--case-root", required=True)
    barrier.add_argument("--output", required=True)

    validate_barrier = subparsers.add_parser(
        "validate-barrier",
        help="Replay a complete prediction barrier against all exact case seals.",
    )
    validate_barrier.add_argument("--adapter-repo", required=True)
    validate_barrier.add_argument("--lock", required=True)
    validate_barrier.add_argument("--h2-commit", required=True)
    validate_barrier.add_argument("--expected-h1", required=True)
    validate_barrier.add_argument("--case-root", required=True)
    validate_barrier.add_argument("--barrier", required=True)

    for name, help_text in (
        (
            "compatibility",
            "Build target-free compatibility inputs for the frozen outcome stages.",
        ),
        (
            "validate-compatibility",
            "Replay the post-barrier compatibility binding without opening a target.",
        ),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--adapter-repo", required=True)
        command.add_argument("--lock", required=True)
        command.add_argument("--h2-commit", required=True)
        command.add_argument("--expected-h1", required=True)
        command.add_argument("--barrier", required=True)
        command.add_argument("--case-root", required=True)
        command.add_argument("--measurement-root", required=True)
        command.add_argument("--compatibility-root", required=True)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Score every official target after replaying the complete barrier.",
    )
    evaluate.add_argument("--adapter-repo", required=True)
    evaluate.add_argument("--lock", required=True)
    evaluate.add_argument("--h2-commit", required=True)
    evaluate.add_argument("--expected-h1", required=True)
    evaluate.add_argument("--barrier", required=True)
    evaluate.add_argument("--case-root", required=True)
    evaluate.add_argument("--measurement-root", required=True)
    evaluate.add_argument("--compatibility-root", required=True)
    evaluate.add_argument("--authorized-future-root", required=True)
    evaluate.add_argument("--authorized-outcome-root", required=True)
    evaluate.add_argument("--output", required=True)
    return parser


def _measurement_paths(
    root: str | Path,
) -> tuple[Path, dict[int, Path], dict[int, Path]]:
    directory = Path(root).absolute()
    return (
        directory / MANIFEST_FILENAME,
        {
            budget: directory / f"budget-{budget}" / MEASUREMENT_ARCHIVE_FILENAME
            for budget in (4, 8)
        },
        {
            budget: directory / f"budget-{budget}" / UNCERTAINTY_ARCHIVE_FILENAME
            for budget in (4, 8)
        },
    )


def _case_dirs(
    lock_path: str | Path, root: str | Path, expected_h1: str
) -> dict[str, Path]:
    lock = load_confirmation_cohort_lock(
        lock_path,
        expected_implementation_commit_h1=expected_h1,
    )
    base = Path(root).absolute()
    return {case_id: base / case_id for case_id in lock["selected_case_ids"]}


def _prediction(args: argparse.Namespace) -> dict[str, Any]:
    manifest, measurements, uncertainties = _measurement_paths(args.measurement_dir)
    common = (
        args.lock,
        args.h2_commit,
        args.case_id,
        args.output_dir,
        args.physical_archive,
        measurements,
        uncertainties,
    )
    keywords = {
        "measurement_manifest": manifest,
        "expected_h1": args.expected_h1,
    }
    if args.command == "predict":
        return assemble_and_seal_confirmation_prediction(*common, **keywords)
    return seal_retained_confirmation_failure(
        *common,
        args.failure_code,
        **keywords,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    source_bootstrap_file: str | Path | None = None,
) -> None:
    args = build_parser().parse_args(argv)
    if source_bootstrap_file is None:
        raise ValueError(
            "production confirmation CLI requires the canonical source bootstrap"
        )
    validate_confirmation_h2_production_entrypoint(
        args.adapter_repo,
        args.lock,
        args.h2_commit,
        expected_h1=args.expected_h1,
        entrypoint_file=__file__,
        entrypoint_repository_path=ENTRYPOINT_REPOSITORY_PATH,
        source_bootstrap_file=source_bootstrap_file,
        source_bootstrap_repository_path=SOURCE_BOOTSTRAP_REPOSITORY_PATH,
    )
    if args.command == "source-custody":
        result = build_confirmation_source_custody_seal(
            args.lock,
            args.h2_commit,
            args.case_id,
            args.source_episode_dir,
            args.staged_case_dir,
            args.output,
            expected_h1=args.expected_h1,
        )
    elif args.command == "measurement":
        observation = RawCameraObservationConfig()
        uncertainty = RawCameraUncertaintyConfig()
        runtime = AllTrackerPrefixRuntime(
            args.alltracker_source,
            args.checkpoint,
            device=args.device,
            config=observation,
        )
        try:
            result = build_confirmation_nested_measurements(
                args.lock,
                args.h2_commit,
                args.case_id,
                args.physical_archive,
                args.processed_episode_dir,
                args.output_dir,
                runtime,
                physical_manifest=args.physical_manifest,
                physical_prediction_seal=args.physical_prediction_seal,
                source_custody_seal=args.source_custody_seal,
                expected_h1=args.expected_h1,
                observation_config=observation,
                uncertainty_config=uncertainty,
            )
        finally:
            runtime.close()
    elif args.command == "materialize-retained-failure":
        result = materialize_and_seal_retained_confirmation_failure(
            args.adapter_repo,
            args.external_execution_repo,
            args.lock,
            args.h2_commit,
            args.case_id,
            args.staged_case_dir,
            args.processed_episode_dir,
            args.source_custody_seal,
            args.physical_work_dir,
            args.backbone_dir,
            args.measurement_output_dir,
            args.case_output_dir,
            args.failure_code,
            expected_h1=args.expected_h1,
        )
    elif args.command in {"predict", "retain-failure"}:
        result = _prediction(args)
    elif args.command == "validate-case":
        result = validate_confirmation_case_seal(
            args.case_dir,
            args.lock,
            args.h2_commit,
            expected_case_id=args.case_id,
            expected_h1=args.expected_h1,
        )
    elif args.command in {"barrier", "validate-barrier"}:
        cases = _case_dirs(args.lock, args.case_root, args.expected_h1)
        if args.command == "barrier":
            result = create_confirmation_prediction_barrier(
                args.output,
                args.lock,
                args.h2_commit,
                cases,
                expected_h1=args.expected_h1,
            )
        else:
            result = validate_confirmation_prediction_barrier(
                args.barrier,
                args.lock,
                args.h2_commit,
                cases,
                expected_h1=args.expected_h1,
            )
    elif args.command in {"compatibility", "validate-compatibility"}:
        cases = _case_dirs(args.lock, args.case_root, args.expected_h1)
        measurements = _case_dirs(
            args.lock,
            args.measurement_root,
            args.expected_h1,
        )
        common = (
            args.adapter_repo,
            args.lock,
            args.h2_commit,
            args.barrier,
            cases,
            measurements,
            args.compatibility_root,
        )
        if args.command == "compatibility":
            result = build_confirmation_outcome_compatibility(
                *common,
                expected_h1=args.expected_h1,
            )
        else:
            validated = validate_confirmation_outcome_compatibility(
                *common,
                expected_h1=args.expected_h1,
            )
            result = validated.manifest
    else:
        cases = _case_dirs(args.lock, args.case_root, args.expected_h1)
        measurements = _case_dirs(
            args.lock,
            args.measurement_root,
            args.expected_h1,
        )
        futures = _case_dirs(
            args.lock,
            args.authorized_future_root,
            args.expected_h1,
        )
        outcomes = _case_dirs(
            args.lock,
            args.authorized_outcome_root,
            args.expected_h1,
        )
        loader = build_confirmation_case_target_loader(
            args.adapter_repo,
            args.lock,
            args.h2_commit,
            args.barrier,
            cases,
            measurements,
            args.compatibility_root,
            futures,
            outcomes,
            expected_h1=args.expected_h1,
            production_mode=True,
        )
        scoring_attestation = validate_confirmation_case_target_loader_attestation(
            loader,
            args.lock,
            args.h2_commit,
            expected_h1=args.expected_h1,
            require_production=True,
        )
        result = evaluate_adaptive_covariance_confirmation(
            args.lock,
            args.barrier,
            args.h2_commit,
            cases,
            expected_h1=args.expected_h1,
            target_loader=loader,
            evaluation_mode=PRODUCTION_EVALUATION_MODE,
            scoring_attestation=scoring_attestation,
            adapter_repository=args.adapter_repo,
        )
        output = Path(args.output).absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(
        "invoke scripts/remote/run_deform360_adaptive_confirmation_cli.py "
        "instead of python -m"
    )
