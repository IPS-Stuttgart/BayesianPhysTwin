#!/usr/bin/env python3
"""Build the source-only fresh technical-replication artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_artifacts import (
    build_fresh_prediction_cohort,
    build_fresh_prediction_failure_seal,
    build_fresh_processing_cohort,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_processing import (
    build_fresh_processing_protocol,
    validate_fresh_processing_protocol,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    build_exclusion_union,
    build_fresh_download_manifest,
    build_fresh_source_plan,
    build_fresh_technical_lock,
    validate_exclusion_union,
    validate_fresh_download_manifest,
    validate_fresh_source_plan,
    validate_fresh_technical_lock,
    write_json_artifact,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_runtime import (
    build_fresh_runtime_amendment,
    validate_fresh_runtime_amendment,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    union = subparsers.add_parser("union")
    union.add_argument("output", type=Path)
    union.add_argument("manifest", nargs="+", type=Path)

    lock = subparsers.add_parser("lock")
    lock.add_argument("output", type=Path)
    lock.add_argument("--exclusion-union", required=True, type=Path)
    lock.add_argument("--public-catalog", required=True, type=Path)
    lock.add_argument("--metadata", required=True, type=Path)
    lock.add_argument("--source-protocol", required=True, type=Path)
    lock.add_argument("--source-qualification", required=True, type=Path)

    plan = subparsers.add_parser("plan")
    plan.add_argument("output", type=Path)
    plan.add_argument("--technical-lock", required=True, type=Path)
    plan.add_argument("--repository-tree", required=True, type=Path)

    verify = subparsers.add_parser("verify-download")
    verify.add_argument("output", type=Path)
    verify.add_argument("--source-plan", required=True, type=Path)
    verify.add_argument("--download-root", required=True, type=Path)

    processing = subparsers.add_parser("processing-lock")
    processing.add_argument("output", type=Path)
    processing.add_argument("--technical-lock", required=True, type=Path)
    processing.add_argument("--source-plan", required=True, type=Path)
    processing.add_argument("--download-manifest", required=True, type=Path)
    processing.add_argument("--implementation-commit", required=True)

    runtime = subparsers.add_parser("runtime-lock")
    runtime.add_argument("output", type=Path)
    runtime.add_argument("--processing-protocol", required=True, type=Path)
    runtime.add_argument("--failed-artifact", required=True, type=Path)
    runtime.add_argument("--failed-log", required=True, type=Path)
    runtime.add_argument("--runtime-identity", required=True, type=Path)
    runtime.add_argument("--validator-commit", required=True)

    cohort = subparsers.add_parser("processing-cohort")
    cohort.add_argument("output", type=Path)
    cohort.add_argument("--technical-lock", required=True, type=Path)
    cohort.add_argument("--processing-protocol", required=True, type=Path)
    cohort.add_argument("--processed-root", required=True, type=Path)

    failure = subparsers.add_parser("prediction-failure")
    failure.add_argument("output", type=Path)
    failure.add_argument("--technical-lock", required=True, type=Path)
    failure.add_argument("--processing-protocol", required=True, type=Path)
    failure.add_argument("--processing-cohort", required=True, type=Path)
    failure.add_argument("--object-id", required=True)
    failure.add_argument("--episode-id", required=True, type=int)

    barrier = subparsers.add_parser("prediction-cohort")
    barrier.add_argument("output", type=Path)
    barrier.add_argument("--technical-lock", required=True, type=Path)
    barrier.add_argument("--prediction-root", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "union":
        artifact = build_exclusion_union(args.manifest)
        validate_exclusion_union(artifact)
    elif args.command == "lock":
        artifact = build_fresh_technical_lock(
            args.exclusion_union,
            args.public_catalog,
            args.metadata,
            args.source_protocol,
            args.source_qualification,
        )
        validate_fresh_technical_lock(artifact)
    elif args.command == "plan":
        artifact = build_fresh_source_plan(
            args.technical_lock,
            args.repository_tree,
        )
        validate_fresh_source_plan(artifact)
    elif args.command == "verify-download":
        artifact = build_fresh_download_manifest(
            args.source_plan,
            args.download_root,
        )
        validate_fresh_download_manifest(artifact)
    elif args.command == "processing-lock":
        artifact = build_fresh_processing_protocol(
            args.technical_lock,
            args.source_plan,
            args.download_manifest,
            implementation_commit=args.implementation_commit,
        )
        validate_fresh_processing_protocol(artifact)
    elif args.command == "runtime-lock":
        artifact = build_fresh_runtime_amendment(
            args.processing_protocol,
            args.failed_artifact,
            args.failed_log,
            args.runtime_identity,
            validator_commit=args.validator_commit,
        )
        validate_fresh_runtime_amendment(artifact)
    elif args.command == "processing-cohort":
        artifact = build_fresh_processing_cohort(
            args.technical_lock,
            args.processing_protocol,
            args.processed_root,
        )
    elif args.command == "prediction-failure":
        artifact = build_fresh_prediction_failure_seal(
            args.technical_lock,
            args.processing_protocol,
            args.processing_cohort,
            args.output,
            object_id=args.object_id,
            episode_id=args.episode_id,
        )
        return
    else:
        artifact = build_fresh_prediction_cohort(
            args.technical_lock,
            args.prediction_root,
        )
    write_json_artifact(artifact, args.output)


if __name__ == "__main__":
    main()
