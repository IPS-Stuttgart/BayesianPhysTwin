"""Prepare and verify the official-Hub Deform360 Stage-1 control plane."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_calibration_bundle import (
    load_deform360_calibration_bundle,
)
from bayesian_phystwin.deform360_stage1_control import (
    build_deform360_stage1_plan,
    create_deform360_visual_provider_lock,
    derive_deform360_visual_calibration_lock,
    load_deform360_stage1_plan,
    save_deform360_stage1_plan,
    save_deform360_visual_calibration_lock_atomic,
    save_deform360_visual_provider_lock_atomic,
    verify_deform360_calibration_access,
    verify_deform360_confirmation_access,
    verify_deform360_stage1_seal,
)
from bayesian_phystwin.deform360_visual_provider_lock import (
    Deform360VisualCalibrationLockV1,
    Deform360VisualProviderLockV1,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_selection() -> Path:
    return (
        _repository_root()
        / "protocols"
        / "locks"
        / "deform360_official_hub_visuotactile_v1_selection.json"
    )


def _default_amendment() -> Path:
    return (
        _repository_root()
        / "protocols"
        / "amendments"
        / "deform360_official_hub_visuotactile_v1_visual_provider_lock.json"
    )


def _default_calibration_design() -> Path:
    return (
        _repository_root()
        / "protocols"
        / "amendments"
        / "deform360_official_hub_visuotactile_v1_calibration_separation.json"
    )


def _optional_positive_integer(value: str) -> int | None:
    if value == "none":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a positive integer or 'none'"
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return parsed


def _metadata(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    return load_strict_json_object(path, label="operator metadata")


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _provider_lock(args: argparse.Namespace) -> int:
    lock = create_deform360_visual_provider_lock(
        provider_attestation_path=args.provider_attestation,
        motioncrafter_revision=args.motioncrafter_revision,
        model_set_id=args.model_set_id,
        root_seed=args.root_seed,
        seed_policy=args.seed_policy,
        window_size=args.window_size,
        overlap=args.overlap,
        height=args.height,
        width=args.width,
        storage_dtype=args.storage_dtype,
        initial_metric_frame_prior_id=args.initial_metric_frame_prior_id,
        additional_metric_anchor_policy=args.additional_metric_anchor_policy,
        max_gauge_rank=args.max_gauge_rank,
        minimum_retained_gauge_trace=args.minimum_retained_gauge_trace,
        metadata=_metadata(args.metadata),
    )
    save_deform360_visual_provider_lock_atomic(
        lock,
        args.output,
        overwrite=args.overwrite,
    )
    _print_json(
        {
            "artifact_kind": "Deform360VisualProviderLockV1",
            "artifact_id": lock.artifact_id,
            "output": str(args.output.resolve()),
            "provider_revision": lock.provider_revision,
            "provider_manifest_id": lock.provider_manifest_id,
            "provider_attestation_sha256": lock.provider_attestation_sha256,
            "motioncrafter_revision": lock.motioncrafter_revision,
            "model_set_id": lock.model_set_id,
            "selected_raw_payloads_opened": False,
            "target_outcomes_used": False,
        }
    )
    return 0


def _plan(args: argparse.Namespace) -> int:
    plan = build_deform360_stage1_plan(
        selection_path=args.selection,
        provider_lock_path=args.provider_lock,
        amendment_path=args.amendment,
        calibration_design_path=args.calibration_design,
        metadata=_metadata(args.metadata),
    )
    save_deform360_stage1_plan(plan, args.output, overwrite=args.overwrite)
    _print_json({**plan.summary(), "output": str(args.output.resolve())})
    return 0


def _verify_plan(args: argparse.Namespace) -> int:
    plan = load_deform360_stage1_plan(args.plan)
    token = verify_deform360_calibration_access(
        plan,
        expected_plan_id=args.expected_plan_id,
        expected_provider_lock_id=args.expected_provider_lock_id,
        expected_selection_artifact_sha256=(
            args.expected_selection_artifact_sha256
        ),
    )
    _print_json({**plan.summary(), "verified_calibration_access_token": token})
    return 0


def _load_provider_lock(path: Path) -> Deform360VisualProviderLockV1:
    return Deform360VisualProviderLockV1.from_mapping(
        load_strict_json_object(path, label="Deform360 visual-provider lock")
    )


def _load_calibration_lock(path: Path) -> Deform360VisualCalibrationLockV1:
    return Deform360VisualCalibrationLockV1.from_mapping(
        load_strict_json_object(path, label="Deform360 visual calibration lock")
    )


def _seal(args: argparse.Namespace) -> int:
    plan = load_deform360_stage1_plan(args.plan)
    provider_lock = _load_provider_lock(args.provider_lock)
    bundle = load_deform360_calibration_bundle(args.calibration_bundle)
    calibration_lock = derive_deform360_visual_calibration_lock(
        plan=plan,
        provider_lock=provider_lock,
        bundle=bundle,
    )
    save_deform360_visual_calibration_lock_atomic(
        calibration_lock,
        args.output,
        overwrite=args.overwrite,
    )
    summary = verify_deform360_stage1_seal(
        plan=plan,
        provider_lock=provider_lock,
        bundle=bundle,
        calibration_lock=calibration_lock,
    )
    if args.summary_output is not None:
        write_atomic_json(
            summary,
            args.summary_output,
            overwrite=args.overwrite,
        )
    _print_json({**summary, "output": str(args.output.resolve())})
    return 0


def _verify_seal(args: argparse.Namespace) -> int:
    summary = verify_deform360_confirmation_access(
        plan=load_deform360_stage1_plan(args.plan),
        provider_lock=_load_provider_lock(args.provider_lock),
        bundle=load_deform360_calibration_bundle(args.calibration_bundle),
        calibration_lock=_load_calibration_lock(args.calibration_lock),
        expected_plan_id=args.expected_plan_id,
        expected_provider_lock_id=args.expected_provider_lock_id,
        expected_bundle_id=args.expected_bundle_id,
        expected_calibration_lock_id=args.expected_calibration_lock_id,
        expected_selection_artifact_sha256=(
            args.expected_selection_artifact_sha256
        ),
        expected_evidence_use_ledger_id=(
            args.expected_evidence_use_ledger_id
        ),
    )
    _print_json(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create and verify target-blind Deform360 Stage-1 provider, plan, "
            "and calibration locks."
        )
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    provider = subparsers.add_parser(
        "provider-lock",
        help="create the concrete target-blind visual-provider lock",
    )
    provider.add_argument("--provider-attestation", type=Path, required=True)
    provider.add_argument("--motioncrafter-revision", required=True)
    provider.add_argument("--model-set-id", required=True)
    provider.add_argument("--root-seed", type=int, default=20260805)
    provider.add_argument(
        "--seed-policy",
        default="per-object-derived-seed-v1",
    )
    provider.add_argument("--window-size", type=int, default=25)
    provider.add_argument("--overlap", type=int, default=8)
    provider.add_argument("--height", type=int, default=320)
    provider.add_argument("--width", type=int, default=640)
    provider.add_argument(
        "--storage-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    provider.add_argument("--initial-metric-frame-prior-id", required=True)
    provider.add_argument(
        "--additional-metric-anchor-policy",
        choices=("none", "independent_sparse"),
        default="none",
    )
    provider.add_argument(
        "--max-gauge-rank",
        type=_optional_positive_integer,
        default=64,
        metavar="INTEGER|none",
    )
    provider.add_argument(
        "--minimum-retained-gauge-trace",
        type=float,
        default=0.999,
    )
    provider.add_argument("--metadata", type=Path)
    provider.add_argument("--output", type=Path, required=True)
    provider.add_argument("--overwrite", action="store_true")
    provider.set_defaults(handler=_provider_lock)

    plan = subparsers.add_parser(
        "plan",
        help="bind Stage-0, the provider lock, and the finite-group design",
    )
    plan.add_argument("--selection", type=Path, default=_default_selection())
    plan.add_argument("--provider-lock", type=Path, required=True)
    plan.add_argument("--amendment", type=Path, default=_default_amendment())
    plan.add_argument(
        "--calibration-design",
        type=Path,
        default=_default_calibration_design(),
    )
    plan.add_argument("--metadata", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--overwrite", action="store_true")
    plan.set_defaults(handler=_plan)

    verify_plan = subparsers.add_parser(
        "verify-plan",
        help="verify reviewed identities before calibration payload access",
    )
    verify_plan.add_argument("--plan", type=Path, required=True)
    verify_plan.add_argument("--expected-plan-id", required=True)
    verify_plan.add_argument("--expected-provider-lock-id", required=True)
    verify_plan.add_argument(
        "--expected-selection-artifact-sha256",
        required=True,
    )
    verify_plan.set_defaults(handler=_verify_plan)

    seal = subparsers.add_parser(
        "seal",
        help="derive the visual calibration lock from the complete bundle",
    )
    seal.add_argument("--plan", type=Path, required=True)
    seal.add_argument("--provider-lock", type=Path, required=True)
    seal.add_argument("--calibration-bundle", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--summary-output", type=Path)
    seal.add_argument("--overwrite", action="store_true")
    seal.set_defaults(handler=_seal)

    verify_seal = subparsers.add_parser(
        "verify-seal",
        help="verify the exact seal before confirmation payload access",
    )
    verify_seal.add_argument("--plan", type=Path, required=True)
    verify_seal.add_argument("--provider-lock", type=Path, required=True)
    verify_seal.add_argument("--calibration-bundle", type=Path, required=True)
    verify_seal.add_argument("--calibration-lock", type=Path, required=True)
    verify_seal.add_argument("--expected-plan-id", required=True)
    verify_seal.add_argument("--expected-provider-lock-id", required=True)
    verify_seal.add_argument("--expected-bundle-id", required=True)
    verify_seal.add_argument("--expected-calibration-lock-id", required=True)
    verify_seal.add_argument(
        "--expected-selection-artifact-sha256",
        required=True,
    )
    verify_seal.add_argument(
        "--expected-evidence-use-ledger-id",
        required=True,
    )
    verify_seal.set_defaults(handler=_verify_seal)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
