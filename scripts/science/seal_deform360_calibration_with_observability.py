#!/usr/bin/env python3
"""Seal Deform360 Stage 1 and authorize one confirmation opening.

This command wraps the existing calibration sealer in a temporary output root,
retains the successful calibration-source record and supported observability
report as exact source bytes, verifies their complete cross-artifact lineage,
and publishes the final directory only after a content-addressed confirmation-
opening authorization has been created.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin._portable_contracts import write_atomic_json
from bayesian_phystwin.cli import deform360_calibration_execution as legacy_cli
from bayesian_phystwin.deform360_calibration_bundle import (
    DEFORM360_CALIBRATION_ROLES,
    load_deform360_calibration_bundle,
)
from bayesian_phystwin.deform360_calibration_execution import (
    Deform360CalibrationExecutionArtifactsV1,
    file_sha256,
    load_deform360_calibration_artifact_ref,
    load_deform360_calibration_execution_seal,
    load_deform360_stage0_selection,
)
from bayesian_phystwin.deform360_calibration_observability_binding import (
    DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY,
    DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY,
    build_deform360_confirmation_opening_authorization,
    save_deform360_confirmation_opening_authorization,
)
from bayesian_phystwin.deform360_calibration_observability_report import (
    load_deform360_calibration_observability_report,
)
from bayesian_phystwin.deform360_calibration_source_run_record import (
    load_deform360_calibration_source_run_record,
)
from bayesian_phystwin.deform360_visual_provider_lock import (
    load_deform360_visual_calibration_lock,
    load_deform360_visual_provider_lock,
)
from bayesian_phystwin.evidence_use_ledger import load_evidence_use_ledger

_RUN_RECORD_ADDITIONAL_NAME = "calibration-source-run-record.json"
_OBSERVABILITY_ADDITIONAL_NAME = "calibration-observability-report.json"
_BINDING_SOURCE_ADDITIONAL_NAME = (
    "claim-bearing/deform360_calibration_observability_binding.py"
)
_COMMAND_SOURCE_ADDITIONAL_NAME = (
    "claim-bearing/seal_deform360_calibration_with_observability.py"
)
_RESERVED_ADDITIONAL_NAMES = frozenset(
    {
        _RUN_RECORD_ADDITIONAL_NAME,
        _OBSERVABILITY_ADDITIONAL_NAME,
        _BINDING_SOURCE_ADDITIONAL_NAME,
        _COMMAND_SOURCE_ADDITIONAL_NAME,
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--visual-provider-lock", type=Path, required=True)
    parser.add_argument("--evidence-ledger", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        type=legacy_cli._artifact_path,  # noqa: SLF001
        default=[],
        metavar="ROLE=PATH",
        help="Selected calibration artifact reference; repeat for all roles",
    )
    parser.add_argument(
        "--calibration-source-run-record",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration-observability-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument(
        "--additional-source",
        action="append",
        type=legacy_cli._named_path,  # noqa: SLF001
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-payloads-opened",
        action="store_true",
        help="Acknowledge that only the locked calibration payloads were opened",
    )
    return parser


def _claim_sources(
    args: argparse.Namespace,
    *,
    repository_root: Path,
) -> list[tuple[str, Path]]:
    supplied = list(args.additional_source)
    names = [name for name, _path in supplied]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError(f"duplicate additional source names: {duplicates}")
    reserved = sorted(_RESERVED_ADDITIONAL_NAMES.intersection(names))
    if reserved:
        raise ValueError(f"reserved additional source names were supplied: {reserved}")
    return [
        *supplied,
        (
            _RUN_RECORD_ADDITIONAL_NAME,
            args.calibration_source_run_record,
        ),
        (
            _OBSERVABILITY_ADDITIONAL_NAME,
            args.calibration_observability_report,
        ),
        (
            _BINDING_SOURCE_ADDITIONAL_NAME,
            repository_root
            / "src/bayesian_phystwin/"
            "deform360_calibration_observability_binding.py",
        ),
        (
            _COMMAND_SOURCE_ADDITIONAL_NAME,
            repository_root
            / "scripts/science/"
            "seal_deform360_calibration_with_observability.py",
        ),
    ]


def _load_products(root: Path) -> Deform360CalibrationExecutionArtifactsV1:
    return Deform360CalibrationExecutionArtifactsV1(
        visual_calibration_lock=load_deform360_visual_calibration_lock(
            root / "visual-calibration-lock.json"
        ),
        calibration_bundle=load_deform360_calibration_bundle(
            root / "calibration-bundle.json"
        ),
        execution_seal=load_deform360_calibration_execution_seal(
            root / "calibration-execution-seal.json"
        ),
    )


def _append_status(path: Path, summary: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n## Claim-bearing confirmation authorization\n\n")
        handle.write(f"- Authorization ID: `{summary['authorization_id']}`\n")
        handle.write(
            "- Calibration-source run record: "
            f"`{summary['calibration_source_run_record_sha256']}`\n"
        )
        handle.write(
            "- Calibration observability report: "
            f"`{summary['calibration_observability_report_id']}`\n"
        )
        handle.write("- Confirmation payloads opened: `false`\n")
        handle.write("- Target outcomes used: `false`\n\n")
        handle.write(str(summary["claim_boundary"]) + "\n")


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not args.calibration_payloads_opened:
        raise ValueError("--calibration-payloads-opened is required for Stage 1")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    repository_root = args.repository_root.resolve()
    workspace = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.claim-bound.",
            dir=output.parent,
        )
    )
    staged = workspace / "sealed"
    try:
        low_level_args = argparse.Namespace(
            selection_lock=args.selection_lock,
            visual_provider_lock=args.visual_provider_lock,
            evidence_ledger=args.evidence_ledger,
            artifact=args.artifact,
            implementation_revision=args.implementation_revision,
            repository_root=repository_root,
            metadata_json=args.metadata_json,
            additional_source=_claim_sources(
                args,
                repository_root=repository_root,
            ),
            output_dir=staged,
            calibration_payloads_opened=True,
        )
        legacy_cli._run(low_level_args)  # noqa: SLF001

        stage0 = load_deform360_stage0_selection(
            staged / "sources/stage0/selection.json",
            protocol_path=(
                staged
                / "sources/repository/protocols/"
                "deform360_official_hub_visuotactile_v1.json"
            ),
        )
        provider = load_deform360_visual_provider_lock(
            staged / "sources/locks/visual-provider-lock.json"
        )
        ledger = load_evidence_use_ledger(
            staged / "sources/calibration/evidence-use-ledger.json"
        )
        artifacts = tuple(
            load_deform360_calibration_artifact_ref(
                staged / f"sources/calibration/artifacts/{role}.json"
            )
            for role in DEFORM360_CALIBRATION_ROLES
        )
        products = _load_products(staged)
        if products.calibration_bundle.calibration_artifacts != artifacts:
            raise ValueError("published calibration artifacts differ from inputs")

        run_record_path = staged / DEFORM360_CALIBRATION_SOURCE_RUN_RECORD_SOURCE_KEY
        report_path = staged / DEFORM360_CALIBRATION_OBSERVABILITY_SOURCE_KEY
        run_record = load_deform360_calibration_source_run_record(run_record_path)
        report = load_deform360_calibration_observability_report(report_path)
        run_file_sha256 = file_sha256(run_record_path)
        report_file_sha256 = file_sha256(report_path)
        authorization = build_deform360_confirmation_opening_authorization(
            products,
            calibration_source_run_record=run_record,
            calibration_observability_report=report,
            calibration_source_run_record_file_sha256=run_file_sha256,
            calibration_observability_report_file_sha256=report_file_sha256,
            stage0_selection=stage0,
            visual_provider_lock=provider,
            evidence_use_ledger=ledger,
            metadata={
                "statistical_unit": "physical_object",
                "calibration_object_count": len(stage0.calibration_units),
                "confirmation_object_count": len(stage0.confirmation_units),
            },
        )
        save_deform360_confirmation_opening_authorization(
            authorization,
            staged / "confirmation-opening-authorization.json",
        )
        summary = authorization.summary()
        write_atomic_json(
            summary,
            staged / "confirmation-opening-summary.json",
            overwrite=False,
        )
        _append_status(staged / "STATUS.md", summary)
        checksums = staged / "SHA256SUMS"
        if checksums.exists():
            checksums.unlink()
        legacy_cli._write_checksums(staged)  # noqa: SLF001
        if output.exists():
            raise FileExistsError(output)
        os.replace(staged, output)
        return summary
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = _run(args)
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"authorized": False, "error": str(error)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
