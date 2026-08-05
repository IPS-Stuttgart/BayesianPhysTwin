#!/usr/bin/env python3
"""Derive the frozen Deform360 calibration windows without opening scores."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bayesian_phystwin._portable_contracts import load_strict_json_object
from bayesian_phystwin.deform360_official_hub_causal_windows import (
    build_deform360_official_hub_causal_window_manifest,
    save_deform360_official_hub_causal_window_manifest,
)
from bayesian_phystwin.deform360_official_hub_stage1 import (
    EXPECTED_PROCESSING_REVISION,
)
from bayesian_phystwin.deform360_visual_provider_recovery_lock import (
    load_deform360_visual_provider_recovery_lock,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--processing-report", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--provider-lock", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--processing-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _clean_revision(repository: Path, *, expected: str | None, name: str) -> str:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if expected is not None and revision != expected:
        raise ValueError(f"{name} revision changed")
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError(f"{name} checkout is dirty")
    return revision


def main() -> None:
    args = _parse_args()
    repository = args.repository.resolve()
    processing_repository = args.processing_repository.resolve()
    implementation_revision = _clean_revision(
        repository,
        expected=None,
        name="Bayesian-PhysTwin",
    )
    _clean_revision(
        processing_repository,
        expected=EXPECTED_PROCESSING_REVISION,
        name="official Deform360 processing",
    )

    processing_report = load_strict_json_object(
        args.processing_report,
        label="Stage 1 processing report",
    )
    execution_lock = load_strict_json_object(
        args.execution_lock,
        label="visual execution lock",
    )
    provider_lock = load_deform360_visual_provider_recovery_lock(args.provider_lock)

    sys.path.insert(0, str(processing_repository))
    from deform360.processing.episode import load_episode_calibration

    manifest = build_deform360_official_hub_causal_window_manifest(
        processing_report=processing_report,
        processed_root=args.processed_root,
        provider_lock=provider_lock,
        execution_lock=execution_lock,
        implementation_revision=implementation_revision,
        camera_calibration_loader=load_episode_calibration,
    )
    save_deform360_official_hub_causal_window_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "success_count": manifest["success_count"],
                "retained_technical_failure_count": manifest[
                    "retained_technical_failure_count"
                ],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
