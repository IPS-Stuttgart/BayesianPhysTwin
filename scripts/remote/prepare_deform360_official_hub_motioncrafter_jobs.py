#!/usr/bin/env python3
"""Freeze the official-Hub calibration MotionCrafter job schedule."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from bayesian_phystwin._portable_contracts import load_strict_json_object
from bayesian_phystwin.deform360_official_hub_causal_windows import (
    load_deform360_official_hub_causal_window_manifest_v2,
)
from bayesian_phystwin.deform360_official_hub_motioncrafter_jobs import (
    build_deform360_motioncrafter_job_manifest,
    save_deform360_motioncrafter_job_manifest,
)
from bayesian_phystwin.deform360_visual_provider_recovery_lock import (
    load_deform360_visual_provider_recovery_lock,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-window-manifest", type=Path, required=True)
    parser.add_argument("--provider-lock", type=Path, required=True)
    parser.add_argument("--model-set-manifest", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    causal = load_deform360_official_hub_causal_window_manifest_v2(
        args.causal_window_manifest
    )
    provider = load_deform360_visual_provider_recovery_lock(args.provider_lock)
    model_set = load_strict_json_object(
        args.model_set_manifest,
        label="MotionCrafter model-set manifest",
    )
    manifest = build_deform360_motioncrafter_job_manifest(
        causal_window_manifest=causal,
        causal_window_manifest_file_sha256=_sha256(args.causal_window_manifest),
        provider_lock=provider,
        provider_lock_file_sha256=_sha256(args.provider_lock),
        model_set_manifest=model_set,
        model_set_manifest_file_sha256=_sha256(args.model_set_manifest),
        implementation_revision=args.implementation_revision,
        runner_source_sha256=_sha256(args.runner_source),
    )
    save_deform360_motioncrafter_job_manifest(args.output, manifest)
    print(manifest["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
