#!/usr/bin/env python3
"""Download and verify the locked fresh Deform360 raw source object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    build_fresh_download_manifest,
    validate_fresh_download_manifest,
    validate_fresh_source_plan,
    write_json_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_plan", type=Path)
    parser.add_argument("download_root", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    plan = json.loads(args.source_plan.read_text(encoding="utf-8"))
    validate_fresh_source_plan(plan)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise RuntimeError("huggingface_hub is required for source download") from exc
    snapshot_download(
        repo_id=plan["repository"],
        repo_type="dataset",
        revision=plan["revision"],
        local_dir=str(args.download_root.resolve()),
        allow_patterns=[row["path"] for row in plan["download"]["files"]],
        max_workers=args.max_workers,
    )
    manifest = build_fresh_download_manifest(args.source_plan, args.download_root)
    validate_fresh_download_manifest(manifest)
    write_json_artifact(manifest, args.output_manifest)


if __name__ == "__main__":
    main()
