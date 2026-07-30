#!/usr/bin/env python3
"""Build the outcome-blind Cloth Sim2Real V1 dataset lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.cloth_sim2real_protocol import (
    build_cloth_sim2real_dataset_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite dataset lock: {args.output}")
    manifest = build_cloth_sim2real_dataset_manifest(
        args.dataset_root,
        archive_sha256=args.archive_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            manifest.descriptor(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(manifest.artifact_sha256)


if __name__ == "__main__":
    main()
