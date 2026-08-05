#!/usr/bin/env python3
"""Materialize the locked one-episode Deform360 Stage-1 processing view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_official_hub_stage1 import (
    load_official_hub_stage1_lock,
    materialize_official_hub_stage1_processing_view,
)


def _load_object(path: Path, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {name}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    lock = load_official_hub_stage1_lock(
        args.repository,
        args.protocol,
        args.selection,
    )
    preflight = _load_object(args.preflight_manifest, name="preflight manifest")
    download = _load_object(args.download_manifest, name="download manifest")
    result = materialize_official_hub_stage1_processing_view(
        preflight,
        download,
        args.payload_root,
        args.output_root,
        lock=lock,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "processing_view_sha256": result["processing_view_sha256"],
                "object_count": result["object_count"],
                "linked_file_count": result["linked_file_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
