#!/usr/bin/env python3
"""Select fresh public PokeFlex takes without opening target archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_fresh_take_selection import (  # noqa: E402
    build_fresh_take_selection_manifest,
)


def _public_take_ids(dataset_root: Path) -> tuple[str, ...]:
    paths = sorted((dataset_root / "poking").glob("*/*_T*.zip"))
    take_ids = tuple(path.stem for path in paths)
    if not take_ids:
        raise ValueError("public PokeFlex archive inventory is empty")
    if len(set(take_ids)) != len(take_ids):
        raise ValueError("public PokeFlex archive inventory is duplicated")
    return take_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--selection-id", required=True)
    parser.add_argument("--salt-label", required=True)
    parser.add_argument("--created-at-utc", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace selection: {args.output}")
    payload = build_fresh_take_selection_manifest(
        ROOT,
        _public_take_ids(args.dataset_root),
        salt_label=args.salt_label,
        selection_id=args.selection_id,
        created_at_utc=args.created_at_utc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(
        json.dumps(
            {
                "eligible_object_count": payload["eligible_object_count"],
                "referenced_take_count": payload["referenced_take_count"],
                "selected_take_ids": payload["selected_take_ids"],
                "selection_manifest_sha256": payload[
                    "selection_manifest_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
