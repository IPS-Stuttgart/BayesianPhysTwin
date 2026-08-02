#!/usr/bin/env python3
"""Build the hash-only exclusion union for the tactile prospective protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bayesian_phystwin.deform360_exclusion_union import (  # noqa: E402
    build_exclusion_union,
)


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclusion", type=Path, action="append", required=True)
    parser.add_argument("--opened-source", type=Path, action="append", required=True)
    parser.add_argument("--additional-opened-object", action="append", default=[])
    parser.add_argument("--additional-source-artifact", type=Path, action="append", default=[])
    parser.add_argument("--owner", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifact = build_exclusion_union(
        [(path, _json(path)) for path in args.exclusion],
        [(path, _json(path)) for path in args.opened_source],
        additional_opened_object_ids=args.additional_opened_object,
        additional_source_artifacts=args.additional_source_artifact,
        owner=args.owner,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact["accounting"], sort_keys=True))
    print(artifact["exclusion_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
