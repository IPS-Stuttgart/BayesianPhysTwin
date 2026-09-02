#!/usr/bin/env python3
"""Audit the immutable Deform360 carrier subset against the current public tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import deform360_bound_replay_v8 as bound_replay


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-artifact-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        bound_replay.self_test()
        return

    artifact_root = args.confirmation_artifact_root.resolve(strict=True)
    result = bound_replay.verify_bound_replay(
        args.data_root,
        read_json(artifact_root / "confirmation-protocol.json"),
        read_json(artifact_root / "bound-readiness.json"),
        read_json(artifact_root / "result.json"),
    )
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "object_count": result["object_count"],
                "objects_with_additions": result["objects_with_additions"],
                "additive_unbound_file_count": result["additive_unbound_file_count"],
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
