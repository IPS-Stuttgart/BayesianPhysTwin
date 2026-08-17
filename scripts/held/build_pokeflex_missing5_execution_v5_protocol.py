#!/usr/bin/env python3
"""Build the pre-archive-access PokeFlex missing-five V5 execution lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from bayesian_phystwin.pokeflex_missing5_execution_v5 import (  # noqa: E402
    IMPLEMENTATION_FILE_PATHS,
    build_execution_protocol,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--completion-protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_missing5_scale_completion_v5.json"
        ),
    )
    parser.add_argument(
        "--parent-protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_official18_v4.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT / "configs" / "sota" / "pokeflex_missing5_execution_v5.json"
        ),
    )
    parser.add_argument("--locked-at-utc", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace execution lock: {args.output}")
    completion = json.loads(args.completion_protocol.read_text(encoding="utf-8"))
    parent = json.loads(args.parent_protocol.read_text(encoding="utf-8"))
    hashes = {
        relative: file_sha256(REPOSITORY_ROOT / relative)
        for relative in IMPLEMENTATION_FILE_PATHS
    }
    protocol = build_execution_protocol(
        completion,
        parent,
        locked_at_utc=args.locked_at_utc,
        implementation_file_sha256s=hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "execution_protocol_sha256": protocol["execution_protocol_sha256"],
                "implementation_file_count": len(hashes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
