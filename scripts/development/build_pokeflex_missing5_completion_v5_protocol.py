#!/usr/bin/env python3
"""Build the pretarget PokeFlex missing-five V5 completion amendment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_missing5_completion_v5 import (  # noqa: E402
    build_completion_protocol,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parent-protocol",
        type=Path,
        default=(ROOT / "configs" / "sota" / "pokeflex_action_robust_official18_v4.json"),
    )
    parser.add_argument(
        "--source-protocol",
        type=Path,
        default=(ROOT / "configs" / "sota" / "pokeflex_missing5_scale_source_v5.json"),
    )
    parser.add_argument(
        "--source-result",
        type=Path,
        default=(
            ROOT
            / "results"
            / "sota"
            / "pokeflex_missing5_scale_source_v5"
            / "source_result.json"
        ),
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to replace V5 completion lock: {args.output}")
    parent = json.loads(args.parent_protocol.read_text(encoding="utf-8"))
    source_protocol = json.loads(args.source_protocol.read_text(encoding="utf-8"))
    source_result = json.loads(args.source_result.read_text(encoding="utf-8"))
    payload = build_completion_protocol(
        parent,
        source_protocol,
        source_result,
        locked_at_utc=args.locked_at_utc,
        parent_protocol_file_sha256=file_sha256(args.parent_protocol),
        source_protocol_file_sha256=file_sha256(args.source_protocol),
        source_result_file_sha256=file_sha256(args.source_result),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "protocol_sha256": payload["protocol_sha256"],
                "target_effective_scales": payload["method"][
                    "target_effective_scales"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
