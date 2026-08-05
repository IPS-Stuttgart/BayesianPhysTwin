#!/usr/bin/env python3
"""Build the frozen retrospective 78-action PokeFlex audit protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_public_transfer_audit import (  # noqa: E402
    build_public_transfer_protocol,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=(
            ROOT / "configs" / "sota" / "pokeflex_action_robust_scale_all18_v4.json"
        ),
    )
    parser.add_argument(
        "--freshness",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_fresh2_exclusion_audit_v5.json"
        ),
    )
    parser.add_argument("--archive-inventory", type=Path, required=True)
    parser.add_argument("--locked-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = build_public_transfer_protocol(
        json.loads(args.calibration.read_text(encoding="utf-8")),
        json.loads(args.freshness.read_text(encoding="utf-8")),
        json.loads(args.archive_inventory.read_text(encoding="utf-8")),
        archive_inventory_file_sha256=file_sha256(args.archive_inventory),
        locked_at_utc=args.locked_at_utc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
