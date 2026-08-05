#!/usr/bin/env python3
"""Build the frozen final-two PokeFlex freshness audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "src"))

from bayesian_phystwin.pokeflex_action_robust_final_freshness import (  # noqa: E402
    build_final_freshness_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous-audit",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "sota"
            / "pokeflex_action_robust_fresh6_exclusion_audit_v3.json"
        ),
    )
    parser.add_argument("--locked-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    previous = json.loads(args.previous_audit.read_text(encoding="utf-8"))
    payload = build_final_freshness_audit(
        previous,
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
