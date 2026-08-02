#!/usr/bin/env python3
"""Validate the bound V14 no-outcome custody record before prediction work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bayesian_phystwin.deform360_tactile_guard_outcome_sealed import (  # noqa: E402
    load_outcome_sealed_protocol,
    validate_v14_custody,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-decision", type=Path, required=True)
    parser.add_argument("--staging-queue", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_outcome_sealed_protocol(
        args.protocol,
        repository_root=args.repo,
    )
    result = validate_v14_custody(
        protocol,
        source_decision_path=args.source_decision,
        staging_queue_path=args.staging_queue,
    )
    print(
        json.dumps(
            {
                "artifact_sha256": result["artifact_sha256"],
                "case_count": len(result["predictions"]),
                "source_outcome_authorized": result["outcome_blind_gate"][
                    "source_outcome_authorized"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
