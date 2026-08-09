#!/usr/bin/env python3
"""Build the third PokeFlex fresh-take exclusion audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from bayesian_phystwin.pokeflex_action_robust_freshness import (  # noqa: E402
    build_action_robust_freshness_audit,
    validate_action_robust_freshness_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--locked-at-utc", required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to replace freshness audit: {args.output}")
    previous = json.loads(args.previous_audit.read_text(encoding="utf-8"))
    audit = build_action_robust_freshness_audit(
        previous,
        locked_at_utc=args.locked_at_utc,
    )
    validate_action_robust_freshness_audit(
        audit,
        bind_registered_digest=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit_sha256": audit["audit_sha256"],
                "target_take_ids": audit["selection"]["take_ids"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
