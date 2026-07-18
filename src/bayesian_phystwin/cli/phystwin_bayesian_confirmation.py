"""CLI for full-cohort Bayesian residual-anchor evaluation."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_bayesian_confirmation import (
    run_bayesian_anchor_confirmation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate robust Bayesian discrepancy anchoring on PhysTwin."
    )
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    summary = run_bayesian_anchor_confirmation(
        args.data_root,
        args.output_dir,
        force=args.force,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
