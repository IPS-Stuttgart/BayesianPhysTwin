"""CLI for locked full-cohort residual baseline confirmation."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_baseline_confirmation import (
    run_phystwin_baseline_confirmation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched residual comparators on the PhysTwin cohort."
    )
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_phystwin_baseline_confirmation(
        args.data_root,
        args.output_dir,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
