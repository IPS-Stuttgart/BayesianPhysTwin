"""Run the source-learned predictive-coupling development experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from bayesian_phystwin_experiments.deform_predictive_coupling import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.archive, args.config, args.output)
    print(
        f"Complete: {result['case_count']} exploratory holdouts. Results: {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
