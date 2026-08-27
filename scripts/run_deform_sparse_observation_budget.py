"""Run the small, already-open DEFORM observation-budget diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin_experiments.deform_sparse_observation_budget import run_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(
        args.archive, args.config, args.output, require_clean_source=True
    )
    print(
        json.dumps(
            {
                "status": "complete-exploratory",
                "output": str(args.output.resolve()),
                "case_count": result["case_count"],
                "zero_budget_mean_byte_identical": result[
                    "zero_budget_mean_byte_identical"
                ],
                "curves": result["curves"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
