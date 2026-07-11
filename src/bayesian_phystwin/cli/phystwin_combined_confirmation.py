"""CLI for full-cohort hierarchical-mechanics plus discrepancy confirmation."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_combined_confirmation import (
    COMBINED_STAGES,
    run_combined_confirmation_stage,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm hierarchical mechanics plus locked discrepancy."
    )
    parser.add_argument("official_repo")
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument("--cohort", choices=("main", "additional"), required=True)
    parser.add_argument("--stage", choices=COMBINED_STAGES, default="all")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--component-run")
    args = parser.parse_args()
    result = run_combined_confirmation_stage(
        args.official_repo,
        args.data_root,
        args.output_dir,
        cohort=args.cohort,
        stage=args.stage,
        component_run=args.component_run,
        execution_cases=args.cases,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
