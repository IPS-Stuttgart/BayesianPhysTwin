"""CLI for direct additional-cohort spatial-control comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.phystwin_additional_control_comparison import (
    compare_additional_anchor_controls,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a frozen additional-cohort anchor with spatial controls."
    )
    parser.add_argument("candidate_run_dir")
    parser.add_argument("output_json")
    parser.add_argument("reference_run_dirs", nargs="+")
    args = parser.parse_args()
    result = compare_additional_anchor_controls(
        args.candidate_run_dir,
        args.reference_run_dirs,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
