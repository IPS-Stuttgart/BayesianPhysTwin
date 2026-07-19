"""Run the frozen pairwise gate on open-27 raw AllTracker measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_raw_pairwise_correspondence_diagnostic import (
    evaluate_raw_pairwise_correspondence_cohort,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel_root", type=Path)
    parser.add_argument("measurement_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = evaluate_raw_pairwise_correspondence_cohort(
        args.panel_root.resolve(),
        args.measurement_root.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
