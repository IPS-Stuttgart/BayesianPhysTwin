"""Run the bounded open-27 robust-correspondence diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_robust_correspondence_diagnostic import (
    evaluate_deform360_robust_correspondence_cohort,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = evaluate_deform360_robust_correspondence_cohort(
        args.root.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
