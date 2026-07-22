"""CLI for outcome-free Deform360 disjoint-camera guarded predictions."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from bayesian_phystwin.deform360_crossview_guard_artifact import (
    build_crossview_guard_prediction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement_dir")
    parser.add_argument("supplement_dir")
    parser.add_argument("baseline_archive")
    parser.add_argument("baseline_key")
    parser.add_argument("output_dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_crossview_guard_prediction(
        args.measurement_dir,
        args.supplement_dir,
        args.baseline_archive,
        args.baseline_key,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
