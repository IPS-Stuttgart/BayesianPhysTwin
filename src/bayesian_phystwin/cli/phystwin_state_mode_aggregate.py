"""Aggregate frozen PhysTwin state mode-retention diagnostics."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.state_correction_decay import (
    aggregate_state_correction_modes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_json")
    parser.add_argument("result_json", nargs="+")
    args = parser.parse_args()
    result = aggregate_state_correction_modes(args.result_json, args.output_json)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
