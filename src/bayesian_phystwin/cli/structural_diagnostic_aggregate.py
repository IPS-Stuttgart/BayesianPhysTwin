"""CLI for aggregating released structural diagnostics."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.structural_diagnostic_aggregate import (
    write_structural_diagnostic_aggregate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json")
    parser.add_argument("summary_json", nargs="+")
    args = parser.parse_args()
    result = write_structural_diagnostic_aggregate(
        args.output_json, args.summary_json
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
