"""CLI for the locked part-pair source-prefix family gate."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_part_pair_source_gate import (
    run_part_pair_source_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen future-blind gate to part-pair refit runs."
    )
    parser.add_argument("source_root")
    parser.add_argument("output_path")
    parser.add_argument("source_protocol")
    args = parser.parse_args()
    result = run_part_pair_source_gate(
        args.source_root,
        args.output_path,
        args.source_protocol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
