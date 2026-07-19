#!/usr/bin/env python3
"""Create a behavior-equivalent compact proxy for MatPhys's simple decoder."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.matphys_graph_parts import compact_graph_part_proxy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_summary")
    parser.add_argument("output_root")
    args = parser.parse_args()
    result = compact_graph_part_proxy(args.source_summary, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
