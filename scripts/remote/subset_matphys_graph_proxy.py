#!/usr/bin/env python3
"""Assemble a compact MatPhys proxy subset from byte-bound source proxies."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.matphys_graph_parts import (
    materialize_compact_graph_proxy_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--cases", required=True)
    args = parser.parse_args()
    cases = tuple(case.strip() for case in args.cases.split(",") if case.strip())
    result = materialize_compact_graph_proxy_subset(
        args.source,
        args.output_root,
        cases,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
