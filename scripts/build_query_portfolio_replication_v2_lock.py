#!/usr/bin/env python3
"""Write the prospective portfolio-recovery lock."""

from __future__ import annotations

import argparse
from pathlib import Path

from bayesian_phystwin._portable_contracts import write_atomic_json
from bayesian_phystwin.query_portfolio_replication_v2 import protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_atomic_json(protocol(), args.output, overwrite=False)


if __name__ == "__main__":
    main()
