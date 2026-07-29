#!/usr/bin/env python3
"""Lock the metadata-only V14 Deform360 fresh-source staging queue."""

from __future__ import annotations

import argparse
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_direct_depth_cohort import (
    build_v14_staging_queue,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--metadata-preflight", type=Path, required=True)
    parser.add_argument("--exclusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    queue = build_v14_staging_queue(
        args.output,
        protocol_path=args.protocol,
        catalog_path=args.catalog,
        metadata_preflight_path=args.metadata_preflight,
        exclusion_path=args.exclusion,
    )
    print(queue["queue_sha256"])


if __name__ == "__main__":
    main()
