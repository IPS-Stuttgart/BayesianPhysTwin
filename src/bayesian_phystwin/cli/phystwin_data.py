"""CLI for selectively retrieving released PhysTwin evaluation artifacts."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_data import (
    DEFAULT_ADDITIONAL_ARCHIVE,
    DEFAULT_DATA_ARCHIVE,
    DEFAULT_EXPERIMENTS_ARCHIVE,
    fetch_phystwin_additional_evaluation_subset,
    fetch_phystwin_evaluation_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the compact released PhysTwin trajectory-evaluation subset."
    )
    parser.add_argument("output_dir")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--data-archive", default=DEFAULT_DATA_ARCHIVE)
    parser.add_argument("--experiments-archive", default=DEFAULT_EXPERIMENTS_ARCHIVE)
    parser.add_argument("--additional", action="store_true")
    parser.add_argument("--additional-archive", default=DEFAULT_ADDITIONAL_ARCHIVE)
    args = parser.parse_args()
    if args.additional:
        manifest = fetch_phystwin_additional_evaluation_subset(
            args.output_dir,
            cases=args.cases,
            archive_url=args.additional_archive,
        )
    else:
        manifest = fetch_phystwin_evaluation_subset(
            args.output_dir,
            cases=args.cases,
            data_archive_url=args.data_archive,
            experiments_archive_url=args.experiments_archive,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
