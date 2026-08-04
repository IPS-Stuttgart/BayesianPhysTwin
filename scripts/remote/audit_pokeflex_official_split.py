#!/usr/bin/env python3
"""Audit whether the PokeFlex public archive materializes the paper split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.pokeflex_official_split import write_official_split_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-empty-inventory-files",
        action="store_true",
        help="Treat zero-byte .zip placeholders as an availability index.",
    )
    args = parser.parse_args()

    result = write_official_split_audit(
        args.public_root,
        args.output,
        require_nonempty_archives=not args.allow_empty_inventory_files,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
