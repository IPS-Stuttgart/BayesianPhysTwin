#!/usr/bin/env python3
"""Collect or merge portable object-disjoint MatPhys spring fields."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.matphys_loo_spring_fields import (
    collect_loo_spring_fields,
    merge_loo_spring_field_bundles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("workspace_manifest")
    collect.add_argument("output_dir")
    collect.add_argument("--folds", help="Optional comma-separated fold indices.")
    merge = subparsers.add_parser("merge")
    merge.add_argument("output_dir")
    merge.add_argument("manifests", nargs="+")
    args = parser.parse_args()

    if args.command == "collect":
        folds = (
            None
            if args.folds is None
            else tuple(int(value) for value in args.folds.split(",") if value.strip())
        )
        result = collect_loo_spring_fields(
            args.workspace_manifest,
            args.output_dir,
            fold_indices=folds,
        )
    else:
        result = merge_loo_spring_field_bundles(args.manifests, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
