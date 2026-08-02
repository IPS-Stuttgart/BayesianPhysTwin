"""CLI for the open-27 source transfer contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from bayesian_phystwin.deform360_pairwise_bias_aware_transfer import (
    stage_open27_transfer_bundle,
    validate_open27_transfer_bundle,
    write_open27_transfer_manifest,
)


def _add_source_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--measurement-root", required=True)
    parser.add_argument("--uncertainty-root", required=True)
    parser.add_argument("--selected-baseline-root", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory, stage, or validate the exact already-open Deform360 "
            "27-case source inputs without computing an outcome."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory")
    _add_source_roots(inventory)
    inventory.add_argument("--output-manifest", required=True)

    stage = commands.add_parser("stage")
    _add_source_roots(stage)
    stage.add_argument("--destination", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--bundle-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        result = write_open27_transfer_manifest(
            args.source_root,
            args.measurement_root,
            args.uncertainty_root,
            args.selected_baseline_root,
            args.output_manifest,
        )
    elif args.command == "stage":
        result = stage_open27_transfer_bundle(
            args.source_root,
            args.measurement_root,
            args.uncertainty_root,
            args.selected_baseline_root,
            args.destination,
        )
    else:
        result = validate_open27_transfer_bundle(args.bundle_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
