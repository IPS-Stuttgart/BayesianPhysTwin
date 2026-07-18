"""CLI for causal selection across PhysTwin-compatible backbone families."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_backbone_family_gate import (
    run_backbone_family_gate,
)


def _family(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("family must have the form NAME=SUMMARY.json")
    return name.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a PhysTwin backbone family on the permitted validation split."
    )
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--family",
        action="append",
        type=_family,
        required=True,
        help="Ordered NAME=overlay-summary pair; declare the fallback family first.",
    )
    parser.add_argument("--cases", help="Optional comma-separated ordered case subset.")
    parser.add_argument("--development-smoke", action="store_true")
    args = parser.parse_args()
    families = dict(args.family)
    if len(families) != len(args.family):
        parser.error("family names must be unique")
    case_names = None
    if args.cases:
        case_names = tuple(case.strip() for case in args.cases.split(",") if case.strip())
    summary = run_backbone_family_gate(
        args.data_root,
        args.output_dir,
        families,
        case_names=case_names,
        development_smoke=args.development_smoke,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
