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
    parser.add_argument("--registered-subset-protocol")
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.0)
    parser.add_argument("--maximum-metric-regression", type=float)
    parser.add_argument(
        "--stability-control",
        action="append",
        type=_family,
        help="NAME=manifest.json identity replay for a non-reference family.",
    )
    parser.add_argument("--maximum-stability-rmse-m", type=float)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Freeze family choices without opening future metrics.",
    )
    args = parser.parse_args()
    families = dict(args.family)
    if len(families) != len(args.family):
        parser.error("family names must be unique")
    stability_controls = dict(args.stability_control or [])
    if len(stability_controls) != len(args.stability_control or []):
        parser.error("stability-control family names must be unique")
    case_names = None
    if args.cases:
        case_names = tuple(case.strip() for case in args.cases.split(",") if case.strip())
    summary = run_backbone_family_gate(
        args.data_root,
        args.output_dir,
        families,
        case_names=case_names,
        development_smoke=args.development_smoke,
        registered_subset_protocol=args.registered_subset_protocol,
        minimum_relative_improvement=args.minimum_relative_improvement,
        maximum_metric_regression=args.maximum_metric_regression,
        stability_control_manifests=(stability_controls or None),
        maximum_stability_rmse_m=args.maximum_stability_rmse_m,
        evaluate_future=not args.selection_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
