"""CLI for causal Bayesian overlays on an external PhysTwin backbone."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_external_backbone import (
    run_external_backbone_overlay,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a future-blind external trajectory bank and run frozen "
            "Bayesian/last-residual overlays on the official 22 cases."
        )
    )
    parser.add_argument("data_root")
    parser.add_argument("manifest_json")
    parser.add_argument("output_dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--development-smoke",
        action="store_true",
        help="Allow only an ordered subset of the three declared development cases.",
    )
    args = parser.parse_args()
    summary = run_external_backbone_overlay(
        args.data_root,
        args.output_dir,
        args.manifest_json,
        force=args.force,
        workers=args.workers,
        development_smoke=args.development_smoke,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
