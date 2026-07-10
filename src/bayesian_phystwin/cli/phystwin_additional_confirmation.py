"""CLI for the label-free additional PhysTwin confirmation."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_additional_confirmation import (
    SPATIAL_MODES,
    run_additional_anchor_confirmation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm capped persistent anchoring without manual labels."
    )
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--global-translation", action="store_true")
    parser.add_argument("--spatial-mode", choices=SPATIAL_MODES, default="per_point")
    args = parser.parse_args()
    if args.global_translation and args.spatial_mode != "per_point":
        parser.error("do not combine --global-translation and --spatial-mode")
    summary = run_additional_anchor_confirmation(
        args.data_root,
        args.output_dir,
        force=args.force,
        spatial_mode=(
            "global_translation" if args.global_translation else args.spatial_mode
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
