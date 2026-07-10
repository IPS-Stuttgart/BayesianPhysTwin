"""CLI for the label-free additional PhysTwin confirmation."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_additional_confirmation import (
    run_additional_anchor_confirmation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm capped persistent anchoring without manual labels."
    )
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_additional_anchor_confirmation(
        args.data_root,
        args.output_dir,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
