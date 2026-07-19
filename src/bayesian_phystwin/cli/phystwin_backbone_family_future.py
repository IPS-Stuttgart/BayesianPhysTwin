"""CLI for opening futures after PhysTwin backbone-family selection."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_backbone_family_gate import (
    open_backbone_family_future,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument("selection_summary")
    args = parser.parse_args()
    summary = open_backbone_family_future(
        args.data_root,
        args.output_dir,
        args.selection_summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
