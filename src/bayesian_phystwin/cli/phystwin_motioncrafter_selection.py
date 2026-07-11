"""CLI for training-only MotionCrafter camera selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.phystwin_motioncrafter_selection import (
    select_motioncrafter_views,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one MotionCrafter camera per PhysTwin case."
    )
    parser.add_argument("output_json")
    parser.add_argument("summary", nargs="+")
    args = parser.parse_args()
    result = select_motioncrafter_views(args.summary)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
