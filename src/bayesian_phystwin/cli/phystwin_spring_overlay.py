"""Build a PhysTwin checkpoint with a hashed external spring-field overlay."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_spring_overlay import (
    build_spring_overlay_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_checkpoint")
    parser.add_argument("spring_y_npy")
    parser.add_argument("output_checkpoint")
    parser.add_argument("--summary")
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    result = build_spring_overlay_checkpoint(
        args.source_checkpoint,
        args.spring_y_npy,
        args.output_checkpoint,
        summary_path=args.summary,
        strength=args.strength,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
