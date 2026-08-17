#!/usr/bin/env python3
"""Build the frozen per-object shrinkage map from the opened source audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from bayesian_phystwin.pokeflex_instance_shrinkage import (  # noqa: E402
    build_instance_scale_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to replace calibration: {args.output}")
    source = json.loads(args.source_audit.read_text(encoding="utf-8"))
    calibration = build_instance_scale_calibration(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "calibration_sha256": calibration["calibration_sha256"],
                "objects": {
                    name: row["multiplier"]
                    for name, row in calibration["objects"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
