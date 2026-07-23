#!/usr/bin/env python3
"""Analyze the preregistered open27 Deform360 raw-camera budget frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_raw_camera_budget_frontier import (
    analyze_camera_budget_frontier,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "sota"
    / "deform360_raw_camera_budget_frontier_v1_development.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    result = analyze_camera_budget_frontier(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
