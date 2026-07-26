#!/usr/bin/env python3
"""Build a causal static-scene gauge artifact for one PhysTwin case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.phystwin_static_scene_gauge import (
    PhysTwinStaticSceneGaugeConfig,
    build_phystwin_static_scene_gauge,
)
from bayesian_phystwin.static_scene_gauge import StaticSceneGaugeConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cues", type=Path, required=True)
    parser.add_argument("--raw-case-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cotracker-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    summary = build_phystwin_static_scene_gauge(
        args.cues,
        args.raw_case_dir,
        args.checkpoint,
        args.cotracker_root,
        args.output,
        config=PhysTwinStaticSceneGaugeConfig(
            train_end_frame=args.train_end_frame,
            gauge=StaticSceneGaugeConfig(),
        ),
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
