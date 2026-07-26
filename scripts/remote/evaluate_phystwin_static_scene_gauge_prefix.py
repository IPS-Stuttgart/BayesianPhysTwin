#!/usr/bin/env python3
"""Score one static-scene gauge artifact on allowed manual prefix tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.phystwin_static_scene_gauge_competence import (
    StaticSceneGaugeCompetenceConfig,
    evaluate_phystwin_static_scene_gauge_prefix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--cues", type=Path, required=True)
    parser.add_argument("--gauge", type=Path, required=True)
    parser.add_argument("--raw-case-dir", type=Path, required=True)
    parser.add_argument("--final-data", type=Path, required=True)
    parser.add_argument("--manual-tracks", type=Path, required=True)
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate_phystwin_static_scene_gauge_prefix(
        args.cues,
        args.gauge,
        args.raw_case_dir,
        args.final_data,
        args.manual_tracks,
        case=args.case,
        config=StaticSceneGaugeCompetenceConfig(
            train_end_frame=args.train_end_frame,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
