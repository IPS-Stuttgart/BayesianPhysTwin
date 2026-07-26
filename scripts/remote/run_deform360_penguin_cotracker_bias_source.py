#!/usr/bin/env python3
"""Build, seal, or evaluate the source-only penguin CoTracker3 belief study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.cotracker3_prefix import CoTracker3PrefixRuntime
from bayesian_phystwin.deform360_cotracker_bias_source import (
    SOURCE_EPISODE_IDS,
    build_penguin_source_prediction,
    evaluate_penguin_source_predictions,
    penguin_episode_directory,
    seal_penguin_source_predictions,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--staged-root", type=Path, required=True)
    predict.add_argument("--response-root", type=Path, required=True)
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--cotracker-source", type=Path, required=True)
    predict.add_argument("--cotracker-checkpoint", type=Path, required=True)
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--episode-id", type=int, action="append")

    seal = subparsers.add_parser("seal")
    seal.add_argument("--prediction-root", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prediction-root", type=Path, required=True)
    evaluate.add_argument("--staged-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.operation == "predict":
        episodes = tuple(args.episode_id or SOURCE_EPISODE_IDS)
        if any(episode not in SOURCE_EPISODE_IDS for episode in episodes):
            raise ValueError("requested episode is outside the source panel")
        runtime = CoTracker3PrefixRuntime(
            args.cotracker_source,
            args.cotracker_checkpoint,
            device=args.device,
        )
        reports = []
        try:
            for episode_id in episodes:
                reports.append(
                    build_penguin_source_prediction(
                        episode_id=episode_id,
                        episode_dir=penguin_episode_directory(
                            args.staged_root, episode_id
                        ),
                        response_root=args.response_root,
                        output_dir=(
                            args.output_root / f"episode_{episode_id:04d}"
                        ),
                        tracker=runtime,
                    )
                )
        finally:
            runtime.close()
        output = {
            "operation": "predict",
            "episode_ids": list(episodes),
            "reports": [
                {
                    "episode_id": report["episode_id"],
                    "technical_fallback": report["technical_fallback"],
                    "result_sha256": report["result_sha256"],
                }
                for report in reports
            ],
        }
    elif args.operation == "seal":
        output = seal_penguin_source_predictions(args.prediction_root)
    else:
        output = evaluate_penguin_source_predictions(
            prediction_root=args.prediction_root,
            staged_root=args.staged_root,
            output_path=args.output,
        )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
