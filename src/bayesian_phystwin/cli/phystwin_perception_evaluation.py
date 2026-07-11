"""CLI for locked case-held-out evaluation of regenerated perception cues."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_perception_evaluation import (
    PerceptionCueEvaluationProtocol,
    run_perception_cue_confirmation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select rich cue scales on development cases and test on held-out cases."
    )
    parser.add_argument("data_root")
    parser.add_argument("cue_root")
    parser.add_argument("output_dir")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    args = parser.parse_args()
    result = run_perception_cue_confirmation(
        args.data_root,
        args.cue_root,
        args.output_dir,
        protocol=PerceptionCueEvaluationProtocol(
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
