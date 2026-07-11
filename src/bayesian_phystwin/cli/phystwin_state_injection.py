"""CLI for frozen PhysTwin endpoint state-injection comparisons."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_state_injection import (
    run_phystwin_state_injection_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare output correction with PhysTwin state injection."
    )
    parser.add_argument("official_repo")
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--cohort",
        choices=("all", "development", "confirmation"),
        default="all",
    )
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--graph-prior-strength", type=float, default=0.1)
    parser.add_argument("--velocity-history-frames", type=int, default=3)
    parser.add_argument("--replay-endpoint-tolerance-m", type=float, default=0.002)
    parser.add_argument("--repeatability-replays", type=int, default=3)
    parser.add_argument("--atomic-spring-forces", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_phystwin_state_injection_comparison(
        args.official_repo,
        args.data_root,
        args.output_dir,
        cohort=args.cohort,
        cases=args.cases,
        graph_prior_strength=args.graph_prior_strength,
        velocity_history_frames=args.velocity_history_frames,
        replay_endpoint_tolerance_m=args.replay_endpoint_tolerance_m,
        repeatability_replays=args.repeatability_replays,
        deterministic_spring_forces=not args.atomic_spring_forces,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
