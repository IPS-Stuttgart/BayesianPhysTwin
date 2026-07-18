"""CLI for fit-prefix zero-order PhysTwin topology/field search."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_zero_order_topology import (
    ZeroOrderTopologySearchConfig,
    run_zero_order_topology_search,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search regional PhysTwin topology and spring fields."
    )
    parser.add_argument("official_repo")
    parser.add_argument("fit_final_data")
    parser.add_argument("optimal_params")
    parser.add_argument("checkpoint")
    parser.add_argument("partition")
    parser.add_argument("cues")
    parser.add_argument("released_trajectory")
    parser.add_argument("gt_track_3d")
    parser.add_argument("output_dir")
    parser.add_argument("--fit-end-frame", type=int, required=True)
    parser.add_argument("--region-count", type=int, default=5)
    parser.add_argument("--candidates-per-family", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run_zero_order_topology_search(
        official_repo=args.official_repo,
        fit_final_data_path=args.fit_final_data,
        optimal_params_path=args.optimal_params,
        checkpoint_path=args.checkpoint,
        partition_path=args.partition,
        cues_path=args.cues,
        output_dir=args.output_dir,
        fit_end_frame=args.fit_end_frame,
        released_trajectory_path=args.released_trajectory,
        gt_track_path=args.gt_track_3d,
        config=ZeroOrderTopologySearchConfig(
            region_count=args.region_count,
            candidates_per_family=args.candidates_per_family,
            seed=args.seed,
        ),
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
