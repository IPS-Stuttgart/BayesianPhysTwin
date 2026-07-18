"""CLI for a typed piecewise PhysTwin topology proposal."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_piecewise_topology import (
    build_piecewise_topology_from_files,
)


def _floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated numeric vector")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a future-blind piecewise topology candidate artifact."
    )
    parser.add_argument("final_data")
    parser.add_argument("optimal_params")
    parser.add_argument("checkpoint")
    parser.add_argument("partition")
    parser.add_argument("output")
    parser.add_argument("--radius-multipliers", type=_floats, required=True)
    parser.add_argument("--neighbour-multipliers", type=_floats, required=True)
    parser.add_argument("--object-log-scale", type=float, default=0.0)
    parser.add_argument("--controller-log-scale", type=float, default=0.0)
    args = parser.parse_args()
    result = build_piecewise_topology_from_files(
        args.final_data,
        args.optimal_params,
        args.checkpoint,
        args.partition,
        args.output,
        radius_multipliers=args.radius_multipliers,
        neighbour_multipliers=args.neighbour_multipliers,
        object_log_scale=args.object_log_scale,
        controller_log_scale=args.controller_log_scale,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
