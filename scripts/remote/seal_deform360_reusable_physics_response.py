#!/usr/bin/env python3
"""Seal one fresh-panel physical response without opening an object outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_reusable_physics import (
    seal_reusable_physics_response,
    validate_reusable_physics_response,
)
from causal4d_public.deform360_reusable_trust_protocol import (
    load_reusable_trust_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-lock", type=Path, required=True)
    parser.add_argument("--physics-addendum", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--operation", choices=("fit", "held-prediction"), required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--init-spring-y", type=float, required=True)
    parser.add_argument("--drag-damping", type=float, required=True)
    parser.add_argument("--dashpot-damping", type=float, required=True)
    parser.add_argument("--prediction-data", type=Path, required=True)
    parser.add_argument("--simulator-data", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--twin-summary", type=Path, required=True)
    parser.add_argument("--driven-result", type=Path, required=True)
    parser.add_argument("--zero-result", type=Path, required=True)
    parser.add_argument("--prediction-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_reusable_trust_protocol(
        args.parent_lock, args.physics_addendum, args.execution_lock
    )
    payload = seal_reusable_physics_response(
        args.prediction_archive,
        protocol=protocol,
        object_id=args.object_id,
        episode_id=args.episode_id,
        operation=args.operation,
        parameters={
            "init_spring_y": args.init_spring_y,
            "drag_damping": args.drag_damping,
            "dashpot_damping": args.dashpot_damping,
        },
        prediction_data_path=args.prediction_data,
        simulator_data_path=args.simulator_data,
        graph_path=args.graph,
        readout_path=args.readout,
        twin_summary_path=args.twin_summary,
        driven_result_path=args.driven_result,
        zero_result_path=args.zero_result,
    )
    validate_reusable_physics_response(
        payload, protocol=protocol, verify_archive=True
    )
    if args.output.exists():
        raise FileExistsError(f"physical response already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
