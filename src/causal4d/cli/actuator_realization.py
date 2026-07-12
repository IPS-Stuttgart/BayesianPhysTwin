"""Fit source-only command-to-actuator synchronization with PyRecEst."""

from __future__ import annotations

import argparse
import json

from causal4d.actuator_realization import calibrate_actuator_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npz")
    parser.add_argument("output_json")
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--minimum-offset-s", type=float, default=-0.150)
    parser.add_argument("--maximum-offset-s", type=float, default=0.150)
    parser.add_argument("--offset-step-s", type=float, default=0.001)
    parser.add_argument("--maximum-time-delta-s", type=float, default=0.010)
    args = parser.parse_args()
    result = calibrate_actuator_npz(
        args.input_npz,
        args.output_json,
        execution_id=args.execution_id,
        minimum_offset_s=args.minimum_offset_s,
        maximum_offset_s=args.maximum_offset_s,
        offset_step_s=args.offset_step_s,
        maximum_time_delta_s=args.maximum_time_delta_s,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
