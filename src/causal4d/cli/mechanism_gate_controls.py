"""Run controlled placebo and positive-control audits of the v3 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d.mechanism_gate_controls import (
    MechanismGateControlConfig,
    run_mechanism_gate_controls,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--simulation-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    result = run_mechanism_gate_controls(
        MechanismGateControlConfig(
            simulation_count=args.simulation_count,
            random_seed=args.seed,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
