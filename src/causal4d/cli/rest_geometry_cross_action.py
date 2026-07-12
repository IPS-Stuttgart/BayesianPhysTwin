"""CLI for fit-only rest-geometry selection and transfer-plan locking."""

from __future__ import annotations

import argparse
import json

from causal4d.rest_geometry_cross_action import (
    write_rest_geometry_cross_action_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select shared rest-geometry hyperparameters from fit executions "
            "and lock factual/same-grasp/new-contact evaluations."
        )
    )
    parser.add_argument("protocol")
    parser.add_argument("evidence_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = write_rest_geometry_cross_action_selection(
        args.protocol,
        args.evidence_dir,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
