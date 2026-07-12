"""Audit decay of a prefix-state correction in frozen PhysTwin rollouts."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.state_correction_decay import (
    audit_frozen_state_correction_decay,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json")
    parser.add_argument("rollout_npz")
    parser.add_argument("correction_json")
    parser.add_argument("output_json")
    args = parser.parse_args()
    result = audit_frozen_state_correction_decay(
        args.summary_json,
        args.rollout_npz,
        args.correction_json,
        args.output_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
