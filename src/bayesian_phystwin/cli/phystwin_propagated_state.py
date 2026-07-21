"""Run the guarded action-propagated state diagnostic in official Warp."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_propagated_state import (
    evaluate_guarded_propagated_state_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_repo")
    parser.add_argument("localization_case_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--position-step-m", type=float, default=0.005)
    parser.add_argument("--velocity-step-mps", type=float, default=0.05)
    args = parser.parse_args()
    result = evaluate_guarded_propagated_state_case(
        args.official_repo,
        args.localization_case_dir,
        args.output_dir,
        position_step_m=args.position_step_m,
        velocity_step_mps=args.velocity_step_mps,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
