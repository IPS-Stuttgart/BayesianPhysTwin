"""Run a prediction-only physical prior under the Deform360 held lock."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_held_physical_prior import run_held_physical_prior


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-zero-manifest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument(
        "--role", choices=("calibration", "confirmation"), default="calibration"
    )
    parser.add_argument("--upstream-repo", required=True)
    parser.add_argument("--official-phystwin-repo", required=True)
    parser.add_argument("--official-config", required=True)
    parser.add_argument("--deform360-repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_held_physical_prior(
        args.frame_zero_manifest,
        args.lock,
        args.output_dir,
        case_name=args.case_name,
        role=args.role,
        upstream_repo=args.upstream_repo,
        official_phystwin_repo=args.official_phystwin_repo,
        official_config=args.official_config,
        deform360_repo=args.deform360_repo,
        python=args.python,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
