"""CLI for shared-object, trial-controller PhysTwin profile combination."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_joint_profile import combine_joint_profile_files


def _assignments(values: list[str], *, value_type: type = str) -> dict[str, object]:
    result: dict[str, object] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected CASE=VALUE, got {value}")
        case_name, raw = value.split("=", 1)
        if not case_name or not raw or case_name in result:
            raise ValueError(f"invalid or duplicate assignment: {value}")
        result[case_name] = value_type(raw)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine profile grids with shared object stiffness and "
            "trial-specific controller scales."
        )
    )
    parser.add_argument("output_dir")
    parser.add_argument("profiles", nargs="+", metavar="CASE=PROFILE_NPZ")
    parser.add_argument("--temperature", action="append", default=[], metavar="CASE=VALUE")
    parser.add_argument("--object-prior-std", type=float, default=0.15)
    parser.add_argument("--controller-prior-std", type=float, default=0.50)
    args = parser.parse_args()
    summary = combine_joint_profile_files(
        _assignments(args.profiles),
        args.output_dir,
        object_prior_std=args.object_prior_std,
        controller_prior_std=args.controller_prior_std,
        likelihood_temperatures=_assignments(
            args.temperature,
            value_type=float,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
