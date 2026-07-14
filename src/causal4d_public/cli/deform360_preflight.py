"""Audit the pinned Deform360 001-rope cohort before model fitting."""

from __future__ import annotations

import argparse
import json

from causal4d_public.deform360 import (
    load_deform360_protocol_config,
    preflight_deform360_001_rope,
    validate_deform360_preflight,
    write_deform360_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_object_dir")
    parser.add_argument("output_json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--processed-root")
    parser.add_argument(
        "--hash-media",
        action="store_true",
        help="Hash every video and NumPy input instead of metadata only.",
    )
    target_tactile = parser.add_mutually_exclusive_group()
    target_tactile.add_argument(
        "--unlock-target-prefix",
        action="store_true",
        help="Read only the locked, visually triggered six-frame target tactile prefix.",
    )
    target_tactile.add_argument(
        "--unlock-target-oracle",
        action="store_true",
        help="Read target tactile values only after target predictions are sealed.",
    )
    parser.add_argument(
        "--target-prefix-start-frame",
        type=int,
        help=(
            "Start of the target tactile prefix selected by the sealed visual/robot "
            "trigger; required with --unlock-target-prefix."
        ),
    )
    args = parser.parse_args()
    try:
        config = load_deform360_protocol_config(args.config)
        result = preflight_deform360_001_rope(
            args.raw_object_dir,
            config,
            processed_root=args.processed_root,
            hash_media=args.hash_media,
            unlock_target_prefix=args.unlock_target_prefix,
            unlock_target_oracle=args.unlock_target_oracle,
            target_prefix_start_frame=args.target_prefix_start_frame,
        )
        write_deform360_preflight(args.output_json, result)
        validation = validate_deform360_preflight(result)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                **validation,
                "capability_gates": result["capability_gates"],
                "split_counts": result["split"]["counts"],
                "output": args.output_json,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["preflight_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
