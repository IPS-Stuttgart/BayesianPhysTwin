#!/usr/bin/env python3
"""Seal all frozen physical responses for one fresh-panel fit episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_reusable_physics import (
    build_reusable_physics_fit_grid_seal,
    validate_reusable_physics_fit_grid_seal,
)
from causal4d_public.deform360_reusable_trust_protocol import (
    load_reusable_trust_protocol,
)
from causal4d_public.deform360_reusable_trust_state import (
    load_reusable_trust_state_addendum,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-lock", type=Path, required=True)
    parser.add_argument("--physics-addendum", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--mask-addendum", type=Path)
    parser.add_argument("--state-addendum", type=Path)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--response-json", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if (args.mask_addendum is None) != (args.state_addendum is None):
        raise ValueError("mask and state addenda are required together")
    protocol = (
        load_reusable_trust_protocol(
            args.parent_lock, args.physics_addendum, args.execution_lock
        )
        if args.state_addendum is None
        else load_reusable_trust_state_addendum(
            args.parent_lock,
            args.physics_addendum,
            args.execution_lock,
            args.mask_addendum,
            args.state_addendum,
        )
    )
    payload = build_reusable_physics_fit_grid_seal(
        args.response_json,
        protocol=protocol,
        object_id=args.object_id,
        episode_id=args.episode_id,
    )
    validate_reusable_physics_fit_grid_seal(
        payload, protocol=protocol, verify_responses=True
    )
    if args.output.exists():
        raise FileExistsError(f"fit-grid seal already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
