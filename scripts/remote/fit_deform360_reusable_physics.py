#!/usr/bin/env python3
"""Freeze one reusable PhysTwin tuple from a fresh object's fit episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_reusable_physics import (
    fit_reusable_physics_selection,
    validate_reusable_physics_selection,
)
from causal4d_public.deform360_reusable_trust_protocol import (
    load_reusable_trust_protocol,
)
from causal4d_public.deform360_reusable_trust_state import (
    load_reusable_trust_state_addendum,
)


def _episode_paths(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        episode, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"expected EPISODE=PATH, got {value!r}")
        episode_id = int(episode)
        if episode_id in result:
            raise ValueError(f"episode path is repeated: {episode_id}")
        result[episode_id] = Path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-lock", type=Path, required=True)
    parser.add_argument("--physics-addendum", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--mask-addendum", type=Path)
    parser.add_argument("--state-addendum", type=Path)
    parser.add_argument("--trust-artifact", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--response-json", action="append", default=[], required=True)
    parser.add_argument("--target", action="append", default=[], required=True)
    parser.add_argument("--robot", action="append", default=[], required=True)
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
    payload = fit_reusable_physics_selection(
        args.response_json,
        target_paths=_episode_paths(args.target),
        robot_paths=_episode_paths(args.robot),
        protocol=protocol,
        trust_artifact_path=args.trust_artifact,
        object_id=args.object_id,
    )
    validate_reusable_physics_selection(payload, protocol=protocol)
    if args.output.exists():
        raise FileExistsError(f"physical selection already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
