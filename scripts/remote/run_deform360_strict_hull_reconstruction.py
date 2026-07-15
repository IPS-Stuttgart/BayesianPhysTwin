#!/usr/bin/env python3
"""Run source-only Deform360 Splatfacto with a strict thin-object hull seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from causal4d_public.deform360_dense_source import sha256_file
from deform360.processing import reconstruct_stage


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--minimum-hull-points", type=int, default=512)
    parser.add_argument("--voxel-resolution", type=int, default=120)
    parser.add_argument("--first-frame-iterations", type=int, default=500)
    parser.add_argument("--warm-start-iterations", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.minimum_hull_points < 1:
        raise ValueError("minimum hull point count must be positive")
    episode_dir = args.aligned_dir / f"episode_{args.episode:04d}"
    source_manifest = episode_dir / "dense_source_smoke.manifest.json"
    source_boundary = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not source_boundary.get("source_only"):
        raise ValueError("strict reconstruction accepts only source-only data")

    original = reconstruct_stage.visual_hull_points

    def strict_visual_hull_points(*call_args: object, **call_kwargs: object):
        call_kwargs["min_points"] = args.minimum_hull_points
        return original(*call_args, **call_kwargs)

    reconstruct_stage.visual_hull_points = strict_visual_hull_points
    try:
        outputs = reconstruct_stage.process_reconstruction_episode(
            args.aligned_dir,
            args.episode,
            first_frame_iterations=args.first_frame_iterations,
            warm_start_iterations=args.warm_start_iterations,
            voxel_resolution=args.voxel_resolution,
            overwrite=True,
        )
    finally:
        reconstruct_stage.visual_hull_points = original

    payload = {
        "schema": "bayesian-phystwin/deform360-strict-hull-reconstruction/v1",
        "source_only": True,
        "source_manifest_sha256": sha256_file(source_manifest),
        "minimum_hull_points": args.minimum_hull_points,
        "voxel_resolution": args.voxel_resolution,
        "first_frame_iterations": args.first_frame_iterations,
        "warm_start_iterations": args.warm_start_iterations,
        "outputs": {
            str(frame): {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for frame, path in sorted(outputs.items())
        },
        "released_default_minimum_hull_points": int(
            reconstruct_stage.DEFAULT_MIN_HULL_POINTS
        ),
        "claim_boundary": (
            "source-only reconstruction control; no calibration or target "
            "episode was read"
        ),
    }
    payload["result_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    output_path = episode_dir / "strict_hull_reconstruction.meta.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
