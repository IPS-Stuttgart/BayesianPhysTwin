#!/usr/bin/env python3
"""Run source-only Deform360 Splatfacto with a strict thin-object hull seed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from causal4d_public.deform360_dense_source import sha256_file
from causal4d_public.deform360_independent_source import (
    validate_independent_source_prediction_seal,
)
from causal4d_public.deform360_reusable_physics import (
    validate_reusable_physics_fit_grid_seal,
)
from causal4d_public.deform360_reusable_trust_protocol import (
    authorize_reusable_trust_held_outcome,
    load_reusable_trust_protocol,
    validate_reusable_trust_prediction_cohort_seal,
)
from causal4d_public.deform360_reusable_trust_state import (
    load_reusable_trust_state_addendum,
)
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
    parser.add_argument(
        "--frame-zero-only",
        action="store_true",
        help="Build only the causal frame-zero splat before prediction sealing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing splats and train only missing frames.",
    )
    parser.add_argument(
        "--prediction-seal",
        type=Path,
        help="Required before post-initial object reconstruction may begin.",
    )
    parser.add_argument("--fresh-parent-lock", type=Path)
    parser.add_argument("--physics-addendum", type=Path)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--mask-addendum", type=Path)
    parser.add_argument("--state-addendum", type=Path)
    parser.add_argument("--fresh-fit-grid-seal", type=Path)
    parser.add_argument("--fresh-held-cohort-seal", type=Path)
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
    prediction_seal: dict[str, Any] | None = None
    future_access_seal: dict[str, Any] | None = None
    future_access_path: Path | None = None
    fresh_values = (
        args.fresh_parent_lock,
        args.physics_addendum,
        args.execution_lock,
    )
    if any(value is not None for value in fresh_values) and not all(
        value is not None for value in fresh_values
    ):
        raise ValueError(
            "fresh parent, physics, and execution locks are required together"
        )
    if (args.mask_addendum is None) != (args.state_addendum is None):
        raise ValueError("mask and state addenda are required together")
    if args.state_addendum is not None and args.fresh_parent_lock is None:
        raise ValueError("state addendum requires the fresh protocol locks")
    fresh_protocol = (
        None
        if args.fresh_parent_lock is None
        else (
            load_reusable_trust_protocol(
                args.fresh_parent_lock,
                args.physics_addendum,
                args.execution_lock,
            )
            if args.state_addendum is None
            else load_reusable_trust_state_addendum(
                args.fresh_parent_lock,
                args.physics_addendum,
                args.execution_lock,
                args.mask_addendum,
                args.state_addendum,
            )
        )
    )
    if args.frame_zero_only:
        if any(
            value is not None
            for value in (
                args.prediction_seal,
                args.fresh_fit_grid_seal,
                args.fresh_held_cohort_seal,
            )
        ):
            raise ValueError(
                "frame-zero reconstruction must precede prediction sealing"
            )
    else:
        if fresh_protocol is None:
            if args.prediction_seal is None:
                raise ValueError("full reconstruction requires a sealed prediction")
            if (
                args.fresh_fit_grid_seal is not None
                or args.fresh_held_cohort_seal is not None
            ):
                raise ValueError("fresh seals require the fresh protocol")
            prediction_seal = json.loads(
                args.prediction_seal.read_text(encoding="utf-8")
            )
            validate_independent_source_prediction_seal(
                prediction_seal, verify_archive=True
            )
            if prediction_seal.get("object_id") != source_boundary.get(
                "object_id"
            ) or int(prediction_seal.get("episode_id", -1)) != int(
                source_boundary.get("episode_index", -2)
            ):
                raise ValueError("prediction seal belongs to another source episode")
            future_access_seal = prediction_seal
            future_access_path = args.prediction_seal
        else:
            if args.prediction_seal is not None:
                raise ValueError("fresh future access cannot use a legacy seal")
            supplied = [
                value
                for value in (
                    args.fresh_fit_grid_seal,
                    args.fresh_held_cohort_seal,
                )
                if value is not None
            ]
            if len(supplied) != 1:
                raise ValueError(
                    "fresh full reconstruction requires exactly one future-access seal"
                )
            future_access_path = supplied[0]
            future_access_seal = json.loads(
                future_access_path.read_text(encoding="utf-8")
            )
            object_id = str(source_boundary.get("object_id"))
            episode_id = int(source_boundary.get("episode_index", -1))
            if args.fresh_fit_grid_seal is not None:
                validated = validate_reusable_physics_fit_grid_seal(
                    future_access_seal,
                    protocol=fresh_protocol,
                    verify_responses=True,
                )
                if (
                    validated["object_id"] != object_id
                    or validated["episode_id"] != episode_id
                ):
                    raise ValueError("fit-grid seal belongs to another episode")
            else:
                validate_reusable_trust_prediction_cohort_seal(
                    future_access_seal,
                    protocol=fresh_protocol,
                    verify_predictions=True,
                )
                authorize_reusable_trust_held_outcome(
                    fresh_protocol,
                    future_access_seal,
                    object_id=object_id,
                    episode_id=episode_id,
                )

    original = reconstruct_stage.visual_hull_points
    original_frame_count = reconstruct_stage.camera_frame_count

    def strict_visual_hull_points(*call_args: object, **call_kwargs: object):
        call_kwargs["min_points"] = args.minimum_hull_points
        return original(*call_args, **call_kwargs)

    def frame_zero_count(*_args: object, **_kwargs: object) -> int:
        return 1

    reconstruct_stage.visual_hull_points = strict_visual_hull_points
    if args.frame_zero_only:
        reconstruct_stage.camera_frame_count = frame_zero_count
    try:
        outputs = reconstruct_stage.process_reconstruction_episode(
            args.aligned_dir,
            args.episode,
            first_frame_iterations=args.first_frame_iterations,
            warm_start_iterations=args.warm_start_iterations,
            voxel_resolution=args.voxel_resolution,
            overwrite=not args.resume,
        )
    finally:
        reconstruct_stage.visual_hull_points = original
        reconstruct_stage.camera_frame_count = original_frame_count
    if args.frame_zero_only and set(outputs) != {0}:
        raise ValueError(
            "frame-zero reconstruction returned post-initial object geometry"
        )

    payload = {
        "schema": "bayesian-phystwin/deform360-strict-hull-reconstruction/v1",
        "source_only": True,
        "source_manifest_sha256": sha256_file(source_manifest),
        "minimum_hull_points": args.minimum_hull_points,
        "voxel_resolution": args.voxel_resolution,
        "first_frame_iterations": args.first_frame_iterations,
        "warm_start_iterations": args.warm_start_iterations,
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "nerfstudio", "gsplat")
        },
        "frame_zero_only": args.frame_zero_only,
        "resumed": args.resume,
        "prediction_seal": (
            None
            if prediction_seal is None
            else {
                "path": str(args.prediction_seal.resolve()),
                "file_sha256": sha256_file(args.prediction_seal),
                "result_sha256": prediction_seal["result_sha256"],
            }
        ),
        "future_access_seal": (
            None
            if future_access_seal is None
            else {
                "path": str(future_access_path.resolve()),
                "file_sha256": sha256_file(future_access_path),
                "result_sha256": future_access_seal["result_sha256"],
                "artifact_kind": future_access_seal.get("artifact_kind"),
            }
        ),
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
        "information_boundary": {
            "object_observation_frames_used": (
                [0] if args.frame_zero_only else list(range(len(outputs)))
            ),
            "prediction_must_be_sealed_before_future_reconstruction": (
                args.frame_zero_only
            ),
            "prediction_seal_verified_before_future_reconstruction": (
                future_access_seal is not None
            ),
        },
        "claim_boundary": (
            "source-only reconstruction control; no calibration or target "
            "episode was read"
        ),
    }
    payload["result_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    output_path = episode_dir / (
        "strict_hull_reconstruction_frame_zero.meta.json"
        if args.frame_zero_only
        else "strict_hull_reconstruction_full.meta.json"
    )
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
