from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_frame_zero_assets import (
    FrameZeroAssetConfig,
    PinnedFrameZeroSam2Runtime,
    load_generic_held_lock,
    run_frame_zero_asset_builder,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a checksumed Deform360 frame-zero bundle from one RGB frame "
            "per camera and immutable calibration."
        )
    )
    parser.add_argument("--lock", required=True)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sam2-repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--role", choices=("calibration", "confirmation"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    lock = load_generic_held_lock(args.lock)
    config = FrameZeroAssetConfig()
    runtime = PinnedFrameZeroSam2Runtime(
        args.sam2_repository,
        args.checkpoint,
        config=config.sam2,
        device=args.device,
    )
    try:
        manifest = run_frame_zero_asset_builder(
            args.episode_dir,
            args.case_name,
            lock,
            args.output_dir,
            runtime,
            role=args.role,
            lock_file_sha256=sha256_file(args.lock),
            config=config,
        )
    finally:
        runtime.close()
    print(
        json.dumps(
            {
                "manifest_artifact_sha256": manifest["artifact_sha256"],
                "bundle": manifest["bundle"],
                "selected_action_bundle": manifest["action_alignment"][
                    "selected_action_bundle"
                ],
                "geometry_qa": manifest["geometry_qa"],
                "runtime": manifest["runtime"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
