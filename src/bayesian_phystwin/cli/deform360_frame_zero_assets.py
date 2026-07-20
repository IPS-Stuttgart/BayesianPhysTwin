from __future__ import annotations

import argparse
import json
import os

from bayesian_phystwin.deform360_frame_zero_assets import (
    APPROVED_CALIBRATION_SMOKE_CASE,
    FrameZeroAssetConfig,
    PinnedFrameZeroSam2Runtime,
    load_generic_held_lock,
    run_frame_zero_asset_builder,
)
from bayesian_phystwin.deform360_frame_zero_semantic_gate import (
    PinnedFrameZeroSemanticGateRuntime,
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
    parser.add_argument(
        "--role", choices=("calibration", "confirmation"), required=True
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help=(
            "Restrict this invocation to the pre-approved 083-blanket-cloth-ep0000 "
            "calibration smoke without narrowing the lock-bound builder."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--semantic-model",
        help="Absolute path to the sealed pinned SigLIP2 snapshot.",
    )
    parser.add_argument(
        "--semantic-model-lock",
        help="Absolute path to the immutable SigLIP2 snapshot lock.",
    )
    parser.add_argument(
        "--deform360-code",
        help="Absolute path to the clean pinned official Deform360 repository.",
    )
    args = parser.parse_args()
    if args.smoke_only and not (
        args.role == "calibration" and args.case_name == APPROVED_CALIBRATION_SMOKE_CASE
    ):
        parser.error(
            "--smoke-only permits only calibration case "
            f"{APPROVED_CALIBRATION_SMOKE_CASE}"
        )
    optional_paths = (
        args.semantic_model,
        args.semantic_model_lock,
        args.deform360_code,
    )
    if any(optional_paths) and not all(optional_paths):
        parser.error(
            "--semantic-model, --semantic-model-lock, and --deform360-code "
            "must be supplied together"
        )
    semantic_runtime = None
    if all(optional_paths):
        # This prerequisite must be present before the SAM runtime first opens
        # CUDA/CuBLAS; setting it only when SigLIP is loaded is too late.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        semantic_runtime = PinnedFrameZeroSemanticGateRuntime(
            args.semantic_model,
            args.semantic_model_lock,
            args.deform360_code,
            device=args.device,
        )

    lock = load_generic_held_lock(args.lock)
    config = FrameZeroAssetConfig()
    runtime = PinnedFrameZeroSam2Runtime(
        args.sam2_repository,
        args.checkpoint,
        config=config.sam2,
        immutable_bindings=lock["immutable_bindings"],
        device=args.device,
    )
    try:
        manifest = run_frame_zero_asset_builder(
            args.episode_dir,
            args.case_name,
            args.lock,
            args.output_dir,
            runtime,
            role=args.role,
            config=config,
            semantic_runtime=semantic_runtime,
        )
    finally:
        runtime.close()
        if semantic_runtime is not None:
            semantic_runtime.close()
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
