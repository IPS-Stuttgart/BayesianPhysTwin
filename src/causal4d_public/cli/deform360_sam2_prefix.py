"""Generate source-locked SAM2 masks for the sealed Deform360 target prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360 import (
    load_deform360_protocol_config,
    validate_deform360_preflight,
)
from causal4d_public.deform360_contact import load_contact_artifact
from causal4d_public.deform360_sam2 import RopeSam2VideoPredictor
from causal4d_public.deform360_sam2_prefix import (
    build_sam2_prefix_mask_audit,
    segment_target_prefix_camera,
    select_source_locked_prefix_cameras,
    target_prefix_bounds,
    validate_sam2_prefix_mask_artifact,
    write_sam2_prefix_mask_audit,
)
from causal4d_public.deform360_sam2_views import load_sam2_view_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root")
    parser.add_argument("contact_prediction_seal_json")
    parser.add_argument("source_view_audit_json")
    parser.add_argument("preflight_json")
    parser.add_argument("output_mask_dir")
    parser.add_argument("output_audit_json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sam2-repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-sync-reliability", type=float, default=0.85)
    parser.add_argument("--minimum-camera-count", type=int, default=8)
    args = parser.parse_args()

    predictor = None
    try:
        config = load_deform360_protocol_config(args.config)
        contact_seal = load_contact_artifact(
            args.contact_prediction_seal_json,
            expected_kind="Deform360TargetContactPredictionSeal",
        )
        source_view_audit = load_sam2_view_audit(args.source_view_audit_json)
        preflight = json.loads(Path(args.preflight_json).read_text(encoding="utf-8"))
        validate_deform360_preflight(preflight)
        camera_policy = select_source_locked_prefix_cameras(
            source_view_audit,
            preflight,
            minimum_synchronization_reliability=args.minimum_sync_reliability,
            minimum_camera_count=config.minimum_tracking_camera_count,
        )
        start, stop = target_prefix_bounds(config, contact_seal)
        predictor = RopeSam2VideoPredictor(
            args.sam2_repository,
            args.checkpoint,
            device=args.device,
        )
        target_index = config.target_episode_ids[0]
        episode_dir = Path(args.processed_root) / f"episode_{target_index:04d}"
        mask_dir = Path(args.output_mask_dir)
        outputs = []
        failures = []
        for camera in camera_policy["selected_cameras"]:
            video_path = episode_dir / camera / "undistorted.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(f"target camera video is missing: {video_path}")
            try:
                outputs.append(
                    segment_target_prefix_camera(
                        predictor,
                        video_path,
                        mask_dir / f"{camera}.npy",
                        start_frame=start,
                        stop_frame_exclusive=stop,
                    )
                )
            except (RuntimeError, ValueError) as error:
                failures.append(
                    {
                        "camera": camera,
                        "reason": type(error).__name__,
                        "message": str(error),
                    }
                )
        result = build_sam2_prefix_mask_audit(
            config=config,
            contact_prediction_seal=contact_seal,
            camera_policy=camera_policy,
            predictor=predictor,
            camera_outputs=outputs,
            camera_failures=failures,
            minimum_camera_count=args.minimum_camera_count,
        )
        write_sam2_prefix_mask_audit(args.output_audit_json, result)
        validation = validate_sam2_prefix_mask_artifact(result)
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    finally:
        if predictor is not None:
            predictor.close()
    print(
        json.dumps(
            {**validation, "output": args.output_audit_json},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
