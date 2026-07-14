"""Select reliable Deform360 rope views using pinned SAM2 and calibration."""

from __future__ import annotations

import argparse
import json

from causal4d_public.deform360 import load_deform360_protocol_config
from causal4d_public.deform360_sam2 import (
    RopeSam2VideoPredictor,
    validate_sam2_episode_access,
)
from causal4d_public.deform360_sam2_views import (
    CrossViewMaskReliabilityConfig,
    build_sam2_view_audit,
    multiview_mask_consistency,
    validate_sam2_view_audit,
    write_sam2_view_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root")
    parser.add_argument("episode_index", type=int)
    parser.add_argument("output_json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sam2-repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera", action="append", dest="cameras")
    parser.add_argument("--held-out-prediction-seal-sha256")
    args = parser.parse_args()
    predictor = None
    try:
        config = load_deform360_protocol_config(args.config)
        access = validate_sam2_episode_access(
            args.episode_index,
            config,
            held_out_prediction_seal_sha256=(args.held_out_prediction_seal_sha256),
        )
        try:
            from deform360.layout import resolve_episode_dir
            from deform360.processing.episode import (
                episode_cameras,
                load_episode_calibration,
            )
        except ImportError as error:
            raise RuntimeError(
                "the pinned Deform360 processing environment is required"
            ) from error
        episode_dir = resolve_episode_dir(args.processed_root, args.episode_index)
        cameras = args.cameras or episode_cameras(episode_dir)
        intrinsics, extrinsics = load_episode_calibration(episode_dir)
        predictor = RopeSam2VideoPredictor(
            args.sam2_repository,
            args.checkpoint,
            device=args.device,
        )
        masks = {}
        automatic_diagnostics = []
        for camera in cameras:
            video_path = episode_dir / camera / "undistorted.mp4"
            try:
                mask, diagnostic = predictor.select_initial_mask(video_path)
            except ValueError as error:
                automatic_diagnostics.append(
                    {
                        "camera": camera,
                        "automatic_selected": False,
                        "error": str(error),
                    }
                )
                continue
            masks[camera] = mask
            automatic_diagnostics.append(
                {
                    "camera": camera,
                    "automatic_selected": True,
                    "diagnostic": diagnostic,
                }
            )
        reliability_config = CrossViewMaskReliabilityConfig()
        consistency = multiview_mask_consistency(
            masks,
            intrinsics,
            extrinsics,
            reliability_config,
        )
        result = build_sam2_view_audit(
            protocol_id=config.protocol_id,
            episode_access=access,
            automatic_view_diagnostics=automatic_diagnostics,
            consistency=consistency,
            reliability_config=reliability_config,
        )
        write_sam2_view_audit(args.output_json, result)
        validation = validate_sam2_view_audit(result)
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    finally:
        if predictor is not None:
            predictor.close()
    print(
        json.dumps(
            {**validation, "output": args.output_json},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
