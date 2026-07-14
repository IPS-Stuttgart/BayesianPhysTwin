"""Generate pinned public-SAM2 fallback masks for one Deform360 rope episode."""

from __future__ import annotations

import argparse
import json

from causal4d_public.deform360 import load_deform360_protocol_config
from causal4d_public.deform360_sam2 import (
    RopeSam2VideoPredictor,
    build_sam2_mask_audit,
    validate_sam2_episode_access,
    validate_sam2_mask_artifact,
    write_sam2_mask_audit,
)
from causal4d_public.deform360_sam2_views import load_sam2_view_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root")
    parser.add_argument("episode_index", type=int)
    parser.add_argument("output_audit_json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sam2-repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera", action="append", dest="cameras")
    parser.add_argument("--view-audit-json")
    parser.add_argument("--held-out-prediction-seal-sha256")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()
    predictor = None
    try:
        config = load_deform360_protocol_config(args.config)
        access = validate_sam2_episode_access(
            args.episode_index,
            config,
            held_out_prediction_seal_sha256=(args.held_out_prediction_seal_sha256),
        )
        cameras = args.cameras
        view_audit_result_sha256 = None
        if args.view_audit_json is not None:
            if cameras is not None:
                raise ValueError("use either --camera or --view-audit-json, not both")
            view_audit = load_sam2_view_audit(args.view_audit_json)
            if view_audit["protocol_id"] != config.protocol_id:
                raise ValueError("view audit belongs to a different protocol")
            if view_audit["episode_access"] != access:
                raise ValueError("view audit episode or information boundary mismatch")
            cameras = view_audit["cross_view_consistency"]["accepted_cameras"]
            view_audit_result_sha256 = view_audit["result_sha256"]
        try:
            from deform360.processing.masks import process_masks_episode
        except ImportError as error:
            raise RuntimeError(
                "the pinned Deform360 processing environment is required"
            ) from error
        predictor = RopeSam2VideoPredictor(
            args.sam2_repository,
            args.checkpoint,
            device=args.device,
        )
        outputs = process_masks_episode(
            args.processed_root,
            args.episode_index,
            prompt="striped rope",
            predictor=predictor,
            cameras=cameras,
            overwrite=not args.no_overwrite,
            preview=args.preview,
        )
        result = build_sam2_mask_audit(
            protocol_id=config.protocol_id,
            episode_access=access,
            predictor=predictor,
            output_paths=outputs,
            view_audit_result_sha256=view_audit_result_sha256,
        )
        write_sam2_mask_audit(args.output_audit_json, result)
        validation = validate_sam2_mask_artifact(result)
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    finally:
        if predictor is not None:
            predictor.close()
    print(
        json.dumps(
            {
                **validation,
                "camera_count": len(result["outputs"]),
                "output": args.output_audit_json,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
