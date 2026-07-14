"""Propagate sealed target-prefix masks after rope predictions are immutable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360 import load_deform360_protocol_config
from causal4d_public.deform360_rope_evaluation import (
    validate_held_out_rope_prediction_seal,
)
from causal4d_public.deform360_sam2 import RopeSam2VideoPredictor
from causal4d_public.deform360_sam2_prefix import validate_sam2_prefix_mask_artifact
from causal4d_public.deform360_sam2_suffix import (
    build_sam2_suffix_mask_audit,
    segment_target_suffix_camera,
    validate_sam2_suffix_mask_artifact,
    write_sam2_suffix_mask_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root")
    parser.add_argument("held_out_prediction_seal_json")
    parser.add_argument("prefix_mask_audit_json")
    parser.add_argument("output_mask_dir")
    parser.add_argument("output_audit_json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sam2-repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    predictor = None
    try:
        protocol = load_deform360_protocol_config(args.config)
        prediction_seal = json.loads(
            Path(args.held_out_prediction_seal_json).read_text(encoding="utf-8")
        )
        validate_held_out_rope_prediction_seal(prediction_seal)
        prefix_audit = json.loads(
            Path(args.prefix_mask_audit_json).read_text(encoding="utf-8")
        )
        validate_sam2_prefix_mask_artifact(prefix_audit)
        target_index = protocol.target_episode_ids[0]
        episode_dir = Path(args.processed_root) / f"episode_{target_index:04d}"
        try:
            from deform360.processing.episode import camera_frame_count
        except ImportError as error:
            raise RuntimeError(
                "the pinned Deform360 processing environment is required"
            ) from error
        cameras = list(prefix_audit["camera_policy"]["selected_cameras"])
        stop = int(camera_frame_count(episode_dir, cameras[0]))
        start = int(prefix_audit["target_prefix"]["start_frame"])
        prefix_outputs = {row["camera"]: row for row in prefix_audit["outputs"]}
        predictor = RopeSam2VideoPredictor(
            args.sam2_repository, args.checkpoint, device=args.device
        )
        outputs = []
        for camera in cameras:
            outputs.append(
                segment_target_suffix_camera(
                    predictor,
                    episode_dir / camera / "undistorted.mp4",
                    prefix_outputs[camera],
                    Path(args.output_mask_dir) / f"{camera}.npy",
                    start_frame=start,
                    stop_frame_exclusive=stop,
                    prefix_frame_count=protocol.prefix_frame_count,
                )
            )
        artifact = build_sam2_suffix_mask_audit(
            protocol=protocol,
            held_out_prediction_seal=prediction_seal,
            prefix_mask_audit=prefix_audit,
            predictor=predictor,
            camera_outputs=outputs,
        )
        write_sam2_suffix_mask_audit(args.output_audit_json, artifact)
        validation = validate_sam2_suffix_mask_artifact(artifact)
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
