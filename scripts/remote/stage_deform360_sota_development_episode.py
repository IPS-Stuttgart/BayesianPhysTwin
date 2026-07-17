#!/usr/bin/env python3
"""Propagate and stage one locked Deform360 development episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2VideoPredictor,
)
from causal4d_public.deform360_reusable_sota_protocol import (
    load_reusable_sota_config,
)
from causal4d_public.deform360_sota_processing import (
    authorize_development_processing,
    propagate_development_masks,
    stage_development_processing_episode,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--role", choices=("fit", "held-development"), required=True)
    parser.add_argument("--reference-annotation-root", type=Path, required=True)
    parser.add_argument("--reference-panel", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--processing-root", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite-stage", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = load_reusable_sota_config(args.protocol)
    authorization = authorize_development_processing(
        protocol,
        object_id=args.object_id,
        episode_id=args.episode_id,
        role=args.role,
    )
    predictor = DeformableObjectSam2VideoPredictor(
        args.sam2_repository,
        args.checkpoint,
        device=args.device,
    )
    try:
        mask_panel = propagate_development_masks(
            authorization=authorization,
            aligned_object_root=args.aligned_root / args.object_id,
            reference_annotation_root=args.reference_annotation_root,
            reference_panel_path=args.reference_panel,
            output_annotation_root=args.annotation_root,
            predictor=predictor,
        )
    finally:
        predictor.close()
    staging = stage_development_processing_episode(
        authorization=authorization,
        aligned_object_root=args.aligned_root / args.object_id,
        annotation_root=args.annotation_root,
        processing_root=args.processing_root,
        overwrite=args.overwrite_stage,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "object_id": args.object_id,
                "episode_id": args.episode_id,
                "camera_count": mask_panel["camera_count"],
                "frame_count": mask_panel["frame_count"],
                "mask_panel_result_sha256": mask_panel["result_sha256"],
                "staging_result_sha256": staging["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
