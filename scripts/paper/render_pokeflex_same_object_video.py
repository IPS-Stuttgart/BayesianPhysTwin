#!/usr/bin/env python3
"""Render one clear video for the frozen PokeFlex same-object replication."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


_REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bayesian_phystwin.pokeflex_same_object_reporting import (  # noqa: E402
    load_json_object,
    sha256_file,
    validate_bounded_result,
    write_json,
)
from pokeflex_same_object_video_capture import (  # noqa: E402
    capture_frozen_predictions,
    choose_take,
    take_decisions,
)
from pokeflex_same_object_video_encode import write_video  # noqa: E402
from pokeflex_same_object_video_plot import (  # noqa: E402
    canonical_projection,
    render_rgb_frame,
    video_limits,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prospective-result", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--take-id")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--repeats-per-state", type=int, default=2)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--prospective-protocol",
        type=Path,
        default=(
            _REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_independent_depth_regret_guard_prospective_v1.json"
        ),
    )
    parser.add_argument(
        "--independent-depth-protocol",
        type=Path,
        default=(
            _REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_independent_depth_source_validation_v2.json"
        ),
    )
    parser.add_argument(
        "--registration-protocol",
        type=Path,
        default=(
            _REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_bayesian_registration_v1.json"
        ),
    )
    args = parser.parse_args()
    _require(args.fps >= 1, "fps must be positive")
    _require(args.repeats_per_state >= 1, "frame repeat must be positive")

    result_path = args.prospective_result.resolve()
    result = load_json_object(result_path)
    bounded = validate_bounded_result(result)
    take_id = choose_take(result, args.take_id)
    take_root = (args.dataset_root.resolve() / take_id).resolve()
    _require(take_root.is_dir(), f"PokeFlex take directory is missing: {take_root}")

    prospective_protocol = load_json_object(args.prospective_protocol.resolve())
    candidate_runner = (
        _REPOSITORY_ROOT
        / "scripts"
        / "remote"
        / "run_pokeflex_checkpoint_registration_independent_depth.py"
    )
    expected_runner = prospective_protocol["payload"]["parent_method"][
        "candidate_runner_sha256"
    ]
    _require(
        sha256_file(candidate_runner) == expected_runner,
        "frozen PokeFlex candidate runner changed",
    )

    captured, reproduction = capture_frozen_predictions(
        take_root=take_root,
        prospective_result=result,
        take_id=take_id,
        independent_depth_protocol_path=args.independent_depth_protocol.resolve(),
        registration_protocol_path=args.registration_protocol.resolve(),
        upstream_checkout=args.upstream_checkout.resolve(),
        checkpoint_root=args.checkpoint_root.resolve(),
    )
    decisions, _ = take_decisions(result, take_id)
    target_frames = [int(value["target_frame"]) for value in decisions]
    first_target = captured[target_frames[0]]["target_vertices_m"]
    center, basis = canonical_projection(first_target)
    limits, distance_limit = video_limits(captured, center, basis)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    slug = take_id.lower().replace("_", "-")
    video_path = output_root / f"pokeflex_same_object_{slug}.mp4"
    poster_path = output_root / "pokeflex_same_object_video_poster.png"
    metadata_path = output_root / "pokeflex_same_object_video_metadata.json"

    def rendered_frames() -> Iterable[np.ndarray]:
        from PIL import Image

        middle = len(target_frames) // 2
        for index, target_frame in enumerate(target_frames):
            current = captured[target_frame]
            frame = render_rgb_frame(
                take_id=take_id,
                frame_index=index,
                frame_count=len(target_frames),
                target_frame=target_frame,
                target_vertices=current["target_vertices_m"],
                baseline_vertices=current["baseline_vertices_m"],
                guarded_vertices=current["guarded_vertices_m"],
                center=center,
                basis=basis,
                limits=limits,
                distance_limit_mm=distance_limit,
                decisions=decisions,
            )
            if index == middle:
                Image.fromarray(frame).save(poster_path)
            yield frame

    video_metadata = write_video(
        video_path,
        rendered_frames(),
        fps=args.fps,
        repeats_per_state=args.repeats_per_state,
    )
    take_result = next(
        value for value in bounded["takes"] if value["take_id"] == take_id
    )
    metadata = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexSameObjectPaperVideoManifestV1",
        "analysis_role": (
            "post-outcome visual exemplar selected as the prospective take with "
            "the largest take-level improvement; not an additional test unit"
        ),
        "take_id": take_id,
        "take_result": take_result,
        "prospective_result": {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        },
        "prospective_protocol_sha256": prospective_protocol["protocol_sha256"],
        "candidate_runner_sha256": expected_runner,
        "reproduction": reproduction,
        "projection": {
            "center_m": center.tolist(),
            "basis": basis.tolist(),
            "limits_m": list(limits),
            "distance_color_limit_mm": distance_limit,
        },
        "video": {
            **video_metadata,
            "path": video_path.name,
            "sha256": sha256_file(video_path),
        },
        "poster": {
            "path": poster_path.name,
            "sha256": sha256_file(poster_path),
        },
        "claim": bounded["claim"],
        "excluded_claims": bounded["excluded_claims"],
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
