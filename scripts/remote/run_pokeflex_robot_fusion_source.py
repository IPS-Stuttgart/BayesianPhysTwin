#!/usr/bin/env python3
"""Run the locked PokeFlex force/tool checkpoint source-fusion study."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_registration_protocol import (  # noqa: E402
    load_pokeflex_registration_protocol,
)
from bayesian_phystwin.pokeflex_released_checkpoint import (  # noqa: E402
    PokeFlexReleasedCheckpoint,
)
from bayesian_phystwin.pokeflex_released_robot_checkpoint import (  # noqa: E402
    PokeFlexReleasedRobotCheckpoint,
)
from bayesian_phystwin.pokeflex_robot_fusion import (  # noqa: E402
    PokeFlexRobotFusionConfig,
    pokeflex_robot_fusion_candidates,
    pokeflex_robot_fusion_features,
)
from bayesian_phystwin.pokeflex_robot_fusion_protocol import (  # noqa: E402
    load_pokeflex_robot_fusion_source_protocol,
)
from run_pokeflex_bayesian_registration_smoke import (  # noqa: E402
    _cd_ul1_mm,
    _load_mesh,
    _surface_sample,
    _template_frame,
    _view_points,
)
from run_pokeflex_checkpoint_registration_independent_depth import (  # noqa: E402
    _load_official_template,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_CD_UL1_mm": float(np.mean(array)),
        "median_CD_UL1_mm": float(np.median(array)),
        "p90_CD_UL1_mm": float(np.quantile(array, 0.9)),
    }


def _checkpoint_hashes(
    root: Path,
    filenames: tuple[str, ...],
) -> dict[str, str]:
    return {filename: _sha256(root / filename) for filename in filenames}


def run_source_take(
    take_root: Path,
    protocol_path: Path,
    parent_protocol_path: Path,
    upstream_checkout: Path,
    pointcloud_checkpoint_root: Path,
    robot_checkpoint_root: Path,
    *,
    device: str | None,
    maximum_frame: int | None,
) -> dict[str, object]:
    """Evaluate one already-open source take without future method inputs."""

    protocol = load_pokeflex_robot_fusion_source_protocol(protocol_path)
    parent = load_pokeflex_registration_protocol(parent_protocol_path)
    expected_parent = protocol["payload"]["parent_protocol"]["protocol_sha256"]
    if parent["protocol_sha256"] != expected_parent:
        raise ValueError("robot-fusion parent protocol changed")

    object_name, separator, take_number = take_root.name.rpartition("_T")
    if (
        not separator
        or object_name not in protocol["development_objects"]
        or f"T{take_number}" not in protocol["source_takes"]
    ):
        raise ValueError(f"take is outside the locked source cohort: {take_root.name}")

    robot_hashes = _checkpoint_hashes(
        robot_checkpoint_root,
        ("attention_model.pth", "decoder.pth", "force_encoder.pth"),
    )
    expected_robot_hashes = {
        name: value["sha256"]
        for name, value in protocol["payload"]["upstream"][
            "released_robot_checkpoint"
        ].items()
    }
    if robot_hashes != expected_robot_hashes:
        raise ValueError("released robot checkpoint hashes changed")
    pointcloud_hashes = _checkpoint_hashes(
        pointcloud_checkpoint_root,
        ("attention_model.pth", "decoder.pth", "pointcloud_encoder.pth"),
    )
    expected_pointcloud_hashes = {
        name: value["sha256"]
        for name, value in parent["payload"]["upstream"][
            "released_kinect_checkpoint"
        ].items()
    }
    if pointcloud_hashes != expected_pointcloud_hashes:
        raise ValueError("released point-cloud checkpoint hashes changed")

    robot_path = take_root / "robot_data.json"
    robot_records = json.loads(robot_path.read_text(encoding="utf-8"))
    robot_by_frame = {int(record["frame"]): record for record in robot_records}
    active = [
        frame
        for frame, record in sorted(robot_by_frame.items())
        if float(record["forces"][1]) > 3.0
    ]
    template_frame = _template_frame(active)
    template_path = take_root / "meshes" / f"mesh-f{template_frame:05d}.obj"
    template_vertices, template_faces, template_preprocessing = (
        _load_official_template(template_path)
    )
    frame_limit = maximum_frame or max(robot_by_frame)
    valid_targets = sorted(frame for frame in active if 6 <= frame <= frame_limit)
    if not valid_targets:
        raise ValueError("source interval contains no causal target frames")

    pointcloud_checkpoint = PokeFlexReleasedCheckpoint.load(
        template_vertices,
        upstream_checkout=upstream_checkout,
        checkpoint_root=pointcloud_checkpoint_root,
        device=device,
    )
    robot_checkpoint = PokeFlexReleasedRobotCheckpoint.load(
        template_vertices,
        upstream_checkout=upstream_checkout,
        checkpoint_root=robot_checkpoint_root,
        device=device,
    )
    if pointcloud_checkpoint.device != robot_checkpoint.device:
        raise ValueError("released checkpoints were loaded on different devices")

    encoded_by_frame = {}
    preprocessing_by_frame = {}
    for frame in range(1, frame_limit):
        views = tuple(
            _view_points(take_root, frame, camera, template_vertices)
            for camera in (0, 1)
        )
        encoded, preprocessing = pointcloud_checkpoint.encode_frame(views)
        encoded_by_frame[frame] = encoded
        preprocessing_by_frame[frame] = preprocessing

    scales = tuple(
        map(float, protocol["payload"]["candidate_lock"]["scales"])
    )
    fusion_config = PokeFlexRobotFusionConfig(scales=scales)
    baseline_errors: list[float] = []
    robot_errors: list[float] = []
    candidate_errors = {
        f"robot_convex_scale_{scale:g}": [] for scale in scales
    }
    target_records = []
    sample_count = int(
        parent["payload"]["evaluation"]["sampling"]["surface_points"]
    )
    base_seed = int(parent["payload"]["evaluation"]["sampling"]["seed"])
    started = time.monotonic()

    for target_frame in valid_targets:
        history_frames = tuple(range(target_frame - 5, target_frame))
        baseline = pointcloud_checkpoint.predict_from_encoded_history(
            [encoded_by_frame[frame] for frame in history_frames],
            [preprocessing_by_frame[frame] for frame in history_frames],
        ).vertices_m
        history_records = [robot_by_frame[frame] for frame in history_frames]
        robot = robot_checkpoint.predict_from_records(history_records).vertices_m
        candidates = pokeflex_robot_fusion_candidates(
            baseline,
            robot,
            config=fusion_config,
        )
        features = pokeflex_robot_fusion_features(
            baseline,
            robot,
            template_vertices,
            history_records,
        )

        # Target geometry is loaded only after every candidate and feature is frozen.
        target_mesh = _load_mesh(
            take_root / "meshes" / f"mesh-f{target_frame:05d}.obj"
        )
        target_sample = _surface_sample(
            np.asarray(target_mesh.vertices, dtype=np.float64) / 1000.0,
            np.asarray(target_mesh.faces, dtype=np.int64),
            sample_count,
            base_seed + target_frame,
        )

        def score(vertices_m: np.ndarray) -> float:
            sample = _surface_sample(
                vertices_m,
                template_faces,
                sample_count,
                base_seed + target_frame,
            )
            return _cd_ul1_mm(sample, target_sample)

        baseline_error = score(baseline)
        robot_error = score(robot)
        baseline_errors.append(baseline_error)
        robot_errors.append(robot_error)
        frame_candidates = {}
        for name, candidate in candidates.items():
            error = score(candidate)
            candidate_errors[name].append(error)
            frame_candidates[f"{name}_CD_UL1_mm"] = error
        target_records.append(
            {
                "target_frame": target_frame,
                "history_frames": list(history_frames),
                "released_checkpoint_CD_UL1_mm": baseline_error,
                "robot_checkpoint_CD_UL1_mm": robot_error,
                "fusion_features": features,
                **frame_candidates,
            }
        )

    aggregates = {
        "released_checkpoint": _summary(baseline_errors),
        "robot_checkpoint": _summary(robot_errors),
        **{
            name: _summary(values) for name, values in candidate_errors.items()
        },
    }
    best_candidate = min(
        candidate_errors,
        key=lambda name: aggregates[name]["mean_CD_UL1_mm"],
    )
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexRobotFusionSourceTake",
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": protocol["protocol_sha256"],
            "parent_path": str(parent_protocol_path.resolve()),
            "parent_sha256": parent["protocol_sha256"],
        },
        "take": {
            "id": take_root.name,
            "object": object_name,
            "take": f"T{take_number}",
            "robot_data_sha256": _sha256(robot_path),
            "template_frame": template_frame,
            "template_sha256": _sha256(template_path),
            "causal_target_frame_count": len(target_records),
            "maximum_frame": frame_limit,
        },
        "upstream": {
            "checkout": str(upstream_checkout.resolve()),
            "code_commit": protocol["payload"]["upstream"]["code_commit"],
            "pointcloud_checkpoint_sha256": pointcloud_hashes,
            "robot_checkpoint_sha256": robot_hashes,
        },
        "template_preprocessing": template_preprocessing,
        "candidate_config": {
            "family": "robot_convex",
            "scales": list(scales),
            "exact_fallback": "robot_convex_scale_0",
        },
        "causal_boundary": {
            "future_observation_used": False,
            "history": "f-5 through f-1 for both released checkpoints",
            "target_geometry_loaded_after_candidate_construction": True,
            "target_objects_opened": False,
        },
        "device": pointcloud_checkpoint.device,
        "runtime_seconds": float(time.monotonic() - started),
        "aggregates": aggregates,
        "best_source_candidate": best_candidate,
        "targets": target_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--pointcloud-checkpoint-root", type=Path, required=True)
    parser.add_argument("--robot-checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_robot_fusion_source_v1.json"
        ),
    )
    parser.add_argument(
        "--parent-protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_bayesian_registration_v1.json"
        ),
    )
    parser.add_argument("--device")
    parser.add_argument("--maximum-frame", type=int)
    args = parser.parse_args()
    result = run_source_take(
        args.take_root.resolve(),
        args.protocol.resolve(),
        args.parent_protocol.resolve(),
        args.upstream_checkout.resolve(),
        args.pointcloud_checkpoint_root.resolve(),
        args.robot_checkpoint_root.resolve(),
        device=args.device,
        maximum_frame=args.maximum_frame,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing source artifact differs: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "best_source_candidate": result["best_source_candidate"],
                "aggregates": result["aggregates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
