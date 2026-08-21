#!/usr/bin/env python3
"""Build causal frame-zero DINO graph parts for one Deform360 MatPhys run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.deform360_matphys_part_features_v1 import (
    DEFORM360_MATPHYS_PART_CLAIM_BOUNDARY,
    DEFORM360_MATPHYS_PART_CONTRACT,
    DEFORM360_MATPHYS_PART_SCHEMA,
    DEFORM360_MATPHYS_PART_VERSION,
    aggregate_direct_node_features,
    array_sha256,
    build_part_arrays,
    ordinary_file,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    _load_calibration,
    frame_zero_camera_support,
)
from bayesian_phystwin.matphys_dino_features import (
    CausalDinoNodeExtractor,
    sha256_file,
)

ARRAY_FILENAME = "deform360_matphys_part_features.npz"
MANIFEST_FILENAME = "deform360_matphys_part_features.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-graph", required=True)
    parser.add_argument("--prefix-episode", required=True)
    parser.add_argument("--prefix-manifest", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--target-object-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="dinov2_vitl14_reg")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--depth-tolerance-m", type=float, default=0.02)
    parser.add_argument("--part-count", type=int, default=5)
    parser.add_argument("--semantic-edge-weight", type=float, default=4.0)
    parser.add_argument(
        "--camera-id",
        action="append",
        default=None,
        help="Use only this provider camera; repeat for a disjoint panel.",
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_prefix_manifest(path: Path, *, case_id: str, object_id: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("prefix manifest must contain a JSON object")
    if value.get("case") != case_id or value.get("object_id") != object_id:
        raise ValueError("prefix manifest identity differs from the invocation")
    if value.get("staged_frame_zero_frame_count") != 1:
        raise ValueError("prefix manifest must bind exactly one frame-zero image")
    if not isinstance(value.get("staged_prefix_frame_count"), int) or int(
        value["staged_prefix_frame_count"]
    ) < 1:
        raise ValueError("prefix manifest omits a nonempty causal prefix")
    boundary = value.get("information_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("prefix manifest omits its information boundary")
    forbidden = (
        "future_dense_reconstruction_read",
        "future_particle_tracks_read",
        "source_object_frames_after_prefix_read",
        "target_metric_read",
    )
    if any(boundary.get(field) is not False for field in forbidden):
        raise ValueError("prefix manifest crossed a future or target boundary")
    if value.get("stratum") not in {"sheet", "volumetric"}:
        raise ValueError("prefix manifest has an unsupported material stratum")
    return value


def _decode_first_rgb(path: Path):
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open frame-zero video {path}")
    try:
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"cannot decode frame zero from {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _sample_tokens(extractor: CausalDinoNodeExtractor, image, pixels: np.ndarray) -> np.ndarray:
    import torch.nn.functional as functional

    tokens = extractor._patch_tokens(image)  # noqa: SLF001 - shared pinned extractor
    width, height = image.size
    normalized = np.asarray(pixels, dtype=np.float32).copy()
    normalized[:, 0] = normalized[:, 0] / max(width - 1, 1) * 2.0 - 1.0
    normalized[:, 1] = normalized[:, 1] / max(height - 1, 1) * 2.0 - 1.0
    grid = extractor.torch.from_numpy(normalized).to(
        extractor.device,
        dtype=tokens.dtype,
    ).view(1, -1, 1, 2)
    return (
        functional.grid_sample(tokens, grid, mode="bilinear", align_corners=False)
        .squeeze(0)
        .squeeze(-1)
        .transpose(0, 1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def main() -> None:
    args = _parse_args()
    graph_path = ordinary_file(args.episode_graph, name="episode graph")
    prefix_root = Path(args.prefix_episode).resolve(strict=True)
    if not prefix_root.is_dir() or prefix_root.is_symlink():
        raise ValueError("prefix episode must be an ordinary directory")
    prefix_manifest_path = ordinary_file(args.prefix_manifest, name="prefix manifest")
    prefix_manifest = _load_prefix_manifest(
        prefix_manifest_path,
        case_id=args.case_id,
        object_id=args.target_object_id,
    )
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    with np.load(graph_path, allow_pickle=False) as archive:
        if not {"vertices", "springs"}.issubset(archive.files):
            raise ValueError("episode graph omits vertices or springs")
        points = np.asarray(archive["vertices"], dtype=np.float32)
        edges = np.asarray(archive["springs"], dtype=np.int64)
    intrinsics, extrinsics = _load_calibration(prefix_root)
    selected_cameras: tuple[str, ...] | None = None
    if args.camera_id is not None:
        selected_cameras = tuple(sorted(str(value) for value in args.camera_id))
        if (
            len(selected_cameras) < 2
            or len(selected_cameras) != len(set(selected_cameras))
        ):
            raise ValueError("provider camera panel must contain unique cameras")
        if not set(selected_cameras) <= set(intrinsics) or not set(
            selected_cameras
        ) <= set(extrinsics):
            raise ValueError("provider camera panel contains an unavailable camera")
        intrinsics = {camera: intrinsics[camera] for camera in selected_cameras}
        extrinsics = {camera: extrinsics[camera] for camera in selected_cameras}
    cameras, support, projected = frame_zero_camera_support(
        points,
        prefix_root,
        intrinsics,
        extrinsics,
        depth_tolerance_m=float(args.depth_tolerance_m),
    )
    if selected_cameras is None and len(cameras) != int(
        prefix_manifest.get("camera_count", -1)
    ):
        raise ValueError("available calibrated cameras differ from the prefix manifest")
    if selected_cameras is not None and cameras != selected_cameras:
        raise ValueError("resolved provider camera panel changed")

    extractor = CausalDinoNodeExtractor(
        model_name=args.model_name,
        image_size=int(args.image_size),
        device=args.device,
        depth_tolerance_m=float(args.depth_tolerance_m),
        relative_depth_tolerance=0.0,
    )
    sampled_by_camera: dict[str, np.ndarray] = {}
    support_by_camera: dict[str, np.ndarray] = {}
    source_records: list[dict[str, object]] = []
    for camera_index, camera in enumerate(cameras):
        camera_root = prefix_root / camera
        video_path = ordinary_file(camera_root / "undistorted.mp4", name=f"{camera} RGB")
        mask_path = ordinary_file(camera_root / "mask_refined.h5", name=f"{camera} mask")
        depth_path = ordinary_file(
            camera_root / "rendered_depth.h5",
            name=f"{camera} depth",
        )
        image = _decode_first_rgb(video_path)
        sampled_by_camera[camera] = _sample_tokens(
            extractor,
            image,
            projected[camera],
        )
        support_by_camera[camera] = support[:, camera_index]
        source_records.append(
            {
                "camera": camera,
                "frame_index": 0,
                "direct_node_count": int(np.sum(support[:, camera_index])),
                "rgb_prefix_video": {
                    "path": str(video_path),
                    "sha256": sha256_file(video_path),
                },
                "frame_zero_mask": {
                    "path": str(mask_path),
                    "sha256": sha256_file(mask_path),
                },
                "frame_zero_rendered_depth": {
                    "path": str(depth_path),
                    "sha256": sha256_file(depth_path),
                },
            }
        )

    direct_features, contributor_count = aggregate_direct_node_features(
        sampled_by_camera,
        support_by_camera,
    )
    arrays = build_part_arrays(
        points,
        edges,
        direct_features,
        contributor_count,
        stratum=str(prefix_manifest["stratum"]),
        part_count=int(args.part_count),
        semantic_edge_weight=float(args.semantic_edge_weight),
    )
    array_path = output / ARRAY_FILENAME
    np.savez_compressed(
        array_path,
        point_part=arrays.point_part,
        part_features=arrays.part_features,
        material_distribution=arrays.material_distribution,
        node_features=arrays.node_features,
        contributor_count=arrays.contributor_count,
        nearest_direct_node=arrays.nearest_direct_node,
        part_seeds=arrays.partition.seeds,
        part_counts=arrays.partition.part_counts,
    )
    array_record = {
        "path": str(array_path),
        "sha256": _file_sha256(array_path),
        "byte_count": array_path.stat().st_size,
        "arrays": {
            "point_part": array_sha256(arrays.point_part),
            "part_features": array_sha256(arrays.part_features),
            "material_distribution": array_sha256(arrays.material_distribution),
            "node_features": array_sha256(arrays.node_features),
            "contributor_count": array_sha256(arrays.contributor_count),
            "nearest_direct_node": array_sha256(arrays.nearest_direct_node),
        },
    }
    identity: dict[str, object] = {
        "schema": DEFORM360_MATPHYS_PART_SCHEMA,
        "schema_version": DEFORM360_MATPHYS_PART_VERSION,
        "contract": DEFORM360_MATPHYS_PART_CONTRACT,
        "case_id": args.case_id,
        "target_object_id": args.target_object_id,
        "stratum": prefix_manifest["stratum"],
        "part_count": int(args.part_count),
        "semantic_edge_weight": float(args.semantic_edge_weight),
        "depth_tolerance_m": float(args.depth_tolerance_m),
        "camera_count": len(cameras),
        "direct_observed_node_count": int(np.sum(contributor_count > 0)),
        "graph_node_count": len(points),
        "direct_observation_fraction": float(np.mean(contributor_count > 0)),
        "connected_component_count": arrays.partition.connected_component_count,
        "boundary_edge_fraction": arrays.partition.boundary_edge_fraction,
        "dino": {
            "repository": "facebookresearch/dinov2",
            "model_name": extractor.model_name,
            "image_size": extractor.image_size,
            "checkpoints": extractor.checkpoints,
        },
        "inputs": {
            "episode_graph": {
                "path": str(graph_path),
                "sha256": sha256_file(graph_path),
            },
            "prefix_manifest": {
                "path": str(prefix_manifest_path),
                "sha256": sha256_file(prefix_manifest_path),
            },
            "intrinsics": {
                "path": str((prefix_root / "undistorted_intrinsics.npy").resolve()),
                "sha256": sha256_file(prefix_root / "undistorted_intrinsics.npy"),
            },
            "extrinsics": {
                "path": str((prefix_root / "extrinsics.npy").resolve()),
                "sha256": sha256_file(prefix_root / "extrinsics.npy"),
            },
            "cameras": source_records,
        },
        "output": array_record,
        "information_boundary": {
            "rgb_frames_read": [0],
            "mask_frames_read": [0],
            "rendered_depth_frames_read": [0],
            "future_object_observations_read": False,
            "target_metrics_read": False,
            "target_outcomes_read": False,
            "changes_frozen_deform_mean": False,
        },
        "claim_boundary": DEFORM360_MATPHYS_PART_CLAIM_BOUNDARY,
    }
    if selected_cameras is not None:
        identity["camera_selection"] = {
            "mode": "explicit-disjoint-provider-panel",
            "camera_ids": list(selected_cameras),
        }
    manifest = {**identity, "artifact_id": content_id(identity)}
    write_atomic_json(manifest, output / MANIFEST_FILENAME, overwrite=False)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
