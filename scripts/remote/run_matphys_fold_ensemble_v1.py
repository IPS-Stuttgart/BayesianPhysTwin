#!/usr/bin/env python3
"""Run target-excluded MatPhys folds on one causal video prefix and graph."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_MATPHYS_PYDEPS = os.environ.get("MATPHYS_PYDEPS")
if _MATPHYS_PYDEPS:
    sys.path.append(_MATPHYS_PYDEPS)

from run_matphys_causal import (  # noqa: E402
    _configure_matphys_imports,
    _install_torchvision_nms_stub,
    _source_commit,
)

from bayesian_phystwin._portable_contracts import (  # noqa: E402
    content_id,
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.matphys_fold_ensemble_v1 import (  # noqa: E402
    MATPHYS_CAUSAL_VIDEO_CONTRACT,
    MATPHYS_FOLD_ENSEMBLE_CLAIM_BOUNDARY,
    MATPHYS_FOLD_PARAMETERIZATION,
    MATPHYS_GRAPH_FEATURE_CONTRACT,
    MATPHYS_PART_MODEL_CONTRACT,
    apply_bounded_spring_residual,
    assert_target_excluded,
    causal_frame_indices,
    install_matphys_warp_warning_compatibility,
    matphys_graph_features,
    validate_matphys_fold_ensemble_source,
)
from bayesian_phystwin.matphys_graph_parts import (  # noqa: E402
    graph_semantic_parts,
)
from bayesian_phystwin.matphys_part_model import (  # noqa: E402
    install_part_aware_simple_model,
)

PREDICTION_SCHEMA = "bayesian-phystwin.matphys-fold-ensemble-prediction"
PREDICTION_VERSION = 1
ARRAY_FILENAME = "matphys_fold_ensemble_springs.npz"
MANIFEST_FILENAME = "matphys_fold_ensemble_prediction.json"
ZERO_FEATURE_CONTROL = "geometry-voronoi-zero-part-feature-control-v1"
REGISTERED_PART_FEATURES = "registered-1024d-part-features-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matphys-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--target-object-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--episode-graph", required=True)
    parser.add_argument("--prefix-video", required=True)
    parser.add_argument("--evidence-end-frame-exclusive", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--part-artifact")
    parser.add_argument("--part-count", type=int, default=5)
    parser.add_argument("--incumbent-spring-y-pa", type=float, default=10000.0)
    parser.add_argument("--proposal-strength", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    if (
        not source.is_file()
        or source.is_symlink()
        or any(parent.is_symlink() for parent in source.parents)
    ):
        raise ValueError(f"{name} must be an ordinary non-symlink file")
    return source.resolve(strict=True)


def _decode_video_prefix(
    path: Path,
    *,
    evidence_end_frame_exclusive: int,
    frame_count: int,
    image_size: int,
    device: Any,
):
    import cv2
    import torch
    from PIL import Image
    from torchvision import transforms

    indices = causal_frame_indices(
        evidence_end_frame_exclusive,
        frame_count=frame_count,
    )
    selected = set(int(value) for value in indices)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot decode prefix video {path}")
    frames: dict[int, Any] = {}
    frame_index = 0
    try:
        while frame_index < evidence_end_frame_exclusive:
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index in selected:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames[frame_index] = Image.fromarray(rgb)
            frame_index += 1
    finally:
        capture.release()
    if tuple(sorted(frames)) != tuple(int(value) for value in indices):
        raise ValueError("prefix video ended before every causal frame was decoded")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    tensor = torch.stack([transform(frames[int(index)]) for index in indices])
    return tensor.unsqueeze(0).to(device), indices


def _part_inputs(
    *,
    graph_path: Path,
    points_m: np.ndarray,
    edges: np.ndarray,
    part_artifact_path: Path | None,
    part_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, object]]:
    if part_artifact_path is not None:
        with np.load(part_artifact_path, allow_pickle=False) as archive:
            required = {"point_part", "part_features", "material_distribution"}
            if not required.issubset(archive.files):
                raise ValueError("part artifact omits registered arrays")
            point_part = np.asarray(archive["point_part"], dtype=np.int64)
            part_features = np.asarray(archive["part_features"], dtype=np.float32)
            material = np.asarray(
                archive["material_distribution"], dtype=np.float32
            )
        policy = REGISTERED_PART_FEATURES
        identity = {
            "path": str(part_artifact_path),
            "sha256": _sha256(part_artifact_path),
        }
    else:
        if part_count < 1:
            raise ValueError("part count must be positive")
        centered = points_m - np.mean(points_m, axis=0, keepdims=True)
        scale = np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-6)
        node_features = np.concatenate(
            (centered / scale, np.ones((len(points_m), 1))), axis=1
        )
        partition = graph_semantic_parts(
            points_m,
            edges,
            node_features,
            part_count=part_count,
        )
        point_part = partition.assignments
        part_features = np.zeros((part_count, 1024), dtype=np.float32)
        material = np.full((part_count, 10), 0.1, dtype=np.float32)
        policy = ZERO_FEATURE_CONTROL
        identity = {
            "path": None,
            "sha256": None,
            "source_graph": {"path": str(graph_path), "sha256": _sha256(graph_path)},
        }
    if point_part.shape != (len(points_m),):
        raise ValueError("point_part does not cover the graph")
    part_total = int(np.max(point_part)) + 1
    if part_features.shape != (part_total, 1024):
        raise ValueError("part_features must have shape (K,1024)")
    if material.shape != (part_total, 10):
        raise ValueError("material_distribution must have shape (K,10)")
    if not np.all(np.isfinite(part_features)):
        raise ValueError("part features are non-finite")
    if not np.all(np.isfinite(material)) or not np.allclose(
        np.sum(material, axis=1), 1.0, atol=1e-5, rtol=0.0
    ):
        raise ValueError("material rows must be finite probability distributions")
    return point_part, part_features, material, policy, identity


def _checkpoint_args(checkpoint: dict[str, object]) -> dict[str, object]:
    raw = checkpoint.get("args")
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "__dict__"):
        return vars(raw)
    raise ValueError("MatPhys checkpoint omits its model arguments")


def main() -> None:
    args = _parse_args()
    matphys_root = Path(args.matphys_root).resolve()
    source_path = _ordinary_file(args.source_manifest, name="source manifest")
    graph_path = _ordinary_file(args.episode_graph, name="episode graph")
    video_path = _ordinary_file(args.prefix_video, name="prefix video")
    part_path = (
        _ordinary_file(args.part_artifact, name="part artifact")
        if args.part_artifact
        else None
    )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = validate_matphys_fold_ensemble_source(
        load_strict_json_object(source_path, label="MatPhys fold source"),
        verify_files=True,
    )
    assert_target_excluded(source, target_object_id=args.target_object_id)
    if _source_commit(matphys_root) != source["source_revision"]:
        raise ValueError("MatPhys checkout revision differs from the source lock")

    with np.load(graph_path, allow_pickle=False) as graph:
        required = {"vertices", "springs"}
        if not required.issubset(graph.files):
            raise ValueError("episode graph omits vertices or springs")
        points_m = np.asarray(graph["vertices"], dtype=np.float32)
        edges = np.asarray(graph["springs"], dtype=np.int64)
    point_part, part_features, material, part_policy, part_identity = _part_inputs(
        graph_path=graph_path,
        points_m=points_m,
        edges=edges,
        part_artifact_path=part_path,
        part_count=int(args.part_count),
    )
    features = matphys_graph_features(points_m, edges, point_part)
    incumbent = np.full(
        len(edges),
        float(args.incumbent_spring_y_pa),
        dtype=np.float32,
    )
    if not np.isfinite(incumbent[0]) or incumbent[0] <= 0.0:
        raise ValueError("incumbent spring stiffness must be finite and positive")

    _configure_matphys_imports(matphys_root)
    _install_torchvision_nms_stub()
    import torch

    warp_warning_compatibility_applied = (
        install_matphys_warp_warning_compatibility()
    )
    import train_model_video_material_simple as training

    install_part_aware_simple_model(
        training,
        part_feature_dim=1024,
        part_feature_scale=1.0,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    first_checkpoint = torch.load(
        source["members"][0]["checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    first_model_args = _checkpoint_args(first_checkpoint)
    video, frame_indices = _decode_video_prefix(
        video_path,
        evidence_end_frame_exclusive=int(args.evidence_end_frame_exclusive),
        frame_count=int(first_model_args.get("num_video_frames", 16)),
        image_size=int(first_model_args.get("videomae_image_size", 224)),
        device=device,
    )
    del first_checkpoint

    z_geo = torch.as_tensor(features.edge_features, dtype=torch.float32, device=device)
    edge_part = torch.as_tensor(
        features.edge_part_index, dtype=torch.long, device=device
    )
    scene = torch.as_tensor(features.scene_features, dtype=torch.float32, device=device)
    part_tensor = torch.as_tensor(part_features, dtype=torch.float32, device=device)
    material_tensor = torch.as_tensor(material, dtype=torch.float32, device=device)
    empty_rest = torch.empty((0, 1), dtype=torch.float32, device=device)
    empty_part = torch.empty(0, dtype=torch.long, device=device)

    member_raw = []
    member_springs = []
    member_records = []
    for member in source["members"]:
        checkpoint_path = Path(member["checkpoint"]["path"])
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model_args = _checkpoint_args(checkpoint)
        compared = (
            "videomae_model",
            "videomae_image_size",
            "num_video_frames",
            "d_motion",
            "mat_codebook_dim",
            "hidden_dim",
            "num_materials",
            "logk_residual_scale",
            "logk_soft_clamp",
        )
        if any(model_args.get(key) != first_model_args.get(key) for key in compared):
            raise ValueError("fold checkpoints disagree on model architecture")
        model = training.SimpleVideoMaterialPhysicsModel(
            videomae_model=str(model_args["videomae_model"]),
            d_motion=int(model_args["d_motion"]),
            d_mat=int(model_args["mat_codebook_dim"]),
            hidden_dim=int(model_args["hidden_dim"]),
            num_materials=int(model_args["num_materials"]),
            logk_residual_scale=float(model_args.get("logk_residual_scale", 1.0)),
            logk_soft_clamp=float(model_args.get("logk_soft_clamp", 0.25)),
            part_feature_dim=1024,
            part_feature_scale=1.0,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            output = model(
                pixel_values=video,
                z_geo=z_geo,
                material_dist=material_tensor,
                edge_part_idx=edge_part,
                part_features=part_tensor,
                geo_stats=scene,
                ctrl_rest_length=empty_rest,
                ctrl_part_idx=empty_part,
            )
        raw = np.asarray(
            output["log_k_raw"].detach().cpu(), dtype=np.float32
        ).reshape(-1)
        if raw.shape != incumbent.shape:
            raise ValueError("fold output does not match the canonical graph edge order")
        spring = apply_bounded_spring_residual(
            incumbent,
            raw,
            proposal_strength=float(args.proposal_strength),
        )
        member_raw.append(raw)
        member_springs.append(spring)
        ratio = spring.astype(np.float64) / incumbent.astype(np.float64)
        member_records.append(
            {
                "fold_index": int(member["fold_index"]),
                "held_out_object_id": str(member["held_out_object_id"]),
                "checkpoint_sha256": str(member["checkpoint"]["sha256"]),
                "spring_ratio_mean": float(np.mean(ratio)),
                "spring_ratio_std": float(np.std(ratio)),
                "spring_ratio_minimum": float(np.min(ratio)),
                "spring_ratio_maximum": float(np.max(ratio)),
            }
        )
        del output, model, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_array = np.stack(member_raw).astype(np.float32)
    spring_array = np.stack(member_springs).astype(np.float32)
    array_path = output_dir / ARRAY_FILENAME
    np.savez_compressed(
        array_path,
        incumbent_spring_y_pa=incumbent,
        member_raw_log_residual=raw_array,
        member_spring_y_pa=spring_array,
        member_mean_spring_y_pa=np.mean(spring_array, axis=0).astype(np.float32),
        member_log_ratio_std=np.std(
            np.log(spring_array.astype(np.float64) / incumbent[None]), axis=0
        ).astype(np.float32),
        graph_points_m=points_m,
        graph_edges=edges,
        point_part=point_part,
        edge_part_index=features.edge_part_index,
        edge_features=features.edge_features,
        scene_features=features.scene_features,
        causal_frame_indices=frame_indices,
        part_features=part_features,
        material_distribution=material,
    )
    mean_member_std = float(
        np.mean(
            np.std(
                np.log(spring_array.astype(np.float64) / incumbent[None]),
                axis=0,
            )
        )
    )
    identity = {
        "schema": PREDICTION_SCHEMA,
        "schema_version": PREDICTION_VERSION,
        "case_id": str(args.case_id),
        "target_object_id": str(args.target_object_id),
        "source_ensemble_id": source["ensemble_id"],
        "source_revision": source["source_revision"],
        "parameterization": MATPHYS_FOLD_PARAMETERIZATION,
        "part_model_contract": MATPHYS_PART_MODEL_CONTRACT,
        "graph_feature_contract": MATPHYS_GRAPH_FEATURE_CONTRACT,
        "causal_video_contract": MATPHYS_CAUSAL_VIDEO_CONTRACT,
        "proposal_strength": float(args.proposal_strength),
        "evidence_end_frame_exclusive": int(args.evidence_end_frame_exclusive),
        "causal_frame_indices": frame_indices.tolist(),
        "part_feature_policy": part_policy,
        "member_count": len(member_records),
        "members": member_records,
        "mean_edge_epistemic_log_ratio_std": mean_member_std,
        "runtime_compatibility": {
            "warp_private_warning_alias_restored": (
                warp_warning_compatibility_applied
            )
        },
        "inputs": {
            "source_manifest": {"path": str(source_path), "sha256": _sha256(source_path)},
            "episode_graph": {"path": str(graph_path), "sha256": _sha256(graph_path)},
            "prefix_video": {"path": str(video_path), "sha256": _sha256(video_path)},
            "part_artifact": part_identity,
        },
        "graph_sha256": features.graph_sha256,
        "output": {"path": str(array_path), "sha256": _sha256(array_path)},
        "information_boundary": {
            "target_future_observations_used": False,
            "target_future_outcomes_opened": False,
            "target_object_used_for_checkpoint_training": False,
            "known_future_action_used_by_material_proposal": False,
            "official_warp_replay_completed": False,
            "calibration_claim_authorized": False,
        },
        "claim_boundary": MATPHYS_FOLD_ENSEMBLE_CLAIM_BOUNDARY,
    }
    manifest = {**identity, "prediction_id": content_id(identity)}
    write_atomic_json(manifest, output_dir / MANIFEST_FILENAME, overwrite=False)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
