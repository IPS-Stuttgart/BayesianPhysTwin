#!/usr/bin/env python3
"""Seal one native PhysTwin interaction for target-excluded MatPhys replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.matphys_dino_features import CausalDinoNodeExtractor
from bayesian_phystwin.matphys_fold_ensemble_v1 import causal_frame_indices
from bayesian_phystwin.matphys_graph_parts import graph_semantic_parts
from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)

SCHEMA = "bayesian-phystwin.matphys-native-phystwin-case"
VERSION = 1
PART_POLICY = "causal-dino-graph-voronoi-cloth-v1"
MATERIAL_CLASS_COUNT = 10
MATPHYS_CLOTH_CLASS_INDEX = 2


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
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


def _file_record(path: str | Path) -> dict[str, object]:
    source = _ordinary_file(path, name="input")
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "byte_count": source.stat().st_size,
    }


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _case_arrays(
    final_data: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(final_data["object_points"], dtype=np.float32)
    surface = np.asarray(final_data["surface_points"], dtype=np.float32)
    interior = np.asarray(final_data["interior_points"], dtype=np.float32)
    controller = np.asarray(final_data["controller_points"], dtype=np.float32)
    if observed.ndim != 3 or observed.shape[-1] != 3 or len(observed) < 3:
        raise ValueError("object_points must have shape (T,N,3) with T>=3")
    if surface.ndim != 2 or surface.shape[1:] != (3,):
        raise ValueError("surface_points must have shape (S,3)")
    if interior.ndim != 2 or interior.shape[1:] != (3,):
        raise ValueError("interior_points must have shape (I,3)")
    if (
        controller.ndim != 3
        or controller.shape[0] != observed.shape[0]
        or controller.shape[-1] != 3
    ):
        raise ValueError("controller_points must align with object frames")
    structure = np.concatenate((observed[0], surface, interior), axis=0)
    for name, value in {
        "object points": observed,
        "structure points": structure,
        "controller points": controller,
    }.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
    return observed, surface, interior, controller, structure


def _checkpoint_spring_fields(
    checkpoint: dict[str, object],
    *,
    total_spring_count: int,
    object_spring_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = checkpoint.get("spring_Y")
    if raw is None:
        raise ValueError("released checkpoint omits spring_Y")
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    complete = np.asarray(raw, dtype=np.float32).reshape(-1)
    if complete.shape != (total_spring_count,):
        raise ValueError("released checkpoint and reconstructed graph disagree")
    checkpoint_count = checkpoint.get("num_object_springs")
    if hasattr(checkpoint_count, "item"):
        checkpoint_count = checkpoint_count.item()
    if int(checkpoint_count) != object_spring_count:
        raise ValueError("released checkpoint object-spring count changed")
    if not np.all(np.isfinite(complete)) or not np.all(complete > 0.0):
        raise ValueError("released checkpoint spring field must be finite and positive")
    return (
        np.ascontiguousarray(complete[:object_spring_count]),
        np.ascontiguousarray(complete),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--optimal-params", type=Path, required=True)
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--prefix-video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part-count", type=int, default=5)
    parser.add_argument("--semantic-edge-weight", type=float, default=4.0)
    parser.add_argument("--dino-keyframes", type=int, default=4)
    parser.add_argument("--dino-model", default="dinov2_vitl14_reg")
    parser.add_argument("--dino-image-size", type=int, default=518)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    case_dir = args.case_dir.resolve(strict=True)
    final_data_path = _ordinary_file(case_dir / "final_data.pkl", name="final data")
    split_path = _ordinary_file(args.split or case_dir / "split.json", name="split")
    video_path = _ordinary_file(
        args.prefix_video or case_dir / "color" / "0.mp4", name="prefix video"
    )
    checkpoint_path = _ordinary_file(args.checkpoint, name="checkpoint")
    optimal_path = _ordinary_file(args.optimal_params, name="optimal parameters")
    baseline_path = _ordinary_file(args.baseline_trajectory, name="baseline trajectory")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    final_data = _load_pickle(final_data_path)
    optimal = _load_pickle(optimal_path)
    observed, surface, interior, controller, structure = _case_arrays(final_data)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train = tuple(int(value) for value in split["train"])
    test = tuple(int(value) for value in split["test"])
    if train[0] != 0 or train[1] != test[0] or test[1] != len(observed):
        raise ValueError("released split must be one contiguous causal prefix/future")
    baseline = np.asarray(_load_pickle(baseline_path), dtype=np.float32)
    if baseline.shape != (len(observed), len(structure), 3):
        raise ValueError("released baseline and reconstructed structure disagree")

    graph = build_phystwin_spring_graph(
        structure,
        controller[0],
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    object_spring_y, complete_spring_y = _checkpoint_spring_fields(
        checkpoint,
        total_spring_count=len(graph.springs),
        object_spring_count=graph.num_object_springs,
    )
    object_edges = np.asarray(graph.springs[: graph.num_object_springs], dtype=np.int64)
    graph_path = output / "episode_graph.npz"
    np.savez_compressed(
        graph_path,
        vertices=np.asarray(structure, dtype=np.float32),
        springs=object_edges,
        rest_lengths=np.asarray(
            graph.rest_lengths[: graph.num_object_springs], dtype=np.float32
        ),
        masses=np.asarray(graph.masses[: len(structure)], dtype=np.float32),
    )
    object_field_path = output / "incumbent_object_spring_y_pa.npy"
    complete_field_path = output / "incumbent_complete_spring_y_pa.npy"
    np.save(object_field_path, object_spring_y, allow_pickle=False)
    np.save(complete_field_path, complete_spring_y, allow_pickle=False)

    frame_ids = causal_frame_indices(train[1], frame_count=int(args.dino_keyframes))
    extractor = CausalDinoNodeExtractor(
        model_name=str(args.dino_model),
        image_size=int(args.dino_image_size),
        device=str(args.device),
    )
    node_features, contributor_count, dino_provenance = extractor.extract_case(
        case_dir, frame_ids.tolist()
    )
    partition = graph_semantic_parts(
        structure,
        object_edges,
        node_features,
        part_count=int(args.part_count),
        semantic_edge_weight=float(args.semantic_edge_weight),
    )
    material = np.zeros((int(args.part_count), MATERIAL_CLASS_COUNT), dtype=np.float32)
    material[:, MATPHYS_CLOTH_CLASS_INDEX] = 1.0
    part_path = output / "part_artifact.npz"
    np.savez_compressed(
        part_path,
        point_part=partition.assignments.astype(np.int64),
        part_features=partition.part_features.astype(np.float32),
        material_distribution=material,
        contributor_count=contributor_count.astype(np.int32),
        seed_node_indices=partition.seeds.astype(np.int64),
    )

    outputs = {
        "episode_graph": _file_record(graph_path),
        "incumbent_object_spring_field": _file_record(object_field_path),
        "incumbent_complete_spring_field": _file_record(complete_field_path),
        "part_artifact": _file_record(part_path),
    }
    identity = {
        "schema": SCHEMA,
        "schema_version": VERSION,
        "case_id": str(args.case_id),
        "split": {"prefix": list(train), "future": list(test)},
        "graph": {
            "structure_node_count": len(structure),
            "original_node_count": observed.shape[1],
            "surface_node_count": len(surface),
            "interior_node_count": len(interior),
            "object_spring_count": graph.num_object_springs,
            "controller_spring_count": len(graph.springs) - graph.num_object_springs,
        },
        "part_policy": {
            "name": PART_POLICY,
            "part_count": int(args.part_count),
            "semantic_edge_weight": float(args.semantic_edge_weight),
            "material_class": "cloth",
            "material_class_index": MATPHYS_CLOTH_CLASS_INDEX,
            "direct_observed_node_count": int(np.sum(contributor_count > 0)),
            "dino": dino_provenance,
        },
        "inputs": {
            "final_data": _file_record(final_data_path),
            "split": _file_record(split_path),
            "checkpoint": _file_record(checkpoint_path),
            "optimal_params": _file_record(optimal_path),
            "baseline_trajectory": _file_record(baseline_path),
            "prefix_video": _file_record(video_path),
        },
        "outputs": outputs,
        "information_boundary": {
            "prefix_end_frame_exclusive": train[1],
            "dino_frame_indices": frame_ids.tolist(),
            "future_rgb_decoded": False,
            "future_observations_used_by_matphys": False,
            "future_outcomes_opened_by_preparer": False,
            "source_only": True,
        },
        "claim_boundary": (
            "This is a source-only native PhysTwin carrier. The released fitted "
            "checkpoint and baseline are unchanged; MatPhys receives only the "
            "registered causal RGB prefix and graph/material inputs."
        ),
    }
    manifest = {**identity, "case_input_id": content_id(identity)}
    write_atomic_json(
        manifest,
        output / "matphys_native_phystwin_case.json",
        overwrite=False,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
