#!/usr/bin/env python3
"""Run the locked PokeFlex Bayesian registration development smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_bayesian_registration import (  # noqa: E402
    PokeFlexBayesianRegistrationConfig,
    crop_points_to_template,
    depth_image_to_world_points,
    register_pokeflex_graph_posterior,
)
from bayesian_phystwin.pokeflex_registration_protocol import (  # noqa: E402
    load_pokeflex_registration_protocol,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"expected one triangle mesh: {path}")
    return mesh


def _template_frame(active_frames: list[int]) -> int:
    if not active_frames:
        raise ValueError("take has no active deformation frames")
    if active_frames[0] != 1:
        return 1
    previous = active_frames[0]
    for frame in active_frames[1:]:
        if frame - previous > 5:
            return int((frame + previous) / 2)
        previous = frame
    raise ValueError("upstream template-selection rule found no inactive gap")


def _surface_sample(
    vertices: np.ndarray, faces: np.ndarray, count: int, seed: int
) -> np.ndarray:
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if not np.all(np.isfinite(areas)) or float(np.sum(areas)) <= 0.0:
        raise ValueError("mesh has invalid surface area")
    generator = np.random.default_rng(seed)
    face_indices = generator.choice(len(faces), size=count, p=areas / np.sum(areas))
    first = generator.random(count)
    second = generator.random(count)
    reflected = first + second > 1.0
    first[reflected] = 1.0 - first[reflected]
    second[reflected] = 1.0 - second[reflected]
    chosen = triangles[face_indices]
    return (
        chosen[:, 0]
        + first[:, None] * (chosen[:, 1] - chosen[:, 0])
        + second[:, None] * (chosen[:, 2] - chosen[:, 0])
    )


def _cd_ul1_mm(prediction: np.ndarray, target: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    indices = cKDTree(target).query(prediction, k=1)[1]
    return float(1000.0 * np.mean(np.sum(np.abs(prediction - target[indices]), axis=1)))


def _view_points(
    root: Path,
    frame: int,
    camera: int,
    template_vertices_m: np.ndarray,
) -> np.ndarray:
    camera_root = root / "kinect" / str(camera)
    parameters = json.loads(
        (camera_root / "camera_parameters.json").read_text(encoding="utf-8")
    )
    depth = cv2.imread(
        str(camera_root / "depth" / f"{frame:05d}.png"), cv2.IMREAD_UNCHANGED
    )
    if depth is None:
        raise FileNotFoundError(f"missing depth frame {frame} for camera {camera}")
    points = depth_image_to_world_points(
        depth,
        np.asarray(parameters["depth_intrinsics"], dtype=np.float64),
        np.asarray(parameters["depth_extrinsics"], dtype=np.float64),
    )
    return crop_points_to_template(points, template_vertices_m)


def run_smoke(
    take_root: Path,
    protocol_path: Path,
    *,
    velocity_scales: tuple[float, ...],
    maximum_frame: int | None,
) -> dict[str, object]:
    protocol = load_pokeflex_registration_protocol(protocol_path)
    expected_take = protocol["payload"]["cohort"]["development_smoke_take"]
    if take_root.name != expected_take:
        raise ValueError(f"smoke take differs from protocol: {take_root.name}")

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
    template = _load_mesh(template_path)
    template_vertices = np.asarray(template.vertices, dtype=np.float64) / 1000.0
    template_faces = np.asarray(template.faces, dtype=np.int64)
    frame_limit = maximum_frame or max(robot_by_frame)
    valid_targets = {
        frame for frame in active if 6 <= frame <= frame_limit
    }
    if not valid_targets:
        raise ValueError("smoke interval contains no causal target frames")

    config = PokeFlexBayesianRegistrationConfig()
    state = template_vertices
    predictions: dict[float, list[float]] = {scale: [] for scale in velocity_scales}
    template_errors = []
    oracle_persistence = []
    update_records = []
    target_records = []
    sample_count = int(protocol["payload"]["evaluation"]["sampling"]["surface_points"])
    base_seed = int(protocol["payload"]["evaluation"]["sampling"]["seed"])

    for source_frame in range(1, frame_limit):
        views = tuple(
            _view_points(take_root, source_frame, camera, template_vertices)
            for camera in (0, 1)
        )
        record = robot_by_frame[source_frame]
        action_supported = float(record["forces"][1]) > 3.0
        result = register_pokeflex_graph_posterior(
            state,
            views,
            action_supported=action_supported,
            config=config,
        )
        next_state = result.posterior_vertices_m
        velocity = next_state - state
        state = next_state
        update_records.append(
            {
                "frame": source_frame,
                "accepted": result.accepted,
                "reason": result.reason,
                "action_supported": action_supported,
                "rms_update_m": result.diagnostics.get("rms_update_m", 0.0),
                "associated_points": result.diagnostics.get("association_count", 0),
                "camera_biases_m": result.camera_biases_m.tolist(),
            }
        )

        target_frame = source_frame + 1
        if target_frame not in valid_targets:
            continue
        target_mesh = _load_mesh(
            take_root / "meshes" / f"mesh-f{target_frame:05d}.obj"
        )
        target_sample = _surface_sample(
            np.asarray(target_mesh.vertices, dtype=np.float64) / 1000.0,
            np.asarray(target_mesh.faces, dtype=np.int64),
            sample_count,
            base_seed + target_frame,
        )
        template_sample = _surface_sample(
            template_vertices,
            template_faces,
            sample_count,
            base_seed,
        )
        previous_mesh = _load_mesh(
            take_root / "meshes" / f"mesh-f{source_frame:05d}.obj"
        )
        previous_sample = _surface_sample(
            np.asarray(previous_mesh.vertices, dtype=np.float64) / 1000.0,
            np.asarray(previous_mesh.faces, dtype=np.int64),
            sample_count,
            base_seed + source_frame,
        )
        frame_errors: dict[str, float] = {}
        template_error = _cd_ul1_mm(template_sample, target_sample)
        oracle_error = _cd_ul1_mm(previous_sample, target_sample)
        template_errors.append(template_error)
        oracle_persistence.append(oracle_error)
        for scale in velocity_scales:
            prediction = state + scale * velocity
            prediction_sample = _surface_sample(
                prediction,
                template_faces,
                sample_count,
                base_seed + target_frame,
            )
            error = _cd_ul1_mm(prediction_sample, target_sample)
            predictions[scale].append(error)
            frame_errors[f"registration_velocity_{scale:g}"] = error
        target_records.append(
            {
                "target_frame": target_frame,
                "template_CD_UL1_mm": template_error,
                "oracle_previous_mesh_CD_UL1_mm": oracle_error,
                **frame_errors,
            }
        )

    def summarize(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean_CD_UL1_mm": float(np.mean(array)),
            "median_CD_UL1_mm": float(np.median(array)),
            "p90_CD_UL1_mm": float(np.quantile(array, 0.9)),
        }

    aggregates = {
        "template": summarize(template_errors),
        "oracle_previous_mesh": summarize(oracle_persistence),
        **{
            f"registration_velocity_{scale:g}": summarize(values)
            for scale, values in predictions.items()
        },
    }
    best_candidate = min(
        (name for name in aggregates if name.startswith("registration_velocity_")),
        key=lambda name: aggregates[name]["mean_CD_UL1_mm"],
    )
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexBayesianRegistrationDevelopmentSmoke",
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": protocol["protocol_sha256"],
        },
        "take": {
            "id": take_root.name,
            "robot_sha256": _sha256(robot_path),
            "template_frame": template_frame,
            "template_sha256": _sha256(template_path),
            "causal_target_frame_count": len(target_records),
            "maximum_frame": frame_limit,
        },
        "registration_config": config.as_dict(),
        "causal_history": "all recursive updates stop at f-1 before scoring f",
        "future_observation_used": False,
        "aggregates": aggregates,
        "best_development_candidate": best_candidate,
        "published_kinect_reference_CD_UL1_mm": 6.498,
        "updates": update_records,
        "targets": target_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_bayesian_registration_v1.json"
        ),
    )
    parser.add_argument(
        "--velocity-scales",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--maximum-frame", type=int)
    args = parser.parse_args()
    result = run_smoke(
        args.take_root.resolve(),
        args.protocol.resolve(),
        velocity_scales=tuple(args.velocity_scales),
        maximum_frame=args.maximum_frame,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing smoke artifact differs: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **result["aggregates"]}, indent=2))


if __name__ == "__main__":
    main()
