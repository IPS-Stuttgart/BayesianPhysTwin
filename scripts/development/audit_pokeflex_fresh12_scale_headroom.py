#!/usr/bin/env python3
"""Measure post-open scale headroom in sealed fresh-take predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scipy.spatial import cKDTree


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPOSITORY_ROOT = _repository_root()
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "remote"))

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (  # noqa: E402
    TARGET_PROTOCOL_FRESH12_PUBLIC_V1,
    canonical_payload_sha256,
    file_sha256,
    load_pokeflex_shrinkage_target_protocol,
    surface_sample,
    target_take_ids_for_protocol,
    validate_prediction_barrier,
    validate_prediction_seal,
)
from bayesian_phystwin.pokeflex_fresh12_staging import (  # noqa: E402
    STAGE_MANIFEST_NAME,
    validate_pokeflex_fresh12_stage_manifest,
    validate_staged_file,
)

BASE_SCALE = 0.125


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_multipliers(value: str) -> tuple[float, ...]:
    multipliers = tuple(float(item) for item in value.split(","))
    _require(bool(multipliers), "scale bank is empty")
    _require(
        all(np.isfinite(item) and item >= 0.0 for item in multipliers),
        "scale bank is invalid",
    )
    _require(len(set(multipliers)) == len(multipliers), "scale bank repeats")
    _require(
        0.0 in multipliers and 1.0 in multipliers,
        "scale bank must include zero and one",
    )
    return tuple(sorted(multipliers))


def _locate_take(dataset_root: Path, take_id: str) -> Path:
    candidates = sorted(path for path in dataset_root.rglob(take_id) if path.is_dir())
    _require(len(candidates) == 1, f"expected one take root for {take_id}")
    return candidates[0]


def _cd_ul1_mm_with_tree(
    prediction: np.ndarray,
    target: np.ndarray,
    tree: cKDTree,
) -> float:
    indices = tree.query(prediction, k=1)[1]
    return float(1000.0 * np.mean(np.sum(np.abs(prediction - target[indices]), axis=1)))


def _summarize(
    per_take: list[dict[str, Any]],
    multipliers: tuple[float, ...],
) -> list[dict[str, Any]]:
    summaries = []
    for multiplier in multipliers:
        key = str(multiplier)
        baseline = np.asarray(
            [row["mean_CD_UL1_mm_by_multiplier"]["0.0"] for row in per_take],
            dtype=np.float64,
        )
        candidate = np.asarray(
            [row["mean_CD_UL1_mm_by_multiplier"][key] for row in per_take],
            dtype=np.float64,
        )
        differences = candidate - baseline
        tolerance = 1e-12
        summaries.append(
            {
                "multiplier": multiplier,
                "effective_scale": BASE_SCALE * multiplier,
                "object_balanced_CD_UL1_mm": float(np.mean(candidate)),
                "object_balanced_relative_improvement": float(
                    (np.mean(baseline) - np.mean(candidate)) / np.mean(baseline)
                ),
                "object_win_count": int(np.sum(differences < -tolerance)),
                "object_tie_count": int(np.sum(np.abs(differences) <= tolerance)),
                "object_loss_count": int(np.sum(differences > tolerance)),
                "worst_object_relative_improvement": float(
                    np.min((baseline - candidate) / baseline)
                ),
            }
        )
    return summaries


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    from run_pokeflex_bayesian_registration_smoke import _load_mesh
    from scipy.spatial import cKDTree

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("barrier", type=Path)
    parser.add_argument("registered_result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "sota"
            / "pokeflex_conservative_shrinkage_fresh12_public_v1.json"
        ),
    )
    parser.add_argument(
        "--multipliers",
        default="0,0.5,1,1.5,2,3,4",
        help="multipliers of the sealed 0.125 correction",
    )
    args = parser.parse_args()

    protocol = load_pokeflex_shrinkage_target_protocol(args.protocol)
    _require(
        protocol["protocol_id"] == TARGET_PROTOCOL_FRESH12_PUBLIC_V1,
        "audit requires the fresh12 protocol",
    )
    multipliers = _parse_multipliers(args.multipliers)
    barrier = json.loads(args.barrier.read_text(encoding="utf-8"))
    validate_prediction_barrier(barrier, protocol)
    registered = json.loads(args.registered_result.read_text(encoding="utf-8"))
    _require(
        registered.get("barrier_sha256") == barrier["barrier_sha256"],
        "registered result uses another barrier",
    )
    registered_by_take = {
        str(row["take_id"]): row for row in registered.get("objects", [])
    }
    barrier_by_take = {
        str(row["take_id"]): row for row in barrier.get("predictions", [])
    }

    count = int(protocol["evaluation"]["surface_sample_count"])
    seed = int(protocol["evaluation"]["surface_sample_seed"])
    per_take = []
    per_frame_oracle_baseline = []
    per_frame_oracle_candidate = []
    for take_id in target_take_ids_for_protocol(protocol):
        archive = validate_prediction_seal(
            args.prediction_root / take_id / "seal.json",
            protocol,
        )
        barrier_row = barrier_by_take[take_id]
        _require(
            file_sha256(archive.seal_path) == barrier_row["seal_file_sha256"],
            f"prediction seal changed: {take_id}",
        )
        _require(
            file_sha256(archive.npz_path) == barrier_row["prediction_npz_sha256"],
            f"prediction archive changed: {take_id}",
        )
        take_root = _locate_take(args.dataset_root, take_id)
        stage = validate_pokeflex_fresh12_stage_manifest(
            take_root / STAGE_MANIFEST_NAME,
            protocol,
            expected_take_id=take_id,
        )
        staged_files = stage["files_by_path"]
        robot_path = take_root / "robot_data.json"
        validate_staged_file(robot_path, take_root, staged_files)
        robot = json.loads(robot_path.read_text(encoding="utf-8"))
        active = sorted(
            int(row["frame"])
            for row in robot
            if int(row["frame"]) >= 6 and float(row["forces"][1]) > 3.0
        )
        frame_to_index = {
            int(frame): index for index, frame in enumerate(archive.target_frames)
        }
        _require(
            all(frame in frame_to_index for frame in active),
            "prediction frame missing",
        )
        errors = {str(multiplier): [] for multiplier in multipliers}
        for frame in active:
            index = frame_to_index[frame]
            mesh_path = take_root / "meshes" / f"mesh-f{frame:05d}.obj"
            validate_staged_file(mesh_path, take_root, staged_files)
            mesh = _load_mesh(mesh_path)
            target_sample = surface_sample(
                np.asarray(mesh.vertices, dtype=np.float64) / 1000.0,
                np.asarray(mesh.faces, dtype=np.int64),
                count,
                seed + frame,
            )
            tree = cKDTree(target_sample)
            baseline = archive.baseline_vertices_m[index]
            delta = archive.candidate_vertices_m[index] - baseline
            frame_errors = []
            for multiplier in multipliers:
                vertices = baseline + multiplier * delta
                prediction_sample = surface_sample(
                    vertices,
                    archive.faces,
                    count,
                    seed + frame,
                )
                error = _cd_ul1_mm_with_tree(prediction_sample, target_sample, tree)
                errors[str(multiplier)].append(error)
                frame_errors.append(error)
            per_frame_oracle_baseline.append(frame_errors[0])
            per_frame_oracle_candidate.append(min(frame_errors))

        means = {key: float(np.mean(value)) for key, value in errors.items()}
        registered_row = registered_by_take[take_id]
        _require(
            abs(means["0.0"] - float(registered_row["baseline_mean_CD_UL1_mm"]))
            < 1e-10,
            f"baseline reproduction failed: {take_id}",
        )
        _require(
            abs(means["1.0"] - float(registered_row["candidate_mean_CD_UL1_mm"]))
            < 1e-10,
            f"sealed scale reproduction failed: {take_id}",
        )
        best_multiplier = min(multipliers, key=lambda value: means[str(value)])
        per_take.append(
            {
                "take_id": take_id,
                "scored_frame_count": len(active),
                "supported_frame_count": int(
                    np.sum(
                        [
                            archive.update_supported[frame_to_index[frame]]
                            for frame in active
                        ]
                    )
                ),
                "mean_CD_UL1_mm_by_multiplier": means,
                "postopen_best_multiplier": best_multiplier,
                "postopen_best_CD_UL1_mm": means[str(best_multiplier)],
            }
        )

    summaries = _summarize(per_take, multipliers)
    best_uniform = min(summaries, key=lambda row: row["object_balanced_CD_UL1_mm"])
    audit: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexFresh12PostopenScaleHeadroomAudit",
        "status": "post-open diagnostic; not prospective evidence",
        "protocol_sha256": protocol["protocol_sha256"],
        "prediction_revision": barrier["implementation_revision"],
        "prediction_barrier_sha256": barrier["barrier_sha256"],
        "prediction_barrier_file_sha256": file_sha256(args.barrier),
        "registered_result_file_sha256": file_sha256(args.registered_result),
        "base_effective_scale": BASE_SCALE,
        "multipliers": list(multipliers),
        "uniform_scale_results": summaries,
        "postopen_best_uniform": best_uniform,
        "per_frame_scale_oracle_relative_improvement": float(
            (np.mean(per_frame_oracle_baseline) - np.mean(per_frame_oracle_candidate))
            / np.mean(per_frame_oracle_baseline)
        ),
        "takes": per_take,
        "claim_boundary": (
            "Opened fresh12 outcomes are used only to diagnose scale headroom. "
            "No multiplier selected here may be claimed on this cohort."
        ),
    }
    audit["audit_sha256"] = canonical_payload_sha256(
        audit,
        digest_field="audit_sha256",
    )
    _write_json(args.output, audit)
    print(
        json.dumps(
            {
                "audit_sha256": audit["audit_sha256"],
                "postopen_best_uniform": best_uniform,
                "per_frame_scale_oracle_relative_improvement": audit[
                    "per_frame_scale_oracle_relative_improvement"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
