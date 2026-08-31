#!/usr/bin/env python3
"""Exact object-identity reassignment test for sealed PokeFlex predictions.

The registered source-side rule selected one correction multiplier per physical
object from two opened interactions.  This retrospective diagnostic reuses the
sealed official13 prediction archives, reconstructs the registered multiplier
bank from the sealed baseline/global correction direction, and scores every
unique reassignment of the multiplier multiset across target object identities.
No model is refit and no prediction archive is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

SCHEMA: Final = "bayesian-phystwin/pokeflex-object-identity-placebo-v1"
RESULT_SCHEMA: Final = "bayesian-phystwin/pokeflex-object-identity-placebo-result-v1"
EXPECTED_PROTOCOL_ID: Final = "pokeflex-object-identity-placebo-v1"
EXPECTED_TARGET_PROTOCOL_ID: Final = "pokeflex-action-robust-official13-public-v1"
EXPECTED_ALLOCATION_COUNT: Final = 51_480


class ExperimentError(ValueError):
    """Raised when an immutable experiment contract is violated."""


def require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ExperimentError(message)


def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ExperimentError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
        )
    except ExperimentError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExperimentError(f"cannot read JSON: {path}") from error
    require(type(value) is dict, f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(payload: Mapping[str, object], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    path.write_text(rendered, encoding="utf-8", newline="\n")


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path.resolve())
    require(protocol.get("schema") == SCHEMA, "unexpected protocol schema")
    require(protocol.get("schema_version") == 1, "unsupported protocol version")
    require(protocol.get("protocol_id") == EXPECTED_PROTOCOL_ID, "protocol ID changed")
    require(
        protocol.get("status")
        == "retrospective-real-data-relation-specificity-diagnostic",
        "protocol status changed",
    )
    take_order = protocol.get("target_take_order")
    require(
        type(take_order) is list
        and len(take_order) == 13
        and len(set(take_order)) == 13
        and all(type(item) is str and "_T" in item for item in take_order),
        "target take order is invalid",
    )
    mapping = protocol.get("same_object_multiplier")
    require(type(mapping) is dict and set(mapping) == set(take_order), "multiplier map changed")
    multipliers = [float(mapping[take]) for take in take_order]
    require(
        Counter(multipliers) == Counter({1.0: 7, 2.0: 1, 3.0: 1, 4.0: 4}),
        "registered multiplier multiset changed",
    )
    randomization = protocol.get("randomization_test")
    require(type(randomization) is dict, "randomization contract is missing")
    require(
        randomization.get("expected_allocation_count") == EXPECTED_ALLOCATION_COUNT,
        "allocation count changed",
    )
    boundary = protocol.get("information_boundary")
    require(
        boundary
        == {
            "fresh_confirmation": False,
            "all_thirteen_target_outcomes_previously_opened": True,
            "new_model_prediction": False,
            "sealed_prediction_bytes_modified": False,
            "target_mesh_rescoring": True,
            "post_outcome_relation_diagnostic": True,
            "paper_claim_authorized": False,
        },
        "information boundary changed",
    )
    return protocol


def bind_registered_inputs(
    protocol: Mapping[str, Any], repository_root: Path
) -> dict[str, Path]:
    contract = protocol["input_contract"]
    result: dict[str, Path] = {}
    registrations = {
        "prediction_barrier": "prediction_barrier_file_sha256",
        "target_result": "target_result_file_sha256",
        "source_scale_calibration": "source_scale_calibration_file_sha256",
        "registered_target_protocol": "registered_target_protocol_file_sha256",
    }
    for path_key, digest_key in registrations.items():
        path = (repository_root / str(contract[path_key])).resolve()
        require(path.is_file(), f"registered input is missing: {path_key}")
        require(sha256_file(path) == contract[digest_key], f"registered input changed: {path_key}")
        result[path_key] = path
    target_protocol = read_json(result["registered_target_protocol"])
    require(
        target_protocol.get("protocol_id") == EXPECTED_TARGET_PROTOCOL_ID,
        "registered target protocol changed",
    )
    barrier = read_json(result["prediction_barrier"])
    require(barrier.get("prediction_count") == 13, "prediction barrier count changed")
    require(barrier.get("target_mesh_opened") is False, "prediction barrier reports target access")
    target_result = read_json(result["target_result"])
    require(len(target_result.get("objects", [])) == 13, "target result cohort changed")
    return result


def discover_prediction_archives(
    root: Path,
    barrier: Mapping[str, Any],
    take_order: Sequence[str],
) -> dict[str, tuple[Path, Path, dict[str, Any]]]:
    require(root.is_dir(), f"scoring root is unavailable: {root}")
    expected = {
        str(row["take_id"]): {
            "seal_file_sha256": str(row["seal_file_sha256"]),
            "prediction_npz_sha256": str(row["prediction_npz_sha256"]),
        }
        for row in barrier["predictions"]
    }
    require(set(expected) == set(take_order), "barrier take inventory changed")
    candidates: dict[str, list[tuple[Path, Path, dict[str, Any]]]] = {
        take: [] for take in take_order
    }
    for seal_path in root.rglob("seal.json"):
        if not seal_path.is_file() or seal_path.stat().st_size > 1_048_576:
            continue
        try:
            seal = read_json(seal_path)
        except ExperimentError:
            continue
        take_id = seal.get("take_id")
        if take_id not in candidates:
            continue
        npz_path = seal_path.parent / str(seal.get("prediction_npz", "prediction.npz"))
        if not npz_path.is_file():
            continue
        if sha256_file(seal_path) != expected[str(take_id)]["seal_file_sha256"]:
            continue
        if sha256_file(npz_path) != expected[str(take_id)]["prediction_npz_sha256"]:
            continue
        require(seal.get("future_mesh_read") is False, f"seal reports future mesh read: {take_id}")
        candidates[str(take_id)].append((seal_path, npz_path, seal))
    selected: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
    for take_id in take_order:
        rows = candidates[take_id]
        require(rows, f"sealed prediction archive not found: {take_id}")
        identities = {(sha256_file(row[0]), sha256_file(row[1])) for row in rows}
        require(len(identities) == 1, f"non-identical sealed prediction duplicates: {take_id}")
        selected[take_id] = min(rows, key=lambda row: (len(str(row[0])), str(row[0])))
    return selected


def discover_take_roots(
    roots: Sequence[Path],
    take_order: Sequence[str],
    active_frames: Mapping[str, Sequence[int]],
) -> dict[str, Path]:
    wanted = set(take_order)
    candidates: dict[str, list[Path]] = {take: [] for take in take_order}
    ignored = {".git", ".venv", "venv", "site-packages", "__pycache__", "predictions"}
    for search_root in roots:
        if not search_root.is_dir():
            continue
        for directory, names, _files in os.walk(search_root):
            names[:] = [name for name in names if name not in ignored]
            path = Path(directory)
            if path.name not in wanted:
                continue
            meshes = path / "meshes"
            frames = active_frames[path.name]
            if meshes.is_dir() and all(
                (meshes / f"mesh-f{int(frame):05d}.obj").is_file() for frame in frames
            ):
                candidates[path.name].append(path.resolve())
            names[:] = []
    selected: dict[str, Path] = {}
    for take_id in take_order:
        rows = sorted(set(candidates[take_id]), key=str)
        require(rows, f"target take root not found: {take_id}")
        reference = rows[0]
        reference_hashes = {
            int(frame): sha256_file(reference / "meshes" / f"mesh-f{int(frame):05d}.obj")
            for frame in active_frames[take_id]
        }
        for row in rows[1:]:
            observed = {
                int(frame): sha256_file(row / "meshes" / f"mesh-f{int(frame):05d}.obj")
                for frame in active_frames[take_id]
            }
            require(observed == reference_hashes, f"target mesh duplicates differ: {take_id}")
        selected[take_id] = reference
    return selected


def surface_sample(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = points[np.asarray(faces, dtype=np.int64)]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    require(np.all(np.isfinite(areas)), "mesh has non-finite area")
    require(float(np.sum(areas)) > 0.0, "mesh has zero surface area")
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


def load_target_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(path, process=False)
    require(isinstance(mesh, trimesh.Trimesh), f"expected one triangle mesh: {path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64) / 1000.0
    faces = np.asarray(mesh.faces, dtype=np.int64)
    require(vertices.ndim == 2 and vertices.shape[1] == 3, "target vertices changed")
    require(faces.ndim == 2 and faces.shape[1] == 3, "target faces changed")
    return vertices, faces


def load_prediction_arrays(path: Path) -> dict[str, np.ndarray]:
    required = {
        "baseline_vertices_m",
        "candidate_vertices_m",
        "global_candidate_vertices_m",
        "faces",
        "target_frames",
        "source_frames",
        "history_start_frames",
        "history_end_frames",
        "update_supported",
        "update_accepted",
        "action_supported",
        "robot_history_supported",
        "correction_rms_m",
    }
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == required, f"prediction schema changed: {path}")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    baseline = np.asarray(arrays["baseline_vertices_m"], dtype=np.float64)
    global_candidate = np.asarray(arrays["global_candidate_vertices_m"], dtype=np.float64)
    candidate = np.asarray(arrays["candidate_vertices_m"], dtype=np.float64)
    require(baseline.shape == global_candidate.shape == candidate.shape, "prediction shape changed")
    require(np.all(np.isfinite(baseline)), "baseline contains non-finite values")
    require(np.all(np.isfinite(global_candidate)), "global candidate contains non-finite values")
    require(np.all(np.isfinite(candidate)), "candidate contains non-finite values")
    supported = np.asarray(arrays["update_supported"])
    require(supported.dtype == np.bool_, "update support must be Boolean")
    require(np.array_equal(global_candidate[~supported], baseline[~supported]), "global fallback changed")
    require(np.array_equal(candidate[~supported], baseline[~supported]), "selected fallback changed")
    return arrays


def score_take(
    take_id: str,
    npz_path: Path,
    take_root: Path,
    active_frames: Sequence[int],
    multipliers: Sequence[float],
    selected_multiplier: float,
    sample_count: int,
    base_seed: int,
) -> dict[str, object]:
    from scipy.spatial import cKDTree

    arrays = load_prediction_arrays(npz_path)
    baseline = np.asarray(arrays["baseline_vertices_m"], dtype=np.float64)
    global_candidate = np.asarray(arrays["global_candidate_vertices_m"], dtype=np.float64)
    stored_candidate = np.asarray(arrays["candidate_vertices_m"], dtype=np.float64)
    faces = np.asarray(arrays["faces"], dtype=np.int64)
    frames = np.asarray(arrays["target_frames"], dtype=np.int64)
    frame_to_index = {int(frame): index for index, frame in enumerate(frames)}
    require(all(int(frame) in frame_to_index for frame in active_frames), f"active frame missing: {take_id}")
    scores: dict[float, list[float]] = {float(value): [] for value in multipliers}
    baseline_scores: list[float] = []
    stored_scores: list[float] = []
    for frame_raw in active_frames:
        frame = int(frame_raw)
        index = frame_to_index[frame]
        target_vertices, target_faces = load_target_mesh(
            take_root / "meshes" / f"mesh-f{frame:05d}.obj"
        )
        target_sample = surface_sample(target_vertices, target_faces, sample_count, base_seed + frame)
        tree = cKDTree(target_sample)

        def cd(vertices: np.ndarray) -> float:
            sample = surface_sample(vertices, faces, sample_count, base_seed + frame)
            nearest = tree.query(sample, k=1)[1]
            return float(1000.0 * np.mean(np.sum(np.abs(sample - target_sample[nearest]), axis=1)))

        baseline_scores.append(cd(baseline[index]))
        stored_scores.append(cd(stored_candidate[index]))
        direction = global_candidate[index] - baseline[index]
        for multiplier in multipliers:
            value = float(multiplier)
            vertices = global_candidate[index] if value == 1.0 else baseline[index] + value * direction
            scores[value].append(cd(vertices))
    means = {str(value): float(np.mean(scores[float(value)])) for value in multipliers}
    return {
        "take_id": take_id,
        "scored_frame_count": len(active_frames),
        "baseline_mean_CD_UL1_mm": float(np.mean(baseline_scores)),
        "stored_candidate_mean_CD_UL1_mm": float(np.mean(stored_scores)),
        "reconstructed_mean_CD_UL1_mm_by_multiplier": means,
        "selected_multiplier": selected_multiplier,
        "selected_reconstructed_mean_CD_UL1_mm": means[str(float(selected_multiplier))],
    }


def unique_allocations(size: int) -> Iterable[tuple[float, ...]]:
    require(size == 13, "allocation generator is registered for thirteen targets")
    positions = tuple(range(size))
    for four_positions in itertools.combinations(positions, 4):
        four_set = set(four_positions)
        remaining_after_four = [index for index in positions if index not in four_set]
        for three_position in remaining_after_four:
            remaining_after_three = [
                index for index in remaining_after_four if index != three_position
            ]
            for two_position in remaining_after_three:
                allocation = [1.0] * size
                for index in four_positions:
                    allocation[index] = 4.0
                allocation[three_position] = 3.0
                allocation[two_position] = 2.0
                yield tuple(allocation)


def exact_randomization(
    take_order: Sequence[str],
    score_matrix: Mapping[str, Mapping[str, float]],
    observed_allocation: Sequence[float],
) -> dict[str, object]:
    values: list[float] = []
    observed = float(
        np.mean(
            [
                score_matrix[take_id][str(float(multiplier))]
                for take_id, multiplier in zip(take_order, observed_allocation, strict=True)
            ]
        )
    )
    observed_count = 0
    tolerance = 1e-12
    for allocation in unique_allocations(len(take_order)):
        mean = float(
            np.mean(
                [
                    score_matrix[take_id][str(float(multiplier))]
                    for take_id, multiplier in zip(take_order, allocation, strict=True)
                ]
            )
        )
        values.append(mean)
        if allocation == tuple(observed_allocation):
            observed_count += 1
    require(len(values) == EXPECTED_ALLOCATION_COUNT, "exact allocation count changed")
    require(observed_count == 1, "observed allocation multiplicity changed")
    array = np.asarray(values, dtype=np.float64)
    no_worse = int(np.sum(array <= observed + tolerance))
    strictly_better = int(np.sum(array < observed - tolerance))
    return {
        "allocation_count": len(values),
        "observed_object_balanced_CD_UL1_mm": observed,
        "strictly_better_allocation_count": strictly_better,
        "no_worse_allocation_count": no_worse,
        "exact_lower_tail_p_value": float(no_worse / len(values)),
        "observed_rank_lower_is_better": strictly_better + 1,
        "minimum_CD_UL1_mm": float(np.min(array)),
        "quantile_025_CD_UL1_mm": float(np.quantile(array, 0.025)),
        "median_CD_UL1_mm": float(np.median(array)),
        "quantile_975_CD_UL1_mm": float(np.quantile(array, 0.975)),
        "maximum_CD_UL1_mm": float(np.max(array)),
        "observed_minus_median_mm": float(observed - np.median(array)),
    }


def build_result(protocol_path: Path, output: Path, repository_revision: str) -> dict[str, object]:
    protocol = validate_protocol(protocol_path)
    repository_root = protocol_path.resolve().parents[1]
    inputs = bind_registered_inputs(protocol, repository_root)
    barrier = read_json(inputs["prediction_barrier"])
    target_result = read_json(inputs["target_result"])
    source_calibration = read_json(inputs["source_scale_calibration"])
    take_order = [str(value) for value in protocol["target_take_order"]]
    selected_map = {str(key): float(value) for key, value in protocol["same_object_multiplier"].items()}
    target_rows = {str(row["take_id"]): row for row in target_result["objects"]}
    require(set(target_rows) == set(take_order), "target result order inventory changed")
    active_frames = {
        take_id: [int(frame["target_frame"]) for frame in target_rows[take_id]["frames"]]
        for take_id in take_order
    }
    scoring_root = Path(protocol["input_contract"]["scoring_root"]).resolve()
    archives = discover_prediction_archives(scoring_root, barrier, take_order)
    search_roots = [Path(value).resolve() for value in protocol["input_contract"]["dataset_search_roots"]]
    take_roots = discover_take_roots(search_roots, take_order, active_frames)
    scoring = protocol["scoring"]
    multipliers = tuple(float(value) for value in (1.0, 2.0, 3.0, 4.0))
    per_take: list[dict[str, object]] = []
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    for index, take_id in enumerate(take_order, start=1):
        _seal, npz_path, _seal_payload = archives[take_id]
        row = score_take(
            take_id,
            npz_path,
            take_roots[take_id],
            active_frames[take_id],
            multipliers,
            selected_map[take_id],
            int(scoring["surface_sample_count"]),
            int(scoring["surface_sample_seed"]),
        )
        expected = target_rows[take_id]
        selected_tolerance = float(scoring["stored_candidate_reproduction_absolute_tolerance_mm"])
        global_tolerance = float(scoring["stored_global_reproduction_absolute_tolerance_mm"])
        row["stored_candidate_reference_CD_UL1_mm"] = float(expected["candidate_mean_CD_UL1_mm"])
        row["stored_global_reference_CD_UL1_mm"] = float(expected["global_candidate_mean_CD_UL1_mm"])
        row["stored_candidate_reproduction_error_mm"] = abs(
            float(row["stored_candidate_mean_CD_UL1_mm"])
            - float(expected["candidate_mean_CD_UL1_mm"])
        )
        row["selected_reconstruction_error_mm"] = abs(
            float(row["selected_reconstructed_mean_CD_UL1_mm"])
            - float(expected["candidate_mean_CD_UL1_mm"])
        )
        row["global_reproduction_error_mm"] = abs(
            float(row["reconstructed_mean_CD_UL1_mm_by_multiplier"]["1.0"])
            - float(expected["global_candidate_mean_CD_UL1_mm"])
        )
        require(float(row["stored_candidate_reproduction_error_mm"]) <= selected_tolerance, f"stored candidate score drift: {take_id}")
        require(float(row["selected_reconstruction_error_mm"]) <= selected_tolerance, f"selected reconstruction drift: {take_id}")
        require(float(row["global_reproduction_error_mm"]) <= global_tolerance, f"global score drift: {take_id}")
        row["prediction_npz_path"] = str(npz_path)
        row["target_take_root"] = str(take_roots[take_id])
        per_take.append(row)
        write_json(
            progress_path,
            {
                "schema": "bayesian-phystwin/pokeflex-object-identity-placebo-progress-v1",
                "completed_take_count": index,
                "total_take_count": len(take_order),
                "last_completed_take": take_id,
                "target_outcomes_previously_opened": True,
            },
        )
        print(f"[{index}/{len(take_order)}] scored {take_id}", flush=True)

    score_matrix = {
        str(row["take_id"]): {
            str(key): float(value)
            for key, value in row["reconstructed_mean_CD_UL1_mm_by_multiplier"].items()
        }
        for row in per_take
    }
    observed_allocation = [selected_map[take_id] for take_id in take_order]
    randomization = exact_randomization(take_order, score_matrix, observed_allocation)
    global_mean = float(np.mean([score_matrix[take]["1.0"] for take in take_order]))
    observed_mean = float(randomization["observed_object_balanced_CD_UL1_mm"])
    oracle_mean = float(
        np.mean([min(score_matrix[take].values()) for take in take_order])
    )
    adjusted = [take for take in take_order if selected_map[take] != 1.0]
    adjusted_wins = sum(
        score_matrix[take][str(selected_map[take])] < score_matrix[take]["1.0"]
        for take in adjusted
    )
    adjusted_ties = sum(
        math.isclose(
            score_matrix[take][str(selected_map[take])],
            score_matrix[take]["1.0"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for take in adjusted
    )

    source_objects = source_calibration["objects"]
    correlation_rows: list[tuple[float, float]] = []
    for take_id in take_order:
        object_name, _, _take = take_id.rpartition("_T")
        if object_name not in source_objects:
            continue
        source_minimum = float(source_objects[object_name]["minimum_source_relative_improvement"])
        heldout_gain = float(
            (score_matrix[take_id]["1.0"] - score_matrix[take_id][str(selected_map[take_id])])
            / score_matrix[take_id]["1.0"]
        )
        correlation_rows.append((source_minimum, heldout_gain))
    from scipy.stats import spearmanr

    correlation = spearmanr(
        [row[0] for row in correlation_rows],
        [row[1] for row in correlation_rows],
    )
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "repository_revision": repository_revision,
        "runner_name": os.environ.get("RUNNER_NAME"),
        "take_order": take_order,
        "same_object_multiplier": selected_map,
        "per_take": per_take,
        "score_matrix_CD_UL1_mm": score_matrix,
        "primary": {
            **randomization,
            "global_multiplier_1_object_balanced_CD_UL1_mm": global_mean,
            "same_object_relative_improvement_over_global": float(
                (global_mean - observed_mean) / global_mean
            ),
            "same_object_minus_global_mm": float(observed_mean - global_mean),
            "oracle_per_object_multiplier_CD_UL1_mm": oracle_mean,
            "same_object_oracle_regret_mm": float(observed_mean - oracle_mean),
            "relation_specificity_supported_at_0_05": bool(
                float(randomization["exact_lower_tail_p_value"]) <= 0.05
                and observed_mean < global_mean
            ),
        },
        "secondary": {
            "adjusted_take_count": len(adjusted),
            "adjusted_take_win_count_vs_global": int(adjusted_wins),
            "adjusted_take_tie_count_vs_global": int(adjusted_ties),
            "source_target_correlation_take_count": len(correlation_rows),
            "source_minimum_vs_heldout_incremental_gain_spearman_rho": float(correlation.statistic),
            "source_minimum_vs_heldout_incremental_gain_two_sided_p": float(correlation.pvalue),
        },
        "input_identity": {
            "prediction_barrier_sha256": sha256_file(inputs["prediction_barrier"]),
            "target_result_sha256": sha256_file(inputs["target_result"]),
            "source_scale_calibration_sha256": sha256_file(inputs["source_scale_calibration"]),
            "registered_target_protocol_sha256": sha256_file(inputs["registered_target_protocol"]),
            "prediction_npz_sha256": {
                take_id: sha256_file(archives[take_id][1]) for take_id in take_order
            },
        },
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = canonical_id(result, "result_id")
    return result


def write_outputs(output: Path, result: Mapping[str, object]) -> None:
    write_json(output / "result.json", result)
    rows = result["per_take"]
    with (output / "score_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "take_id",
                "selected_multiplier",
                "baseline_CD_UL1_mm",
                "multiplier_1_CD_UL1_mm",
                "multiplier_2_CD_UL1_mm",
                "multiplier_3_CD_UL1_mm",
                "multiplier_4_CD_UL1_mm",
            ]
        )
        for row in rows:
            matrix = row["reconstructed_mean_CD_UL1_mm_by_multiplier"]
            writer.writerow(
                [
                    row["take_id"],
                    row["selected_multiplier"],
                    row["baseline_mean_CD_UL1_mm"],
                    matrix["1.0"],
                    matrix["2.0"],
                    matrix["3.0"],
                    matrix["4.0"],
                ]
            )
    primary = result["primary"]
    secondary = result["secondary"]
    lines = [
        "# PokeFlex object-identity placebo v1",
        "",
        f"- Result ID: `{result['result_id']}`",
        f"- Same-object mean CD: `{primary['observed_object_balanced_CD_UL1_mm']:.9f} mm`",
        f"- Global multiplier-1 mean CD: `{primary['global_multiplier_1_object_balanced_CD_UL1_mm']:.9f} mm`",
        f"- Relative improvement over global: `{100.0 * primary['same_object_relative_improvement_over_global']:.4f}%`",
        f"- Exact allocation count: `{primary['allocation_count']}`",
        f"- Exact lower-tail p-value: `{primary['exact_lower_tail_p_value']:.8f}`",
        f"- Rank among allocations: `{primary['observed_rank_lower_is_better']}/{primary['allocation_count']}`",
        f"- Placebo median CD: `{primary['median_CD_UL1_mm']:.9f} mm`",
        f"- Adjusted-object wins/ties: `{secondary['adjusted_take_win_count_vs_global']}/{secondary['adjusted_take_tie_count_vs_global']}` of `{secondary['adjusted_take_count']}`",
        f"- Source/held-out Spearman rho: `{secondary['source_minimum_vs_heldout_incremental_gain_spearman_rho']:.6f}`",
        f"- Relation-specificity diagnostic passed at 0.05: `{str(primary['relation_specificity_supported_at_0_05']).lower()}`",
        "",
        str(result["claim_boundary"]),
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "SHA256SUMS":
            hashes.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")


def self_test() -> None:
    allocations = list(unique_allocations(13))
    require(len(allocations) == EXPECTED_ALLOCATION_COUNT, "self-test allocation count")
    require(len(set(allocations)) == EXPECTED_ALLOCATION_COUNT, "self-test duplicate allocation")
    require(all(Counter(row) == Counter({1.0: 7, 2.0: 1, 3.0: 1, 4.0: 4}) for row in allocations), "self-test multiset")
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.asarray([[0, 1, 2]])
    first = surface_sample(vertices, faces, 32, 7)
    second = surface_sample(vertices, faces, 32, 7)
    require(np.array_equal(first, second), "surface sampler is nondeterministic")
    print("self-test passed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    run = subparsers.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repository-revision", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.command == "self-test":
        self_test()
        return 0
    require(
        len(arguments.repository_revision) == 40
        and all(character in "0123456789abcdef" for character in arguments.repository_revision),
        "repository revision must be a full lowercase SHA",
    )
    require(not arguments.output.exists(), "output path already exists")
    result = build_result(
        arguments.protocol,
        arguments.output,
        arguments.repository_revision,
    )
    write_outputs(arguments.output, result)
    print(json.dumps({"result_id": result["result_id"], **result["primary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentError as error:
        print(f"PokeFlex object-identity placebo failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
