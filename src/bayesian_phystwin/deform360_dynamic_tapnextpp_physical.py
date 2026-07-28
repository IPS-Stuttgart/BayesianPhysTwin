"""Physical-backbone artifacts for the dynamic TAPNext++ protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_object_exclusion import file_sha256
from .observation_belief import array_sha256
from .tapnextpp_dynamic_multiview import PROTOCOL_ID

PHYSICAL_ARCHIVE_FILENAME = "dynamic_tapnextpp_physical.npz"
PHYSICAL_MANIFEST_FILENAME = "dynamic_tapnextpp_physical.json"
PHYSICAL_ARTIFACT_KIND = "Deform360DynamicTAPNextPPPhysicalBackbone"
PHYSICAL_FRAME_COUNT = 76
GRAPH_BASIS_RANK = 8

_ARCHIVE_FIELDS = frozenset(
    {
        "action_support",
        "driven_readout_m",
        "frame_zero_points_m",
        "graph_basis",
        "persistence_prediction_m",
        "physical_prediction_m",
        "zero_action_readout_m",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(
    payload: Mapping[str, Any],
    *,
    digest_key: str,
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonicalize_columns(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def build_readout_graph_basis(
    vertices: np.ndarray,
    springs: np.ndarray,
    readout_weights: np.ndarray,
    *,
    rank: int = GRAPH_BASIS_RANK,
) -> np.ndarray:
    """Lift deterministic low-frequency physical modes to material identities."""

    graph_vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(springs, dtype=np.int64)
    weights = np.asarray(readout_weights, dtype=np.float64)
    _require(
        graph_vertices.ndim == 2
        and graph_vertices.shape[1] == 3
        and np.all(np.isfinite(graph_vertices)),
        "physical graph vertices are invalid",
    )
    node_count = len(graph_vertices)
    _require(
        edges.ndim == 2
        and edges.shape[1] == 2
        and len(edges) > 0
        and np.all((edges >= 0) & (edges < node_count)),
        "physical graph springs are invalid",
    )
    _require(np.all(edges[:, 0] != edges[:, 1]), "physical graph has self edges")
    _require(
        weights.ndim == 2
        and weights.shape[1] == node_count
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-6),
        "material readout weights are invalid",
    )
    _require(
        1 <= rank <= 3 * min(node_count, len(weights)),
        "graph basis rank is invalid",
    )

    adjacency = np.zeros((node_count, node_count), dtype=np.float64)
    adjacency[edges[:, 0], edges[:, 1]] = 1.0
    adjacency[edges[:, 1], edges[:, 0]] = 1.0
    laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency
    _, modes = np.linalg.eigh(laplacian)
    modes = _canonicalize_columns(modes)
    chosen: list[np.ndarray] = []
    for scalar_mode in modes.T:
        lifted = weights @ scalar_mode
        for axis in range(3):
            candidate = np.zeros((len(weights), 3), dtype=np.float64)
            candidate[:, axis] = lifted
            vector = candidate.reshape(-1)
            for previous in chosen:
                vector -= previous * float(previous @ vector)
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-10:
                continue
            vector /= norm
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0.0:
                vector *= -1.0
            chosen.append(vector)
            if len(chosen) == rank:
                break
        if len(chosen) == rank:
            break
    _require(len(chosen) == rank, "lifted graph basis is rank deficient")
    basis = np.column_stack(chosen).reshape(len(weights), 3, rank)
    _require(
        np.allclose(
            basis.reshape(-1, rank).T @ basis.reshape(-1, rank),
            np.eye(rank),
            atol=1e-8,
        ),
        "lifted graph basis is not orthonormal",
    )
    return basis


def _normalized_archive(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    _require(set(arrays) == _ARCHIVE_FIELDS, "physical archive fields changed")
    result = {
        name: np.ascontiguousarray(np.asarray(value))
        for name, value in arrays.items()
    }
    physical = np.asarray(result["physical_prediction_m"], dtype=np.float64)
    persistence = np.asarray(
        result["persistence_prediction_m"],
        dtype=np.float64,
    )
    frame_zero = np.asarray(result["frame_zero_points_m"], dtype=np.float64)
    basis = np.asarray(result["graph_basis"], dtype=np.float64)
    _require(
        physical.ndim == 3
        and physical.shape[0] == PHYSICAL_FRAME_COUNT
        and physical.shape[2] == 3
        and persistence.shape == physical.shape
        and frame_zero.shape == physical.shape[1:]
        and np.all(np.isfinite(physical))
        and np.all(np.isfinite(persistence))
        and np.all(np.isfinite(frame_zero)),
        "physical prediction arrays are invalid",
    )
    _require(
        basis.shape == (len(frame_zero), 3, GRAPH_BASIS_RANK)
        and np.all(np.isfinite(basis)),
        "physical graph basis is invalid",
    )
    for name in ("driven_readout_m", "zero_action_readout_m"):
        values = np.asarray(result[name], dtype=np.float64)
        _require(
            values.shape == physical.shape and np.all(np.isfinite(values)),
            f"{name} is invalid",
        )
        _require(
            np.array_equal(values[0], frame_zero),
            f"{name} changed frame-zero material identities",
        )
    support = np.asarray(result["action_support"], dtype=np.float64)
    _require(
        support.shape == (len(frame_zero),)
        and np.all(np.isfinite(support))
        and np.all((support >= 0.0) & (support <= 1.0)),
        "physical action support is invalid",
    )
    _require(
        np.array_equal(physical[0], frame_zero)
        and np.array_equal(persistence[0], frame_zero),
        "physical backbone changed frame-zero material identities",
    )
    flat_basis = basis.reshape(-1, GRAPH_BASIS_RANK)
    _require(
        np.allclose(
            flat_basis.T @ flat_basis,
            np.eye(GRAPH_BASIS_RANK),
            atol=1e-6,
        ),
        "physical graph basis is not orthonormal",
    )
    return result


def write_dynamic_physical_artifacts(
    output_dir: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    protocol_path: str | Path,
    cohort_lock_path: str | Path,
    case_record: Mapping[str, Any],
    partition: str,
    physical_mode: str,
    code_revision: str,
    input_files: Mapping[str, str | Path],
    runtime_provenance: Mapping[str, Any],
    fallback_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal a dynamic-protocol physical prediction before camera outcomes."""

    _require(partition in {"source", "target"}, "physical partition is invalid")
    _require(
        len(code_revision) == 40
        and all(character in "0123456789abcdef" for character in code_revision),
        "physical code revision is invalid",
    )
    _require(
        physical_mode in {"warp_twin", "source_admission_persistence_fallback"},
        "physical mode changed",
    )
    _require(
        (physical_mode == "source_admission_persistence_fallback")
        == (fallback_diagnostics is not None),
        "fallback diagnostics disagree with physical mode",
    )
    _require(
        not ({"protocol", "cohort_lock"} & set(input_files)),
        "physical input files use reserved names",
    )
    normalized = _normalized_archive(arrays)
    output = Path(output_dir).resolve()
    _require(not output.exists(), "physical output directory already exists")
    output.mkdir(parents=True)
    archive = output / PHYSICAL_ARCHIVE_FILENAME
    temporary = output / (PHYSICAL_ARCHIVE_FILENAME + ".tmp.npz")
    np.savez_compressed(temporary, **normalized)
    temporary.replace(archive)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PHYSICAL_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "partition": partition,
        **dict(case_record),
        "physical_mode": physical_mode,
        "physical_admitted": physical_mode == "warp_twin",
        "code_revision": code_revision,
        "fallback_diagnostics": (
            None if fallback_diagnostics is None else dict(fallback_diagnostics)
        ),
        "physical_archive": {
            "filename": PHYSICAL_ARCHIVE_FILENAME,
            "file_sha256": file_sha256(archive),
            "array_sha256": {
                name: array_sha256(value)
                for name, value in sorted(normalized.items())
            },
        },
        "inputs_sha256": {
            "protocol": file_sha256(protocol_path),
            "cohort_lock": file_sha256(cohort_lock_path),
            **{
                name: file_sha256(path)
                for name, path in sorted(input_files.items())
            },
        },
        "runtime_provenance": dict(runtime_provenance),
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_object_track_read": False,
            "future_tactile_read": False,
            "provider_outcome_or_metric_read": False,
            "held_v8_target_query_score_barrier_or_outcome_access": False,
        },
    }
    manifest["result_sha256"] = _canonical_sha256(
        manifest,
        digest_key="result_sha256",
    )
    (output / PHYSICAL_MANIFEST_FILENAME).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_dynamic_physical_artifacts(output)
    return manifest


def validate_dynamic_physical_artifacts(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate one physical manifest and its exact numeric archive."""

    output = Path(output_dir).resolve()
    manifest = json.loads(
        (output / PHYSICAL_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    _require(
        manifest.get("artifact_kind") == PHYSICAL_ARTIFACT_KIND
        and manifest.get("protocol_id") == PROTOCOL_ID,
        "physical artifact belongs to another protocol",
    )
    _require(
        manifest.get("partition") in {"source", "target"}
        and manifest.get("physical_mode")
        in {"warp_twin", "source_admission_persistence_fallback"}
        and manifest.get("physical_admitted")
        is (manifest.get("physical_mode") == "warp_twin"),
        "physical artifact status is invalid",
    )
    _require(
        (manifest.get("fallback_diagnostics") is not None)
        == (
            manifest.get("physical_mode")
            == "source_admission_persistence_fallback"
        ),
        "physical fallback evidence is invalid",
    )
    _require(
        manifest.get("result_sha256")
        == _canonical_sha256(manifest, digest_key="result_sha256"),
        "physical manifest checksum changed",
    )
    archive = output / PHYSICAL_ARCHIVE_FILENAME
    _require(
        manifest.get("physical_archive", {}).get("file_sha256")
        == file_sha256(archive),
        "physical archive checksum changed",
    )
    with np.load(archive, allow_pickle=False) as stored:
        arrays = _normalized_archive(
            {name: np.asarray(stored[name]) for name in stored.files}
        )
    _require(
        manifest["physical_archive"]["array_sha256"]
        == {
            name: array_sha256(value)
            for name, value in sorted(arrays.items())
        },
        "physical archive array checksum changed",
    )
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("future_object_rgb_read") is False
        and boundary.get("future_object_geometry_read") is False
        and boundary.get("future_object_track_read") is False
        and boundary.get("future_tactile_read") is False
        and boundary.get("provider_outcome_or_metric_read") is False
        and boundary.get(
            "held_v8_target_query_score_barrier_or_outcome_access"
        )
        is False,
        "physical artifact crossed its prediction boundary",
    )
    return manifest, arrays


__all__ = [
    "GRAPH_BASIS_RANK",
    "PHYSICAL_ARCHIVE_FILENAME",
    "PHYSICAL_ARTIFACT_KIND",
    "PHYSICAL_FRAME_COUNT",
    "PHYSICAL_MANIFEST_FILENAME",
    "build_readout_graph_basis",
    "validate_dynamic_physical_artifacts",
    "write_dynamic_physical_artifacts",
]
