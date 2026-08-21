"""Target-excluded MatPhys fold ensembles for physical-backend uncertainty.

The public MatPhys checkpoint collapses on the released PhysTwin interactions.
The useful transferable asset is instead the independently trained, object-held-
out fold family.  This module gives that family a strict, backend-facing
contract without claiming that checkpoint disagreement is calibrated by itself.

Every member predicts a bounded log-stiffness residual around an incumbent
spring field.  Official Warp replays remain a separate step.  The resulting
trajectory ensemble can then expose physical-backend epistemic spread while a
zero-strength proposal is exactly the incumbent array.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

from ._canonical_contracts import plain_json
from ._portable_contracts import (
    canonical_sorted_strings,
    content_id,
    exact_revision,
    nonempty_string,
    repository_name,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
)

MATPHYS_FOLD_ENSEMBLE_SCHEMA: Final = (
    "bayesian-phystwin.matphys-fold-ensemble-source"
)
MATPHYS_FOLD_ENSEMBLE_VERSION: Final = 1
MATPHYS_FOLD_ENSEMBLE_PROTOCOL: Final = (
    "target-excluded-source-supervised-matphys-fold-ensemble-v1"
)
MATPHYS_FOLD_TRAINING_CONTRACT: Final = "source-supervised-meta"
MATPHYS_FOLD_PARAMETERIZATION: Final = (
    "released-phystwin-bounded-logk-residual-v1"
)
MATPHYS_FOLD_SOURCE_REPOSITORY: Final = "Yrainy0615/MatPhys"
MATPHYS_PART_MODEL_CONTRACT: Final = "simple-videomae-dino-part-conditioning-v1"
MATPHYS_GRAPH_FEATURE_CONTRACT: Final = "matphys-part-aware-geometry-11d-v1"
MATPHYS_CAUSAL_VIDEO_CONTRACT: Final = "numeric-prefix-linspace-floor-v1"
MATPHYS_ENSEMBLE_MOMENT_CONTRACT: Final = (
    "equal-unique-member-population-moments-v1"
)

MATPHYS_FOLD_ENSEMBLE_CLAIM_BOUNDARY: Final = (
    "The object-held-out MatPhys folds propose bounded spring residuals around "
    "an unchanged physical incumbent. Fold disagreement is physical-backend "
    "epistemic evidence, not calibrated uncertainty by itself. Accuracy, NLL, "
    "coverage, and decision-safety claims require source calibration and a "
    "fresh, prediction-sealed target evaluation."
)

_SOURCE_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "protocol",
        "source_repository",
        "source_revision",
        "training_contract",
        "parameterization",
        "part_model_contract",
        "graph_feature_contract",
        "causal_video_contract",
        "training_universe_object_ids",
        "member_count",
        "members",
        "source_artifacts",
        "claim_boundary",
        "ensemble_id",
    }
)
_MEMBER_FIELDS: Final = frozenset(
    {
        "fold_index",
        "held_out_object_id",
        "training_object_ids",
        "checkpoint",
        "training_audit",
    }
)
_FILE_FIELDS: Final = frozenset({"path", "sha256", "byte_count"})

FloatArray = npt.NDArray[np.floating[Any]]
IntegerArray = npt.NDArray[np.integer[Any]]


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[object], value)


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    _require(
        source.is_file()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} must be an ordinary non-symlink file",
    )
    return source.resolve(strict=True)


def _file_record(path: str | Path, *, name: str) -> dict[str, object]:
    source = _ordinary_file(path, name=name)
    return {
        "path": str(source),
        "sha256": _file_sha256(source),
        "byte_count": source.stat().st_size,
    }


def _normalize_file_record(
    value: object,
    *,
    name: str,
    verify_file: bool,
) -> dict[str, object]:
    record = _mapping(value, name=name)
    require_exact_fields(record, expected=_FILE_FIELDS, name=name)
    path = nonempty_string(record.get("path"), name=f"{name}.path")
    digest = sha256_digest(record.get("sha256"), name=f"{name}.sha256")
    byte_count = _positive_integer(record.get("byte_count"), name=f"{name}.byte_count")
    if verify_file:
        source = _ordinary_file(path, name=name)
        _require(_file_sha256(source) == digest, f"{name} SHA-256 changed")
        _require(source.stat().st_size == byte_count, f"{name} byte count changed")
        path = str(source)
    return {"path": path, "sha256": digest, "byte_count": byte_count}


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def causal_frame_indices(
    evidence_end_frame_exclusive: int,
    *,
    frame_count: int = 16,
) -> npt.NDArray[np.int64]:
    """Return the exact causal linspace sampling used by the fold training.

    NumPy's integer cast intentionally floors intermediate locations.  The
    endpoint is always the final permitted frame, never a future frame.
    """

    evidence_end = _positive_integer(
        evidence_end_frame_exclusive,
        name="evidence_end_frame_exclusive",
    )
    count = _positive_integer(frame_count, name="frame_count")
    count = min(count, evidence_end)
    indices: npt.NDArray[np.int64] = np.linspace(
        0, evidence_end - 1, count
    ).astype(np.int64)
    _require(len(np.unique(indices)) == count, "causal frame selection duplicated a frame")
    _require(
        int(indices[-1]) < evidence_end,
        "causal frame selection crossed the evidence boundary",
    )
    return indices


def _validated_graph(
    points_m: np.ndarray,
    edges: np.ndarray,
    point_part: np.ndarray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    points = np.asarray(points_m, dtype=np.float64)
    links = np.asarray(edges, dtype=np.int64)
    parts = np.asarray(point_part, dtype=np.int64).reshape(-1)
    _require(points.ndim == 2 and points.shape[1] == 3, "points_m must have shape (N,3)")
    _require(len(points) > 1 and np.all(np.isfinite(points)), "points_m must be finite")
    _require(links.ndim == 2 and links.shape[1] == 2 and len(links) > 0, "edges must have shape (E,2)")
    _require(np.all(links >= 0) and np.all(links < len(points)), "edge endpoint exceeds points")
    _require(np.all(links[:, 0] != links[:, 1]), "self edges are not supported")
    canonical = np.sort(links, axis=1)
    _require(
        len(np.unique(canonical, axis=0)) == len(links),
        "duplicate undirected edges are not supported",
    )
    _require(len(parts) == len(points) and np.all(parts >= 0), "point_part must cover every point")
    unique_parts = np.unique(parts)
    _require(
        np.array_equal(unique_parts, np.arange(len(unique_parts))),
        "point_part labels must be contiguous from zero",
    )
    return points, links, parts


@dataclass(frozen=True)
class MatPhysGraphFeatures:
    edge_features: npt.NDArray[np.float32]
    scene_features: npt.NDArray[np.float32]
    edge_part_index: npt.NDArray[np.int64]
    graph_sha256: str


def matphys_graph_features(
    points_m: np.ndarray,
    edges: np.ndarray,
    point_part: np.ndarray,
    *,
    density_k: int = 16,
) -> MatPhysGraphFeatures:
    """Reproduce MatPhys's part-aware 11-D geometry and six scene features."""

    points, links, parts = _validated_graph(points_m, edges, point_part)
    density_neighbors = _positive_integer(density_k, name="density_k")
    edge_part = parts[links[:, 0]]
    edge_i, edge_j = links[:, 0], links[:, 1]
    vector = points[edge_j] - points[edge_i]
    rest_length = np.linalg.norm(vector, axis=1, keepdims=True)
    _require(np.all(rest_length > 1e-8), "graph contains a zero-length edge")
    absolute_direction = np.abs(vector / rest_length)
    part_count = int(parts.max()) + 1

    normalized_length = np.zeros_like(rest_length)
    for part in range(part_count):
        selected = edge_part == part
        if np.any(selected):
            normalized_length[selected] = rest_length[selected] / (
                np.median(rest_length[selected]) + 1e-8
            )

    degree: npt.NDArray[np.float64] = np.zeros(len(points), dtype=np.float64)
    same_part = parts[edge_i] == parts[edge_j]
    np.add.at(degree, edge_i[same_part], 1.0)
    np.add.at(degree, edge_j[same_part], 1.0)
    degree_normalized: npt.NDArray[np.float64] = np.zeros(
        len(points), dtype=np.float64
    )
    for part in range(part_count):
        selected = parts == part
        degree_normalized[selected] = degree[selected] / (
            np.mean(degree[selected]) + 1e-8
        )
    degree_i, degree_j = degree_normalized[edge_i], degree_normalized[edge_j]

    local_density: npt.NDArray[np.float64] = np.zeros(
        len(points), dtype=np.float64
    )
    for part in range(part_count):
        selected_indices = np.flatnonzero(parts == part)
        part_points = points[selected_indices]
        if len(part_points) < 2:
            local_density[selected_indices] = 1.0
            continue
        distance_squared = np.sum(
            (part_points[:, None] - part_points[None]) ** 2,
            axis=-1,
        )
        np.fill_diagonal(distance_squared, np.inf)
        neighbor_count = min(density_neighbors, len(part_points) - 1)
        nearest = np.partition(
            distance_squared,
            kth=neighbor_count - 1,
            axis=1,
        )[:, :neighbor_count]
        density = 1.0 / (np.sqrt(np.mean(nearest, axis=1)) + 1e-8)
        local_density[selected_indices] = density / (np.median(density) + 1e-8)

    density_i, density_j = local_density[edge_i], local_density[edge_j]
    pca_ratio_1: npt.NDArray[np.float64] = np.zeros(
        part_count, dtype=np.float64
    )
    pca_ratio_2: npt.NDArray[np.float64] = np.zeros(
        part_count, dtype=np.float64
    )
    pca_spread: npt.NDArray[np.float64] = np.zeros(
        part_count, dtype=np.float64
    )
    for part in range(part_count):
        part_points = points[parts == part]
        if len(part_points) < 3:
            pca_ratio_1[part] = 1.0
            pca_ratio_2[part] = 1.0
            continue
        centered = part_points - np.mean(part_points, axis=0)
        covariance = centered.T @ centered / (len(part_points) - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1].clip(min=1e-12)
        pca_ratio_1[part] = eigenvalues[1] / eigenvalues[0]
        pca_ratio_2[part] = eigenvalues[2] / eigenvalues[0]
        pca_spread[part] = np.log(eigenvalues[0] + 1e-8)
    nonzero_spread = pca_spread[pca_spread != 0.0]
    spread_scale = np.median(np.abs(nonzero_spread)) + 1e-8
    pca_spread /= spread_scale

    edge_features = np.concatenate(
        (
            normalized_length,
            absolute_direction,
            np.minimum(degree_i, degree_j)[:, None],
            np.maximum(degree_i, degree_j)[:, None],
            ((density_i + density_j) / 2.0)[:, None],
            np.abs(density_i - density_j)[:, None],
            pca_ratio_1[edge_part, None],
            pca_ratio_2[edge_part, None],
            pca_spread[edge_part, None],
        ),
        axis=1,
    ).astype(np.float32)
    bounding_box = np.max(points, axis=0) - np.min(points, axis=0)
    scene_features: npt.NDArray[np.float32] = np.asarray(
        (
            float(np.mean(rest_length)),
            float(np.std(rest_length) + 1e-6),
            float(np.log(len(points) + 1)),
            float(np.log(len(links) + 1)),
            float(np.linalg.norm(bounding_box)),
            float(2.0 * len(links) / len(points)),
        ),
        dtype=np.float32,
    )
    _require(edge_features.shape == (len(links), 11), "MatPhys edge feature width changed")
    graph_digest = hashlib.sha256()
    graph_digest.update(_array_sha256(points.astype(np.float32)).encode("ascii"))
    graph_digest.update(_array_sha256(links).encode("ascii"))
    return MatPhysGraphFeatures(
        edge_features=edge_features,
        scene_features=scene_features,
        edge_part_index=edge_part.astype(np.int64),
        graph_sha256=graph_digest.hexdigest(),
    )


def apply_bounded_spring_residual(
    incumbent_spring_y_pa: np.ndarray,
    raw_log_residual: np.ndarray,
    *,
    proposal_strength: float,
    maximum_abs_log_ratio: float = float(np.log(2.0)),
) -> np.ndarray:
    """Apply a fold residual around the incumbent with an exact zero identity."""

    incumbent = np.asarray(incumbent_spring_y_pa)
    raw = np.asarray(raw_log_residual)
    strength = _finite_number(
        proposal_strength,
        name="proposal_strength",
        minimum=0.0,
        maximum=1.0,
    )
    bound = _finite_number(
        maximum_abs_log_ratio,
        name="maximum_abs_log_ratio",
        minimum=0.0,
    )
    _require(
        incumbent.ndim == 1
        and np.issubdtype(incumbent.dtype, np.floating)
        and len(incumbent) > 0,
        "incumbent_spring_y_pa must be a floating vector",
    )
    _require(raw.shape == incumbent.shape, "raw_log_residual must match the spring field")
    _require(
        np.all(np.isfinite(incumbent)) and np.all(incumbent > 0.0),
        "incumbent spring values must be finite and positive",
    )
    _require(np.all(np.isfinite(raw)), "raw log residual must be finite")
    if strength == 0.0 or bound == 0.0:
        return incumbent
    log_ratio = strength * bound * np.tanh(raw.astype(np.float64))
    result = incumbent.astype(np.float64) * np.exp(log_ratio)
    return result.astype(incumbent.dtype, copy=False)


@dataclass(frozen=True)
class MatPhysEnsembleMoments:
    mean_m: npt.NDArray[np.float64]
    covariance_m2: npt.NDArray[np.float64]
    unique_member_indices: npt.NDArray[np.int64]
    unique_member_sha256s: tuple[str, ...]


def trajectory_ensemble_moments(
    member_trajectories_m: np.ndarray,
) -> MatPhysEnsembleMoments:
    """Compute equal-weight moments after collapsing byte-identical members."""

    members = np.asarray(member_trajectories_m)
    _require(
        members.ndim == 4 and members.shape[-1] == 3 and len(members) > 0,
        "member_trajectories_m must have shape (M,T,N,3)",
    )
    _require(
        np.issubdtype(members.dtype, np.floating) and np.all(np.isfinite(members)),
        "member trajectories must be finite floating arrays",
    )
    indices: list[int] = []
    digests: list[str] = []
    seen: set[str] = set()
    for index, member in enumerate(members):
        digest = _array_sha256(member)
        if digest not in seen:
            seen.add(digest)
            indices.append(index)
            digests.append(digest)
    unique = members[np.asarray(indices, dtype=np.int64)].astype(np.float64)
    mean = np.mean(unique, axis=0)
    centered = unique - mean[None]
    covariance = np.einsum("mtni,mtnj->tnij", centered, centered) / len(unique)
    covariance = (covariance + np.swapaxes(covariance, -1, -2)) * 0.5
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    _require(minimum_eigenvalue >= -1e-12, "ensemble covariance is not PSD")
    return MatPhysEnsembleMoments(
        mean_m=mean,
        covariance_m2=covariance,
        unique_member_indices=np.asarray(indices, dtype=np.int64),
        unique_member_sha256s=tuple(digests),
    )


def _normalized_member(
    value: object,
    *,
    universe: tuple[str, ...],
    verify_files: bool,
) -> dict[str, object]:
    member = _mapping(value, name="ensemble member")
    require_exact_fields(member, expected=_MEMBER_FIELDS, name="ensemble member")
    fold_index = _nonnegative_integer(member.get("fold_index"), name="fold_index")
    held_out = nonempty_string(
        member.get("held_out_object_id"),
        name="held_out_object_id",
    )
    _require(held_out in universe, "held-out object is outside the training universe")
    training = canonical_sorted_strings(
        _sequence(member.get("training_object_ids"), name="training_object_ids"),
        name="training_object_ids",
    )
    _require(
        training == tuple(item for item in universe if item != held_out),
        "member training objects must equal the universe minus its held-out object",
    )
    return {
        "fold_index": fold_index,
        "held_out_object_id": held_out,
        "training_object_ids": list(training),
        "checkpoint": _normalize_file_record(
            member.get("checkpoint"),
            name=f"fold {fold_index} checkpoint",
            verify_file=verify_files,
        ),
        "training_audit": _normalize_file_record(
            member.get("training_audit"),
            name=f"fold {fold_index} training audit",
            verify_file=verify_files,
        ),
    }


def validate_matphys_fold_ensemble_source(
    value: object,
    *,
    verify_files: bool,
) -> dict[str, object]:
    """Validate and canonicalize one target-excluded fold source manifest."""

    source = _mapping(value, name="MatPhys fold ensemble source")
    require_exact_fields(source, expected=_SOURCE_FIELDS, name="MatPhys fold ensemble source")
    _require(source.get("schema") == MATPHYS_FOLD_ENSEMBLE_SCHEMA, "source schema changed")
    _require(source.get("schema_version") == MATPHYS_FOLD_ENSEMBLE_VERSION, "source schema version changed")
    _require(source.get("protocol") == MATPHYS_FOLD_ENSEMBLE_PROTOCOL, "source protocol changed")
    repository = repository_name(source.get("source_repository"), name="source_repository")
    _require(repository == MATPHYS_FOLD_SOURCE_REPOSITORY, "source repository changed")
    revision = exact_revision(source.get("source_revision"), name="source_revision")
    _require(source.get("training_contract") == MATPHYS_FOLD_TRAINING_CONTRACT, "training contract changed")
    _require(source.get("parameterization") == MATPHYS_FOLD_PARAMETERIZATION, "parameterization changed")
    _require(source.get("part_model_contract") == MATPHYS_PART_MODEL_CONTRACT, "part model contract changed")
    _require(source.get("graph_feature_contract") == MATPHYS_GRAPH_FEATURE_CONTRACT, "graph feature contract changed")
    _require(source.get("causal_video_contract") == MATPHYS_CAUSAL_VIDEO_CONTRACT, "causal video contract changed")
    universe = canonical_sorted_strings(
        _sequence(
            source.get("training_universe_object_ids"),
            name="training_universe_object_ids",
        ),
        name="training_universe_object_ids",
    )
    _require(len(universe) >= 2, "training universe needs at least two objects")
    raw_members = _sequence(source.get("members"), name="members")
    member_count = _positive_integer(source.get("member_count"), name="member_count")
    _require(len(raw_members) == member_count, "member count changed")
    members: tuple[dict[str, object], ...] = tuple(
        _normalized_member(item, universe=universe, verify_files=verify_files)
        for item in raw_members
    )
    fold_indices = tuple(
        _nonnegative_integer(member["fold_index"], name="fold_index")
        for member in members
    )
    _require(
        fold_indices == tuple(range(member_count)),
        "fold indices must be sorted and contiguous from zero",
    )
    held_out = tuple(str(member["held_out_object_id"]) for member in members)
    _require(
        len(set(held_out)) == member_count
        and set(held_out) == set(universe)
        and member_count == len(universe),
        "ensemble must contain exactly one held-out fold per training object",
    )
    checkpoint_hashes = tuple(
        sha256_digest(
            _mapping(
                member["checkpoint"],
                name=f"fold {member['fold_index']} checkpoint",
            ).get("sha256"),
            name=f"fold {member['fold_index']} checkpoint.sha256",
        )
        for member in members
    )
    _require(
        len(set(checkpoint_hashes)) == member_count,
        "ensemble checkpoint SHA-256 values must be unique",
    )
    artifacts = source_artifact_mapping(
        _mapping(source.get("source_artifacts"), name="source_artifacts"),
        name="source_artifacts",
    )
    _require(
        source.get("claim_boundary") == MATPHYS_FOLD_ENSEMBLE_CLAIM_BOUNDARY,
        "claim boundary changed",
    )
    identity = {
        "schema": MATPHYS_FOLD_ENSEMBLE_SCHEMA,
        "schema_version": MATPHYS_FOLD_ENSEMBLE_VERSION,
        "protocol": MATPHYS_FOLD_ENSEMBLE_PROTOCOL,
        "source_repository": repository,
        "source_revision": revision,
        "training_contract": MATPHYS_FOLD_TRAINING_CONTRACT,
        "parameterization": MATPHYS_FOLD_PARAMETERIZATION,
        "part_model_contract": MATPHYS_PART_MODEL_CONTRACT,
        "graph_feature_contract": MATPHYS_GRAPH_FEATURE_CONTRACT,
        "causal_video_contract": MATPHYS_CAUSAL_VIDEO_CONTRACT,
        "training_universe_object_ids": list(universe),
        "member_count": member_count,
        "members": list(members),
        "source_artifacts": plain_json(artifacts),
        "claim_boundary": MATPHYS_FOLD_ENSEMBLE_CLAIM_BOUNDARY,
    }
    _require(source.get("ensemble_id") == content_id(identity), "ensemble_id changed")
    return {**identity, "ensemble_id": content_id(identity)}


def build_matphys_fold_ensemble_source(
    *,
    source_revision: str,
    training_universe_object_ids: Sequence[str],
    members: Sequence[Mapping[str, object]],
    source_artifacts: Mapping[str, str],
) -> dict[str, object]:
    """Build a strict source manifest from checked checkpoint/audit files."""

    universe = canonical_sorted_strings(
        training_universe_object_ids,
        name="training_universe_object_ids",
    )
    normalized_members: list[dict[str, object]] = []
    for member in members:
        fold_index = _nonnegative_integer(member.get("fold_index"), name="fold_index")
        held_out = nonempty_string(
            member.get("held_out_object_id"),
            name="held_out_object_id",
        )
        training = canonical_sorted_strings(
            _sequence(member.get("training_object_ids"), name="training_object_ids"),
            name="training_object_ids",
        )
        normalized_members.append(
            {
                "fold_index": fold_index,
                "held_out_object_id": held_out,
                "training_object_ids": list(training),
                "checkpoint": _file_record(
                    nonempty_string(member.get("checkpoint_path"), name="checkpoint_path"),
                    name=f"fold {fold_index} checkpoint",
                ),
                "training_audit": _file_record(
                    nonempty_string(member.get("training_audit_path"), name="training_audit_path"),
                    name=f"fold {fold_index} training audit",
                ),
            }
        )
    identity = {
        "schema": MATPHYS_FOLD_ENSEMBLE_SCHEMA,
        "schema_version": MATPHYS_FOLD_ENSEMBLE_VERSION,
        "protocol": MATPHYS_FOLD_ENSEMBLE_PROTOCOL,
        "source_repository": MATPHYS_FOLD_SOURCE_REPOSITORY,
        "source_revision": source_revision,
        "training_contract": MATPHYS_FOLD_TRAINING_CONTRACT,
        "parameterization": MATPHYS_FOLD_PARAMETERIZATION,
        "part_model_contract": MATPHYS_PART_MODEL_CONTRACT,
        "graph_feature_contract": MATPHYS_GRAPH_FEATURE_CONTRACT,
        "causal_video_contract": MATPHYS_CAUSAL_VIDEO_CONTRACT,
        "training_universe_object_ids": list(universe),
        "member_count": len(normalized_members),
        "members": normalized_members,
        "source_artifacts": plain_json(
            source_artifact_mapping(source_artifacts, name="source_artifacts")
        ),
        "claim_boundary": MATPHYS_FOLD_ENSEMBLE_CLAIM_BOUNDARY,
    }
    candidate = {**identity, "ensemble_id": content_id(identity)}
    return validate_matphys_fold_ensemble_source(candidate, verify_files=True)


def assert_target_excluded(
    source: object,
    *,
    target_object_id: str,
) -> None:
    """Reject any ensemble member trained with the target physical object."""

    manifest = validate_matphys_fold_ensemble_source(source, verify_files=False)
    target = nonempty_string(target_object_id, name="target_object_id")
    members = _sequence(manifest["members"], name="members")
    for raw_member in members:
        member = _mapping(raw_member, name="ensemble member")
        training = _sequence(
            member.get("training_object_ids"),
            name="training_object_ids",
        )
        if target in training:
            raise ValueError(
                f"fold {member['fold_index']} checkpoint training includes the target"
            )
