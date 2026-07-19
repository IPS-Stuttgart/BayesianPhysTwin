"""Permit-gated scoring for the prospective held Deform360 protocol.

This module contains no outcome reconstruction code.  A caller supplies one
outcome callback per *locked* calibration case.  The callback is invoked only
through :func:`deform360_held_protocol.run_outcome_operation`, after the
complete 15-case online-prediction barrier has been authorized.

The official Deform360 reconstruction and the new frame-zero visual hull need
not contain the same material identities.  Their identities are therefore
joined by a frozen, one-to-one, sparse minimum-cost assignment at frame zero:

* every sealed point is assigned to a distinct visible and valid official
  identity;
* assignment edges are limited to 15 mm, the already-frozen frame-zero depth
  tolerance;
* coverage must be exactly 100 percent and collisions exactly zero; and
* the transported frame-zero coordinates are replaced bit-exactly by the
  sealed coordinates only after the raw assignment distances are recorded.

This produces a transported reconstruction proxy.  It is not native official
material identity and it is not Deform360 Table-4 parity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import numpy as np

from .deform360_held_protocol import (
    CALIBRATION_CASE_NAMES,
    DATASET_REVISION,
    FRAME_COUNT,
    METRIC_LOCK,
    OutcomePhasePermit,
    PROTOCOL_ID,
    create_calibration_gate_decision,
    held_artifact_sha256,
    run_outcome_operation,
    validate_frame_zero_bundle_manifest,
    validate_online_prediction_seal,
    validate_physical_prior_seal,
    validate_prefix_stage_authorization,
)
from .deform360_online_belief_evaluation import (
    score_deform360_hidden_trajectory,
)


MAXIMUM_FRAME_ZERO_ASSIGNMENT_DISTANCE_M = 0.015
MAXIMUM_SPARSE_ASSIGNMENT_EDGE_COUNT = 5_000_000
ASSIGNMENT_CHUNK_SIZE = 256
TARGET_ARTIFACT_KIND = "Deform360OfficialReconstructionTarget"
OUTCOME_ARTIFACT_KIND = "Deform360HeldOfficialOutcome"
SCORE_EVIDENCE_KIND = "Deform360HeldCalibrationScoreEvidence"

ONLINE_ARRAY_NAMES = frozenset(
    {
        "center_ids",
        "primary_prediction_m",
        "selected_raw_backbone_m",
        "frame_zero_points_m",
    }
)
TARGET_ARRAY_NAMES = frozenset(
    {
        "object_points",
        "object_visibilities",
        "object_motions_valid",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = f"{array.dtype.str}:{','.join(map(str, array.shape))}".encode()
    return hashlib.sha256(descriptor + b"\0" + array.tobytes()).hexdigest()


def _write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination.resolve()


def _bound_path(record: Mapping[str, Any], *, label: str) -> Path:
    value = record.get("path")
    _require(isinstance(value, str) and bool(value), f"{label} path is missing")
    return Path(value).resolve()


def _validate_outcome_file_binding(
    record: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    _require(isinstance(record, Mapping), f"{label} file binding is missing")
    _require(
        set(record) == {"path", "sha256", "size_bytes"},
        f"{label} file binding fields changed",
    )
    value = record.get("path")
    _require(isinstance(value, str) and bool(value), f"{label} path is missing")
    path = Path(value)
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label} is not a regular non-symlink file",
    )
    resolved = path.resolve()
    observed = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    _require(dict(record) == observed, f"{label} file binding changed")
    return observed


def scored_frames() -> tuple[int, ...]:
    """Expand and revalidate the frozen half-open score intervals."""

    intervals = METRIC_LOCK.get("scored_frame_intervals_half_open")
    _require(
        intervals == [[20, 38], [39, 57], [58, 76]],
        "held scored-frame intervals changed",
    )
    frames = tuple(
        frame for start, stop in intervals for frame in range(int(start), int(stop))
    )
    _require(len(frames) == 54 and frames[-1] < FRAME_COUNT, "invalid score frames")
    return frames


@dataclass(frozen=True)
class SealedCasePredictions:
    """The only sealed prediction arrays consumed by the gate scorer."""

    case_name: str
    center_ids: np.ndarray
    primary_prediction_m: np.ndarray
    selected_raw_backbone_m: np.ndarray
    frame_zero_points_m: np.ndarray
    seal_path: Path
    archive_path: Path
    bindings: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class OfficialTarget:
    """Outcome-side official reconstruction returned by a permitted callback."""

    object_points: np.ndarray
    object_visibilities: np.ndarray
    object_motions_valid: np.ndarray
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class TargetOperation:
    """One callback that creates or reads an outcome behind a live permit."""

    operation: Literal["create", "read"]
    callback: Callable[[], OfficialTarget | Mapping[str, Any]]


@dataclass(frozen=True)
class TransportedTarget:
    """Official trajectories transported onto the sealed point identities."""

    object_points: np.ndarray
    object_visibilities: np.ndarray
    object_motions_valid: np.ndarray
    official_identity_ids: np.ndarray
    assignment_distance_m: np.ndarray
    diagnostics: Mapping[str, Any]


def _load_float32_array(
    stored: Any,
    name: str,
    *,
    shape: tuple[int | None, ...],
) -> np.ndarray:
    value = np.asarray(stored[name])
    _require(value.dtype == np.dtype(np.float32), f"{name} must be float32")
    _require(value.ndim == len(shape), f"{name} rank changed")
    for axis, expected in enumerate(shape):
        if expected is not None:
            _require(value.shape[axis] == expected, f"{name} shape changed")
    _require(np.all(np.isfinite(value)), f"{name} contains non-finite coordinates")
    return value.copy()


def load_sealed_case_predictions(
    permit: OutcomePhasePermit,
    case_name: str,
) -> SealedCasePredictions:
    """Load only the four frozen arrays from one validated online seal."""

    _require(permit.role == "calibration", "gate scoring requires calibration permit")
    seal_paths = dict(permit.seal_paths)
    _require(case_name in seal_paths, "case is outside the permit seal set")
    seal_path = Path(seal_paths[case_name]).resolve()
    seal = validate_online_prediction_seal(
        seal_path,
        permit.lock_path,
        expected_case_name=case_name,
        expected_role="calibration",
    )
    authorization_path = _bound_path(
        seal["prefix_authorization"], label="prefix authorization"
    )
    authorization = validate_prefix_stage_authorization(
        authorization_path, permit.lock_path
    )
    physical_seal_path = _bound_path(
        authorization["physical_prior_seal"], label="physical-prior seal"
    )
    physical = validate_physical_prior_seal(
        physical_seal_path,
        permit.lock_path,
        expected_case_name=case_name,
        expected_role="calibration",
    )
    frame_zero_manifest_path = _bound_path(
        physical["frame_zero_manifest"], label="frame-zero manifest"
    )
    frame_zero_manifest = validate_frame_zero_bundle_manifest(
        frame_zero_manifest_path,
        permit.lock_path,
        expected_case_name=case_name,
        expected_role="calibration",
    )

    archive_record = seal["online_artifacts"]["online_prediction_archive"]
    archive_path = _bound_path(archive_record, label="online prediction archive")
    _require(archive_path.suffix == ".npz", "online prediction archive must be NPZ")
    with np.load(archive_path, allow_pickle=False) as stored:
        _require(
            ONLINE_ARRAY_NAMES.issubset(stored.files),
            "online prediction archive lacks frozen score arrays",
        )
        frame_zero = _load_float32_array(stored, "frame_zero_points_m", shape=(None, 3))
        point_count = len(frame_zero)
        primary = _load_float32_array(
            stored,
            "primary_prediction_m",
            shape=(FRAME_COUNT, point_count, 3),
        )
        comparator = _load_float32_array(
            stored,
            "selected_raw_backbone_m",
            shape=(FRAME_COUNT, point_count, 3),
        )
        centers_raw = np.asarray(stored["center_ids"])
    _require(centers_raw.dtype.kind in "iu", "center_ids must be integers")
    centers = centers_raw.astype(np.int64, copy=True)
    _require(centers.ndim == 1 and len(centers) > 0, "center_ids must be a vector")
    _require(len(np.unique(centers)) == len(centers), "center_ids must be unique")
    _require(
        np.all((centers >= 0) & (centers < point_count)),
        "center_ids exceed the sealed point set",
    )
    _require(point_count > len(centers), "center exclusion leaves no score points")
    _require(
        np.array_equal(primary[0], frame_zero),
        "primary frame zero differs from its sealed identity",
    )
    _require(
        np.array_equal(comparator[0], frame_zero),
        "selected-raw frame zero differs from its sealed identity",
    )

    bundle_path = _bound_path(frame_zero_manifest["bundle"], label="frame-zero bundle")
    with np.load(bundle_path, allow_pickle=False) as stored:
        _require(
            "object_points_world_m" in stored.files,
            "frame-zero bundle lacks object points",
        )
        bundle_frame_zero = np.asarray(stored["object_points_world_m"])
    _require(
        bundle_frame_zero.dtype == np.dtype(np.float32)
        and np.array_equal(bundle_frame_zero, frame_zero),
        "online archive differs from the frame-zero bundle identity",
    )

    physical_archive_record = physical["physical_artifacts"][
        "physical_prediction_archive"
    ]
    physical_archive_path = _bound_path(
        physical_archive_record, label="physical prediction archive"
    )
    with np.load(physical_archive_path, allow_pickle=False) as stored:
        _require(
            "frame_zero_points_m" in stored.files,
            "physical archive lacks frame-zero points",
        )
        physical_frame_zero = np.asarray(stored["frame_zero_points_m"])
    _require(
        physical_frame_zero.dtype == np.dtype(np.float32)
        and np.array_equal(physical_frame_zero, frame_zero),
        "online archive differs from the physical frame-zero identity",
    )
    return SealedCasePredictions(
        case_name=case_name,
        center_ids=centers,
        primary_prediction_m=primary,
        selected_raw_backbone_m=comparator,
        frame_zero_points_m=frame_zero,
        seal_path=seal_path,
        archive_path=archive_path,
        bindings={
            "online_prediction_seal": {
                "path": str(seal_path),
                "sha256": _sha256_file(seal_path),
            },
            "online_prediction_archive": dict(archive_record),
            "physical_prediction_archive": dict(physical_archive_record),
            "frame_zero_bundle": dict(frame_zero_manifest["bundle"]),
        },
    )


def normalize_official_target(
    value: OfficialTarget | Mapping[str, Any],
) -> OfficialTarget:
    """Validate the outcome callback payload without changing its masks."""

    if isinstance(value, OfficialTarget):
        target = value
    else:
        _require(isinstance(value, Mapping), "outcome callback returned no target")
        _require(
            TARGET_ARRAY_NAMES.issubset(value),
            "official target lacks required arrays",
        )
        target = OfficialTarget(
            object_points=np.asarray(value["object_points"]),
            object_visibilities=np.asarray(value["object_visibilities"]),
            object_motions_valid=np.asarray(value["object_motions_valid"]),
            provenance=dict(value.get("provenance", {})),
        )
    points = np.asarray(target.object_points)
    visible = np.asarray(target.object_visibilities)
    valid = np.asarray(target.object_motions_valid)
    _require(
        np.issubdtype(points.dtype, np.floating),
        "official object_points must be floating point",
    )
    _require(
        points.ndim == 3 and points.shape[0] == FRAME_COUNT and points.shape[2] == 3,
        "official object_points must have shape (76, M, 3)",
    )
    _require(points.shape[1] > 0, "official target has no material identities")
    _require(visible.dtype == np.dtype(bool), "official visibility must be boolean")
    _require(valid.dtype == np.dtype(bool), "official validity must be boolean")
    _require(
        visible.shape == points.shape[:2] and valid.shape == points.shape[:2],
        "official visibility/validity masks must have shape (76, M)",
    )
    _require(
        np.all(np.isfinite(points[0])),
        "official frame-zero coordinates must be finite",
    )
    return OfficialTarget(
        object_points=points.copy(),
        object_visibilities=visible.copy(),
        object_motions_valid=valid.copy(),
        provenance=dict(target.provenance),
    )


def official_target_array_sha256(target: OfficialTarget) -> dict[str, str]:
    """Return the three exact array hashes required in outcome provenance."""

    normalized = normalize_official_target(target)
    return {
        "object_points": _sha256_array(normalized.object_points),
        "object_visibilities": _sha256_array(normalized.object_visibilities),
        "object_motions_valid": _sha256_array(normalized.object_motions_valid),
    }


def validate_permitted_target_provenance(
    target_value: OfficialTarget | Mapping[str, Any],
    permit: OutcomePhasePermit,
    case_name: str,
) -> OfficialTarget:
    """Bind one callback result to the live cohort, case, and exact arrays."""

    target = normalize_official_target(target_value)
    provenance = target.provenance
    object_id, encoded_episode = case_name.rsplit("-ep", maxsplit=1)
    required = {
        "target_artifact_kind",
        "outcome_artifact_kind",
        "case_name",
        "object_id",
        "episode_id",
        "dataset_revision",
        "cohort_barrier_sha256",
        "target_file",
        "outcome_file",
        "array_sha256",
        "information_boundary",
    }
    _require(
        isinstance(provenance, Mapping) and required.issubset(provenance),
        "permitted official target provenance is incomplete",
    )
    _require(
        provenance.get("target_artifact_kind") == TARGET_ARTIFACT_KIND,
        "official target artifact kind changed",
    )
    _require(
        provenance.get("outcome_artifact_kind") == OUTCOME_ARTIFACT_KIND,
        "held outcome artifact kind changed",
    )
    _require(provenance.get("case_name") == case_name, "target binds another case")
    _require(provenance.get("object_id") == object_id, "target object changed")
    _require(
        provenance.get("episode_id") == int(encoded_episode),
        "target episode changed",
    )
    _require(
        provenance.get("dataset_revision") == DATASET_REVISION,
        "target dataset revision changed",
    )
    _require(
        provenance.get("cohort_barrier_sha256") == permit.cohort_barrier_sha256,
        "target was opened under another cohort barrier",
    )
    target_file = _validate_outcome_file_binding(
        provenance.get("target_file", {}), label="official target"
    )
    outcome_file = _validate_outcome_file_binding(
        provenance.get("outcome_file", {}), label="held outcome"
    )
    _require(
        target_file["path"] != outcome_file["path"],
        "target and outcome provenance bind the same file",
    )
    _require(
        provenance.get("array_sha256") == official_target_array_sha256(target),
        "official target array checksums changed",
    )
    boundary = provenance.get("information_boundary", {})
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("complete_cohort_barrier_validated_before_future_open") is True
        and boundary.get("official_target_constructed_or_read_after_barrier") is True
        and boundary.get("prediction_metric_computed_during_target_construction")
        is False,
        "official target provenance crossed the cohort outcome boundary",
    )
    return target


def sparse_min_cost_frame_zero_assignment(
    sealed_frame_zero_m: np.ndarray,
    official_frame_zero_m: np.ndarray,
    *,
    maximum_distance_m: float = MAXIMUM_FRAME_ZERO_ASSIGNMENT_DISTANCE_M,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return a full one-to-one minimum-cost assignment inside a fixed radius."""

    sealed = np.asarray(sealed_frame_zero_m, dtype=np.float64)
    official = np.asarray(official_frame_zero_m, dtype=np.float64)
    _require(
        sealed.ndim == 2 and sealed.shape[1] == 3 and len(sealed) > 0,
        "sealed frame zero must have shape (N, 3)",
    )
    _require(
        official.ndim == 2 and official.shape[1] == 3 and len(official) >= len(sealed),
        "official frame zero must have at least N points with shape (M, 3)",
    )
    _require(
        np.all(np.isfinite(sealed)) and np.all(np.isfinite(official)),
        "frame-zero assignment inputs must be finite",
    )
    _require(
        maximum_distance_m == MAXIMUM_FRAME_ZERO_ASSIGNMENT_DISTANCE_M,
        "frame-zero assignment distance is frozen at 15 mm",
    )
    try:
        import scipy
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import min_weight_full_bipartite_matching
        from scipy.spatial import cKDTree
    except (ImportError, ValueError) as error:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "SciPy graph runtime is required for identity transport"
        ) from error

    tree = cKDTree(official)
    edge_rows: list[np.ndarray] = []
    edge_columns: list[np.ndarray] = []
    edge_distances: list[np.ndarray] = []
    edge_count = 0
    for start in range(0, len(sealed), ASSIGNMENT_CHUNK_SIZE):
        stop = min(start + ASSIGNMENT_CHUNK_SIZE, len(sealed))
        neighbours = tree.query_ball_point(
            sealed[start:stop], r=maximum_distance_m, workers=1, return_sorted=True
        )
        for local_row, candidate_values in enumerate(neighbours):
            row = start + local_row
            candidates = np.asarray(candidate_values, dtype=np.int64)
            _require(
                len(candidates) > 0,
                f"sealed frame-zero point {row} has no official identity within 15 mm",
            )
            distances = np.linalg.norm(official[candidates] - sealed[row], axis=1)
            keep = distances <= maximum_distance_m
            candidates = candidates[keep]
            distances = distances[keep]
            _require(len(candidates) > 0, f"assignment radius emptied row {row}")
            order = np.lexsort((candidates, distances))
            candidates = candidates[order]
            distances = distances[order]
            edge_count += len(candidates)
            _require(
                edge_count <= MAXIMUM_SPARSE_ASSIGNMENT_EDGE_COUNT,
                "frame-zero assignment graph exceeds the frozen edge budget",
            )
            edge_rows.append(np.full(len(candidates), row, dtype=np.int64))
            edge_columns.append(candidates)
            edge_distances.append(distances)

    rows = np.concatenate(edge_rows)
    columns = np.concatenate(edge_columns)
    distances = np.concatenate(edge_distances)
    epsilon = np.finfo(np.float64).eps * max(1.0, maximum_distance_m)
    # Strict positivity prevents sparse zero-weight edges from disappearing.
    # The edge-specific deterministic perturbation only resolves sub-ulp ties.
    tie_code = (
        ((rows.astype(np.uint64) + 1) * np.uint64(2_654_435_761))
        ^ ((columns.astype(np.uint64) + 1) * np.uint64(2_246_822_519))
    ) % np.uint64(1_000_003)
    costs = distances + epsilon * (1.0 + tie_code.astype(np.float64) / 1_000_003.0)
    graph = csr_matrix((costs, (rows, columns)), shape=(len(sealed), len(official)))
    try:
        matched_rows, matched_columns = min_weight_full_bipartite_matching(graph)
    except ValueError as error:
        raise ValueError(
            "no collision-free full frame-zero assignment exists within 15 mm"
        ) from error
    _require(
        len(matched_rows) == len(sealed)
        and np.array_equal(np.sort(matched_rows), np.arange(len(sealed))),
        "sparse assignment did not cover every sealed point",
    )
    assigned = np.empty(len(sealed), dtype=np.int64)
    assigned[matched_rows] = matched_columns
    _require(
        len(np.unique(assigned)) == len(assigned),
        "frame-zero identity assignment contains collisions",
    )
    assigned_distance = np.linalg.norm(official[assigned] - sealed, axis=1)
    _require(
        np.all(assigned_distance <= maximum_distance_m),
        "frame-zero assignment exceeds 15 mm",
    )
    diagnostics = {
        "algorithm": "scipy-sparse-minimum-weight-full-bipartite-matching",
        "scipy_version": str(scipy.__version__),
        "maximum_assignment_distance_m": maximum_distance_m,
        "candidate_edge_count": int(edge_count),
        "sealed_point_coverage_fraction": 1.0,
        "assigned_official_identity_collision_count": 0,
        "assigned_official_identity_count": int(len(assigned)),
        "official_identity_count": int(len(official)),
        "mean_assignment_distance_m": float(np.mean(assigned_distance)),
        "p95_assignment_distance_m": float(np.quantile(assigned_distance, 0.95)),
        "observed_maximum_assignment_distance_m": float(np.max(assigned_distance)),
        "assignment_ids_sha256": _sha256_array(assigned),
        "assignment_distances_sha256": _sha256_array(assigned_distance),
    }
    return assigned, assigned_distance, diagnostics


def transport_official_target(
    sealed_frame_zero_m: np.ndarray,
    target_value: OfficialTarget | Mapping[str, Any],
) -> TransportedTarget:
    """Transport a permitted official target onto sealed point identities."""

    target = normalize_official_target(target_value)
    eligible = (
        target.object_visibilities[0]
        & target.object_motions_valid[0]
        & np.all(np.isfinite(target.object_points[0]), axis=1)
    )
    eligible_ids = np.flatnonzero(eligible)
    _require(
        len(eligible_ids) >= len(sealed_frame_zero_m),
        "too few visible and valid official frame-zero identities",
    )
    local_ids, distances, diagnostics = sparse_min_cost_frame_zero_assignment(
        sealed_frame_zero_m,
        target.object_points[0, eligible_ids],
    )
    official_ids = eligible_ids[local_ids]
    points = target.object_points[:, official_ids].copy()
    visible = target.object_visibilities[:, official_ids].copy()
    valid = target.object_motions_valid[:, official_ids].copy()
    points[0] = np.asarray(sealed_frame_zero_m, dtype=points.dtype)
    _require(
        np.array_equal(points[0].astype(np.float32), np.asarray(sealed_frame_zero_m)),
        "transported frame zero is not bit-exact in float32",
    )
    diagnostics = {
        **diagnostics,
        "eligible_official_frame_zero_identity_count": int(len(eligible_ids)),
        "official_identity_ids_sha256": _sha256_array(official_ids),
        "raw_official_frame_zero_sha256": _sha256_array(target.object_points[0]),
        "sealed_frame_zero_sha256": _sha256_array(sealed_frame_zero_m),
        "transported_frame_zero_replaced_with_sealed_identity": True,
        "claim_limitation": (
            "one-to-one transported official reconstruction proxy; not native "
            "official material identity and not Deform360 Table-4 parity"
        ),
    }
    return TransportedTarget(
        object_points=points,
        object_visibilities=visible,
        object_motions_valid=valid,
        official_identity_ids=official_ids,
        assignment_distance_m=distances,
        diagnostics=diagnostics,
    )


def score_sealed_case(
    predictions: SealedCasePredictions,
    target_value: OfficialTarget | Mapping[str, Any],
) -> dict[str, Any]:
    """Score the primary and frozen comparator with no method selection."""

    _require(
        METRIC_LOCK.get("primary") == "post_update_hidden_symmetric_chamfer_m"
        and METRIC_LOCK.get("secondary") == "post_update_hidden_identity_rmse_m"
        and METRIC_LOCK.get("comparator") == "selected_raw_backbone"
        and METRIC_LOCK.get(
            "assimilation_centers_excluded_from_both_chamfer_directions"
        )
        is True
        and METRIC_LOCK.get("assimilation_centers_excluded_from_identity_metric")
        is True,
        "held metric lock changed",
    )
    transported = transport_official_target(
        predictions.frame_zero_points_m, target_value
    )
    frames = scored_frames()
    primary = score_deform360_hidden_trajectory(
        predictions.primary_prediction_m,
        transported.object_points,
        transported.object_visibilities,
        transported.object_motions_valid,
        center_ids=predictions.center_ids,
        scored_frames=frames,
    )
    comparator = score_deform360_hidden_trajectory(
        predictions.selected_raw_backbone_m,
        transported.object_points,
        transported.object_visibilities,
        transported.object_motions_valid,
        center_ids=predictions.center_ids,
        scored_frames=frames,
    )
    gate_score = {
        "primary_chamfer_m": float(primary["post_update_hidden_symmetric_chamfer_m"]),
        "comparator_chamfer_m": float(
            comparator["post_update_hidden_symmetric_chamfer_m"]
        ),
        "primary_identity_rmse_m": float(primary["post_update_hidden_identity_rmse_m"]),
        "comparator_identity_rmse_m": float(
            comparator["post_update_hidden_identity_rmse_m"]
        ),
    }
    return {
        "case_name": predictions.case_name,
        "gate_score": gate_score,
        "scored_frames": list(frames),
        "permanently_excluded_center_ids": predictions.center_ids.tolist(),
        "identity_transport": dict(transported.diagnostics),
        "scores": {
            "primary": primary,
            "selected_raw_backbone": comparator,
        },
        "sealed_inputs": dict(predictions.bindings),
        "method_selection_or_tuning_performed": False,
    }


def score_calibration_cohort(
    permit: OutcomePhasePermit,
    target_operations: Mapping[str, TargetOperation],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    """Score all 15 cases, opening each target only through its live permit."""

    _require(permit.role == "calibration", "calibration scorer requires its permit")
    _require(
        set(target_operations) == set(CALIBRATION_CASE_NAMES),
        "target operations must contain all 15 locked calibration cases",
    )
    gate_scores: dict[str, dict[str, float]] = {}
    records: dict[str, dict[str, Any]] = {}
    for case_name in CALIBRATION_CASE_NAMES:
        predictions = load_sealed_case_predictions(permit, case_name)
        operation = target_operations[case_name]
        _require(
            isinstance(operation, TargetOperation),
            f"invalid target operation for {case_name}",
        )
        target = run_outcome_operation(
            permit,
            case_name=case_name,
            operation=operation.operation,
            callback=operation.callback,
        )
        permitted_target = validate_permitted_target_provenance(
            target, permit, case_name
        )
        record = score_sealed_case(predictions, permitted_target)
        record["outcome_provenance"] = dict(permitted_target.provenance)
        gate_scores[case_name] = dict(record["gate_score"])
        records[case_name] = record
    return gate_scores, records


def calibration_score_evidence(
    permit: OutcomePhasePermit,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build canonical score evidence suitable for a later protocol binding."""

    _require(permit.role == "calibration", "score evidence requires calibration")
    _require(
        set(records) == set(CALIBRATION_CASE_NAMES),
        "score evidence must contain all 15 locked calibration cases",
    )
    ordered: dict[str, Mapping[str, Any]] = {}
    for case_name in CALIBRATION_CASE_NAMES:
        record = records[case_name]
        _require(record.get("case_name") == case_name, "score evidence case changed")
        _require(
            record.get("method_selection_or_tuning_performed") is False,
            "score evidence performed method selection",
        )
        gate_score = record.get("gate_score", {})
        _require(
            isinstance(gate_score, Mapping)
            and set(gate_score)
            == {
                "primary_chamfer_m",
                "comparator_chamfer_m",
                "primary_identity_rmse_m",
                "comparator_identity_rmse_m",
            },
            "score evidence gate fields changed",
        )
        ordered[case_name] = dict(record)
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": SCORE_EVIDENCE_KIND,
        "protocol_id": PROTOCOL_ID,
        "role": "calibration",
        "cohort_barrier_sha256": permit.cohort_barrier_sha256,
        "lock": {
            "path": str(Path(permit.lock_path).resolve()),
            "sha256": _sha256_file(permit.lock_path),
        },
        "ordered_case_names": list(CALIBRATION_CASE_NAMES),
        "metric_lock": dict(METRIC_LOCK),
        "case_records": ordered,
        "information_boundary": {
            "all_15_online_predictions_sealed_before_any_outcome": True,
            "outcomes_opened_only_through_live_permit": True,
            "method_selection_or_tuning_performed": False,
            "confirmation_payload_read": False,
        },
    }
    # Also verifies that every nested value is finite JSON data.
    json.dumps(artifact, sort_keys=True, allow_nan=False)
    artifact["artifact_sha256"] = held_artifact_sha256(artifact)
    return artifact


def write_calibration_score_evidence(
    output_path: str | Path,
    permit: OutcomePhasePermit,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist immutable canonical evidence before the GO/NO-GO decision."""

    artifact = calibration_score_evidence(permit, records)
    _write_new_json(output_path, artifact)
    return artifact


def score_and_create_calibration_gate(
    decision_path: str | Path,
    permit: OutcomePhasePermit,
    target_operations: Mapping[str, TargetOperation],
    *,
    evidence_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Run the frozen scorer and feed its exact records to the frozen gate."""

    gate_scores, records = score_calibration_cohort(permit, target_operations)
    destination = (
        Path(evidence_path)
        if evidence_path is not None
        else Path(decision_path).with_name("calibration-score-evidence.json")
    )
    evidence = write_calibration_score_evidence(destination, permit, records)
    decision = create_calibration_gate_decision(decision_path, permit, gate_scores)
    return decision, evidence, records


__all__ = [
    "MAXIMUM_FRAME_ZERO_ASSIGNMENT_DISTANCE_M",
    "OfficialTarget",
    "OUTCOME_ARTIFACT_KIND",
    "SCORE_EVIDENCE_KIND",
    "SealedCasePredictions",
    "TargetOperation",
    "TARGET_ARTIFACT_KIND",
    "TransportedTarget",
    "calibration_score_evidence",
    "load_sealed_case_predictions",
    "normalize_official_target",
    "official_target_array_sha256",
    "score_and_create_calibration_gate",
    "score_calibration_cohort",
    "score_sealed_case",
    "scored_frames",
    "sparse_min_cost_frame_zero_assignment",
    "transport_official_target",
    "validate_permitted_target_provenance",
    "write_calibration_score_evidence",
]
