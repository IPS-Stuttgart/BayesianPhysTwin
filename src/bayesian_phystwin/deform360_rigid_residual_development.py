"""Open27 ablation of total displacement versus rigid-plus-residual fields.

This module is development-only.  It reuses the audited open27 loader, the
permanent center/anchor split, and the scoring contract from
``deform360_query_field_development``.  It has no held-protocol imports and
cannot consume a held outcome.

For each arm and frame, the ablation fits a proper Kabsch transform from the
frame-zero field anchors to their predicted positions.  The transform is
evaluated at each query and a Gaussian fixed-k interpolation of the remaining
anchor residual is added.  Exact anchor queries bypass the decoder and return
the corresponding nodal trajectory bit-exactly.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import deform360_query_field_development as query_development
from .deform360_frozen_query_field import (
    FrameZeroQuerySet,
    FrozenFieldGeometry,
    FrozenNodalDisplacementField,
    query_frozen_nodal_field,
)
from .deform360_online_belief_evaluation import (
    CENTER_COUNT,
    EXPECTED_SOURCE_EPISODES,
    PROTOCOL_ID as SOURCE_PROTOCOL_ID,
    _post_update_scored_frames,
)


PROTOCOL_ID = "deform360-open27-rigid-residual-field-v1-development"
ARTIFACT_KIND = "Deform360Open27RigidResidualFieldDevelopmentAblation"
TOTAL_OPERATOR_ID = "total-displacement-gaussian-v1"
RIGID_RESIDUAL_OPERATOR_ID = "proper-kabsch-plus-gaussian-residual-v1"
OPERATOR_IDS = (TOTAL_OPERATOR_ID, RIGID_RESIDUAL_OPERATOR_ID)
_DETERMINANT_TOLERANCE = 1e-10


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fit_proper_kabsch_transform(
    frame_zero_anchor_positions_m: np.ndarray,
    predicted_anchor_positions_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``y = R x + t`` with ``det(R)=+1`` in float64.

    Returned points use row-vector storage, so callers evaluate the transform
    as ``points @ rotation.T + translation``.
    """

    source = np.asarray(frame_zero_anchor_positions_m, dtype=np.float64)
    target = np.asarray(predicted_anchor_positions_m, dtype=np.float64)
    _require(
        source.shape == target.shape
        and source.ndim == 2
        and source.shape[1] == 3
        and len(source) >= 3,
        "Kabsch source and target must share shape (N, 3) with N >= 3",
    )
    _require(
        np.all(np.isfinite(source)) and np.all(np.isfinite(target)),
        "Kabsch source and target must be finite",
    )
    if np.array_equal(source, target):
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    source_center = np.mean(source, axis=0, dtype=np.float64)
    target_center = np.mean(target, axis=0, dtype=np.float64)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transpose = np.linalg.svd(covariance, full_matrices=False)
    rotation = right_transpose.T @ left.T
    if float(np.linalg.det(rotation)) < 0.0:
        right_transpose = right_transpose.copy()
        right_transpose[-1] *= -1.0
        rotation = right_transpose.T @ left.T
    determinant = float(np.linalg.det(rotation))
    _require(
        determinant > 0.0 and abs(determinant - 1.0) <= _DETERMINANT_TOLERANCE,
        "Kabsch fit did not produce a proper rotation",
    )
    translation = target_center - rotation @ source_center
    return rotation, translation


def query_proper_kabsch_residual_trajectory(
    nodal_trajectory_m: np.ndarray,
    frame_zero_anchor_positions_m: np.ndarray,
    query_positions_m: np.ndarray,
    neighbor_anchor_indices: np.ndarray,
    neighbor_weights: np.ndarray,
    exact_anchor_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one nodal arm with a proper-rigid base and local residual."""

    nodal = np.asarray(nodal_trajectory_m)
    anchors = np.asarray(frame_zero_anchor_positions_m)
    queries = np.asarray(query_positions_m)
    neighbors = np.asarray(neighbor_anchor_indices)
    weights = np.asarray(neighbor_weights)
    exact = np.asarray(exact_anchor_indices)
    _require(
        nodal.dtype == anchors.dtype == queries.dtype == np.dtype(np.float32),
        "rigid-residual trajectories, anchors, and queries must have dtype float32",
    )
    _require(
        nodal.ndim == 3
        and nodal.shape[1:] == anchors.shape
        and anchors.ndim == 2
        and anchors.shape[1] == 3
        and len(nodal) > 0
        and len(anchors) >= 3,
        "rigid-residual nodal inputs have incompatible shapes",
    )
    _require(
        queries.ndim == 2 and queries.shape[1] == 3 and len(queries) > 0,
        "rigid-residual queries must have nonempty shape (M, 3)",
    )
    _require(
        neighbors.dtype == np.dtype(np.int64)
        and neighbors.ndim == 2
        and neighbors.shape[0] == len(queries)
        and neighbors.shape == weights.shape,
        "rigid-residual neighbor indices and weights differ in shape",
    )
    _require(
        exact.dtype == np.dtype(np.int64) and exact.shape == (len(queries),),
        "exact-anchor indices must have shape (M,) and dtype int64",
    )
    _require(
        np.all((0 <= neighbors) & (neighbors < len(anchors)))
        and np.all((-1 <= exact) & (exact < len(anchors))),
        "rigid-residual anchor index is out of range",
    )
    _require(
        np.all(np.isfinite(nodal))
        and np.all(np.isfinite(anchors))
        and np.all(np.isfinite(queries))
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.allclose(
            np.sum(weights, axis=1, dtype=np.float64),
            1.0,
            rtol=0.0,
            atol=1e-15,
        ),
        "rigid-residual inputs or normalized weights are invalid",
    )
    _require(
        np.array_equal(nodal[0], anchors),
        "rigid-residual nodal trajectory must equal anchors at frame zero",
    )

    anchors64 = anchors.astype(np.float64)
    queries64 = queries.astype(np.float64)
    output = np.empty((len(nodal), len(queries), 3), dtype=np.float32)
    determinants = np.empty(len(nodal), dtype=np.float64)
    for frame, predicted in enumerate(nodal):
        rotation, translation = fit_proper_kabsch_transform(anchors, predicted)
        determinants[frame] = np.linalg.det(rotation)
        rigid_anchors = anchors64 @ rotation.T + translation
        rigid_queries = queries64 @ rotation.T + translation
        anchor_residual = predicted.astype(np.float64) - rigid_anchors
        interpolated_residual = np.sum(
            anchor_residual[neighbors] * weights[:, :, None],
            axis=1,
            dtype=np.float64,
        )
        output[frame] = (rigid_queries + interpolated_residual).astype(np.float32)

    for query_index, anchor_index in enumerate(exact):
        if anchor_index >= 0:
            output[:, query_index] = nodal[:, anchor_index]
    output.setflags(write=False)
    determinants.setflags(write=False)
    return output, determinants


def _local_neighbor_indices(
    anchor_ids: np.ndarray,
    neighbor_anchor_ids: np.ndarray,
) -> np.ndarray:
    anchors = np.asarray(anchor_ids, dtype=np.int64)
    neighbors = np.asarray(neighbor_anchor_ids, dtype=np.int64)
    local = np.searchsorted(anchors, neighbors)
    _require(
        np.all(local < len(anchors)) and np.array_equal(anchors[local], neighbors),
        "queried neighbor is absent from the sorted field anchors",
    )
    return local.astype(np.int64, copy=False)


def _exact_anchor_indices(
    anchor_ids: np.ndarray,
    exact_anchor_mask: np.ndarray,
    nearest_anchor_ids: np.ndarray,
) -> np.ndarray:
    anchors = np.asarray(anchor_ids, dtype=np.int64)
    exact_mask = np.asarray(exact_anchor_mask, dtype=bool)
    nearest = np.asarray(nearest_anchor_ids, dtype=np.int64)
    result = np.full(len(exact_mask), -1, dtype=np.int64)
    if np.any(exact_mask):
        result[exact_mask] = np.searchsorted(anchors, nearest[exact_mask])
        _require(
            np.array_equal(anchors[result[exact_mask]], nearest[exact_mask]),
            "exact queried anchor is absent from the field",
        )
    return result


def _score_operator(
    primary_prediction_m: np.ndarray,
    comparator_prediction_m: np.ndarray,
    primary_native_m: np.ndarray,
    comparator_native_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    query_ids: np.ndarray,
    supported_query_mask: np.ndarray,
    *,
    scored_frames: Sequence[int],
) -> dict[str, object]:
    support = np.asarray(supported_query_mask, dtype=bool)
    query = np.asarray(query_ids, dtype=np.int64)
    fidelity_available = np.broadcast_to(
        support[None], (len(target_m), len(query))
    ).copy()
    target_available = (
        np.asarray(visibility, dtype=bool)[:, query]
        & np.asarray(validity, dtype=bool)[:, query]
        & np.all(np.isfinite(target_m[:, query]), axis=2)
        & fidelity_available
    )
    primary_target = query_development._trajectory_metrics(
        primary_prediction_m,
        target_m[:, query],
        target_available,
        scored_frames=scored_frames,
    )
    comparator_target = query_development._trajectory_metrics(
        comparator_prediction_m,
        target_m[:, query],
        target_available,
        scored_frames=scored_frames,
    )
    primary_fidelity = query_development._trajectory_metrics(
        primary_prediction_m,
        primary_native_m[:, query],
        fidelity_available,
        scored_frames=scored_frames,
    )
    comparator_fidelity = query_development._trajectory_metrics(
        comparator_prediction_m,
        comparator_native_m[:, query],
        fidelity_available,
        scored_frames=scored_frames,
    )
    return {
        "target_scores": {
            "primary": primary_target,
            "comparator": comparator_target,
            "shared_mask": (
                "fixed support AND future visibility AND future motion-validity"
            ),
        },
        "field_native_fidelity": {
            "primary": primary_fidelity,
            "comparator": comparator_fidelity,
            "equal_arm_identity_rmse_m": 0.5
            * (
                float(primary_fidelity["identity_rmse_m"])
                + float(comparator_fidelity["identity_rmse_m"])
            ),
            "shared_mask": "fixed geometric support; no future target value or mask",
        },
    }


def _metric_deltas(
    rigid_result: Mapping[str, Any],
    total_result: Mapping[str, Any],
) -> dict[str, float]:
    rigid = query_development._flatten_result(rigid_result)
    total = query_development._flatten_result(total_result)
    return {
        f"rigid_minus_total_{metric}": float(rigid[metric] - total[metric])
        for metric in query_development._AGGREGATE_METRICS
    }


def evaluate_rigid_residual_case_arrays(
    primary_native_m: np.ndarray,
    comparator_native_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    center_ids: np.ndarray,
    *,
    anchor_count: int,
    candidate: Any,
    scored_frames: Sequence[int],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Evaluate both field operators on one frozen center/anchor/query split."""

    _require(
        candidate.operator_id == "gaussian-knn-normalized-v1",
        "rigid-residual ablation accepts only the frozen Gaussian candidates",
    )
    total_result, selected = query_development.evaluate_query_field_case_arrays(
        primary_native_m,
        comparator_native_m,
        target_m,
        visibility,
        validity,
        center_ids,
        anchor_count=anchor_count,
        candidate=candidate,
        scored_frames=scored_frames,
    )
    anchor_ids = selected["anchor_ids"]
    query_ids = selected["query_ids"]
    object_scale = query_development._object_scale_m(comparator_native_m[0])
    config = candidate.config(object_scale)
    geometry = FrozenFieldGeometry(
        anchor_ids=anchor_ids,
        anchor_positions_m=comparator_native_m[0, anchor_ids],
        assimilation_anchor_ids=np.empty(0, dtype=np.int64),
    )
    field = FrozenNodalDisplacementField(
        geometry=geometry,
        primary_nodal_trajectory_m=primary_native_m[:, anchor_ids],
        comparator_nodal_trajectory_m=comparator_native_m[:, anchor_ids],
        config=config,
    )
    queries = FrameZeroQuerySet(
        identity_ids=query_ids,
        positions_m=target_m[0, query_ids],
    )
    total_query = query_frozen_nodal_field(field, queries)
    _require(
        np.array_equal(
            total_query.supported_identity_mask,
            selected["supported_query_mask"],
        ),
        "reconstructed Gaussian query changed the frozen support mask",
    )
    neighbor_local = _local_neighbor_indices(
        anchor_ids, total_query.neighbor_anchor_ids
    )
    exact_local = _exact_anchor_indices(
        anchor_ids,
        total_query.exact_anchor_mask,
        total_query.nearest_anchor_ids,
    )
    primary_rigid, primary_determinants = query_proper_kabsch_residual_trajectory(
        field.primary_nodal_trajectory_m,
        geometry.anchor_positions_m,
        queries.positions_m,
        neighbor_local,
        total_query.neighbor_weights,
        exact_local,
    )
    comparator_rigid, comparator_determinants = query_proper_kabsch_residual_trajectory(
        field.comparator_nodal_trajectory_m,
        geometry.anchor_positions_m,
        queries.positions_m,
        neighbor_local,
        total_query.neighbor_weights,
        exact_local,
    )
    rigid_result = _score_operator(
        primary_rigid,
        comparator_rigid,
        primary_native_m,
        comparator_native_m,
        target_m,
        visibility,
        validity,
        query_ids,
        total_query.supported_identity_mask,
        scored_frames=scored_frames,
    )
    rigid_result.update(
        {
            "candidate": candidate.descriptor(),
            "resolved_config_m": copy.deepcopy(total_result["resolved_config_m"]),
            "geometry": copy.deepcopy(total_result["geometry"]),
            "proper_kabsch": {
                "fit": "per arm and frame on frame-zero anchors",
                "row_vector_evaluation": "x @ R.T + translation",
                "reflection_policy": "flip final right-singular vector",
                "primary_determinant": {
                    "minimum": float(np.min(primary_determinants)),
                    "maximum": float(np.max(primary_determinants)),
                },
                "comparator_determinant": {
                    "minimum": float(np.min(comparator_determinants)),
                    "maximum": float(np.max(comparator_determinants)),
                },
                "exact_anchor_bypass_count": int(np.sum(exact_local >= 0)),
            },
        }
    )
    result = {
        "candidate": candidate.descriptor(),
        "operators": {
            TOTAL_OPERATOR_ID: total_result,
            RIGID_RESIDUAL_OPERATOR_ID: rigid_result,
        },
        "matched_comparison": _metric_deltas(rigid_result, total_result),
    }
    return result, selected


def _gaussian_candidates() -> tuple[Any, ...]:
    candidates = tuple(
        value
        for value in query_development._candidate_grid()
        if value.operator_id == "gaussian-knn-normalized-v1"
    )
    _require(len(candidates) == 9, "frozen Gaussian k/f candidate grid changed")
    return candidates


def _target_superiority(
    records: Sequence[tuple[str, str, Mapping[str, float]]],
) -> dict[str, object]:
    _require(bool(records), "cannot compare target scores for an empty panel")
    primary_identity = np.asarray(
        [value["primary_target_identity_rmse_m"] for _, _, value in records],
        dtype=np.float64,
    )
    comparator_identity = np.asarray(
        [value["comparator_target_identity_rmse_m"] for _, _, value in records],
        dtype=np.float64,
    )
    primary_chamfer = np.asarray(
        [value["primary_target_symmetric_chamfer_m"] for _, _, value in records],
        dtype=np.float64,
    )
    comparator_chamfer = np.asarray(
        [value["comparator_target_symmetric_chamfer_m"] for _, _, value in records],
        dtype=np.float64,
    )
    identity_wins = primary_identity < comparator_identity
    chamfer_wins = primary_chamfer < comparator_chamfer
    return {
        "case_count": len(records),
        "primary_identity_win_count": int(np.sum(identity_wins)),
        "primary_chamfer_win_count": int(np.sum(chamfer_wins)),
        "primary_dual_metric_win_count": int(np.sum(identity_wins & chamfer_wins)),
    }


def _aggregate_operator_records(
    records: Mapping[str, Mapping[int, Sequence[tuple[str, str, Mapping[str, float]]]]],
    candidates: Sequence[Any],
) -> tuple[dict[str, object], dict[str, object]]:
    aggregates: dict[str, object] = {}
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selection_rows = []
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        by_anchor: dict[str, object] = {}
        for anchor_count in query_development.ANCHOR_COUNTS:
            values = records[candidate_id][anchor_count]
            aggregate = query_development._aggregate_records(values)
            aggregate["primary_vs_comparator_target"] = _target_superiority(values)
            by_anchor[str(anchor_count)] = aggregate
        across_anchor = {
            metric: float(
                np.mean(
                    [
                        by_anchor[str(anchor_count)]["equal_object_mean"][metric]
                        for anchor_count in query_development.ANCHOR_COUNTS
                    ]
                )
            )
            for metric in query_development._AGGREGATE_METRICS
        }
        aggregates[candidate_id] = {
            "candidate": candidate.descriptor(),
            "by_anchor_count": by_anchor,
            "equal_anchor_count_mean_of_equal_object_means": across_anchor,
        }
        selection_rows.append(
            {
                "candidate_id": candidate_id,
                "selection_objective_m": across_anchor[
                    query_development.SELECTION_METRIC
                ],
            }
        )
    ranking = query_development._rank_with_tolerance(
        selection_rows,
        value_key="selection_objective_m",
        candidates=candidate_by_id,
    )
    selected_id = str(ranking[0]["candidate_id"])
    selected_aggregate = aggregates[selected_id]
    selected_across_anchor = selected_aggregate[
        "equal_anchor_count_mean_of_equal_object_means"
    ]
    primary_identity = float(selected_across_anchor["primary_target_identity_rmse_m"])
    comparator_identity = float(
        selected_across_anchor["comparator_target_identity_rmse_m"]
    )
    primary_chamfer = float(
        selected_across_anchor["primary_target_symmetric_chamfer_m"]
    )
    comparator_chamfer = float(
        selected_across_anchor["comparator_target_symmetric_chamfer_m"]
    )
    selection = {
        "metric": query_development.SELECTION_METRIC,
        "rule": (
            "minimize equal-object equal-arm field-vs-native identity RMSE, "
            "averaged equally across A=64,128,256"
        ),
        "ranking": ranking,
        "selected_candidate_id": selected_id,
        "selected_config": candidate_by_id[selected_id].descriptor(),
        "selected_objective_m": float(ranking[0]["selection_objective_m"]),
        "future_target_scores_used_for_selection": False,
        "future_target_masks_used_for_selection": False,
        "selected_descriptive_metrics": copy.deepcopy(selected_across_anchor),
        "selected_primary_vs_comparator_target": {
            "status": "descriptive after fidelity-only selection",
            "identity_rmse_relative_reduction": float(
                (comparator_identity - primary_identity) / comparator_identity
            ),
            "symmetric_chamfer_relative_reduction": float(
                (comparator_chamfer - primary_chamfer) / comparator_chamfer
            ),
            "by_anchor_count_case_wins": {
                str(anchor_count): copy.deepcopy(
                    selected_aggregate["by_anchor_count"][str(anchor_count)][
                        "primary_vs_comparator_target"
                    ]
                )
                for anchor_count in query_development.ANCHOR_COUNTS
            },
        },
    }
    return aggregates, selection


def _matched_aggregate_comparisons(
    total: Mapping[str, Any],
    rigid: Mapping[str, Any],
    candidates: Sequence[Any],
) -> dict[str, object]:
    by_cell: dict[str, object] = {}
    delta_values = {metric: [] for metric in query_development._AGGREGATE_METRICS}
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        for anchor_count in query_development.ANCHOR_COUNTS:
            total_values = total[candidate_id]["by_anchor_count"][str(anchor_count)][
                "equal_object_mean"
            ]
            rigid_values = rigid[candidate_id]["by_anchor_count"][str(anchor_count)][
                "equal_object_mean"
            ]
            deltas = {
                f"rigid_minus_total_{metric}": float(
                    rigid_values[metric] - total_values[metric]
                )
                for metric in query_development._AGGREGATE_METRICS
            }
            for metric in query_development._AGGREGATE_METRICS:
                delta_values[metric].append(
                    float(rigid_values[metric] - total_values[metric])
                )
            by_cell[f"A{anchor_count}-{candidate_id}"] = {
                "anchor_count": anchor_count,
                "candidate_id": candidate_id,
                **deltas,
            }
    summary = {}
    for metric, values in delta_values.items():
        array = np.asarray(values, dtype=np.float64)
        summary[metric] = {
            "cell_count": len(array),
            "rigid_better_cell_count": int(np.sum(array < 0.0)),
            "mean_rigid_minus_total_m": float(np.mean(array)),
            "median_rigid_minus_total_m": float(np.median(array)),
            "minimum_rigid_minus_total_m": float(np.min(array)),
            "maximum_rigid_minus_total_m": float(np.max(array)),
        }
    return {"by_cell": by_cell, "summary": summary}


def _pure_rigid_sanity() -> dict[str, object]:
    rng = np.random.default_rng(731)
    anchors = rng.normal(size=(32, 3)).astype(np.float32)
    extra_queries = rng.normal(size=(19, 3)).astype(np.float32)
    queries = np.concatenate((anchors[:4], extra_queries), axis=0)
    axis = np.asarray([0.3, -0.5, 0.8], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    rotation = (
        np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    )
    translation = np.asarray([0.4, -0.2, 0.7], dtype=np.float64)
    predicted = (anchors.astype(np.float64) @ rotation.T + translation).astype(
        np.float32
    )
    truth = (queries.astype(np.float64) @ rotation.T + translation).astype(np.float32)
    nodal = np.stack((anchors, predicted))

    squared_distance = np.sum(
        np.square(
            queries.astype(np.float64)[:, None] - anchors.astype(np.float64)[None]
        ),
        axis=2,
    )
    anchor_ids = np.arange(len(anchors), dtype=np.int64)
    neighbors = np.empty((len(queries), 4), dtype=np.int64)
    weights = np.empty((len(queries), 4), dtype=np.float64)
    exact = np.full(len(queries), -1, dtype=np.int64)
    length_scale = 0.1 * query_development._object_scale_m(anchors)
    for query_index, distances in enumerate(squared_distance):
        order = np.lexsort((anchor_ids, distances))[:4]
        neighbors[query_index] = order
        relative = np.maximum(distances[order] - distances[order[0]], 0.0)
        raw = np.exp(-relative / (2.0 * length_scale**2))
        weights[query_index] = raw / np.sum(raw)
        if query_index < 4:
            exact[query_index] = query_index
            weights[query_index] = 0.0
            weights[query_index, int(np.flatnonzero(order == query_index)[0])] = 1.0
    output, determinants = query_proper_kabsch_residual_trajectory(
        nodal,
        anchors,
        queries,
        neighbors,
        weights,
        exact,
    )
    maximum_query_error = float(
        np.max(np.abs(output[1].astype(np.float64) - truth.astype(np.float64)))
    )
    exact_anchor_equal = bool(np.array_equal(output[:, :4], nodal[:, :4]))
    return {
        "seed": 731,
        "maximum_query_coordinate_error_m": maximum_query_error,
        "exact_anchor_trajectory_bit_exact": exact_anchor_equal,
        "fitted_determinant": {
            "minimum": float(np.min(determinants)),
            "maximum": float(np.max(determinants)),
        },
        "pass_tolerance_m": 1e-6,
        "passed": bool(maximum_query_error <= 1e-6 and exact_anchor_equal),
    }


def build_rigid_residual_development_ablation(
    source_root: str | Path,
    audited_run_dir: str | Path,
) -> dict[str, object]:
    """Build the deterministic 27-cell open27 ablation entirely in memory."""

    source = Path(source_root).resolve()
    run_dir = Path(audited_run_dir).resolve()
    _require(source.is_dir(), f"source root is not a directory: {source}")
    _require(run_dir.is_dir(), f"audited run is not a directory: {run_dir}")
    expected = tuple(sorted(query_development._expected_case_names()))
    observed_source = tuple(
        sorted(path.name for path in source.iterdir() if path.is_dir())
    )
    _require(
        observed_source == expected,
        "source root is not exactly the fixed open27 episode whitelist",
    )
    summary_path = run_dir / "summary.json"
    _, artifact_records = query_development._validate_summary(
        summary_path, source, run_dir
    )
    candidates = _gaussian_candidates()
    records: dict[
        str,
        dict[str, dict[int, list[tuple[str, str, dict[str, float]]]]],
    ] = {
        operator: {
            candidate.candidate_id: {
                count: [] for count in query_development.ANCHOR_COUNTS
            }
            for candidate in candidates
        }
        for operator in OPERATOR_IDS
    }
    cases: dict[str, object] = {}
    input_hashes: dict[str, object] = {}
    determinant_minimum = np.inf
    determinant_maximum = -np.inf
    for case_name in expected:
        metadata, loaded = query_development._load_audited_case(
            source,
            run_dir,
            case_name,
            artifact_records[case_name],
        )
        input_hashes[case_name] = metadata["input_hashes"]
        case_record: dict[str, object] = {
            "object_id": metadata["object_id"],
            "episode_id": metadata["episode_id"],
            "object_scale_m": query_development._object_scale_m(
                loaded["comparator"][0]
            ),
            "anchor_counts": {},
        }
        for anchor_count in query_development.ANCHOR_COUNTS:
            anchor_record: dict[str, object] = {"candidates": {}}
            common_arrays: Mapping[str, np.ndarray] | None = None
            for candidate in candidates:
                result, selected = evaluate_rigid_residual_case_arrays(
                    loaded["primary"],
                    loaded["comparator"],
                    loaded["target"],
                    loaded["visibility"],
                    loaded["validity"],
                    loaded["centers"],
                    anchor_count=anchor_count,
                    candidate=candidate,
                    scored_frames=metadata["scored_frames"],
                )
                if common_arrays is None:
                    common_arrays = selected
                    anchor_record.update(
                        {
                            "anchor_count": anchor_count,
                            "query_count": len(selected["query_ids"]),
                            "anchor_ids": selected["anchor_ids"].tolist(),
                            "anchor_ids_sha256": query_development._sha256_array(
                                selected["anchor_ids"]
                            ),
                            "query_identity_ids_sha256": (
                                query_development._sha256_array(selected["query_ids"])
                            ),
                            "assimilation_center_ids_sha256": (
                                query_development._sha256_array(loaded["centers"])
                            ),
                        }
                    )
                else:
                    for key in ("anchor_ids", "query_ids", "supported_query_mask"):
                        _require(
                            np.array_equal(selected[key], common_arrays[key]),
                            "candidate changed the frozen split or support mask",
                        )
                rigid_kabsch = result["operators"][RIGID_RESIDUAL_OPERATOR_ID][
                    "proper_kabsch"
                ]
                for arm in ("primary_determinant", "comparator_determinant"):
                    determinant_minimum = min(
                        determinant_minimum, float(rigid_kabsch[arm]["minimum"])
                    )
                    determinant_maximum = max(
                        determinant_maximum, float(rigid_kabsch[arm]["maximum"])
                    )
                anchor_record["candidates"][candidate.candidate_id] = result
                for operator in OPERATOR_IDS:
                    flat = query_development._flatten_result(
                        result["operators"][operator]
                    )
                    records[operator][candidate.candidate_id][anchor_count].append(
                        (
                            case_name,
                            str(metadata["object_id"]),
                            flat,
                        )
                    )
            case_record["anchor_counts"][str(anchor_count)] = anchor_record
        cases[case_name] = case_record

    aggregates: dict[str, object] = {}
    selections: dict[str, object] = {}
    for operator in OPERATOR_IDS:
        aggregates[operator], selections[operator] = _aggregate_operator_records(
            records[operator], candidates
        )
    total_objective = float(selections[TOTAL_OPERATOR_ID]["selected_objective_m"])
    rigid_objective = float(
        selections[RIGID_RESIDUAL_OPERATOR_ID]["selected_objective_m"]
    )
    selected_operator = (
        RIGID_RESIDUAL_OPERATOR_ID
        if rigid_objective < total_objective
        else TOTAL_OPERATOR_ID
    )
    sanity = _pure_rigid_sanity()
    _require(bool(sanity["passed"]), "pure-rigid Kabsch sanity check failed")
    decision: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "inputs": {
            "source_root": str(source),
            "audited_run_dir": str(run_dir),
            "audited_summary": {
                "path": str(summary_path),
                "sha256": query_development._sha256(summary_path),
            },
            "case_artifact_hashes": input_hashes,
        },
        "fixed_protocol": {
            "physical_objects": {
                key: list(value) for key, value in EXPECTED_SOURCE_EPISODES.items()
            },
            "case_count": len(expected),
            "assimilation_center_count": CENTER_COUNT,
            "assimilation_centers": "audited 16 IDs; excluded before anchor FPS",
            "anchor_counts": list(query_development.ANCHOR_COUNTS),
            "candidate_count_per_anchor": len(candidates),
            "matched_cell_count": len(candidates)
            * len(query_development.ANCHOR_COUNTS),
            "gaussian_neighbor_counts": list(
                query_development.GAUSSIAN_NEIGHBOR_COUNTS
            ),
            "gaussian_length_scale_fractions": list(
                query_development.GAUSSIAN_LENGTH_SCALE_FRACTIONS
            ),
            "object_scale": {
                "rule": "5th-to-95th percentile frame-zero bbox diagonal",
                "quantile_method": "linear",
            },
            "support_radius_fraction": query_development.SUPPORT_RADIUS_FRACTION,
            "anchor_selection": (
                "deterministic frame-zero FPS over non-center identities, then "
                "sort selected IDs"
            ),
            "query_identities": "all permanent non-center, non-anchor identities",
            "scored_frames": list(_post_update_scored_frames(76)),
            "operators": {
                TOTAL_OPERATOR_ID: (
                    "normalized Gaussian fixed-k interpolation of total nodal "
                    "displacement"
                ),
                RIGID_RESIDUAL_OPERATOR_ID: (
                    "per-arm/per-frame proper Kabsch base plus the identical "
                    "Gaussian interpolation of anchor residuals"
                ),
            },
            "exact_anchor_rule": "bit-exact nodal trajectory bypass",
            "primary_native_arm": query_development.PRIMARY_ARM,
            "comparator_native_arm": query_development.COMPARATOR_ARM,
            "target_score_mask": (
                "fixed support AND future visibility AND future validity; shared "
                "by arms and operators"
            ),
            "fidelity_mask": "fixed geometric support only",
            "selection_uses": "field-vs-native fidelity only",
            "selection_excludes": "all future target coordinates and masks",
        },
        "candidate_grid": [candidate.descriptor() for candidate in candidates],
        "pure_rigid_sanity": sanity,
        "open27_proper_rotation_determinants": {
            "minimum": float(determinant_minimum),
            "maximum": float(determinant_maximum),
        },
        "case_results": cases,
        "aggregates": aggregates,
        "selection_by_operator": selections,
        "matched_operator_comparison": _matched_aggregate_comparisons(
            aggregates[TOTAL_OPERATOR_ID],
            aggregates[RIGID_RESIDUAL_OPERATOR_ID],
            candidates,
        ),
        "operator_decision": {
            "metric": query_development.SELECTION_METRIC,
            "total_selected_objective_m": total_objective,
            "rigid_residual_selected_objective_m": rigid_objective,
            "rigid_minus_total_selected_objective_m": (
                rigid_objective - total_objective
            ),
            "rigid_minus_total_selected_objective_fraction": float(
                (rigid_objective - total_objective) / total_objective
            ),
            "selected_operator_id": selected_operator,
            "rigid_residual_improves_fidelity_selection": bool(
                rigid_objective < total_objective
            ),
        },
        "claim_boundary": (
            "post-hoc development ablation on the already-open audited "
            "independent-source Deform360-27 panel; not held evidence, not the "
            "native Deform360 evaluator, and not a state-of-the-art claim"
        ),
    }
    json.dumps(decision, sort_keys=True, allow_nan=False)
    return decision


def write_rigid_residual_development_ablation(
    source_root: str | Path,
    audited_run_dir: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Build and exclusively write a strict deterministic JSON artifact."""

    decision = build_rigid_residual_development_ablation(source_root, audited_run_dir)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return decision


__all__ = [
    "ARTIFACT_KIND",
    "PROTOCOL_ID",
    "RIGID_RESIDUAL_OPERATOR_ID",
    "TOTAL_OPERATOR_ID",
    "build_rigid_residual_development_ablation",
    "evaluate_rigid_residual_case_arrays",
    "fit_proper_kabsch_transform",
    "query_proper_kabsch_residual_trajectory",
    "write_rigid_residual_development_ablation",
]
