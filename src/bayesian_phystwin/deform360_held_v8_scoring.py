"""Pure direct-identity scoring for the prospective Deform360 held v8.

This module consumes predictions which have already been queried at the
official frame-zero identities.  It deliberately has no artifact I/O,
outcome reconstruction, identity transport, assignment, or field-decoder
dependency.  Both prediction arms are scored on one shared, predeclared mask.

The optional source-node count is reporting metadata only.  In particular,
the scorer neither requires nor assumes any ordering or cardinality relation
between the source nodes used by the frozen field and the official identities
on which that field was queried.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np


PROTOCOL_ID = "deform360-held-online-belief-v8.3"
SCORER_ID = "deform360-held-v8-direct-official-identity-v1"
FRAME_COUNT = 76
CENTER_COUNT = 16
SCORED_FRAME_INTERVALS_HALF_OPEN = ((20, 38), (39, 57), (58, 76))
SCORED_FRAMES = tuple(
    frame
    for start, stop in SCORED_FRAME_INTERVALS_HALF_OPEN
    for frame in range(start, stop)
)

MINIMUM_SUPPORT_COVERAGE_FRACTION = 0.90
MINIMUM_HIDDEN_SUPPORTED_IDENTITY_COUNT = 32
MINIMUM_MEAN_CHAMFER_IMPROVEMENT_FRACTION = 0.05
MAXIMUM_CASE_CHAMFER_REGRESSION_FRACTION = 0.10
CALIBRATION_CASE_COUNT = 15
CALIBRATION_MINIMUM_CHAMFER_WINS = 10
CONFIRMATION_CASE_COUNT = 6
CONFIRMATION_REQUIRED_CHAMFER_WINS = 6
CONFIRMATION_ONE_SIDED_SIGN_TEST_P = 1.0 / 64.0

_SCORE_KEYS = (
    "primary_chamfer_m",
    "comparator_chamfer_m",
    "primary_identity_rmse_m",
    "comparator_identity_rmse_m",
)
_NEAREST_DISTANCE_CHUNK_SIZE = 128


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    descriptor = f"{array.dtype.str}:{','.join(map(str, array.shape))}".encode()
    return hashlib.sha256(descriptor + b"\0" + array.tobytes()).hexdigest()


def _bit_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return (
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and np.ascontiguousarray(left_array).tobytes()
        == np.ascontiguousarray(right_array).tobytes()
    )


def _mean_nearest_euclidean_distance_m(
    source_m: np.ndarray,
    destination_m: np.ndarray,
) -> float:
    """Mean source-to-nearest-destination Euclidean distance."""

    source = np.asarray(source_m, dtype=np.float64)
    destination = np.asarray(destination_m, dtype=np.float64)
    _require(
        source.ndim == destination.ndim == 2
        and source.shape[1:] == destination.shape[1:] == (3,)
        and len(source) > 0
        and len(destination) > 0,
        "nearest-distance inputs must have nonempty shape (N, 3)",
    )
    _require(
        np.all(np.isfinite(source)) and np.all(np.isfinite(destination)),
        "nearest-distance inputs must be finite",
    )
    nearest: list[float] = []
    for start in range(0, len(source), _NEAREST_DISTANCE_CHUNK_SIZE):
        chunk = source[start : start + _NEAREST_DISTANCE_CHUNK_SIZE]
        squared_distance = np.sum(
            np.square(chunk[:, None, :] - destination[None, :, :]),
            axis=2,
        )
        nearest.extend(np.sqrt(np.min(squared_distance, axis=1)).tolist())
    return math.fsum(nearest) / len(nearest)


def _score_arm(
    prediction_m: np.ndarray,
    target_m: np.ndarray,
    evaluation_mask: np.ndarray,
) -> dict[str, Any]:
    identity_rmse: list[float] = []
    prediction_to_target: list[float] = []
    target_to_prediction: list[float] = []
    symmetric_chamfer: list[float] = []
    identity_count: list[int] = []
    for frame in SCORED_FRAMES:
        mask = evaluation_mask[frame]
        count = int(np.sum(mask))
        _require(count > 0, f"no supported official identity at frame {frame}")
        predicted = prediction_m[frame, mask].astype(np.float64)
        target = target_m[frame, mask].astype(np.float64)
        residual = predicted - target
        identity_rmse.append(float(np.sqrt(np.mean(np.square(residual)))))
        forward = _mean_nearest_euclidean_distance_m(predicted, target)
        reverse = _mean_nearest_euclidean_distance_m(target, predicted)
        prediction_to_target.append(forward)
        target_to_prediction.append(reverse)
        symmetric_chamfer.append(0.5 * (forward + reverse))
        identity_count.append(count)

    frame_count = len(SCORED_FRAMES)
    return {
        "coordinate_rmse_m": math.fsum(identity_rmse) / frame_count,
        "symmetric_euclidean_chamfer_m": (math.fsum(symmetric_chamfer) / frame_count),
        "by_frame": {
            "frame_indices": list(SCORED_FRAMES),
            "coordinate_rmse_m": identity_rmse,
            "prediction_to_target_euclidean_chamfer_m": prediction_to_target,
            "target_to_prediction_euclidean_chamfer_m": target_to_prediction,
            "symmetric_euclidean_chamfer_m": symmetric_chamfer,
            "identity_count": identity_count,
        },
    }


def _improvement_fraction(primary: float, comparator: float) -> float | None:
    if comparator == 0.0:
        return None
    return (comparator - primary) / comparator


def _paired_metrics(
    *,
    primary_chamfer_m: float,
    comparator_chamfer_m: float,
    primary_identity_rmse_m: float,
    comparator_identity_rmse_m: float,
) -> dict[str, float | None]:
    return {
        "primary_minus_comparator_chamfer_m": (
            primary_chamfer_m - comparator_chamfer_m
        ),
        "chamfer_improvement_m": comparator_chamfer_m - primary_chamfer_m,
        "chamfer_improvement_fraction": _improvement_fraction(
            primary_chamfer_m, comparator_chamfer_m
        ),
        "primary_minus_comparator_identity_rmse_m": (
            primary_identity_rmse_m - comparator_identity_rmse_m
        ),
        "identity_rmse_improvement_m": (
            comparator_identity_rmse_m - primary_identity_rmse_m
        ),
        "identity_rmse_improvement_fraction": _improvement_fraction(
            primary_identity_rmse_m, comparator_identity_rmse_m
        ),
    }


def _source_cardinality_relation(
    official_identity_count: int,
    source_node_count: int | None,
) -> str:
    if source_node_count is None:
        return "source-node-count-not-provided"
    if official_identity_count < source_node_count:
        return "official-identity-count-less-than-source-node-count"
    if official_identity_count > source_node_count:
        return "official-identity-count-greater-than-source-node-count"
    return "official-identity-count-equals-source-node-count"


def score_direct_official_identity_case(
    *,
    case_name: str,
    object_id: str,
    primary_prediction_m: np.ndarray,
    selected_raw_backbone_m: np.ndarray,
    queried_identity_ids: np.ndarray,
    target_identity_ids: np.ndarray,
    official_frame_zero_m: np.ndarray,
    target_points_m: np.ndarray,
    object_visibilities: np.ndarray,
    object_motions_valid: np.ndarray,
    shared_support_mask: np.ndarray,
    center_exclusion_mask: np.ndarray,
    frame_indices: np.ndarray,
    source_node_count: int | None = None,
) -> dict[str, Any]:
    """Score one v8 case directly on its exact official identities.

    The target may contain non-finite future coordinates; these are removed by
    the one shared per-frame mask.  A non-finite value anywhere in either
    prediction is instead a hard error, including on an unsupported identity
    or an unscored frame.
    """

    _require(isinstance(case_name, str) and bool(case_name), "case_name is missing")
    _require(isinstance(object_id, str) and bool(object_id), "object_id is missing")

    primary = np.asarray(primary_prediction_m)
    comparator = np.asarray(selected_raw_backbone_m)
    target = np.asarray(target_points_m)
    query_ids = np.asarray(queried_identity_ids)
    official_ids = np.asarray(target_identity_ids)
    frame_zero = np.asarray(official_frame_zero_m)
    visible = np.asarray(object_visibilities)
    valid = np.asarray(object_motions_valid)
    support = np.asarray(shared_support_mask)
    excluded = np.asarray(center_exclusion_mask)
    frames = np.asarray(frame_indices)

    _require(
        primary.dtype == comparator.dtype == target.dtype == np.dtype(np.float32),
        "both predictions and the target must have dtype float32",
    )
    _require(
        primary.ndim == 3
        and primary.shape[0] == FRAME_COUNT
        and primary.shape[2] == 3
        and comparator.shape == primary.shape
        and target.shape == primary.shape,
        "both predictions and the target must share shape (76, M, 3)",
    )
    identity_count = primary.shape[1]
    _require(identity_count > CENTER_COUNT, "official identity set is too small")
    _require(
        query_ids.dtype == official_ids.dtype == np.dtype(np.int64)
        and query_ids.shape == official_ids.shape == (identity_count,),
        "queried and target identity IDs must have int64 shape (M,)",
    )
    _require(
        len(np.unique(query_ids)) == identity_count
        and len(np.unique(official_ids)) == identity_count,
        "official identity IDs must be unique",
    )
    _require(
        np.array_equal(query_ids, official_ids),
        "queried and target official identity order differs",
    )
    _require(
        frame_zero.dtype == np.dtype(np.float32)
        and frame_zero.shape == (identity_count, 3)
        and np.all(np.isfinite(frame_zero)),
        "official_frame_zero_m must have finite float32 shape (M, 3)",
    )
    _require(
        visible.dtype == valid.dtype == np.dtype(bool)
        and visible.shape == valid.shape == (FRAME_COUNT, identity_count),
        "visibility and validity must have bool shape (76, M)",
    )
    _require(
        support.dtype == excluded.dtype == np.dtype(bool)
        and support.shape == excluded.shape == (identity_count,),
        "shared support and center-exclusion masks must have bool shape (M,)",
    )
    excluded_count = int(np.sum(excluded))
    _require(
        frames.dtype == np.dtype(np.int64)
        and frames.shape == (FRAME_COUNT,)
        and np.array_equal(frames, np.arange(FRAME_COUNT, dtype=np.int64)),
        "frame_indices must be int64 arange(76)",
    )
    _require(
        np.all(np.isfinite(primary)) and np.all(np.isfinite(comparator)),
        "both queried predictions must be finite globally",
    )
    _require(
        _bit_equal(primary[0], frame_zero),
        "primary frame zero is not bit-equal to official x0",
    )
    _require(
        _bit_equal(comparator[0], frame_zero),
        "comparator frame zero is not bit-equal to official x0",
    )
    _require(
        _bit_equal(target[0], frame_zero),
        "target frame zero is not bit-equal to official x0",
    )
    if source_node_count is not None:
        _require(
            isinstance(source_node_count, (int, np.integer))
            and not isinstance(source_node_count, (bool, np.bool_))
            and int(source_node_count) > 0,
            "source_node_count must be a positive integer when provided",
        )
        source_node_count = int(source_node_count)

    base_mask = support & ~excluded
    _require(np.any(base_mask), "shared mask leaves no hidden supported identity")
    finite_target = np.all(np.isfinite(target), axis=2)
    evaluation_mask = base_mask[None, :] & visible & valid & finite_target

    primary_score = _score_arm(primary, target, evaluation_mask)
    comparator_score = _score_arm(comparator, target, evaluation_mask)
    primary_chamfer = float(primary_score["symmetric_euclidean_chamfer_m"])
    comparator_chamfer = float(comparator_score["symmetric_euclidean_chamfer_m"])
    primary_identity = float(primary_score["coordinate_rmse_m"])
    comparator_identity = float(comparator_score["coordinate_rmse_m"])
    gate_score = {
        "primary_chamfer_m": primary_chamfer,
        "comparator_chamfer_m": comparator_chamfer,
        "primary_identity_rmse_m": primary_identity,
        "comparator_identity_rmse_m": comparator_identity,
    }
    supported_count = int(np.sum(support))
    hidden_supported_count = int(np.sum(base_mask))
    scored_counts = [int(np.sum(evaluation_mask[frame])) for frame in SCORED_FRAMES]
    support_coverage = supported_count / identity_count
    return {
        "protocol_id": PROTOCOL_ID,
        "scorer_id": SCORER_ID,
        "case_name": case_name,
        "object_id": object_id,
        "scored_frame_intervals_half_open": [
            list(interval) for interval in SCORED_FRAME_INTERVALS_HALF_OPEN
        ],
        "scored_frames": list(SCORED_FRAMES),
        "gate_score": gate_score,
        "scores": {
            "primary": primary_score,
            "selected_raw_backbone": comparator_score,
        },
        "paired": _paired_metrics(
            primary_chamfer_m=primary_chamfer,
            comparator_chamfer_m=comparator_chamfer,
            primary_identity_rmse_m=primary_identity,
            comparator_identity_rmse_m=comparator_identity,
        ),
        "mask_evidence": {
            "official_identity_count": identity_count,
            "supported_identity_count": supported_count,
            "support_coverage_fraction": support_coverage,
            "assimilation_center_count": CENTER_COUNT,
            "center_excluded_identity_count": excluded_count,
            "hidden_supported_identity_count": hidden_supported_count,
            "scored_identity_count_per_frame": scored_counts,
            "minimum_scored_identity_count": min(scored_counts),
            "maximum_scored_identity_count": max(scored_counts),
            "shared_support_mask_sha256": _sha256_array(support),
            "center_exclusion_mask_sha256": _sha256_array(excluded),
            "shared_base_mask_sha256": _sha256_array(base_mask),
            "per_frame_evaluation_mask_sha256": _sha256_array(evaluation_mask),
            "single_shared_mask_for_both_arms": True,
            "arm_specific_dropping_performed": False,
            "density_weighting_performed": False,
            "support_gate": {
                "minimum_support_coverage_fraction": (
                    MINIMUM_SUPPORT_COVERAGE_FRACTION
                ),
                "minimum_hidden_supported_identity_count": (
                    MINIMUM_HIDDEN_SUPPORTED_IDENTITY_COUNT
                ),
                "support_coverage_passed": (
                    support_coverage >= MINIMUM_SUPPORT_COVERAGE_FRACTION
                ),
                "hidden_supported_count_passed": (
                    hidden_supported_count >= MINIMUM_HIDDEN_SUPPORTED_IDENTITY_COUNT
                ),
            },
        },
        "direct_identity": {
            "semantics": "exact-official-identity-order-and-frame-zero-v1",
            "official_identity_ids_sha256": _sha256_array(official_ids),
            "queried_identity_ids_sha256": _sha256_array(query_ids),
            "official_frame_zero_sha256": _sha256_array(frame_zero),
            "frame_indices_sha256": _sha256_array(frames),
            "official_identity_count": identity_count,
            "source_node_count": source_node_count,
            "cardinality_relation": _source_cardinality_relation(
                identity_count, source_node_count
            ),
            "source_node_count_used_for_scoring": False,
            "identity_order_exact": True,
            "primary_x0_bit_exact": True,
            "comparator_x0_bit_exact": True,
            "target_x0_bit_exact": True,
            "transport_performed": False,
            "assignment_performed": False,
            "query_performed": False,
        },
        "metric_contract": {
            "identity_metric": "per-frame coordinate RMSE in metres",
            "chamfer_metric": (
                "per-frame symmetric mean nearest-neighbour Euclidean distance "
                "in metres"
            ),
            "temporal_aggregation": "equal mean over the 54 frozen scored frames",
            "identity_density_weighting": False,
            "lower_is_better": True,
        },
        "method_selection_or_tuning_performed": False,
    }


def _normalize_expected_case_objects(
    expected_case_to_object: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    _require(
        isinstance(expected_case_to_object, Mapping) and bool(expected_case_to_object),
        "expected_case_to_object must be a nonempty mapping",
    )
    case_names = tuple(expected_case_to_object)
    _require(
        all(isinstance(case, str) and bool(case) for case in case_names),
        "expected case names must be nonempty strings",
    )
    objects: list[str] = []
    for case_name in case_names:
        object_id = expected_case_to_object[case_name]
        _require(
            isinstance(object_id, str) and bool(object_id),
            f"expected object ID is invalid for {case_name}",
        )
        if object_id not in objects:
            objects.append(object_id)
    return case_names, tuple(objects)


def _normalize_case_records(
    case_records: Mapping[str, Mapping[str, Any]],
    expected_case_to_object: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...], tuple[str, ...]]:
    case_names, object_ids = _normalize_expected_case_objects(expected_case_to_object)
    _require(
        isinstance(case_records, Mapping) and set(case_records) == set(case_names),
        "case records do not match the exact declared cohort",
    )
    normalized: dict[str, dict[str, Any]] = {}
    for case_name in case_names:
        record = case_records[case_name]
        _require(isinstance(record, Mapping), f"invalid record for {case_name}")
        _require(record.get("case_name") == case_name, "case record name changed")
        _require(
            record.get("object_id") == expected_case_to_object[case_name],
            f"case object changed for {case_name}",
        )
        score = record.get("gate_score")
        _require(
            isinstance(score, Mapping) and set(score) == set(_SCORE_KEYS),
            f"gate score fields changed for {case_name}",
        )
        normalized_score = {key: float(score[key]) for key in _SCORE_KEYS}
        _require(
            all(
                math.isfinite(value) and value >= 0.0
                for value in normalized_score.values()
            ),
            f"invalid gate score for {case_name}",
        )
        masks = record.get("mask_evidence")
        _require(isinstance(masks, Mapping), f"mask evidence missing for {case_name}")
        required_mask_keys = {
            "official_identity_count",
            "supported_identity_count",
            "support_coverage_fraction",
            "assimilation_center_count",
            "center_excluded_identity_count",
            "hidden_supported_identity_count",
        }
        _require(
            required_mask_keys <= set(masks),
            f"mask evidence fields missing for {case_name}",
        )
        integer_keys = (
            "official_identity_count",
            "supported_identity_count",
            "assimilation_center_count",
            "center_excluded_identity_count",
            "hidden_supported_identity_count",
        )
        _require(
            all(
                isinstance(masks[key], (int, np.integer))
                and not isinstance(masks[key], (bool, np.bool_))
                for key in integer_keys
            ),
            f"mask counts are invalid for {case_name}",
        )
        official_count = int(masks["official_identity_count"])
        supported_count = int(masks["supported_identity_count"])
        assimilation_center_count = int(masks["assimilation_center_count"])
        excluded_count = int(masks["center_excluded_identity_count"])
        hidden_count = int(masks["hidden_supported_identity_count"])
        coverage = float(masks["support_coverage_fraction"])
        _require(
            official_count > CENTER_COUNT
            and 0 <= hidden_count <= supported_count <= official_count
            and assimilation_center_count == CENTER_COUNT
            and 0 <= excluded_count <= official_count
            and 0 <= supported_count - hidden_count <= excluded_count
            and math.isfinite(coverage)
            and math.isclose(
                coverage,
                supported_count / official_count,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"mask counts are inconsistent for {case_name}",
        )
        normalized[case_name] = {
            "case_name": case_name,
            "object_id": expected_case_to_object[case_name],
            "gate_score": normalized_score,
            "support_coverage_fraction": coverage,
            "hidden_supported_identity_count": hidden_count,
        }
    return normalized, case_names, object_ids


def _mean_scores(
    records: list[dict[str, Any]],
) -> dict[str, float]:
    return {
        key: math.fsum(record["gate_score"][key] for record in records) / len(records)
        for key in _SCORE_KEYS
    }


def _score_summary_with_pairing(scores: Mapping[str, float]) -> dict[str, Any]:
    normalized = {key: float(scores[key]) for key in _SCORE_KEYS}
    return {
        **normalized,
        "paired": _paired_metrics(
            primary_chamfer_m=normalized["primary_chamfer_m"],
            comparator_chamfer_m=normalized["comparator_chamfer_m"],
            primary_identity_rmse_m=normalized["primary_identity_rmse_m"],
            comparator_identity_rmse_m=normalized["comparator_identity_rmse_m"],
        ),
    }


def aggregate_equal_case_and_object(
    case_records: Mapping[str, Mapping[str, Any]],
    *,
    expected_case_to_object: Mapping[str, str],
) -> dict[str, Any]:
    """Return both equal-case and equal-object summaries.

    Equal-object metrics first average cases within each object and then give
    every object equal weight.  Neither summary weights by point count,
    visibility count, support count, or target sampling density.
    """

    normalized, case_names, object_ids = _normalize_case_records(
        case_records, expected_case_to_object
    )
    records = [normalized[case] for case in case_names]
    by_object: dict[str, dict[str, Any]] = {}
    for object_id in object_ids:
        object_records = [
            normalized[case]
            for case in case_names
            if normalized[case]["object_id"] == object_id
        ]
        by_object[object_id] = {
            "case_names": [record["case_name"] for record in object_records],
            "case_count": len(object_records),
            **_score_summary_with_pairing(_mean_scores(object_records)),
        }
    equal_object_scores = {
        key: math.fsum(by_object[object_id][key] for object_id in object_ids)
        / len(object_ids)
        for key in _SCORE_KEYS
    }
    return {
        "ordered_case_names": list(case_names),
        "ordered_object_ids": list(object_ids),
        "case_count": len(case_names),
        "object_count": len(object_ids),
        "equal_case_mean": _score_summary_with_pairing(_mean_scores(records)),
        "by_object_equal_case_mean": by_object,
        "equal_object_mean": _score_summary_with_pairing(equal_object_scores),
        "weighting": {
            "case": "equal",
            "object": "equal after equal-case mean within object",
            "point_or_visibility_density": "none",
        },
    }


def _gate_common(
    case_records: Mapping[str, Mapping[str, Any]],
    expected_case_to_object: Mapping[str, str],
    *,
    expected_case_count: int,
) -> tuple[
    dict[str, dict[str, Any]],
    tuple[str, ...],
    dict[str, Any],
    dict[str, Any],
]:
    normalized, case_names, _ = _normalize_case_records(
        case_records, expected_case_to_object
    )
    _require(
        len(case_names) == expected_case_count,
        f"gate requires exactly {expected_case_count} declared cases",
    )
    aggregation = aggregate_equal_case_and_object(
        case_records, expected_case_to_object=expected_case_to_object
    )
    equal_case = aggregation["equal_case_mean"]
    comparator_chamfer = float(equal_case["comparator_chamfer_m"])
    _require(
        comparator_chamfer > 0.0,
        "comparator equal-case mean Chamfer must be positive",
    )
    wins = sum(
        normalized[case]["gate_score"]["primary_chamfer_m"]
        < normalized[case]["gate_score"]["comparator_chamfer_m"]
        for case in case_names
    )
    no_large_regression = all(
        normalized[case]["gate_score"]["primary_chamfer_m"]
        <= (1.0 + MAXIMUM_CASE_CHAMFER_REGRESSION_FRACTION)
        * normalized[case]["gate_score"]["comparator_chamfer_m"]
        for case in case_names
    )
    coverage_failures = [
        case
        for case in case_names
        if normalized[case]["support_coverage_fraction"]
        < MINIMUM_SUPPORT_COVERAGE_FRACTION
    ]
    hidden_count_failures = [
        case
        for case in case_names
        if normalized[case]["hidden_supported_identity_count"]
        < MINIMUM_HIDDEN_SUPPORTED_IDENTITY_COUNT
    ]
    common = {
        "case_chamfer_wins": wins,
        "no_case_over_maximum_chamfer_regression": no_large_regression,
        "support_coverage_failures": coverage_failures,
        "hidden_supported_count_failures": hidden_count_failures,
        "minimum_observed_support_coverage_fraction": min(
            normalized[case]["support_coverage_fraction"] for case in case_names
        ),
        "minimum_observed_hidden_supported_identity_count": min(
            normalized[case]["hidden_supported_identity_count"] for case in case_names
        ),
    }
    return normalized, case_names, aggregation, common


def evaluate_calibration_gate(
    case_records: Mapping[str, Mapping[str, Any]],
    *,
    expected_case_to_object: Mapping[str, str],
) -> dict[str, Any]:
    """Apply the v7-equivalent 15-case GO/NO-GO checks plus v8 support."""

    _, _, aggregation, common = _gate_common(
        case_records,
        expected_case_to_object,
        expected_case_count=CALIBRATION_CASE_COUNT,
    )
    equal_case = aggregation["equal_case_mean"]
    chamfer_improvement = equal_case["paired"]["chamfer_improvement_fraction"]
    checks = {
        "all_cases_support_coverage_at_least_0_90": not common[
            "support_coverage_failures"
        ],
        "all_cases_hidden_supported_count_at_least_32": not common[
            "hidden_supported_count_failures"
        ],
        "mean_chamfer_improvement_at_least_5_percent": (
            chamfer_improvement is not None
            and chamfer_improvement >= MINIMUM_MEAN_CHAMFER_IMPROVEMENT_FRACTION
        ),
        "aggregate_identity_improves": (
            equal_case["primary_identity_rmse_m"]
            < equal_case["comparator_identity_rmse_m"]
        ),
        "at_least_10_of_15_chamfer_wins": (
            common["case_chamfer_wins"] >= CALIBRATION_MINIMUM_CHAMFER_WINS
        ),
        "no_case_over_10_percent_chamfer_regression": common[
            "no_case_over_maximum_chamfer_regression"
        ],
    }
    return {
        "gate": "v8-calibration-go-no-go-v1",
        "thresholds": {
            "case_count": CALIBRATION_CASE_COUNT,
            "minimum_support_coverage_fraction": (MINIMUM_SUPPORT_COVERAGE_FRACTION),
            "minimum_hidden_supported_identity_count": (
                MINIMUM_HIDDEN_SUPPORTED_IDENTITY_COUNT
            ),
            "minimum_equal_case_mean_chamfer_improvement_fraction": (
                MINIMUM_MEAN_CHAMFER_IMPROVEMENT_FRACTION
            ),
            "minimum_case_chamfer_wins": CALIBRATION_MINIMUM_CHAMFER_WINS,
            "maximum_case_chamfer_regression_fraction": (
                MAXIMUM_CASE_CHAMFER_REGRESSION_FRACTION
            ),
        },
        "aggregation": aggregation,
        **common,
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate_confirmation_gate(
    case_records: Mapping[str, Mapping[str, Any]],
    *,
    expected_case_to_object: Mapping[str, str],
) -> dict[str, Any]:
    """Apply the v7-equivalent exact-six confirmation checks plus v8 support."""

    _, case_names, aggregation, common = _gate_common(
        case_records,
        expected_case_to_object,
        expected_case_count=CONFIRMATION_CASE_COUNT,
    )
    equal_case = aggregation["equal_case_mean"]
    chamfer_improvement = equal_case["paired"]["chamfer_improvement_fraction"]
    wins = int(common["case_chamfer_wins"])
    sign_test_p = math.fsum(
        math.comb(len(case_names), successes)
        for successes in range(wins, len(case_names) + 1)
    ) / (2 ** len(case_names))
    checks = {
        "all_cases_support_coverage_at_least_0_90": not common[
            "support_coverage_failures"
        ],
        "all_cases_hidden_supported_count_at_least_32": not common[
            "hidden_supported_count_failures"
        ],
        "all_6_cases_chamfer_win": wins == CONFIRMATION_REQUIRED_CHAMFER_WINS,
        "one_sided_sign_test_p_is_1_over_64": (
            sign_test_p == CONFIRMATION_ONE_SIDED_SIGN_TEST_P
        ),
        "mean_chamfer_improvement_at_least_5_percent": (
            chamfer_improvement is not None
            and chamfer_improvement >= MINIMUM_MEAN_CHAMFER_IMPROVEMENT_FRACTION
        ),
        "aggregate_identity_improves": (
            equal_case["primary_identity_rmse_m"]
            < equal_case["comparator_identity_rmse_m"]
        ),
        "no_case_over_10_percent_chamfer_regression": common[
            "no_case_over_maximum_chamfer_regression"
        ],
    }
    return {
        "gate": "v8-exact-six-confirmation-v1",
        "thresholds": {
            "case_count": CONFIRMATION_CASE_COUNT,
            "required_case_chamfer_wins": CONFIRMATION_REQUIRED_CHAMFER_WINS,
            "one_sided_sign_test_p": CONFIRMATION_ONE_SIDED_SIGN_TEST_P,
            "minimum_support_coverage_fraction": (MINIMUM_SUPPORT_COVERAGE_FRACTION),
            "minimum_hidden_supported_identity_count": (
                MINIMUM_HIDDEN_SUPPORTED_IDENTITY_COUNT
            ),
            "minimum_equal_case_mean_chamfer_improvement_fraction": (
                MINIMUM_MEAN_CHAMFER_IMPROVEMENT_FRACTION
            ),
            "maximum_case_chamfer_regression_fraction": (
                MAXIMUM_CASE_CHAMFER_REGRESSION_FRACTION
            ),
        },
        "aggregation": aggregation,
        **common,
        "one_sided_sign_test_p": sign_test_p,
        "checks": checks,
        "passed": all(checks.values()),
    }
