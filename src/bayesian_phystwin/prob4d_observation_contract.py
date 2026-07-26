"""Validate legacy and joint-gauge Prob4D causal observation contracts.

This module deliberately re-implements the producer checks instead of importing
Prob4D.  Bayesian-PhysTwin must be able to reject semantically inconsistent
artifacts before an innovation is formed, including when the producer package is
not installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .observation_belief import ObservationBeliefV1

PROB4D_SOURCE_REPOSITORY = "FlorianPfaff/Prob4D"
PROB4D_CAUSAL_STREAM_ID = "prob4d:causal-overlap-window-points"
PROB4D_CAUSAL_LINEAGE_VERSION = 1

# Retained public name for compatibility with the original per-window contract.
PROB4D_LEGACY_GAUGE_FACTOR_NAMES = tuple(
    f"gauge_latent_{index}" for index in range(7)
)
PROB4D_GAUGE_FACTOR_NAMES = PROB4D_LEGACY_GAUGE_FACTOR_NAMES
PROB4D_JOINT_GAUGE_FACTOR_PREFIX = "joint_gauge_latent_"
PROB4D_JOINT_GAUGE_MODEL = "sequential_joint_spanning_tree_v1"
PROB4D_FIXED_LAG_GAUGE_MODEL = "fixed_lag_block_diagonal_approximation_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, *, name: str) -> str:
    result = str(value)
    _require(
        len(result) == 64
        and all(character in "0123456789abcdef" for character in result),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return result


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _require_integer(value: Any, *, name: str) -> int:
    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    return int(value)


def _require_probability(value: Any, *, name: str, positive: bool = False) -> float:
    result = float(value)
    lower_ok = result > 0.0 if positive else result >= 0.0
    _require(
        np.isfinite(result) and lower_ok and result <= 1.0,
        f"{name} must lie in {'(0, 1]' if positive else '[0, 1]'}",
    )
    return result


def _validate_metric_anchor(
    metadata: Mapping[str, Any],
    *,
    coordinate_frame: str,
    first_window: str,
) -> tuple[str, str]:
    anchor = _require_mapping(
        metadata.get("metric_gauge_anchor"),
        name="metric_gauge_anchor",
    )
    anchor_id = _require_sha256(
        anchor.get("artifact_id", ""),
        name="metric gauge-anchor artifact_id",
    )
    anchor_source_sha256 = _require_sha256(
        anchor.get("source_artifact_sha256", ""),
        name="metric gauge-anchor source_artifact_sha256",
    )
    _require(
        anchor.get("window_id") == first_window,
        "metric gauge anchor does not identify the first window",
    )

    # New Prob4D artifacts carry a content-addressed external metric prior.
    # Older artifacts used the more verbose calibration fields below.  Validate
    # every field that is present and preserve support for both encodings.
    if "schema_name" in anchor:
        _require(
            anchor.get("schema_name") == "prob4d.metric-gauge-anchor",
            "unsupported metric gauge-anchor schema",
        )
    if "schema_version" in anchor:
        _require(
            _require_integer(
                anchor.get("schema_version"),
                name="metric gauge-anchor schema_version",
            )
            == 1,
            "unsupported metric gauge-anchor version",
        )
    if "metric_units" in anchor:
        _require(
            anchor.get("metric_units") == "m",
            "metric gauge anchor must use metres",
        )
    anchor_frame = anchor.get("coordinate_frame", anchor.get("world_frame_id"))
    if anchor_frame is not None:
        _require(
            anchor_frame == coordinate_frame,
            "metric gauge-anchor frame differs from observation frame",
        )

    if "source_kind" in anchor:
        _require(
            bool(str(anchor.get("source_kind", ""))),
            "metric gauge anchor has no source kind",
        )
    if "calibration_artifact_sha256" in anchor:
        _require_sha256(
            anchor.get("calibration_artifact_sha256", ""),
            name="metric gauge-anchor calibration_artifact_sha256",
        )
    if "covariance_treatment" in anchor:
        _require(
            anchor.get("covariance_treatment")
            in {"fixed_external_calibration", "propagated_external_prior"},
            "portable Prob4D causal artifact requires a fixed metric anchor or propagated external prior",
        )

    return anchor_id, anchor_source_sha256


def _validate_factor_semantics(
    belief: ObservationBeliefV1,
    metadata: Mapping[str, Any],
) -> dict[str, object]:
    factor_names = belief.factor_names
    factor_groups = belief.factor_group_ids
    window_indices = belief.window_indices

    if (
        factor_names == PROB4D_LEGACY_GAUGE_FACTOR_NAMES
        and np.array_equal(factor_groups, window_indices)
    ):
        return {
            "covariance_semantics": "legacy_per_window_sim3_marginals_v1",
            "factor_group_count": len(np.unique(factor_groups)),
            "factor_rank": len(factor_names),
            "cross_window_covariance_preserved": False,
        }

    _require(
        len(factor_names) > 0,
        "Prob4D joint-gauge artifact must contain gauge factors",
    )
    expected_names = tuple(
        f"{PROB4D_JOINT_GAUGE_FACTOR_PREFIX}{index:04d}"
        for index in range(len(factor_names))
    )
    _require(
        factor_names == expected_names,
        "Prob4D joint-gauge factor names are not canonical",
    )
    _require(
        np.array_equal(np.unique(factor_groups), np.asarray([0], dtype=np.int64)),
        "Prob4D joint-gauge artifact must use one shared factor group",
    )
    _require(
        metadata.get("factor_definition") == "one shared joint gauge latent vector",
        "Prob4D joint-gauge factor definition changed",
    )
    _require(
        metadata.get("factor_group_semantics")
        == (
            "all rows use one factor group; each window contributes its block of "
            "the same joint gauge covariance root"
        ),
        "Prob4D joint-gauge factor-group semantics changed",
    )

    posterior = _require_mapping(
        metadata.get("gauge_posterior"),
        name="gauge_posterior",
    )
    model = str(posterior.get("model", ""))
    _require(
        model in {PROB4D_JOINT_GAUGE_MODEL, PROB4D_FIXED_LAG_GAUGE_MODEL},
        "unsupported Prob4D gauge-posterior model",
    )
    window_count = _require_integer(
        posterior.get("window_count"),
        name="gauge-posterior window_count",
    )
    _require(
        window_count == len(belief.window_names),
        "gauge-posterior window count differs from the descriptor",
    )
    _require(
        _require_integer(
            posterior.get("full_dimension"),
            name="gauge-posterior full_dimension",
        )
        == 7 * window_count,
        "gauge-posterior dimension is inconsistent with its windows",
    )
    factor_rank = _require_integer(
        posterior.get("exported_factor_rank"),
        name="gauge-posterior exported_factor_rank",
    )
    _require(
        factor_rank == len(factor_names),
        "gauge-posterior rank differs from the exported factors",
    )
    retained = _require_probability(
        posterior.get("retained_covariance_trace_fraction"),
        name="retained covariance trace fraction",
        positive=True,
    )
    minimum = _require_probability(
        posterior.get("minimum_retained_gauge_trace"),
        name="minimum retained gauge trace",
        positive=True,
    )
    _require(
        retained + 1e-12 >= minimum,
        "Prob4D gauge rank reduction violates its retained-trace threshold",
    )
    max_rank = posterior.get("max_gauge_rank")
    if max_rank is not None:
        _require(
            _require_integer(max_rank, name="gauge-posterior max_gauge_rank")
            >= factor_rank,
            "gauge-posterior max rank is below the exported factor rank",
        )

    cross_window = posterior.get("cross_window_covariance_preserved")
    _require(
        isinstance(cross_window, bool),
        "gauge-posterior cross-window flag must be Boolean",
    )
    _require(
        metadata.get("joint_cross_window_gauge_covariance_represented")
        is cross_window,
        "Prob4D top-level and gauge-posterior covariance flags differ",
    )
    parents = posterior.get("parent_window_ids")
    _require(
        isinstance(parents, list) and len(parents) == window_count,
        "gauge-posterior parent lineage changed length",
    )
    _require(parents[0] is None, "first Prob4D gauge must not have a parent")

    if model == PROB4D_JOINT_GAUGE_MODEL:
        _require(
            cross_window is True,
            "sequential Prob4D gauge tree must preserve cross-window covariance",
        )
        for index, parent in enumerate(parents[1:], start=1):
            _require(
                parent in belief.window_names[:index],
                "Prob4D gauge tree parent must precede its child",
            )
        _require(
            posterior.get("fixed_lag_boundary_covariance_is_approximate") is False,
            "sequential gauge tree cannot declare fixed-lag approximation",
        )
    else:
        _require(
            cross_window is False,
            "fixed-lag block-diagonal gauge model cannot claim cross-window covariance",
        )
        _require(
            posterior.get("fixed_lag_boundary_covariance_is_approximate") is True,
            "fixed-lag gauge covariance must declare its boundary approximation",
        )

    return {
        "covariance_semantics": model,
        "factor_group_count": 1,
        "factor_rank": factor_rank,
        "cross_window_covariance_preserved": cross_window,
        "retained_covariance_trace_fraction": retained,
    }


def is_prob4d_causal_observation_belief(
    belief: ObservationBeliefV1,
) -> bool:
    """Return whether the artifact declares the strict Prob4D causal stream."""

    return (
        belief.source_repository == PROB4D_SOURCE_REPOSITORY
        and belief.stream_id == PROB4D_CAUSAL_STREAM_ID
    )


def validate_prob4d_causal_observation_belief(
    belief: ObservationBeliefV1,
) -> dict[str, object]:
    """Fail closed on inconsistent metric, gauge, or temporal lineage."""

    _require(
        is_prob4d_causal_observation_belief(belief),
        "observation belief is not the strict Prob4D causal stream",
    )
    _require(
        belief.source_revision.lower() != "unknown",
        "Prob4D causal artifact has no exact source revision",
    )

    metadata = belief.metadata
    _require(
        metadata.get("metric_coordinates") is True,
        "Prob4D causal artifact must declare metric coordinates",
    )
    _require(
        metadata.get("metric_units") == "m",
        "Prob4D causal artifact must declare metric units",
    )
    coordinate_frame = str(metadata.get("coordinate_frame", ""))
    _require(
        bool(coordinate_frame),
        "Prob4D causal artifact has no coordinate frame",
    )

    anchor_id, anchor_source_sha256 = _validate_metric_anchor(
        metadata,
        coordinate_frame=coordinate_frame,
        first_window=belief.window_names[0],
    )
    factor_validation = _validate_factor_semantics(belief, metadata)

    lineage = _require_mapping(
        metadata.get("causal_source_lineage"),
        name="causal_source_lineage",
    )
    _require(
        _require_integer(
            lineage.get("schema_version"),
            name="causal lineage schema_version",
        )
        == PROB4D_CAUSAL_LINEAGE_VERSION,
        "unsupported Prob4D causal-lineage version",
    )
    _require(
        lineage.get("producer") == "Prob4D",
        "causal lineage has changed producer",
    )
    _require(
        lineage.get("motioncrafter_lineage_schema_version") == 1,
        "unsupported MotionCrafter temporal-lineage version",
    )
    _require(
        lineage.get("motioncrafter_windowing_model")
        == "motioncrafter_sliding_window_v1",
        "unsupported MotionCrafter windowing model",
    )
    _require(
        lineage.get("source_product")
        == "independently_decoded_overlap_windows",
        "Prob4D causal artifact uses an inadmissible source product",
    )
    _require(
        _require_integer(
            lineage.get("causal_frame_stop_exclusive"),
            name="causal lineage frame stop",
        )
        == belief.causal_frame_stop,
        "causal lineage cutoff differs from the artifact cutoff",
    )
    _require(
        _require_integer(
            lineage.get("future_prediction_payloads_opened"),
            name="future_prediction_payloads_opened",
        )
        == 0,
        "Prob4D causal artifact reports opening future payloads",
    )
    _require(
        lineage.get("admissibility_rule")
        == "source_frame_max < causal_frame_stop_exclusive",
        "Prob4D causal artifact has changed its admission rule",
    )
    lineage_source_sha256 = _require_sha256(
        lineage.get("source_artifact_sha256", ""),
        name="causal lineage source_artifact_sha256",
    )
    _require(
        lineage_source_sha256 == belief.source_artifact_sha256,
        "causal lineage source digest differs from the descriptor",
    )

    selected = lineage.get("selected_windows")
    _require(
        isinstance(selected, list)
        and len(selected) == len(belief.window_names),
        "causal lineage must identify every observation window",
    )
    for window_index, expected_window_id in enumerate(belief.window_names):
        record = _require_mapping(
            selected[window_index],
            name=f"selected_windows[{window_index}]",
        )
        _require(
            record.get("window_id") == expected_window_id,
            "causal lineage window order differs from the descriptor",
        )
        start = _require_integer(
            record.get("source_frame_start"),
            name=f"selected window {expected_window_id} start",
        )
        stop = _require_integer(
            record.get("source_frame_stop_exclusive"),
            name=f"selected window {expected_window_id} stop",
        )
        maximum = _require_integer(
            record.get("source_frame_max"),
            name=f"selected window {expected_window_id} maximum",
        )
        _require(
            0 <= start <= maximum < stop <= belief.causal_frame_stop,
            "selected Prob4D window crosses its causal boundary",
        )
        _require_sha256(
            record.get("frame_indices_sha256", ""),
            name=f"selected window {expected_window_id} frame digest",
        )
        payload_digest = _require_sha256(
            record.get("payload_sha256", ""),
            name=f"selected window {expected_window_id} payload digest",
        )
        if window_index == 0:
            _require(
                payload_digest == anchor_source_sha256,
                "metric anchor is not bound to the first selected payload",
            )
        rows = belief.window_indices == window_index
        if np.any(rows):
            row_frames = belief.frame_ids[rows]
            _require(
                np.all((row_frames >= start) & (row_frames <= maximum)),
                "observation rows exceed their declared source window",
            )

    return {
        "validated": True,
        "schema_version": PROB4D_CAUSAL_LINEAGE_VERSION,
        "causal_frame_stop": belief.causal_frame_stop,
        "window_count": len(belief.window_names),
        "metric_anchor_id": anchor_id,
        "source_artifact_sha256": belief.source_artifact_sha256,
        **factor_validation,
    }


__all__ = [
    "PROB4D_CAUSAL_LINEAGE_VERSION",
    "PROB4D_CAUSAL_STREAM_ID",
    "PROB4D_FIXED_LAG_GAUGE_MODEL",
    "PROB4D_GAUGE_FACTOR_NAMES",
    "PROB4D_JOINT_GAUGE_FACTOR_PREFIX",
    "PROB4D_JOINT_GAUGE_MODEL",
    "PROB4D_LEGACY_GAUGE_FACTOR_NAMES",
    "PROB4D_SOURCE_REPOSITORY",
    "is_prob4d_causal_observation_belief",
    "validate_prob4d_causal_observation_belief",
]
