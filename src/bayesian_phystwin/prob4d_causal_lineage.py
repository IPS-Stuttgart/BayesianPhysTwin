"""Validate the causal metadata of portable Prob4D observation beliefs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .observation_belief import ObservationBeliefV1

PROB4D_SOURCE_REPOSITORY = "FlorianPfaff/Prob4D"
PROB4D_CAUSAL_STREAM_ID = "prob4d:causal-overlap-window-points"
PROB4D_CAUSAL_LINEAGE_VERSION = 1
PROB4D_GAUGE_FACTOR_NAMES = tuple(
    f"gauge_latent_{index}" for index in range(7)
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, *, name: str) -> str:
    result = str(value)
    _require(
        len(result) == 64
        and all(
            character in "0123456789abcdef"
            for character in result
        ),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return result


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _require_integer(value: Any, *, name: str) -> int:
    _require(
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    return int(value)


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
    _require(
        belief.factor_names == PROB4D_GAUGE_FACTOR_NAMES,
        "Prob4D causal artifact has changed gauge factor names",
    )
    _require(
        np.array_equal(
            belief.factor_group_ids,
            belief.window_indices,
        ),
        "Prob4D causal gauge factor groups must equal window indices",
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
    _require_sha256(
        anchor.get("calibration_artifact_sha256", ""),
        name="metric gauge-anchor calibration_artifact_sha256",
    )
    _require(
        anchor.get("window_id") == belief.window_names[0],
        "metric gauge anchor does not identify the first window",
    )
    _require(
        anchor.get("world_frame_id") == coordinate_frame,
        "metric gauge-anchor frame differs from observation frame",
    )
    _require(
        anchor.get("covariance_treatment")
        == "fixed_external_calibration",
        "portable Prob4D causal artifact requires a fixed metric anchor",
    )

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
    for window_index, expected_window_id in enumerate(
        belief.window_names
    ):
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
        frame_digest = _require_sha256(
            record.get("frame_indices_sha256", ""),
            name=f"selected window {expected_window_id} frame digest",
        )
        _require(bool(frame_digest), "selected window frame digest is empty")
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
                np.all(
                    (row_frames >= start)
                    & (row_frames <= maximum)
                ),
                "observation rows exceed their declared source window",
            )

    return {
        "validated": True,
        "schema_version": PROB4D_CAUSAL_LINEAGE_VERSION,
        "causal_frame_stop": belief.causal_frame_stop,
        "window_count": len(belief.window_names),
        "metric_anchor_id": anchor_id,
        "source_artifact_sha256": belief.source_artifact_sha256,
    }


__all__ = [
    "PROB4D_CAUSAL_LINEAGE_VERSION",
    "PROB4D_CAUSAL_STREAM_ID",
    "PROB4D_GAUGE_FACTOR_NAMES",
    "PROB4D_SOURCE_REPOSITORY",
    "is_prob4d_causal_observation_belief",
    "validate_prob4d_causal_observation_belief",
]
