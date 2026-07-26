"""Validate the causal metadata of portable Prob4D observation beliefs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .observation_belief import ObservationBeliefV1

PROB4D_SOURCE_REPOSITORY = "FlorianPfaff/Prob4D"
PROB4D_CAUSAL_STREAM_ID = "prob4d:causal-overlap-window-points"
PROB4D_CAUSAL_LINEAGE_VERSION = 1
PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION = 1
PROB4D_CAUSAL_STREAM_CONTRACT_VERSION = 2
PROB4D_GAUGE_FACTOR_NAMES = tuple(
    f"gauge_latent_{index}" for index in range(7)
)
PROB4D_JOINT_GAUGE_FACTOR_PREFIX = "joint_gauge_latent_"


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


def _require_probability(value: Any, *, name: str) -> float:
    _require(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(np.isfinite(result), f"{name} must be finite")
    _require(0.0 <= result <= 1.0, f"{name} must lie in [0, 1]")
    return result


def _validate_gauge_contract(
    belief: ObservationBeliefV1,
    metadata: Mapping[str, Any],
) -> tuple[int, str]:
    raw_version = metadata.get("prob4d_causal_stream_contract_version")
    version = (
        PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION
        if raw_version is None
        else _require_integer(
            raw_version,
            name="Prob4D causal-stream contract version",
        )
    )
    _require(
        version
        in {
            PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION,
            PROB4D_CAUSAL_STREAM_CONTRACT_VERSION,
        },
        "unsupported Prob4D causal-stream contract version",
    )

    if version == PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION:
        _require(
            belief.factor_names == PROB4D_GAUGE_FACTOR_NAMES,
            "legacy Prob4D causal artifact has changed gauge factor names",
        )
        _require(
            np.array_equal(
                belief.factor_group_ids,
                belief.window_indices,
            ),
            "legacy Prob4D gauge factor groups must equal window indices",
        )
        return version, "per_window_sim3_gauge_marginals"

    expected_names = tuple(
        f"{PROB4D_JOINT_GAUGE_FACTOR_PREFIX}{index:04d}"
        for index in range(len(belief.factor_names))
    )
    _require(
        belief.factor_names == expected_names,
        "Prob4D joint gauge factor names are not canonical",
    )
    _require(
        np.all(belief.factor_group_ids == 0),
        "Prob4D joint gauge factors must use one shared factor group",
    )
    _require(
        metadata.get("gauge_mode") == "sequential",
        "strict Prob4D stream contract v2 requires sequential gauge mode",
    )
    _require(
        metadata.get("joint_cross_window_gauge_covariance_represented") is True,
        "Prob4D stream contract v2 must represent cross-window gauge covariance",
    )
    posterior = _require_mapping(
        metadata.get("gauge_posterior"),
        name="gauge_posterior",
    )
    _require(
        posterior.get("model") == "sequential_joint_spanning_tree_v1",
        "unsupported Prob4D joint gauge-posterior model",
    )
    _require(
        _require_integer(
            posterior.get("window_count"),
            name="gauge-posterior window_count",
        )
        == len(belief.window_names),
        "gauge-posterior window count differs from the descriptor",
    )
    _require(
        _require_integer(
            posterior.get("full_dimension"),
            name="gauge-posterior full_dimension",
        )
        == 7 * len(belief.window_names),
        "gauge-posterior dimension differs from the window gauges",
    )
    _require(
        _require_integer(
            posterior.get("exported_factor_rank"),
            name="gauge-posterior exported_factor_rank",
        )
        == len(belief.factor_names),
        "gauge-posterior rank differs from exported factors",
    )
    retained = _require_probability(
        posterior.get("retained_covariance_trace_fraction"),
        name="retained gauge-covariance trace fraction",
    )
    minimum = _require_probability(
        posterior.get("minimum_retained_gauge_trace"),
        name="minimum retained gauge-covariance trace",
    )
    _require(
        retained + 1e-12 >= minimum,
        "exported gauge factors retain less covariance trace than declared",
    )
    _require(
        posterior.get("cross_window_covariance_preserved") is True,
        "Prob4D stream contract v2 lost cross-window gauge covariance",
    )
    _require(
        posterior.get("fixed_lag_boundary_covariance_is_approximate") is False,
        "strict Prob4D stream contract v2 rejects approximate fixed-lag covariance",
    )
    parent_ids = posterior.get("parent_window_ids")
    _require(
        isinstance(parent_ids, list)
        and len(parent_ids) == len(belief.window_names),
        "gauge-posterior parent lineage must identify every window",
    )
    _require(parent_ids[0] is None, "first Prob4D gauge window must have no parent")
    for index, parent_id in enumerate(parent_ids[1:], start=1):
        _require(
            parent_id in belief.window_names[:index],
            "Prob4D gauge parent must precede its child window",
        )
    return version, "joint_cross_window_sim3_gauge_covariance"


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
    stream_contract_version, gauge_covariance_semantics = _validate_gauge_contract(
        belief,
        metadata,
    )
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
    calibration_digest = anchor.get("calibration_artifact_sha256")
    if calibration_digest is not None:
        _require_sha256(
            calibration_digest,
            name="metric gauge-anchor calibration_artifact_sha256",
        )
    _require(
        anchor.get("window_id") == belief.window_names[0],
        "metric gauge anchor does not identify the first window",
    )
    anchor_frame = str(
        anchor.get("coordinate_frame", anchor.get("world_frame_id", ""))
    )
    _require(
        anchor_frame == coordinate_frame,
        "metric gauge-anchor frame differs from observation frame",
    )
    covariance_treatment = anchor.get("covariance_treatment")
    if (
        stream_contract_version >= PROB4D_CAUSAL_STREAM_CONTRACT_VERSION
        or covariance_treatment is not None
    ):
        _require(
            covariance_treatment == "fixed_external_calibration",
            "portable Prob4D causal artifact requires a fixed metric anchor",
        )
    if stream_contract_version >= PROB4D_CAUSAL_STREAM_CONTRACT_VERSION:
        _require(
            anchor.get("schema_name") == "prob4d.metric-gauge-anchor",
            "unsupported Prob4D metric gauge-anchor schema",
        )
        _require(
            _require_integer(
                anchor.get("schema_version"),
                name="metric gauge-anchor schema_version",
            )
            == 1,
            "unsupported Prob4D metric gauge-anchor version",
        )
        _require(
            anchor.get("metric_units") == "m",
            "Prob4D metric gauge anchor must declare metric units",
        )
        _require(
            bool(str(anchor.get("source_kind", ""))),
            "Prob4D metric gauge anchor has no source kind",
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
        "stream_contract_version": stream_contract_version,
        "gauge_covariance_semantics": gauge_covariance_semantics,
        "causal_frame_stop": belief.causal_frame_stop,
        "window_count": len(belief.window_names),
        "metric_anchor_id": anchor_id,
        "source_artifact_sha256": belief.source_artifact_sha256,
    }


__all__ = [
    "PROB4D_CAUSAL_LINEAGE_VERSION",
    "PROB4D_CAUSAL_STREAM_CONTRACT_VERSION",
    "PROB4D_CAUSAL_STREAM_ID",
    "PROB4D_GAUGE_FACTOR_NAMES",
    "PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION",
    "PROB4D_SOURCE_REPOSITORY",
    "is_prob4d_causal_observation_belief",
    "validate_prob4d_causal_observation_belief",
]
