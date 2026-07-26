"""Validate legacy and joint-gauge Prob4D causal observation contracts.

This module deliberately re-implements the producer checks instead of importing
Prob4D. Bayesian-PhysTwin must reject semantically inconsistent artifacts before
an innovation is formed, including when the producer package is not installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .observation_belief import ObservationBeliefV1

PROB4D_SOURCE_REPOSITORY = "FlorianPfaff/Prob4D"
PROB4D_CAUSAL_STREAM_ID = "prob4d:causal-overlap-window-points"
PROB4D_CAUSAL_LINEAGE_VERSION = 1
PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION = 1
PROB4D_CAUSAL_STREAM_CONTRACT_VERSION = 2

PROB4D_LEGACY_GAUGE_FACTOR_NAMES = tuple(
    f"gauge_latent_{index}" for index in range(7)
)
PROB4D_GAUGE_FACTOR_NAMES = PROB4D_LEGACY_GAUGE_FACTOR_NAMES
PROB4D_JOINT_GAUGE_FACTOR_PREFIX = "joint_gauge_latent_"
PROB4D_JOINT_GAUGE_MODEL = "sequential_joint_spanning_tree_v1"
PROB4D_FIXED_LAG_GAUGE_MODEL = "fixed_lag_block_diagonal_approximation_v1"

FIXED_EXTERNAL_CALIBRATION = "fixed_external_calibration"
PROPAGATED_EXTERNAL_PRIOR = "propagated_external_prior"


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


def _require_probability(value: Any, *, name: str) -> float:
    result = float(value)
    _require(
        np.isfinite(result) and 0.0 < result <= 1.0,
        f"{name} must lie in (0, 1]",
    )
    return result


def _stream_contract_version(
    metadata: Mapping[str, Any],
    factor_names: Sequence[str],
) -> tuple[int, bool]:
    raw = metadata.get("prob4d_causal_stream_contract_version")
    if raw is None:
        if tuple(factor_names) == PROB4D_GAUGE_FACTOR_NAMES:
            return PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION, True
        if factor_names and all(
            name.startswith(PROB4D_JOINT_GAUGE_FACTOR_PREFIX)
            for name in factor_names
        ):
            # Prob4D 0.2.0 emitted the joint representation before the
            # provider-specific stream contract received an explicit version.
            return PROB4D_CAUSAL_STREAM_CONTRACT_VERSION, True
        raise ValueError(
            "Prob4D causal artifact has no recognizable stream contract"
        )
    version = _require_integer(
        raw,
        name="Prob4D causal stream contract version",
    )
    _require(
        version
        in {
            PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION,
            PROB4D_CAUSAL_STREAM_CONTRACT_VERSION,
        },
        "unsupported Prob4D causal stream contract version",
    )
    return version, False


def _validate_joint_gauge_contract(
    belief: ObservationBeliefV1,
    metadata: Mapping[str, Any],
) -> dict[str, object]:
    expected_names = tuple(
        f"{PROB4D_JOINT_GAUGE_FACTOR_PREFIX}{index:04d}"
        for index in range(len(belief.factor_names))
    )
    _require(
        bool(expected_names) and belief.factor_names == expected_names,
        "Prob4D joint gauge factor names are not canonical",
    )
    unique_factor_groups = np.unique(belief.factor_group_ids)
    _require(
        np.array_equal(unique_factor_groups, np.asarray([0], dtype=np.int64)),
        "Prob4D joint gauge factors must use one shared factor group",
    )
    _require(
        metadata.get("factor_definition")
        == "one shared joint gauge latent vector",
        "Prob4D joint gauge factor definition changed",
    )
    _require(
        metadata.get("factor_group_semantics")
        == (
            "all rows use one factor group; each window contributes its block of "
            "the same joint gauge covariance root"
        ),
        "Prob4D joint gauge factor-group semantics changed",
    )
    _require(
        metadata.get("joint_cross_window_gauge_covariance_represented") is True,
        "Prob4D stream contract v2 must represent cross-window gauge covariance",
    )
    _require(
        metadata.get("gauge_mode") == "sequential",
        "Prob4D stream contract v2 requires the causal sequential gauge mode",
    )

    posterior = _require_mapping(
        metadata.get("gauge_posterior"),
        name="gauge_posterior",
    )
    _require(
        posterior.get("model") == PROB4D_JOINT_GAUGE_MODEL,
        "Prob4D stream contract v2 has an unsupported joint gauge model",
    )
    window_count = _require_integer(
        posterior.get("window_count"),
        name="gauge posterior window_count",
    )
    _require(
        window_count == len(belief.window_names),
        "gauge posterior window count differs from the observation descriptor",
    )
    full_dimension = _require_integer(
        posterior.get("full_dimension"),
        name="gauge posterior full_dimension",
    )
    _require(
        full_dimension == 7 * window_count,
        "gauge posterior dimension does not match seven parameters per window",
    )
    factor_rank = _require_integer(
        posterior.get("exported_factor_rank"),
        name="gauge posterior exported_factor_rank",
    )
    _require(
        factor_rank == len(belief.factor_names),
        "gauge posterior factor rank differs from the observation descriptor",
    )
    retained = _require_probability(
        posterior.get("retained_covariance_trace_fraction"),
        name="retained gauge covariance trace fraction",
    )
    minimum_retained = _require_probability(
        posterior.get("minimum_retained_gauge_trace"),
        name="minimum retained gauge covariance trace",
    )
    _require(
        retained + 1e-12 >= minimum_retained,
        "Prob4D joint gauge rank reduction violates its retained-trace threshold",
    )
    max_rank = posterior.get("max_gauge_rank")
    if max_rank is not None:
        _require(
            _require_integer(max_rank, name="gauge posterior max_gauge_rank")
            >= factor_rank,
            "gauge posterior max rank is below the exported factor rank",
        )
    _require(
        posterior.get("cross_window_covariance_preserved") is True,
        "Prob4D stream contract v2 lost cross-window gauge covariance",
    )
    _require(
        posterior.get("fixed_lag_boundary_covariance_is_approximate") is False,
        "Prob4D stream contract v2 cannot use approximate fixed-lag covariance",
    )
    parents = posterior.get("parent_window_ids")
    _require(
        isinstance(parents, list) and len(parents) == window_count,
        "gauge posterior parent lineage changed length",
    )
    _require(
        parents[0] is None,
        "the first Prob4D gauge must be rooted at the metric anchor",
    )
    for index, parent in enumerate(parents[1:], start=1):
        _require(
            parent in belief.window_names[:index],
            "joint gauge parent must identify an earlier retained window",
        )
    alignments = posterior.get("alignments")
    if alignments is not None:
        _require(
            isinstance(alignments, list),
            "gauge posterior alignments must be a list",
        )
        selected = [
            record
            for record in alignments
            if isinstance(record, Mapping)
            and record.get("selected_for_joint_tree") is True
        ]
        _require(
            len(selected) == max(window_count - 1, 0),
            "joint gauge tree must select exactly one edge per non-root window",
        )
    return {
        "gauge_covariance_semantics": (
            "joint_cross_window_sim3_gauge_covariance"
        ),
        "factor_group_count": 1,
        "factor_rank": factor_rank,
        "cross_window_covariance_preserved": True,
        "retained_covariance_trace_fraction": retained,
    }


def _validate_gauge_contract(
    belief: ObservationBeliefV1,
    metadata: Mapping[str, Any],
) -> tuple[int, bool, dict[str, object]]:
    version, inferred = _stream_contract_version(
        metadata,
        belief.factor_names,
    )
    if version == PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION:
        _require(
            belief.factor_names == PROB4D_GAUGE_FACTOR_NAMES,
            "Prob4D legacy gauge factor names changed",
        )
        _require(
            np.array_equal(
                belief.factor_group_ids,
                belief.window_indices,
            ),
            "Prob4D legacy gauge factor groups must equal window indices",
        )
        return (
            version,
            inferred,
            {
                "gauge_covariance_semantics": (
                    "per_window_sim3_gauge_marginals"
                ),
                "factor_group_count": len(
                    np.unique(belief.factor_group_ids)
                ),
                "factor_rank": len(belief.factor_names),
                "cross_window_covariance_preserved": False,
            },
        )
    return version, inferred, _validate_joint_gauge_contract(belief, metadata)


def _validate_metric_anchor(
    belief: ObservationBeliefV1,
    metadata: Mapping[str, Any],
    *,
    coordinate_frame: str,
    stream_version: int,
    stream_version_inferred: bool,
) -> tuple[str, str, str | None, str | None]:
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
        anchor.get("window_id") == belief.window_names[0],
        "metric gauge anchor does not identify the first window",
    )

    anchor_frame = anchor.get(
        "coordinate_frame",
        anchor.get("world_frame_id", coordinate_frame),
    )
    _require(
        anchor_frame == coordinate_frame,
        "metric gauge-anchor frame differs from observation frame",
    )
    if "world_frame_id" in anchor:
        _require(
            anchor.get("world_frame_id") == coordinate_frame,
            "metric gauge-anchor world frame differs from observation frame",
        )
    if anchor.get("case_id") is not None:
        _require(
            anchor.get("case_id") == belief.case_id,
            "metric gauge-anchor case differs from observation case",
        )

    calibration_digest = anchor.get("calibration_artifact_sha256")
    if calibration_digest is not None:
        calibration_digest = _require_sha256(
            calibration_digest,
            name="metric gauge-anchor calibration_artifact_sha256",
        )
    covariance_treatment = anchor.get("covariance_treatment")
    if covariance_treatment is not None:
        _require(
            covariance_treatment
            in {FIXED_EXTERNAL_CALIBRATION, PROPAGATED_EXTERNAL_PRIOR},
            "portable Prob4D causal artifact requires a fixed metric anchor "
            "or propagated external prior",
        )

    explicit_v2 = (
        stream_version == PROB4D_CAUSAL_STREAM_CONTRACT_VERSION
        and not stream_version_inferred
    )
    if explicit_v2:
        _require(
            anchor.get("schema_name") == "prob4d.metric-gauge-anchor"
            and _require_integer(
                anchor.get("schema_version"),
                name="metric gauge-anchor schema_version",
            )
            == 1,
            "Prob4D stream contract v2 requires metric gauge-anchor schema v1",
        )
        _require(
            anchor.get("case_id") == belief.case_id,
            "Prob4D stream contract v2 anchor must identify the observation case",
        )
        _require(
            anchor.get("coordinate_frame") == coordinate_frame
            and anchor.get("world_frame_id") == coordinate_frame,
            "Prob4D stream contract v2 anchor must identify the world frame",
        )
        _require(
            anchor.get("metric_units") == "m",
            "Prob4D stream contract v2 anchor must declare metric units",
        )
        _require(
            bool(str(anchor.get("source_kind", ""))),
            "Prob4D stream contract v2 anchor has no source kind",
        )
        _require(
            calibration_digest is not None,
            "Prob4D stream contract v2 anchor has no "
            "calibration_artifact_sha256",
        )
        _require(
            covariance_treatment
            in {FIXED_EXTERNAL_CALIBRATION, PROPAGATED_EXTERNAL_PRIOR},
            "Prob4D stream contract v2 anchor has no covariance treatment",
        )
        _require(
            metadata.get("metric_anchor_covariance_in_joint_factor") is True,
            "Prob4D stream contract v2 must include metric-anchor covariance "
            "in its joint factor",
        )
    elif stream_version == PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION:
        if covariance_treatment is not None:
            _require(
                covariance_treatment == FIXED_EXTERNAL_CALIBRATION,
                "legacy Prob4D artifact requires a fixed metric anchor",
            )
    elif "schema_name" in anchor:
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

    return (
        anchor_id,
        anchor_source_sha256,
        None if calibration_digest is None else str(calibration_digest),
        None if covariance_treatment is None else str(covariance_treatment),
    )


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
    stream_version, stream_version_inferred, factor_validation = (
        _validate_gauge_contract(belief, metadata)
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

    (
        anchor_id,
        anchor_source_sha256,
        calibration_artifact_sha256,
        covariance_treatment,
    ) = _validate_metric_anchor(
        belief,
        metadata,
        coordinate_frame=coordinate_frame,
        stream_version=stream_version,
        stream_version_inferred=stream_version_inferred,
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
        "stream_contract_version": stream_version,
        "stream_contract_version_inferred": stream_version_inferred,
        "causal_frame_stop": belief.causal_frame_stop,
        "window_count": len(belief.window_names),
        "metric_anchor_id": anchor_id,
        "metric_anchor_covariance_treatment": covariance_treatment,
        "calibration_artifact_sha256": calibration_artifact_sha256,
        "source_artifact_sha256": belief.source_artifact_sha256,
        **factor_validation,
    }


__all__ = [
    "FIXED_EXTERNAL_CALIBRATION",
    "PROB4D_CAUSAL_LINEAGE_VERSION",
    "PROB4D_CAUSAL_STREAM_CONTRACT_VERSION",
    "PROB4D_CAUSAL_STREAM_ID",
    "PROB4D_FIXED_LAG_GAUGE_MODEL",
    "PROB4D_GAUGE_FACTOR_NAMES",
    "PROB4D_JOINT_GAUGE_FACTOR_PREFIX",
    "PROB4D_JOINT_GAUGE_MODEL",
    "PROB4D_LEGACY_CAUSAL_STREAM_CONTRACT_VERSION",
    "PROB4D_LEGACY_GAUGE_FACTOR_NAMES",
    "PROB4D_SOURCE_REPOSITORY",
    "PROPAGATED_EXTERNAL_PRIOR",
    "is_prob4d_causal_observation_belief",
    "validate_prob4d_causal_observation_belief",
]
