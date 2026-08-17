"""Exact adapter from portable observations to named structured covariance.

``ObservationBeliefV1`` stores one local covariance block per observation row and
one low-rank factor matrix whose latent vector is shared only inside each
``factor_group_id``. ``StructuredPointCovarianceV1`` instead stores one root per
named uncertainty component. This module expands factor groups into disjoint
column blocks, preserving the represented covariance exactly while making the
uncertainty budget explicit.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_integer,
    plain_json,
)
from .observation_belief import ObservationBeliefV1
from .structured_point_covariance import (
    SHARED_COVARIANCE_COMPONENTS,
    SharedCovarianceComponent,
    StructuredPointCovarianceV1,
)
from .structured_point_covariance_io import write_structured_point_covariance

OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_SCHEMA: Final = (
    "bayesian_phystwin.observation_structured_covariance_adapter"
)
OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_VERSION: Final = 1
OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_SEMANTICS: Final = (
    "exact-factor-group-disjoint-column-expansion-v1"
)
OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_CLAIM_BOUNDARY: Final = (
    "This adapter preserves the covariance represented by an ObservationBeliefV1 "
    "under an explicit caller-supplied component classification. It does not "
    "establish provider competence, covariance calibration, physical-mechanism "
    "identification, downstream benefit, deployment safety, or state of the art."
)


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a nonempty literal string without surrounding whitespace"
        )
    return value


def _component_mapping(
    observation: ObservationBeliefV1,
    value: Mapping[str, str],
) -> dict[str, SharedCovarianceComponent]:
    if not isinstance(value, Mapping):
        raise TypeError("factor_components must be a mapping")
    factor_names = observation.factor_names
    if len(set(factor_names)) != len(factor_names):
        raise ValueError(
            "observation factor_names must be unique for component classification"
        )
    missing = sorted(set(factor_names) - set(value))
    extra = sorted(set(value) - set(factor_names))
    if missing or extra:
        raise ValueError(
            "factor component mapping must identify every factor exactly; "
            f"missing={missing}, extra={extra}"
        )
    allowed = set(SHARED_COVARIANCE_COMPONENTS)
    result: dict[str, SharedCovarianceComponent] = {}
    for name in factor_names:
        component = value[name]
        if type(component) is not str or component not in allowed:
            raise ValueError(
                f"factor_components[{name!r}] must be one of "
                f"{list(SHARED_COVARIANCE_COMPONENTS)}"
            )
        result[name] = cast(SharedCovarianceComponent, component)
    return result


def observation_point_ids(observation: ObservationBeliefV1) -> tuple[str, ...]:
    """Return deterministic unique IDs for the observation rows."""

    if not isinstance(observation, ObservationBeliefV1):
        raise TypeError("observation must be an ObservationBeliefV1")
    return tuple(
        (
            f"{observation.case_id}|{observation.stream_id}|"
            f"frame={int(frame)}|entity={int(entity)}|"
            f"view={int(view)}|window={int(window)}"
        )
        for frame, entity, view, window in zip(
            observation.frame_ids,
            observation.entity_ids,
            observation.view_indices,
            observation.window_indices,
            strict=True,
        )
    )


def _expanded_component_factors(
    observation: ObservationBeliefV1,
    factor_components: Mapping[str, SharedCovarianceComponent],
    *,
    maximum_expanded_rank: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    factor_names = observation.factor_names
    if not factor_names:
        return {}, {}
    group_ids = tuple(int(value) for value in np.unique(observation.factor_group_ids))
    component_columns = {
        component: tuple(
            index
            for index, name in enumerate(factor_names)
            if factor_components[name] == component
        )
        for component in SHARED_COVARIANCE_COMPONENTS
    }
    expanded_ranks = {
        component: len(group_ids) * len(columns)
        for component, columns in component_columns.items()
        if columns
    }
    total_rank = sum(expanded_ranks.values())
    if total_rank > maximum_expanded_rank:
        raise ValueError(
            "factor-group expansion exceeds maximum_expanded_rank: "
            f"required={total_rank}, maximum={maximum_expanded_rank}"
        )

    factors: dict[str, np.ndarray] = {}
    row_group_ids = np.asarray(observation.factor_group_ids)
    source = np.asarray(observation.low_rank_factor_m)
    row_count = observation.observation_count
    for component in SHARED_COVARIANCE_COMPONENTS:
        columns = component_columns[component]
        if not columns:
            continue
        rank_per_group = len(columns)
        expanded = np.zeros(
            (row_count, 3, len(group_ids) * rank_per_group),
            dtype=np.float64,
        )
        column_index = np.asarray(columns, dtype=np.int64)
        for position, group_id in enumerate(group_ids):
            selected = row_group_ids == group_id
            start = position * rank_per_group
            stop = start + rank_per_group
            expanded[selected, :, start:stop] = source[selected][:, :, column_index]
        factors[component] = expanded
    return factors, expanded_ranks


def structured_covariance_from_observation_belief(
    observation: ObservationBeliefV1,
    *,
    coordinate_frame: str,
    factor_components: Mapping[str, str],
    calibration_artifact_id: str | None = None,
    maximum_expanded_rank: int = 4096,
    metadata: Mapping[str, Any] | None = None,
) -> StructuredPointCovarianceV1:
    """Preserve one observation belief as a named structured covariance budget.

    The caller must classify every factor column. Factor-group independence is
    represented by assigning each group a disjoint column block. No covariance
    approximation, truncation, rescaling, or calibration occurs.
    """

    if not isinstance(observation, ObservationBeliefV1):
        raise TypeError("observation must be an ObservationBeliefV1")
    frame = _literal_string(coordinate_frame, name="coordinate_frame")
    maximum_rank = genuine_integer(
        maximum_expanded_rank,
        name="maximum_expanded_rank",
        minimum=1,
    )
    components = _component_mapping(observation, factor_components)
    factors, expanded_ranks = _expanded_component_factors(
        observation,
        components,
        maximum_expanded_rank=maximum_rank,
    )
    if metadata is None:
        caller_metadata: Mapping[str, Any] = {}
    elif isinstance(metadata, Mapping):
        caller_metadata = metadata
    else:
        raise TypeError("metadata must be a mapping or None")
    frozen_caller_metadata = frozen_finite_json_mapping(
        caller_metadata,
        name="observation structured covariance metadata",
    )
    adapter_metadata: dict[str, Any] = {
        "adapter_schema": OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_SCHEMA,
        "adapter_schema_version": OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_VERSION,
        "adapter_semantics": OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_SEMANTICS,
        "observation_artifact_id": observation.artifact_id,
        "case_id": observation.case_id,
        "stream_id": observation.stream_id,
        "causal_frame_stop": observation.causal_frame_stop,
        "source_repository": observation.source_repository,
        "source_revision": observation.source_revision,
        "source_artifact_sha256": observation.source_artifact_sha256,
        "factor_group_semantics": (
            "rows with one factor_group_id share the declared latent columns; "
            "different factor_group_ids are represented by disjoint columns"
        ),
        "factor_components": dict(components),
        "factor_group_count": int(len(np.unique(observation.factor_group_ids))),
        "expanded_shared_ranks": expanded_ranks,
        "covariance_preserved_exactly": True,
        "caller": plain_json(frozen_caller_metadata),
    }
    return StructuredPointCovarianceV1(
        point_ids=observation_point_ids(observation),
        local_covariance_m2=observation.local_covariance_m2,
        shared_factors_m=factors,
        coordinate_frame=frame,
        source_artifact_id=observation.artifact_id,
        calibration_artifact_id=calibration_artifact_id,
        metadata=adapter_metadata,
    )


def write_observation_structured_covariance(
    path: str | Path,
    observation: ObservationBeliefV1,
    *,
    coordinate_frame: str,
    factor_components: Mapping[str, str],
    calibration_artifact_id: str | None = None,
    maximum_expanded_rank: int = 4096,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> StructuredPointCovarianceV1:
    """Adapt and atomically publish one identity-verified covariance archive."""

    covariance = structured_covariance_from_observation_belief(
        observation,
        coordinate_frame=coordinate_frame,
        factor_components=factor_components,
        calibration_artifact_id=calibration_artifact_id,
        maximum_expanded_rank=maximum_expanded_rank,
        metadata=metadata,
    )
    write_structured_point_covariance(path, covariance, overwrite=overwrite)
    return covariance


__all__ = [
    "OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_CLAIM_BOUNDARY",
    "OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_SCHEMA",
    "OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_SEMANTICS",
    "OBSERVATION_STRUCTURED_COVARIANCE_ADAPTER_VERSION",
    "observation_point_ids",
    "structured_covariance_from_observation_belief",
    "write_observation_structured_covariance",
]
