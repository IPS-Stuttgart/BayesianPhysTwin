"""Sentinel-debiased endpoint measurements for guarded Deform360 updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .deform360_dynamic_tapnextpp_assimilation import (
    BirthAnchoredMeasurements,
)
from .deform360_sentinel_query_schedule import (
    Deform360SentinelQuerySchedule,
)
from .phystwin_sentinel_queries import (
    ACTIVE_QUERY_ROLE,
    SENTINEL_QUERY_ROLE,
    SentinelBiasConfig,
    SentinelBiasEstimate,
    debias_active_displacements,
    estimate_sentinel_common_bias,
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class SentinelDebiasConfig:
    """Frozen requirements for admitting a sentinel-debiased update."""

    require_all_sentinels: bool = True
    shared_correlation_group_id: int = 0
    bias: SentinelBiasConfig = SentinelBiasConfig(
        minimum_reliability=0.05,
        covariance_floor_m2=1e-10,
        maximum_inconsistency_sigma=4.0,
        minimum_correlation_groups=1,
    )


@dataclass(frozen=True)
class SentinelDebiasResult:
    """Debiased active measurements or an exact-persistence abstention."""

    measurements: BirthAnchoredMeasurements
    estimate: SentinelBiasEstimate
    applied: bool
    decision: str
    active_entity_ids: np.ndarray
    sentinel_entity_ids: np.ndarray
    supported_active_count: int
    supported_sentinel_count: int

    def __post_init__(self) -> None:
        active = np.ascontiguousarray(
            np.asarray(self.active_entity_ids, dtype=np.int64)
        )
        sentinel = np.ascontiguousarray(
            np.asarray(self.sentinel_entity_ids, dtype=np.int64)
        )
        _require(
            active.ndim == sentinel.ndim == 1,
            "sentinel debias entity IDs must be one-dimensional",
        )
        _require(
            len(np.unique(active)) == len(active)
            and len(np.unique(sentinel)) == len(sentinel)
            and set(map(int, active)).isdisjoint(map(int, sentinel)),
            "sentinel debias entity roles overlap or repeat",
        )
        _require(
            0 <= self.supported_active_count <= len(active)
            and 0 <= self.supported_sentinel_count <= len(sentinel),
            "sentinel debias support count is invalid",
        )
        _require(
            self.applied == self.estimate.usable,
            "sentinel debias application differs from bias admission",
        )
        if self.applied:
            _require(
                self.supported_sentinel_count == len(sentinel),
                "applied sentinel update lacks complete sentinel support",
            )
        else:
            _require(
                not np.any(self.measurements.available),
                "rejected sentinel update must retain exact persistence",
            )
        active.setflags(write=False)
        sentinel.setflags(write=False)
        object.__setattr__(self, "active_entity_ids", active)
        object.__setattr__(self, "sentinel_entity_ids", sentinel)

    def report(self) -> dict[str, Any]:
        """Return a JSON-safe admission record."""

        return {
            "schema_version": 1,
            "artifact_kind": "Deform360SentinelDebiasAdmission",
            "applied": self.applied,
            "decision": self.decision,
            "active_entity_ids": self.active_entity_ids.tolist(),
            "sentinel_entity_ids": self.sentinel_entity_ids.tolist(),
            "supported_active_count": self.supported_active_count,
            "supported_sentinel_count": self.supported_sentinel_count,
            "bias_estimate": {
                "bias_m": (
                    None
                    if self.estimate.bias_m is None
                    else self.estimate.bias_m.tolist()
                ),
                "covariance_m2": (
                    None
                    if self.estimate.covariance_m2 is None
                    else self.estimate.covariance_m2.tolist()
                ),
                "observation_count": self.estimate.observation_count,
                "correlation_group_count": (
                    self.estimate.correlation_group_count
                ),
                "maximum_inconsistency_sigma": (
                    self.estimate.maximum_inconsistency_sigma
                ),
                "usable": self.estimate.usable,
                "decision": self.estimate.decision,
            },
            "method_contract": {
                "sentinel_residual": (
                    "observed displacement minus physical-prefix displacement"
                ),
                "within_panel_cross_identity_correlation": (
                    "one shared unknown-correlation group"
                ),
                "active_measurement_debiased": self.applied,
                "bias_covariance_added_to_active_covariance": self.applied,
                "sentinels_used_as_state_measurements": False,
                "rejection": "bit-exact persistence",
            },
        }


def _empty_measurements(
    physical_shape: tuple[int, int, int],
    active_entity_ids: np.ndarray,
) -> BirthAnchoredMeasurements:
    frame_count, node_count, coordinate_count = physical_shape
    _require(coordinate_count == 3, "physical coordinate count changed")
    measurement: np.ndarray = np.full(
        (frame_count, node_count, coordinate_count),
        np.nan,
        dtype=np.float64,
    )
    covariance: np.ndarray = np.full(
        (frame_count, node_count, coordinate_count, coordinate_count),
        np.nan,
        dtype=np.float64,
    )
    probability: np.ndarray = np.zeros(
        (frame_count, node_count),
        dtype=np.float64,
    )
    available: np.ndarray = np.zeros(
        (frame_count, node_count),
        dtype=bool,
    )
    return BirthAnchoredMeasurements(
        measurement_m=measurement,
        covariance_m2=covariance,
        prior_reliability=probability,
        association_probability=probability,
        available=available,
        entity_ids=active_entity_ids,
    )


def build_sentinel_debiased_measurements(
    measurements: BirthAnchoredMeasurements,
    schedule: Deform360SentinelQuerySchedule,
    physical_prediction_m: np.ndarray,
    *,
    config: SentinelDebiasConfig | None = None,
) -> SentinelDebiasResult:
    """Use near-static graph identities to remove one coherent gauge term.

    All fused sentinel identities belong to one unknown-correlation group,
    because they use the same camera panel and tracker.  Duplicating identities
    therefore cannot create independent confidence.  Rejection returns a
    measurement object with no available rows, which makes the unchanged
    dynamic predictor emit bit-exact persistence.
    """

    cfg = config or SentinelDebiasConfig()
    physical = np.asarray(physical_prediction_m, dtype=np.float64)
    _require(
        physical.ndim == 3
        and physical.shape[2] == 3
        and np.all(np.isfinite(physical)),
        "physical prediction must have finite shape (T, N, 3)",
    )
    _require(
        measurements.measurement_m.shape == physical.shape,
        "sentinel measurements and physical prediction differ",
    )
    entities = np.asarray(schedule.entity_ids, dtype=np.int64)
    roles = np.asarray(schedule.query_roles)
    _require(
        np.array_equal(measurements.entity_ids, entities),
        "measurement identities differ from the sentinel schedule",
    )
    _require(
        np.all(
            schedule.birth_frames == schedule.config.query_birth_frame
        )
        and np.all(
            schedule.update_frames == schedule.config.query_update_frame
        ),
        "sentinel schedule endpoint semantics changed",
    )
    birth_frame = schedule.config.query_birth_frame
    update_frame = schedule.config.query_update_frame
    active_ids = entities[roles == ACTIVE_QUERY_ROLE]
    sentinel_ids = entities[roles == SENTINEL_QUERY_ROLE]
    _require(
        len(active_ids) == schedule.config.active_query_count
        and len(sentinel_ids) == schedule.config.sentinel_query_count,
        "sentinel schedule role count changed",
    )
    sentinel_available = measurements.available[
        update_frame,
        sentinel_ids,
    ]
    active_available = measurements.available[
        update_frame,
        active_ids,
    ]
    supported_sentinel_count = int(np.sum(sentinel_available))
    supported_active_count = int(np.sum(active_available))
    empty = _empty_measurements(physical.shape, active_ids)

    if cfg.require_all_sentinels and not np.all(sentinel_available):
        estimate = SentinelBiasEstimate(
            bias_m=None,
            covariance_m2=None,
            observation_count=supported_sentinel_count,
            correlation_group_count=0,
            maximum_inconsistency_sigma=None,
            usable=False,
            decision="incomplete-sentinel-endpoint-support",
        )
        return SentinelDebiasResult(
            measurements=empty,
            estimate=estimate,
            applied=False,
            decision="exact-persistence-incomplete-sentinel-support",
            active_entity_ids=active_ids,
            sentinel_entity_ids=sentinel_ids,
            supported_active_count=supported_active_count,
            supported_sentinel_count=supported_sentinel_count,
        )

    selected_sentinels = sentinel_ids[sentinel_available]
    observed_sentinel_displacement = (
        measurements.measurement_m[
            update_frame,
            selected_sentinels,
        ]
        - physical[birth_frame, selected_sentinels]
    )
    predicted_sentinel_displacement = (
        physical[update_frame, selected_sentinels]
        - physical[birth_frame, selected_sentinels]
    )
    estimate = estimate_sentinel_common_bias(
        observed_sentinel_displacement,
        predicted_sentinel_displacement,
        measurements.covariance_m2[
            update_frame,
            selected_sentinels,
        ],
        measurements.prior_reliability[
            update_frame,
            selected_sentinels,
        ],
        np.full(
            len(selected_sentinels),
            cfg.shared_correlation_group_id,
            dtype=np.int64,
        ),
        config=cfg.bias,
    )
    if not estimate.usable:
        return SentinelDebiasResult(
            measurements=empty,
            estimate=estimate,
            applied=False,
            decision=f"exact-persistence-{estimate.decision}",
            active_entity_ids=active_ids,
            sentinel_entity_ids=sentinel_ids,
            supported_active_count=supported_active_count,
            supported_sentinel_count=supported_sentinel_count,
        )

    measurement = np.full(physical.shape, np.nan, dtype=np.float64)
    covariance = np.full((*physical.shape[:2], 3, 3), np.nan, dtype=np.float64)
    reliability = np.zeros(physical.shape[:2], dtype=np.float64)
    association = np.zeros(physical.shape[:2], dtype=np.float64)
    available = np.zeros(physical.shape[:2], dtype=bool)
    selected_active = active_ids[active_available]
    if len(selected_active):
        observed_active_displacement = (
            measurements.measurement_m[update_frame, selected_active]
            - physical[birth_frame, selected_active]
        )
        debiased_displacement, debiased_covariance = (
            debias_active_displacements(
                observed_active_displacement,
                measurements.covariance_m2[
                    update_frame,
                    selected_active,
                ],
                estimate,
            )
        )
        measurement[update_frame, selected_active] = (
            physical[birth_frame, selected_active] + debiased_displacement
        )
        covariance[update_frame, selected_active] = debiased_covariance
        reliability[update_frame, selected_active] = (
            measurements.prior_reliability[
                update_frame,
                selected_active,
            ]
        )
        association[update_frame, selected_active] = (
            measurements.association_probability[
                update_frame,
                selected_active,
            ]
        )
        available[update_frame, selected_active] = True
    debiased = BirthAnchoredMeasurements(
        measurement_m=measurement,
        covariance_m2=covariance,
        prior_reliability=reliability,
        association_probability=association,
        available=available,
        entity_ids=active_ids,
    )
    return SentinelDebiasResult(
        measurements=debiased,
        estimate=estimate,
        applied=True,
        decision="sentinel-common-mode-debiased",
        active_entity_ids=active_ids,
        sentinel_entity_ids=sentinel_ids,
        supported_active_count=supported_active_count,
        supported_sentinel_count=supported_sentinel_count,
    )


__all__ = [
    "SentinelDebiasConfig",
    "SentinelDebiasResult",
    "build_sentinel_debiased_measurements",
]
