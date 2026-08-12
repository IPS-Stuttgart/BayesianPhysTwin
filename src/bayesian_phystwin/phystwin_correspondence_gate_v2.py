"""Prospective fail-closed wrapper for the registered correspondence gate.

The v1 implementation is evidence-bound and must remain byte-for-byte stable.
This module validates public inputs strictly, then delegates valid numerical
work to v1 so the consensus algorithm, tie breaking, thresholds, and fallback
semantics remain identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .phystwin_correspondence_gate import (
    PairwiseCorrespondenceGateConfig as _V1Config,
)
from .phystwin_correspondence_gate import (
    detect_pairwise_consensus_correspondences as _detect_v1,
)
from .phystwin_correspondence_gate import (
    pairwise_distance_strain_m as _pairwise_distance_strain_v1,
)


def _strict_boolean_array(value: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.bool_):
        raise ValueError(f"{name} must contain only booleans")
    return np.asarray(raw, dtype=np.bool_)


def _numeric_position_array(value: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} positions must be numeric")
    return np.asarray(raw, dtype=np.float64)


def _strict_material_ids(value: np.ndarray, *, count: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (count,):
        raise ValueError("material_ids must have shape (K,)")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("material_ids must contain integers")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("material_ids exceed the supported signed 64-bit range")
    ids = raw.astype(np.int64, copy=False)
    if len(np.unique(ids)) != count:
        raise ValueError("material_ids must contain one unique ID per centre")
    return ids


def _strict_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class PairwiseCorrespondenceGateConfig:
    """Strict prospective configuration with v1-compatible defaults."""

    absolute_pair_strain_m: float = 0.030
    relative_pair_strain: float = 0.10
    minimum_inlier_count: int = 9
    minimum_inlier_fraction: float = 0.70
    maximum_exact_center_count: int = 24

    def __post_init__(self) -> None:
        absolute = _strict_real(
            self.absolute_pair_strain_m,
            name="absolute_pair_strain_m",
        )
        if absolute <= 0.0:
            raise ValueError("absolute_pair_strain_m must be positive")
        relative = _strict_real(
            self.relative_pair_strain,
            name="relative_pair_strain",
        )
        if relative < 0.0:
            raise ValueError("relative_pair_strain must be nonnegative")
        minimum_count = _strict_integer(
            self.minimum_inlier_count,
            name="minimum_inlier_count",
        )
        if minimum_count < 2:
            raise ValueError("minimum_inlier_count must be at least two")
        minimum_fraction = _strict_real(
            self.minimum_inlier_fraction,
            name="minimum_inlier_fraction",
        )
        if not 0.0 < minimum_fraction <= 1.0:
            raise ValueError("minimum_inlier_fraction must lie in (0, 1]")
        maximum_count = _strict_integer(
            self.maximum_exact_center_count,
            name="maximum_exact_center_count",
        )
        if maximum_count < minimum_count:
            raise ValueError(
                "maximum_exact_center_count must cover minimum_inlier_count"
            )
        object.__setattr__(self, "absolute_pair_strain_m", absolute)
        object.__setattr__(self, "relative_pair_strain", relative)
        object.__setattr__(self, "minimum_inlier_count", minimum_count)
        object.__setattr__(self, "minimum_inlier_fraction", minimum_fraction)
        object.__setattr__(self, "maximum_exact_center_count", maximum_count)

    def _as_v1(self) -> _V1Config:
        return _V1Config(
            absolute_pair_strain_m=self.absolute_pair_strain_m,
            relative_pair_strain=self.relative_pair_strain,
            minimum_inlier_count=self.minimum_inlier_count,
            minimum_inlier_fraction=self.minimum_inlier_fraction,
            maximum_exact_center_count=self.maximum_exact_center_count,
        )


@dataclass(frozen=True)
class PairwiseCorrespondenceGateResult:
    """Immutable prospective result on the caller's full centre axis."""

    inlier_mask: np.ndarray
    accepted: bool
    decision: str
    available_count: int
    inlier_count: int
    inlier_fraction: float
    pair_count: int
    compatible_pair_fraction: float
    median_inlier_normalized_strain: float | None
    maximum_inlier_normalized_strain: float | None

    def __post_init__(self) -> None:
        mask = _strict_boolean_array(self.inlier_mask, name="inlier_mask").copy()
        if mask.ndim != 1:
            raise ValueError("inlier_mask must be a vector")
        if not isinstance(self.accepted, (bool, np.bool_)):
            raise ValueError("accepted must be a boolean")
        available_count = _strict_integer(
            self.available_count,
            name="available_count",
        )
        inlier_count = _strict_integer(self.inlier_count, name="inlier_count")
        pair_count = _strict_integer(self.pair_count, name="pair_count")
        inlier_fraction = _strict_real(
            self.inlier_fraction,
            name="inlier_fraction",
        )
        compatible_fraction = _strict_real(
            self.compatible_pair_fraction,
            name="compatible_pair_fraction",
        )
        if available_count < 0 or available_count > len(mask):
            raise ValueError("available_count is inconsistent with inlier_mask")
        if inlier_count != int(np.sum(mask)):
            raise ValueError("inlier_count is inconsistent with inlier_mask")
        if inlier_count > available_count:
            raise ValueError("inlier_count exceeds available_count")
        if not 0.0 <= inlier_fraction <= 1.0:
            raise ValueError("inlier_fraction must lie in [0, 1]")
        if pair_count < 0:
            raise ValueError("pair_count must be nonnegative")
        if not 0.0 <= compatible_fraction <= 1.0:
            raise ValueError("compatible_pair_fraction must lie in [0, 1]")
        if not isinstance(self.decision, str) or not self.decision:
            raise ValueError("decision must be a nonempty string")
        for name in (
            "median_inlier_normalized_strain",
            "maximum_inlier_normalized_strain",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _strict_real(value, name=name))
        mask.setflags(write=False)
        object.__setattr__(self, "inlier_mask", mask)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "available_count", available_count)
        object.__setattr__(self, "inlier_count", inlier_count)
        object.__setattr__(self, "inlier_fraction", inlier_fraction)
        object.__setattr__(self, "pair_count", pair_count)
        object.__setattr__(
            self,
            "compatible_pair_fraction",
            compatible_fraction,
        )


def pairwise_distance_strain_m(
    source_positions_m: np.ndarray,
    observed_positions_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return v1 pair-distance diagnostics after strict numeric admission."""

    source = _numeric_position_array(source_positions_m, name="source")
    observed = _numeric_position_array(observed_positions_m, name="observed")
    return _pairwise_distance_strain_v1(source, observed)


def detect_pairwise_consensus_correspondences(
    source_positions_m: np.ndarray,
    observed_positions_m: np.ndarray,
    available: np.ndarray,
    *,
    material_ids: np.ndarray | None = None,
    config: PairwiseCorrespondenceGateConfig | None = None,
) -> PairwiseCorrespondenceGateResult:
    """Run the registered v1 algorithm behind a strict prospective boundary."""

    if config is None:
        cfg = PairwiseCorrespondenceGateConfig()
    elif isinstance(config, PairwiseCorrespondenceGateConfig):
        cfg = config
    else:
        raise TypeError("config must be a PairwiseCorrespondenceGateConfig")
    source = _numeric_position_array(source_positions_m, name="source")
    observed = _numeric_position_array(observed_positions_m, name="observed")
    mask = _strict_boolean_array(available, name="available")
    if source.ndim != 2 or source.shape[1] != 3 or observed.shape != source.shape:
        raise ValueError("source and observed positions must share shape (K, 3)")
    if mask.shape != (len(source),):
        raise ValueError("available must have shape (K,)")
    ids: np.ndarray
    if material_ids is None:
        ids = np.arange(len(source), dtype=np.int64)
    else:
        ids = _strict_material_ids(material_ids, count=len(source))

    result = _detect_v1(
        source,
        observed,
        mask,
        material_ids=ids,
        config=cfg._as_v1(),
    )
    return PairwiseCorrespondenceGateResult(
        inlier_mask=result.inlier_mask,
        accepted=result.accepted,
        decision=result.decision,
        available_count=result.available_count,
        inlier_count=result.inlier_count,
        inlier_fraction=result.inlier_fraction,
        pair_count=result.pair_count,
        compatible_pair_fraction=result.compatible_pair_fraction,
        median_inlier_normalized_strain=result.median_inlier_normalized_strain,
        maximum_inlier_normalized_strain=result.maximum_inlier_normalized_strain,
    )


__all__ = [
    "PairwiseCorrespondenceGateConfig",
    "PairwiseCorrespondenceGateResult",
    "detect_pairwise_consensus_correspondences",
    "pairwise_distance_strain_m",
]
