"""Closed-schema interchange for explicit observation timing priors."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .observation_timing_nuisance import ObservationTimingPrior

OBSERVATION_TIME_CORRECTION_CONVENTION = (
    "aligned_observation_time_s = observation_time_s + offset_s"
)
_FIELDS = frozenset(
    {
        "clock_domain",
        "mean_offset_s",
        "standard_deviation_s",
        "source_artifact_id",
        "offset_convention",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _nonempty_string(value: object, *, name: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{name} must be nonempty")
    result = str(value)
    _require(result == result.strip(), f"{name} has surrounding whitespace")
    return result


def _finite_float(value: object, *, name: str) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be finite")
    result = float(raw.item())
    _require(np.isfinite(result), f"{name} must be finite")
    return result


def observation_timing_prior_from_payload(
    value: Mapping[str, object],
) -> ObservationTimingPrior:
    """Validate a producer payload and construct the internal timing prior.

    The exact sign convention is checked before the scalar mean is admitted.
    This prevents a producer's fitted offset from being silently reversed at the
    BayesianPhysTwin boundary.
    """

    _require(isinstance(value, Mapping), "timing prior payload must be a mapping")
    _require(set(value) == _FIELDS, "timing prior payload fields changed")
    convention = _nonempty_string(
        value["offset_convention"],
        name="offset_convention",
    )
    _require(
        convention == OBSERVATION_TIME_CORRECTION_CONVENTION,
        "observation time-correction convention changed",
    )
    clock_domain = _nonempty_string(
        value["clock_domain"],
        name="clock_domain",
    )
    mean_offset_s = _finite_float(value["mean_offset_s"], name="mean_offset_s")
    standard_deviation_s = _finite_float(
        value["standard_deviation_s"],
        name="standard_deviation_s",
    )
    _require(
        standard_deviation_s > 0.0,
        "standard_deviation_s must be positive",
    )
    source_artifact_id = _nonempty_string(
        value["source_artifact_id"],
        name="source_artifact_id",
    )
    return ObservationTimingPrior(
        clock_domain=clock_domain,
        mean_offset_s=mean_offset_s,
        standard_deviation_s=standard_deviation_s,
        source_artifact_id=source_artifact_id,
    )


def observation_timing_prior_payload(
    prior: ObservationTimingPrior,
) -> dict[str, str | float]:
    """Export an internal timing prior with its explicit sign convention."""

    return {
        "clock_domain": prior.clock_domain,
        "mean_offset_s": float(prior.mean_offset_s),
        "standard_deviation_s": float(prior.standard_deviation_s),
        "source_artifact_id": prior.source_artifact_id,
        "offset_convention": OBSERVATION_TIME_CORRECTION_CONVENTION,
    }


__all__ = [
    "OBSERVATION_TIME_CORRECTION_CONVENTION",
    "observation_timing_prior_from_payload",
    "observation_timing_prior_payload",
]
