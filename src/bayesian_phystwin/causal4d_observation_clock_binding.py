"""Claim-bearing Causal4D clock-prior binding for timestamp lineages.

The timestamp binding records the exact shared clock-prior artifact ID, clock
domain, and time scale.  Claim-bearing use must validate the complete Causal4D
prior record against those values before exposing the compact Gaussian timing
prior used by the numerical update.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .causal4d_observation_clock_prior import (
    causal4d_observation_timing_prior_from_record,
)
from .observation_timing_nuisance import ObservationTimingPrior


class _TimestampBindingLike(Protocol):
    shared_clock_offset_prior_artifact_id: str | None
    clock_domain: str
    time_scale: str


def bind_causal4d_observation_clock_prior(
    binding: _TimestampBindingLike,
    value: Mapping[str, Any],
) -> ObservationTimingPrior:
    """Validate one complete Causal4D prior against timestamp-lineage identity.

    Compact five-field timing payloads are intentionally insufficient here: the
    complete source-execution panel, finite-sample predictive summary,
    information boundary, claim boundary, and Causal4D content ID are all
    revalidated before the numerical prior is returned.
    """

    expected_id = binding.shared_clock_offset_prior_artifact_id
    if expected_id is None:
        raise ValueError("timestamp lineage declares no shared clock prior")
    return causal4d_observation_timing_prior_from_record(
        value,
        expected_artifact_id=expected_id,
        expected_clock_domain=binding.clock_domain,
        expected_time_scale=binding.time_scale,
    )


__all__ = ["bind_causal4d_observation_clock_prior"]
