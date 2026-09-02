"""Finite-group calibration for interventional cause-family adequacy.

The calibration score is the norm of the residual component orthogonal to the
complete registered cause-signature span. A split-conformal order statistic
turns independent source-group scores into a frozen adequacy radius before any
target residual is inspected.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Final

import numpy as np

from .interventional_cause_adequacy_v1 import (
    InterventionalCauseFamilyAdequacyV1,
)

CAUSE_FAMILY_CALIBRATION_SCHEMA: Final = (
    "bayesian_phystwin.interventional_cause_family_calibration"
)
CAUSE_FAMILY_CALIBRATION_VERSION: Final = 1
CAUSE_FAMILY_CALIBRATION_SEMANTICS: Final = (
    "split-conformal-source-group-none-of-the-above-radius-v1"
)
CAUSE_FAMILY_CALIBRATION_CLAIM_BOUNDARY: Final = (
    "Under exchangeability of the registered independent source groups and one "
    "future group, the frozen order statistic controls the probability that the "
    "future orthogonal residual norm exceeds the radius when the same registered "
    "cause-family data-generating assumptions hold. It does not establish that "
    "those assumptions are physically correct or complete, identify a cause, "
    "guarantee detection power, or authorize unseen-domain deployment."
)


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a 64-character lowercase hex digest")
    return value


def _finite(value: object, *, name: str, strictly_positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _canonical_id(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CauseFamilyAdequacyCalibrationV1:
    """Source-frozen split-conformal radius for one exact cause family."""

    cause_family_id: str
    intervention_roster_id: str
    whitening_id: str
    grouping_rule_id: str
    source_group_scores: Mapping[str, float]
    miscoverage_alpha: float
    candidate_family_frozen_before_scores: bool
    target_outcomes_used: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    calibration_id: str | None = None

    source_group_order: tuple[str, ...] = field(init=False)
    source_score_order_statistics: tuple[float, ...] = field(init=False)
    quantile_index_one_based: int = field(init=False)
    noise_radius: float = field(init=False)
    finite_sample_coverage_lower_bound: float = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "cause_family_id",
            "intervention_roster_id",
            "whitening_id",
            "grouping_rule_id",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name=name),
            )
        if type(self.candidate_family_frozen_before_scores) is not bool:
            raise ValueError("candidate_family_frozen_before_scores must be Boolean")
        if not self.candidate_family_frozen_before_scores:
            raise ValueError("candidate cause family must be frozen before scores")
        if type(self.target_outcomes_used) is not bool:
            raise ValueError("target_outcomes_used must be Boolean")
        if self.target_outcomes_used:
            raise ValueError("target outcomes cannot calibrate the adequacy radius")
        if not isinstance(self.source_group_scores, Mapping):
            raise TypeError("source_group_scores must be a mapping")
        group_order = tuple(sorted(self.source_group_scores))
        if len(group_order) < 2:
            raise ValueError("at least two independent source groups are required")
        if any(type(group) is not str or not group for group in group_order):
            raise ValueError("source group IDs must be nonempty literal strings")

        scores: dict[str, float] = {}
        for group in group_order:
            score = _finite(
                self.source_group_scores[group],
                name=f"source_group_scores[{group!r}]",
            )
            if score < 0.0:
                raise ValueError("source-group adequacy scores must be nonnegative")
            scores[group] = score

        alpha = _finite(
            self.miscoverage_alpha,
            name="miscoverage_alpha",
            strictly_positive=True,
        )
        if alpha >= 1.0:
            raise ValueError("miscoverage_alpha must be less than one")
        count = len(group_order)
        quantile_index = int(math.ceil((count + 1) * (1.0 - alpha)))
        if quantile_index > count:
            raise ValueError(
                "too few source groups for a finite conformal radius at this alpha"
            )
        order_statistics = tuple(sorted(scores.values()))
        radius = order_statistics[quantile_index - 1]
        lower_bound = quantile_index / (count + 1)
        if lower_bound + 1e-15 < 1.0 - alpha:
            raise RuntimeError("conformal quantile arithmetic is inconsistent")

        metadata = json.loads(
            json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        )
        object.__setattr__(self, "source_group_scores", scores)
        object.__setattr__(self, "miscoverage_alpha", alpha)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "source_group_order", group_order)
        object.__setattr__(
            self,
            "source_score_order_statistics",
            order_statistics,
        )
        object.__setattr__(self, "quantile_index_one_based", quantile_index)
        object.__setattr__(self, "noise_radius", radius)
        object.__setattr__(
            self,
            "finite_sample_coverage_lower_bound",
            lower_bound,
        )

        expected = _canonical_id(self.descriptor())
        supplied = self.calibration_id
        if supplied is not None:
            supplied = _digest(supplied, name="calibration_id")
            if supplied != expected:
                raise ValueError("calibration_id does not match calibration content")
        object.__setattr__(self, "calibration_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": CAUSE_FAMILY_CALIBRATION_SCHEMA,
            "schema_version": CAUSE_FAMILY_CALIBRATION_VERSION,
            "semantics": CAUSE_FAMILY_CALIBRATION_SEMANTICS,
            "cause_family_id": self.cause_family_id,
            "intervention_roster_id": self.intervention_roster_id,
            "whitening_id": self.whitening_id,
            "grouping_rule_id": self.grouping_rule_id,
            "source_group_order": list(self.source_group_order),
            "source_group_scores": {
                group: self.source_group_scores[group]
                for group in self.source_group_order
            },
            "miscoverage_alpha": self.miscoverage_alpha,
            "candidate_family_frozen_before_scores": (
                self.candidate_family_frozen_before_scores
            ),
            "target_outcomes_used": self.target_outcomes_used,
            "source_score_order_statistics": list(
                self.source_score_order_statistics
            ),
            "quantile_index_one_based": self.quantile_index_one_based,
            "noise_radius": self.noise_radius,
            "finite_sample_coverage_lower_bound": (
                self.finite_sample_coverage_lower_bound
            ),
            "metadata": self.metadata,
            "claim_boundary": CAUSE_FAMILY_CALIBRATION_CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "calibration_id": self.calibration_id}

    def certify(
        self,
        *,
        residual_id: str,
        cause_signature_ids: Mapping[str, str],
        cause_signatures: Mapping[str, np.ndarray],
        whitened_residual: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> InterventionalCauseFamilyAdequacyV1:
        """Apply the frozen radius to one target-closed residual certificate."""
        return InterventionalCauseFamilyAdequacyV1(
            residual_id=residual_id,
            intervention_roster_id=self.intervention_roster_id,
            whitening_id=self.whitening_id,
            cause_signature_ids=cause_signature_ids,
            cause_signatures=cause_signatures,
            whitened_residual=whitened_residual,
            noise_radius=self.noise_radius,
            metadata={
                **({} if metadata is None else dict(metadata)),
                "cause_family_calibration_id": self.calibration_id,
                "source_group_count": len(self.source_group_order),
                "miscoverage_alpha": self.miscoverage_alpha,
            },
        )
