"""Explicit semantics for covariance reported by Bayesian belief updates.

A covariance matrix can be a working IRLS/Gauss--Newton approximation, a local
observed-information approximation, a group-score sandwich correction, or the
exact prior covariance retained after a rejected update. The numerical array
alone does not say which interpretation is valid. This module adds a small
content-addressed contract for that distinction without changing any frozen
solver or covariance bytes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id

POSTERIOR_COVARIANCE_SEMANTICS_SCHEMA = (
    "bayesian_phystwin.posterior_covariance_semantics"
)
POSTERIOR_COVARIANCE_SEMANTICS_VERSION = 1
EXACT_PRIOR_FALLBACK_LIKELIHOOD_SEMANTICS = (
    "not-applicable-exact-prior-fallback-v1"
)

PosteriorCovarianceMethod = Literal[
    "irls_working",
    "laplace_observed_information",
    "group_sandwich",
    "exact_prior_fallback",
]
POSTERIOR_COVARIANCE_METHODS: tuple[PosteriorCovarianceMethod, ...] = (
    "irls_working",
    "laplace_observed_information",
    "group_sandwich",
    "exact_prior_fallback",
)


def _nonempty_literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _method(value: object) -> PosteriorCovarianceMethod:
    if type(value) is not str or value not in POSTERIOR_COVARIANCE_METHODS:
        raise ValueError(
            "method must be one of " f"{list(POSTERIOR_COVARIANCE_METHODS)}"
        )
    return cast(PosteriorCovarianceMethod, value)


def _covariance_dimension(covariance: object) -> int:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("covariance must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be finite")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
        raise ValueError("covariance must be symmetric")
    if np.min(np.linalg.eigvalsh(0.5 * (matrix + matrix.T))) < -1e-9:
        raise ValueError("covariance must be positive semidefinite")
    return len(matrix)


@dataclass(frozen=True, slots=True)
class PosteriorCovarianceSemanticsV1:
    """Machine-readable interpretation of one reported covariance matrix."""

    method: PosteriorCovarianceMethod
    dimension: int
    likelihood_power_semantics: str
    prior_included: bool = True
    generalized_bayes: bool = True
    mixture_curvature_exact: bool = False
    group_score_correction: bool = False
    calibrated: bool = False
    calibration_artifact_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        method = _method(self.method)
        dimension = genuine_integer(
            self.dimension,
            name="dimension",
            minimum=1,
        )
        likelihood_power_semantics = _nonempty_literal_string(
            self.likelihood_power_semantics,
            name="likelihood_power_semantics",
        )
        prior_included = genuine_boolean(
            self.prior_included,
            name="prior_included",
        )
        generalized_bayes = genuine_boolean(
            self.generalized_bayes,
            name="generalized_bayes",
        )
        mixture_curvature_exact = genuine_boolean(
            self.mixture_curvature_exact,
            name="mixture_curvature_exact",
        )
        group_score_correction = genuine_boolean(
            self.group_score_correction,
            name="group_score_correction",
        )
        calibrated = genuine_boolean(self.calibrated, name="calibrated")

        expected = {
            "irls_working": (False, False),
            "laplace_observed_information": (True, False),
            "group_sandwich": (False, True),
            "exact_prior_fallback": (False, False),
        }[method]
        if (mixture_curvature_exact, group_score_correction) != expected:
            raise ValueError(
                "curvature and group-score flags contradict covariance method"
            )

        if method == "exact_prior_fallback":
            if (
                likelihood_power_semantics
                != EXACT_PRIOR_FALLBACK_LIKELIHOOD_SEMANTICS
            ):
                raise ValueError(
                    "exact prior fallback has fixed likelihood-power semantics"
                )
            if not prior_included:
                raise ValueError("exact prior fallback must include the prior")
            if generalized_bayes:
                raise ValueError(
                    "exact prior fallback is not a generalized-Bayes covariance"
                )
            if calibrated:
                raise ValueError("exact prior fallback cannot be marked calibrated")

        calibration_id = self.calibration_artifact_id
        if calibrated:
            calibration_id = literal_lower_hex(
                calibration_id,
                name="calibration_artifact_id",
                lengths={64},
            )
        elif calibration_id is not None:
            raise ValueError(
                "calibration_artifact_id is allowed only when calibrated is true"
            )

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="posterior covariance semantics metadata",
        )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(
            self,
            "likelihood_power_semantics",
            likelihood_power_semantics,
        )
        object.__setattr__(self, "prior_included", prior_included)
        object.__setattr__(self, "generalized_bayes", generalized_bayes)
        object.__setattr__(
            self,
            "mixture_curvature_exact",
            mixture_curvature_exact,
        )
        object.__setattr__(
            self,
            "group_score_correction",
            group_score_correction,
        )
        object.__setattr__(self, "calibrated", calibrated)
        object.__setattr__(self, "calibration_artifact_id", calibration_id)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = literal_lower_hex(
                supplied_id,
                name="artifact_id",
                lengths={64},
            )
            if supplied_id != expected_id:
                raise ValueError(
                    "posterior covariance semantics artifact_id does not match content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    def policy_descriptor(self) -> dict[str, object]:
        """Return the dimension-independent interpretation policy."""

        return {
            "schema": POSTERIOR_COVARIANCE_SEMANTICS_SCHEMA,
            "schema_version": POSTERIOR_COVARIANCE_SEMANTICS_VERSION,
            "method": self.method,
            "likelihood_power_semantics": self.likelihood_power_semantics,
            "prior_included": self.prior_included,
            "generalized_bayes": self.generalized_bayes,
            "mixture_curvature_exact": self.mixture_curvature_exact,
            "group_score_correction": self.group_score_correction,
            "calibrated": self.calibrated,
            "calibration_artifact_id": self.calibration_artifact_id,
        }

    @property
    def policy_id(self) -> str:
        """Return the stable policy ID shared across compatible dimensions."""

        return content_id(self.policy_descriptor())

    def descriptor(self) -> dict[str, object]:
        return {
            **self.policy_descriptor(),
            "dimension": self.dimension,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        name: str = "posterior covariance semantics",
    ) -> PosteriorCovarianceSemanticsV1:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
        expected_fields = {
            "schema",
            "schema_version",
            "method",
            "dimension",
            "likelihood_power_semantics",
            "prior_included",
            "generalized_bayes",
            "mixture_curvature_exact",
            "group_score_correction",
            "calibrated",
            "calibration_artifact_id",
            "metadata",
            "artifact_id",
        }
        if set(value) != expected_fields:
            raise ValueError(f"{name} fields changed")
        if value["schema"] != POSTERIOR_COVARIANCE_SEMANTICS_SCHEMA:
            raise ValueError(f"{name} schema changed")
        version = genuine_integer(
            value["schema_version"],
            name=f"{name} schema_version",
            minimum=1,
        )
        if version != POSTERIOR_COVARIANCE_SEMANTICS_VERSION:
            raise ValueError(f"{name} version changed")
        return cls(
            method=cast(PosteriorCovarianceMethod, value["method"]),
            dimension=cast(int, value["dimension"]),
            likelihood_power_semantics=cast(
                str,
                value["likelihood_power_semantics"],
            ),
            prior_included=cast(bool, value["prior_included"]),
            generalized_bayes=cast(bool, value["generalized_bayes"]),
            mixture_curvature_exact=cast(
                bool,
                value["mixture_curvature_exact"],
            ),
            group_score_correction=cast(
                bool,
                value["group_score_correction"],
            ),
            calibrated=cast(bool, value["calibrated"]),
            calibration_artifact_id=cast(
                str | None,
                value["calibration_artifact_id"],
            ),
            metadata=cast(Mapping[str, Any], value["metadata"]),
            artifact_id=cast(str, value["artifact_id"]),
        )


def working_irls_covariance_semantics(
    covariance: np.ndarray,
    *,
    likelihood_power_semantics: str = (
        "grouped-student-t-generalized-bayes-power-v1"
    ),
    metadata: Mapping[str, Any] | None = None,
) -> PosteriorCovarianceSemanticsV1:
    """Describe the current working IRLS covariance without a calibration claim."""

    return PosteriorCovarianceSemanticsV1(
        method="irls_working",
        dimension=_covariance_dimension(covariance),
        likelihood_power_semantics=likelihood_power_semantics,
        prior_included=True,
        generalized_bayes=True,
        mixture_curvature_exact=False,
        group_score_correction=False,
        calibrated=False,
        metadata=metadata or {},
    )


def exact_prior_fallback_covariance_semantics(
    covariance: np.ndarray,
    *,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> PosteriorCovarianceSemanticsV1:
    """Describe the exact prior covariance retained after a rejected update."""

    fallback_reason = _nonempty_literal_string(reason, name="reason")
    details = dict(metadata or {})
    recorded_reason = details.get("fallback_reason")
    if recorded_reason is not None and recorded_reason != fallback_reason:
        raise ValueError("metadata fallback_reason contradicts reason")
    details["fallback_reason"] = fallback_reason
    return PosteriorCovarianceSemanticsV1(
        method="exact_prior_fallback",
        dimension=_covariance_dimension(covariance),
        likelihood_power_semantics=(
            EXACT_PRIOR_FALLBACK_LIKELIHOOD_SEMANTICS
        ),
        prior_included=True,
        generalized_bayes=False,
        mixture_curvature_exact=False,
        group_score_correction=False,
        calibrated=False,
        metadata=details,
    )


__all__ = [
    "EXACT_PRIOR_FALLBACK_LIKELIHOOD_SEMANTICS",
    "POSTERIOR_COVARIANCE_METHODS",
    "POSTERIOR_COVARIANCE_SEMANTICS_SCHEMA",
    "POSTERIOR_COVARIANCE_SEMANTICS_VERSION",
    "PosteriorCovarianceMethod",
    "PosteriorCovarianceSemanticsV1",
    "exact_prior_fallback_covariance_semantics",
    "working_irls_covariance_semantics",
]
