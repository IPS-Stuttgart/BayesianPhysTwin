"""Calibration-frozen scale-plus-floor transforms for predictive covariance."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    plain_json,
)
from ._portable_contracts import content_id, sha256_digest
from .calibration_domain_guard import CalibrationDomainGuardCertificateV1

DOMAIN_COVARIANCE_CALIBRATION_SCHEMA = "bayesian_phystwin.domain_covariance_calibration"
DOMAIN_COVARIANCE_CALIBRATION_VERSION = 1
DOMAIN_COVARIANCE_DECISION_SCHEMA = (
    "bayesian_phystwin.domain_covariance_calibration_decision"
)
DOMAIN_COVARIANCE_DECISION_VERSION = 1
DOMAIN_COVARIANCE_APPLICATION_SCHEMA = (
    "bayesian_phystwin.domain_covariance_calibration_application"
)
DOMAIN_COVARIANCE_APPLICATION_VERSION = 1
DOMAIN_COVARIANCE_DATA_SCHEMA = "bayesian_phystwin.domain_covariance_calibration_data"
DOMAIN_COVARIANCE_DATA_VERSION = 1


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _strings(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    try:
        source = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of strings") from error
    if not source:
        raise ValueError(f"{name} must not be empty")
    return tuple(
        _canonical_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(source)
    )


def _number(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite number")
    result = float(raw.item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _grid(
    values: Sequence[float],
    *,
    name: str,
    positive: bool,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of finite numbers")
    try:
        source = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of finite numbers") from error
    if not source:
        raise ValueError(f"{name} must not be empty")
    result = tuple(
        _number(
            value,
            name=f"{name}[{index}]",
            positive=positive,
            nonnegative=not positive,
        )
        for index, value in enumerate(source)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(result))


def _residuals(value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2:
        raise ValueError("residuals must be a two-dimensional real numeric array")
    result = np.asarray(raw, dtype=np.float64)
    if result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError("residuals must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError("residuals must contain only finite values")
    return np.array(result, dtype=np.float64, copy=True, order="C")


def _covariances(value: object, *, calibration: bool) -> tuple[np.ndarray, bool]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("covariances must contain real numeric values")
    single = raw.ndim == 2
    if single:
        raw = raw[None, :, :]
    elif raw.ndim != 3:
        raise ValueError("covariances must have shape (d, d) or (m, d, d)")
    if calibration and single:
        raise ValueError("calibration covariances must have shape (events, d, d)")
    result = np.asarray(raw, dtype=np.float64)
    if result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError("covariances must not be empty")
    if result.shape[1] != result.shape[2]:
        raise ValueError("covariance matrices must be square")
    if not np.all(np.isfinite(result)):
        raise ValueError("covariances must contain only finite values")
    return np.array(result, dtype=np.float64, copy=True, order="C"), single


def _validate_covariances(value: np.ndarray, *, tolerance: float) -> None:
    transposed = np.swapaxes(value, -1, -2)
    if not np.allclose(value, transposed, atol=tolerance, rtol=tolerance):
        raise ValueError("covariances must be symmetric")
    eigenvalues = np.linalg.eigvalsh(0.5 * (value + transposed))
    if float(np.min(eigenvalues, initial=0.0)) < -tolerance:
        raise ValueError("covariances must be positive semidefinite")


def _transform(value: np.ndarray, scale: float, floor: float) -> np.ndarray:
    identity = np.eye(value.shape[-1], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        result = scale * value + floor * identity
    if not np.all(np.isfinite(result)):
        raise ValueError("transformed covariance must be finite")
    return 0.5 * (result + np.swapaxes(result, -1, -2))


def _nll(
    residual: np.ndarray,
    covariance: np.ndarray,
    *,
    eigenvalue_floor: float,
) -> float:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
    projected = eigenvectors.T @ residual
    return 0.5 * (
        len(residual) * math.log(2.0 * math.pi)
        + float(np.sum(np.log(eigenvalues)))
        + float(np.sum(np.square(projected) / eigenvalues))
    )


def _event_scores(
    residuals: np.ndarray,
    covariances: np.ndarray,
    *,
    scale: float,
    floor: float,
    eigenvalue_floor: float,
) -> np.ndarray:
    transformed = _transform(covariances, scale, floor)
    return np.asarray(
        [
            _nll(
                residuals[index],
                transformed[index],
                eigenvalue_floor=eigenvalue_floor,
            )
            for index in range(len(residuals))
        ],
        dtype=np.float64,
    )


def _equal_group_mean(values: np.ndarray, groups: Sequence[str]) -> float:
    roster = tuple(groups)
    return float(
        np.mean(
            [
                np.mean(values[[value == group for value in roster]])
                for group in sorted(set(roster))
            ]
        )
    )


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0float64\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _immutable(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )


@dataclass(frozen=True, slots=True)
class DomainCovarianceCalibrationConfigV1:
    """Finite transform grid and cross-fitted support thresholds."""

    covariance_scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    isotropic_variances: Sequence[float] = (0.0, 1e-8, 1e-6, 1e-4)
    minimum_group_count: int = 4
    minimum_mean_loo_nll_improvement: float = 0.0
    maximum_single_group_loo_nll_regression: float = 0.0
    scoring_eigenvalue_floor: float = 1e-12
    covariance_psd_tolerance: float = 1e-10
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        scales = _grid(
            self.covariance_scales,
            name="covariance_scales",
            positive=True,
        )
        floors = _grid(
            self.isotropic_variances,
            name="isotropic_variances",
            positive=False,
        )
        if 1.0 not in scales or 0.0 not in floors:
            raise ValueError("the transform grid must include raw covariance (1, 0)")
        object.__setattr__(self, "covariance_scales", scales)
        object.__setattr__(self, "isotropic_variances", floors)
        object.__setattr__(
            self,
            "minimum_group_count",
            genuine_integer(
                self.minimum_group_count,
                name="minimum_group_count",
                minimum=2,
            ),
        )
        for name in (
            "minimum_mean_loo_nll_improvement",
            "maximum_single_group_loo_nll_regression",
            "covariance_psd_tolerance",
            "numerical_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _number(getattr(self, name), name=name, nonnegative=True),
            )
        object.__setattr__(
            self,
            "scoring_eigenvalue_floor",
            _number(
                self.scoring_eigenvalue_floor,
                name="scoring_eigenvalue_floor",
                positive=True,
            ),
        )

    @property
    def transforms(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (scale, floor)
            for scale in self.covariance_scales
            for floor in self.isotropic_variances
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "covariance_scales": list(self.covariance_scales),
            "isotropic_variances": list(self.isotropic_variances),
            "minimum_group_count": self.minimum_group_count,
            "minimum_mean_loo_nll_improvement": (self.minimum_mean_loo_nll_improvement),
            "maximum_single_group_loo_nll_regression": (
                self.maximum_single_group_loo_nll_regression
            ),
            "scoring_eigenvalue_floor": self.scoring_eigenvalue_floor,
            "covariance_psd_tolerance": self.covariance_psd_tolerance,
            "numerical_tolerance": self.numerical_tolerance,
        }


TransformScore = tuple[float, float, float]
HeldGroupScore = tuple[str, float, float, float, float]


def _transform_score(value: Sequence[object]) -> TransformScore:
    row = tuple(value)
    if len(row) != 3:
        raise ValueError("each transform score must have three fields")
    return (
        _number(row[0], name="covariance_scale", positive=True),
        _number(row[1], name="isotropic_variance", nonnegative=True),
        _number(row[2], name="equal_group_mean_nll"),
    )


def _held_score(value: Sequence[object]) -> HeldGroupScore:
    row = tuple(value)
    if len(row) != 5:
        raise ValueError("each held-group score must have five fields")
    return (
        _canonical_string(row[0], name="held_group_id"),
        _number(row[1], name="selected_covariance_scale", positive=True),
        _number(row[2], name="selected_isotropic_variance", nonnegative=True),
        _number(row[3], name="raw_held_group_nll"),
        _number(row[4], name="calibrated_held_group_nll"),
    )


def _tie_break(score: TransformScore) -> tuple[float, float, float, float]:
    scale, floor, nll = score
    return (nll, floor, abs(math.log(scale)), scale)


@dataclass(frozen=True, slots=True)
class DomainCovarianceCalibrationDecisionV1:
    """Selected transform and held-group support for one domain."""

    domain_id: str
    group_ids: Sequence[str]
    transform_scores: Sequence[TransformScore]
    selected_covariance_scale: float
    selected_isotropic_variance: float
    raw_equal_group_mean_nll: float
    calibrated_equal_group_mean_nll: float
    leave_one_group_out: Sequence[HeldGroupScore]
    guard_supported: bool
    calibration_supported: bool
    reasons: Sequence[str]
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        domain = _canonical_string(self.domain_id, name="domain_id")
        groups = tuple(sorted(_strings(self.group_ids, name="group_ids")))
        if len(set(groups)) != len(groups):
            raise ValueError("group_ids must not contain duplicates")
        scores = tuple(_transform_score(value) for value in self.transform_scores)
        if not scores:
            raise ValueError("transform_scores must not be empty")
        if len({(row[0], row[1]) for row in scores}) != len(scores):
            raise ValueError("transform_scores must not contain duplicate transforms")
        scores = tuple(sorted(scores, key=lambda row: (row[0], row[1])))
        selected = min(scores, key=_tie_break)
        scale = _number(
            self.selected_covariance_scale,
            name="selected_covariance_scale",
            positive=True,
        )
        floor = _number(
            self.selected_isotropic_variance,
            name="selected_isotropic_variance",
            nonnegative=True,
        )
        if (scale, floor) != selected[:2]:
            raise ValueError("selected transform does not minimize the frozen score")
        raw = next(
            (row for row in scores if row[0] == 1.0 and row[1] == 0.0),
            None,
        )
        if raw is None:
            raise ValueError("transform_scores must include raw covariance (1, 0)")
        raw_nll = _number(
            self.raw_equal_group_mean_nll,
            name="raw_equal_group_mean_nll",
        )
        calibrated_nll = _number(
            self.calibrated_equal_group_mean_nll,
            name="calibrated_equal_group_mean_nll",
        )
        if not math.isclose(raw_nll, raw[2], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("raw_equal_group_mean_nll disagrees with raw score")
        if not math.isclose(
            calibrated_nll,
            selected[2],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("calibrated score disagrees with selected transform")
        held = tuple(_held_score(value) for value in self.leave_one_group_out)
        if {row[0] for row in held} != set(groups) or len(held) != len(groups):
            raise ValueError("leave-one-group-out roster must equal group_ids")
        held = tuple(sorted(held, key=lambda row: row[0]))
        reasons = tuple(sorted(_strings(self.reasons, name="reasons")))
        if len(set(reasons)) != len(reasons):
            raise ValueError("reasons must not contain duplicates")
        object.__setattr__(self, "domain_id", domain)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "transform_scores", scores)
        object.__setattr__(self, "selected_covariance_scale", scale)
        object.__setattr__(self, "selected_isotropic_variance", floor)
        object.__setattr__(self, "raw_equal_group_mean_nll", raw_nll)
        object.__setattr__(self, "calibrated_equal_group_mean_nll", calibrated_nll)
        object.__setattr__(self, "leave_one_group_out", held)
        object.__setattr__(
            self,
            "guard_supported",
            genuine_boolean(self.guard_supported, name="guard_supported"),
        )
        object.__setattr__(
            self,
            "calibration_supported",
            genuine_boolean(
                self.calibration_supported,
                name="calibration_supported",
            ),
        )
        object.__setattr__(self, "reasons", reasons)
        expected = content_id(self.descriptor())
        if (
            self.artifact_id is not None
            and sha256_digest(
                self.artifact_id,
                name="artifact_id",
            )
            != expected
        ):
            raise ValueError("artifact_id does not match calibration decision")
        object.__setattr__(self, "artifact_id", expected)

    @property
    def mean_loo_nll_improvement(self) -> float:
        held = cast(Sequence[HeldGroupScore], self.leave_one_group_out)
        return float(np.mean([row[3] - row[4] for row in held]))

    @property
    def worst_loo_nll_regression(self) -> float:
        held = cast(Sequence[HeldGroupScore], self.leave_one_group_out)
        return max(0.0, max(row[4] - row[3] for row in held))

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DOMAIN_COVARIANCE_DECISION_SCHEMA,
            "schema_version": DOMAIN_COVARIANCE_DECISION_VERSION,
            "domain_id": self.domain_id,
            "group_ids": list(self.group_ids),
            "transform_scores": [
                {
                    "covariance_scale": row[0],
                    "isotropic_variance": row[1],
                    "equal_group_mean_nll": row[2],
                }
                for row in self.transform_scores
            ],
            "selected_covariance_scale": self.selected_covariance_scale,
            "selected_isotropic_variance": self.selected_isotropic_variance,
            "raw_equal_group_mean_nll": self.raw_equal_group_mean_nll,
            "calibrated_equal_group_mean_nll": (self.calibrated_equal_group_mean_nll),
            "leave_one_group_out": [
                {
                    "held_group_id": row[0],
                    "selected_covariance_scale": row[1],
                    "selected_isotropic_variance": row[2],
                    "raw_held_group_nll": row[3],
                    "calibrated_held_group_nll": row[4],
                    "nll_improvement": row[3] - row[4],
                    "nll_regression": max(0.0, row[4] - row[3]),
                }
                for row in cast(Sequence[HeldGroupScore], self.leave_one_group_out)
            ],
            "mean_loo_nll_improvement": self.mean_loo_nll_improvement,
            "worst_loo_nll_regression": self.worst_loo_nll_regression,
            "guard_supported": self.guard_supported,
            "calibration_supported": self.calibration_supported,
            "reasons": list(self.reasons),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class DomainCovarianceCalibrationCertificateV1:
    """Content-addressed transform and information-boundary certificate."""

    predictor_id: str
    calibration_partition_id: str
    statistical_unit: str
    residual_semantics: str
    covariance_semantics: str
    guard_certificate_id: str
    guard_deployment_admissible: bool
    config: DomainCovarianceCalibrationConfigV1
    calibration_data_id: str
    decisions: Sequence[DomainCovarianceCalibrationDecisionV1]
    predictor_frozen_before_calibration_outcomes: bool
    transform_grid_frozen_before_calibration_outcomes: bool
    application_outcomes_used_for_calibration_selection: bool
    calibration_groups_independent: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, DomainCovarianceCalibrationConfigV1):
            raise TypeError("config must be a DomainCovarianceCalibrationConfigV1")
        decisions = tuple(sorted(self.decisions, key=lambda item: item.domain_id))
        if not decisions or any(
            not isinstance(item, DomainCovarianceCalibrationDecisionV1)
            for item in decisions
        ):
            raise TypeError(
                "decisions must contain DomainCovarianceCalibrationDecisionV1"
            )
        if len({item.domain_id for item in decisions}) != len(decisions):
            raise ValueError("decisions must not contain duplicate domains")
        for decision in decisions:
            reasons: list[str] = []
            if not decision.guard_supported:
                reasons.append("calibration-domain-guard-rejected")
            if len(decision.group_ids) < self.config.minimum_group_count:
                reasons.append("insufficient-calibration-groups")
            if (
                decision.mean_loo_nll_improvement + self.config.numerical_tolerance
                < self.config.minimum_mean_loo_nll_improvement
            ):
                reasons.append("mean-loo-nll-improvement-below-threshold")
            if (
                decision.worst_loo_nll_regression
                > self.config.maximum_single_group_loo_nll_regression
                + self.config.numerical_tolerance
            ):
                reasons.append("single-group-loo-nll-regression-exceeds-limit")
            supported = not reasons
            if supported:
                reasons.append("calibration-criteria-passed")
            if decision.calibration_supported != supported:
                raise ValueError("calibration decision has invalid support")
            if set(decision.reasons) != set(reasons):
                raise ValueError("calibration decision has invalid reasons")
        for name in (
            "predictor_id",
            "calibration_partition_id",
            "guard_certificate_id",
            "calibration_data_id",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        for name in (
            "statistical_unit",
            "residual_semantics",
            "covariance_semantics",
        ):
            object.__setattr__(
                self,
                name,
                _canonical_string(getattr(self, name), name=name),
            )
        for name in (
            "guard_deployment_admissible",
            "predictor_frozen_before_calibration_outcomes",
            "transform_grid_frozen_before_calibration_outcomes",
            "application_outcomes_used_for_calibration_selection",
            "calibration_groups_independent",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="domain covariance calibration metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if (
            self.artifact_id is not None
            and sha256_digest(
                self.artifact_id,
                name="artifact_id",
            )
            != expected
        ):
            raise ValueError("artifact_id does not match calibration certificate")
        object.__setattr__(self, "artifact_id", expected)

    @property
    def deployment_admissible(self) -> bool:
        return (
            self.guard_deployment_admissible
            and self.predictor_frozen_before_calibration_outcomes
            and self.transform_grid_frozen_before_calibration_outcomes
            and not self.application_outcomes_used_for_calibration_selection
            and self.calibration_groups_independent
        )

    @property
    def supported_domains(self) -> tuple[str, ...]:
        if not self.deployment_admissible:
            return ()
        return tuple(
            decision.domain_id
            for decision in self.decisions
            if decision.calibration_supported
        )

    def decision_for_domain(
        self,
        domain_id: str,
    ) -> DomainCovarianceCalibrationDecisionV1 | None:
        domain = _canonical_string(domain_id, name="domain_id")
        return next(
            (item for item in self.decisions if item.domain_id == domain),
            None,
        )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DOMAIN_COVARIANCE_CALIBRATION_SCHEMA,
            "schema_version": DOMAIN_COVARIANCE_CALIBRATION_VERSION,
            "predictor_id": self.predictor_id,
            "calibration_partition_id": self.calibration_partition_id,
            "statistical_unit": self.statistical_unit,
            "residual_semantics": self.residual_semantics,
            "covariance_semantics": self.covariance_semantics,
            "guard_certificate_id": self.guard_certificate_id,
            "guard_deployment_admissible": self.guard_deployment_admissible,
            "config": self.config.descriptor(),
            "calibration_data_id": self.calibration_data_id,
            "decisions": [item.to_record() for item in self.decisions],
            "information_boundary": {
                "predictor_frozen_before_calibration_outcomes": (
                    self.predictor_frozen_before_calibration_outcomes
                ),
                "transform_grid_frozen_before_calibration_outcomes": (
                    self.transform_grid_frozen_before_calibration_outcomes
                ),
                "application_outcomes_used_for_calibration_selection": (
                    self.application_outcomes_used_for_calibration_selection
                ),
                "calibration_groups_independent": (self.calibration_groups_independent),
                "deployment_admissible": self.deployment_admissible,
            },
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class DomainCovarianceCalibrationApplicationV1:
    """Record of one exact fallback or applied covariance transform."""

    certificate_id: str
    domain_id: str
    decision_id: str | None
    inference_admissible: bool
    certificate_deployment_admissible: bool
    calibration_supported: bool
    applied: bool
    reason: str
    covariance_scale: float
    isotropic_variance: float
    raw_covariance_sha256: str
    output_covariance_sha256: str
    exact_fallback: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "certificate_id",
            sha256_digest(self.certificate_id, name="certificate_id"),
        )
        object.__setattr__(
            self,
            "domain_id",
            _canonical_string(self.domain_id, name="domain_id"),
        )
        if self.decision_id is not None:
            object.__setattr__(
                self,
                "decision_id",
                sha256_digest(self.decision_id, name="decision_id"),
            )
        for name in (
            "inference_admissible",
            "certificate_deployment_admissible",
            "calibration_supported",
            "applied",
            "exact_fallback",
        ):
            object.__setattr__(
                self,
                name,
                genuine_boolean(getattr(self, name), name=name),
            )
        if self.applied == self.exact_fallback:
            raise ValueError("applied and exact_fallback must be logical opposites")
        object.__setattr__(
            self,
            "reason",
            _canonical_string(self.reason, name="reason"),
        )
        object.__setattr__(
            self,
            "covariance_scale",
            _number(self.covariance_scale, name="covariance_scale", positive=True),
        )
        object.__setattr__(
            self,
            "isotropic_variance",
            _number(
                self.isotropic_variance,
                name="isotropic_variance",
                nonnegative=True,
            ),
        )
        for name in ("raw_covariance_sha256", "output_covariance_sha256"):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        if self.applied:
            if not (
                self.inference_admissible
                and self.certificate_deployment_admissible
                and self.calibration_supported
            ):
                raise ValueError("applied calibration requires all admissibility gates")
            if self.decision_id is None:
                raise ValueError("applied calibration requires a domain decision")
            if self.reason != "calibration-domain-authorized":
                raise ValueError("applied calibration has invalid reason")
            if self.covariance_scale == 1.0 and self.isotropic_variance == 0.0:
                raise ValueError("applied calibration must change the transform")
        else:
            if self.covariance_scale != 1.0 or self.isotropic_variance != 0.0:
                raise ValueError("fallback must use raw covariance transform")
            if self.decision_id is None and self.calibration_supported:
                raise ValueError("unknown domain cannot be calibration-supported")
            if not self.inference_admissible:
                expected_reason = "inference-rejected"
            elif self.decision_id is None:
                expected_reason = "unknown-calibration-domain"
            elif not self.certificate_deployment_admissible:
                expected_reason = "calibration-information-boundary-rejected"
            elif not self.calibration_supported:
                expected_reason = "calibration-domain-rejected"
            else:
                expected_reason = "calibration-identity-transform-retained"
            if self.reason != expected_reason:
                raise ValueError("fallback calibration has invalid reason")
        if (
            self.exact_fallback
            and self.raw_covariance_sha256 != self.output_covariance_sha256
        ):
            raise ValueError("exact fallback must preserve covariance identity")
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="domain covariance application metadata",
            ),
        )
        expected = content_id(self.descriptor())
        if (
            self.artifact_id is not None
            and sha256_digest(
                self.artifact_id,
                name="artifact_id",
            )
            != expected
        ):
            raise ValueError("artifact_id does not match calibration application")
        object.__setattr__(self, "artifact_id", expected)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": DOMAIN_COVARIANCE_APPLICATION_SCHEMA,
            "schema_version": DOMAIN_COVARIANCE_APPLICATION_VERSION,
            "certificate_id": self.certificate_id,
            "domain_id": self.domain_id,
            "decision_id": self.decision_id,
            "inference_admissible": self.inference_admissible,
            "certificate_deployment_admissible": (
                self.certificate_deployment_admissible
            ),
            "calibration_supported": self.calibration_supported,
            "applied": self.applied,
            "reason": self.reason,
            "covariance_scale": self.covariance_scale,
            "isotropic_variance": self.isotropic_variance,
            "raw_covariance_sha256": self.raw_covariance_sha256,
            "output_covariance_sha256": self.output_covariance_sha256,
            "exact_fallback": self.exact_fallback,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def _score(
    residuals: np.ndarray,
    covariances: np.ndarray,
    groups: Sequence[str],
    config: DomainCovarianceCalibrationConfigV1,
    transform: tuple[float, float],
) -> TransformScore:
    scale, floor = transform
    values = _event_scores(
        residuals,
        covariances,
        scale=scale,
        floor=floor,
        eigenvalue_floor=config.scoring_eigenvalue_floor,
    )
    return (scale, floor, _equal_group_mean(values, groups))


def _select(
    residuals: np.ndarray,
    covariances: np.ndarray,
    groups: Sequence[str],
    config: DomainCovarianceCalibrationConfigV1,
) -> tuple[TransformScore, tuple[TransformScore, ...]]:
    scores = tuple(
        _score(residuals, covariances, groups, config, transform)
        for transform in config.transforms
    )
    return min(scores, key=_tie_break), scores


def fit_domain_covariance_calibration(
    *,
    predictor_id: str,
    calibration_partition_id: str,
    statistical_unit: str,
    residual_semantics: str,
    covariance_semantics: str,
    event_ids: Sequence[str],
    group_ids: Sequence[str],
    domain_ids: Sequence[str],
    residuals: object,
    covariances: object,
    domain_guard: CalibrationDomainGuardCertificateV1,
    predictor_frozen_before_calibration_outcomes: bool,
    transform_grid_frozen_before_calibration_outcomes: bool,
    application_outcomes_used_for_calibration_selection: bool,
    calibration_groups_independent: bool,
    config: DomainCovarianceCalibrationConfigV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DomainCovarianceCalibrationCertificateV1:
    """Fit one transform per domain using only the frozen calibration data."""

    settings = DomainCovarianceCalibrationConfigV1() if config is None else config
    if not isinstance(settings, DomainCovarianceCalibrationConfigV1):
        raise TypeError("config must be a DomainCovarianceCalibrationConfigV1")
    if not isinstance(domain_guard, CalibrationDomainGuardCertificateV1):
        raise TypeError("domain_guard must be a CalibrationDomainGuardCertificateV1")
    predictor = sha256_digest(predictor_id, name="predictor_id")
    partition = sha256_digest(
        calibration_partition_id,
        name="calibration_partition_id",
    )
    unit = _canonical_string(statistical_unit, name="statistical_unit")
    residual_name = _canonical_string(
        residual_semantics,
        name="residual_semantics",
    )
    covariance_name = _canonical_string(
        covariance_semantics,
        name="covariance_semantics",
    )
    if partition != domain_guard.calibration_partition_id:
        raise ValueError("calibration_partition_id differs from domain guard")
    if unit != domain_guard.statistical_unit:
        raise ValueError("statistical_unit differs from domain guard")
    events = _strings(event_ids, name="event_ids")
    groups = _strings(group_ids, name="group_ids")
    domains = _strings(domain_ids, name="domain_ids")
    if len(set(events)) != len(events):
        raise ValueError("event_ids must not contain duplicates")
    residual_array = _residuals(residuals)
    covariance_array, _ = _covariances(covariances, calibration=True)
    if not (
        len(events)
        == len(groups)
        == len(domains)
        == len(residual_array)
        == len(covariance_array)
    ):
        raise ValueError(
            "event_ids, group_ids, domain_ids, residuals, and covariances "
            "must have equal event counts"
        )
    if residual_array.shape[1] != covariance_array.shape[1]:
        raise ValueError("residual and covariance dimensions differ")
    _validate_covariances(
        covariance_array,
        tolerance=settings.covariance_psd_tolerance,
    )
    group_domains: dict[str, str] = {}
    for group, domain in zip(groups, domains, strict=True):
        if group_domains.setdefault(group, domain) != domain:
            raise ValueError("each calibration group must belong to one domain")
    order = np.asarray(
        sorted(range(len(events)), key=events.__getitem__),
        dtype=np.int64,
    )
    events = tuple(events[index] for index in order)
    groups = tuple(groups[index] for index in order)
    domains = tuple(domains[index] for index in order)
    residual_array = residual_array[order]
    covariance_array = covariance_array[order]
    calibration_data_id = content_id(
        {
            "schema": DOMAIN_COVARIANCE_DATA_SCHEMA,
            "schema_version": DOMAIN_COVARIANCE_DATA_VERSION,
            "predictor_id": predictor,
            "calibration_partition_id": partition,
            "statistical_unit": unit,
            "residual_semantics": residual_name,
            "covariance_semantics": covariance_name,
            "records": [
                {
                    "event_id": events[index],
                    "group_id": groups[index],
                    "domain_id": domains[index],
                    "residual": residual_array[index].tolist(),
                    "covariance": covariance_array[index].tolist(),
                }
                for index in range(len(events))
            ],
        }
    )
    decisions: list[DomainCovarianceCalibrationDecisionV1] = []
    for domain in sorted(set(domains)):
        mask = np.asarray([value == domain for value in domains], dtype=bool)
        domain_residuals = residual_array[mask]
        domain_covariances = covariance_array[mask]
        domain_groups = tuple(
            group for group, selected in zip(groups, mask, strict=True) if selected
        )
        unique_groups = tuple(sorted(set(domain_groups)))
        selected, scores = _select(
            domain_residuals,
            domain_covariances,
            domain_groups,
            settings,
        )
        raw = next(row for row in scores if row[:2] == (1.0, 0.0))
        held_rows: list[HeldGroupScore] = []
        for held_group in unique_groups:
            held = np.asarray(
                [group == held_group for group in domain_groups],
                dtype=bool,
            )
            training = ~held
            held_selected = raw
            if np.any(training):
                held_selected, _ = _select(
                    domain_residuals[training],
                    domain_covariances[training],
                    tuple(
                        group
                        for group, selected_training in zip(
                            domain_groups,
                            training,
                            strict=True,
                        )
                        if selected_training
                    ),
                    settings,
                )
            raw_values = _event_scores(
                domain_residuals[held],
                domain_covariances[held],
                scale=1.0,
                floor=0.0,
                eigenvalue_floor=settings.scoring_eigenvalue_floor,
            )
            calibrated_values = _event_scores(
                domain_residuals[held],
                domain_covariances[held],
                scale=held_selected[0],
                floor=held_selected[1],
                eigenvalue_floor=settings.scoring_eigenvalue_floor,
            )
            held_groups = tuple(
                group
                for group, selected_held in zip(
                    domain_groups,
                    held,
                    strict=True,
                )
                if selected_held
            )
            held_rows.append(
                (
                    held_group,
                    held_selected[0],
                    held_selected[1],
                    _equal_group_mean(raw_values, held_groups),
                    _equal_group_mean(calibrated_values, held_groups),
                )
            )
        guard_decision = domain_guard.decision_for_domain(domain)
        if guard_decision is not None and set(guard_decision.group_ids) != set(
            unique_groups
        ):
            raise ValueError(
                f"calibration group roster for domain {domain!r} "
                "differs from domain guard"
            )
        guard_supported = bool(
            guard_decision is not None and guard_decision.calibration_supported
        )
        mean_improvement = float(np.mean([row[3] - row[4] for row in held_rows]))
        worst_regression = max(0.0, max(row[4] - row[3] for row in held_rows))
        reasons: list[str] = []
        if not guard_supported:
            reasons.append("calibration-domain-guard-rejected")
        if len(unique_groups) < settings.minimum_group_count:
            reasons.append("insufficient-calibration-groups")
        if (
            mean_improvement + settings.numerical_tolerance
            < settings.minimum_mean_loo_nll_improvement
        ):
            reasons.append("mean-loo-nll-improvement-below-threshold")
        if (
            worst_regression
            > settings.maximum_single_group_loo_nll_regression
            + settings.numerical_tolerance
        ):
            reasons.append("single-group-loo-nll-regression-exceeds-limit")
        supported = not reasons
        if supported:
            reasons.append("calibration-criteria-passed")
        decisions.append(
            DomainCovarianceCalibrationDecisionV1(
                domain_id=domain,
                group_ids=unique_groups,
                transform_scores=scores,
                selected_covariance_scale=selected[0],
                selected_isotropic_variance=selected[1],
                raw_equal_group_mean_nll=raw[2],
                calibrated_equal_group_mean_nll=selected[2],
                leave_one_group_out=held_rows,
                guard_supported=guard_supported,
                calibration_supported=supported,
                reasons=reasons,
            )
        )
    return DomainCovarianceCalibrationCertificateV1(
        predictor_id=predictor,
        calibration_partition_id=partition,
        statistical_unit=unit,
        residual_semantics=residual_name,
        covariance_semantics=covariance_name,
        guard_certificate_id=str(domain_guard.artifact_id),
        guard_deployment_admissible=domain_guard.deployment_admissible,
        config=settings,
        calibration_data_id=calibration_data_id,
        decisions=decisions,
        predictor_frozen_before_calibration_outcomes=(
            predictor_frozen_before_calibration_outcomes
        ),
        transform_grid_frozen_before_calibration_outcomes=(
            transform_grid_frozen_before_calibration_outcomes
        ),
        application_outcomes_used_for_calibration_selection=(
            application_outcomes_used_for_calibration_selection
        ),
        calibration_groups_independent=calibration_groups_independent,
        metadata={} if metadata is None else metadata,
    )


def apply_domain_covariance_calibration(
    raw_covariance: np.ndarray,
    certificate: DomainCovarianceCalibrationCertificateV1,
    *,
    domain_id: str,
    inference_admissible: bool,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, DomainCovarianceCalibrationApplicationV1]:
    """Apply a frozen transform or return the exact raw covariance object."""

    if not isinstance(raw_covariance, np.ndarray):
        raise TypeError("raw_covariance must be a numpy.ndarray")
    if not isinstance(certificate, DomainCovarianceCalibrationCertificateV1):
        raise TypeError(
            "certificate must be a DomainCovarianceCalibrationCertificateV1"
        )
    domain = _canonical_string(domain_id, name="domain_id")
    inference_ok = genuine_boolean(
        inference_admissible,
        name="inference_admissible",
    )
    normalized, single = _covariances(raw_covariance, calibration=False)
    _validate_covariances(
        normalized,
        tolerance=certificate.config.covariance_psd_tolerance,
    )
    decision = certificate.decision_for_domain(domain)
    supported = bool(decision is not None and decision.calibration_supported)
    identity_transform = bool(
        decision is not None
        and decision.selected_covariance_scale == 1.0
        and decision.selected_isotropic_variance == 0.0
    )
    applied = (
        inference_ok
        and certificate.deployment_admissible
        and supported
        and not identity_transform
    )
    if not inference_ok:
        reason = "inference-rejected"
    elif decision is None:
        reason = "unknown-calibration-domain"
    elif not certificate.deployment_admissible:
        reason = "calibration-information-boundary-rejected"
    elif not decision.calibration_supported:
        reason = "calibration-domain-rejected"
    elif identity_transform:
        reason = "calibration-identity-transform-retained"
    else:
        reason = "calibration-domain-authorized"
    raw_digest = _array_digest(np.asarray(raw_covariance, dtype=np.float64))
    if applied:
        assert decision is not None
        transformed = _transform(
            normalized,
            decision.selected_covariance_scale,
            decision.selected_isotropic_variance,
        )
        _validate_covariances(
            transformed,
            tolerance=certificate.config.covariance_psd_tolerance,
        )
        output = _immutable(transformed[0] if single else transformed)
        output_digest = _array_digest(output)
        scale = decision.selected_covariance_scale
        floor = decision.selected_isotropic_variance
    else:
        output = raw_covariance
        output_digest = raw_digest
        scale = 1.0
        floor = 0.0
    record = DomainCovarianceCalibrationApplicationV1(
        certificate_id=str(certificate.artifact_id),
        domain_id=domain,
        decision_id=None if decision is None else decision.artifact_id,
        inference_admissible=inference_ok,
        certificate_deployment_admissible=certificate.deployment_admissible,
        calibration_supported=supported,
        applied=applied,
        reason=reason,
        covariance_scale=scale,
        isotropic_variance=floor,
        raw_covariance_sha256=raw_digest,
        output_covariance_sha256=output_digest,
        exact_fallback=not applied,
        metadata={} if metadata is None else metadata,
    )
    return output, record


__all__ = [
    "DOMAIN_COVARIANCE_APPLICATION_SCHEMA",
    "DOMAIN_COVARIANCE_APPLICATION_VERSION",
    "DOMAIN_COVARIANCE_CALIBRATION_SCHEMA",
    "DOMAIN_COVARIANCE_CALIBRATION_VERSION",
    "DOMAIN_COVARIANCE_DATA_SCHEMA",
    "DOMAIN_COVARIANCE_DATA_VERSION",
    "DOMAIN_COVARIANCE_DECISION_SCHEMA",
    "DOMAIN_COVARIANCE_DECISION_VERSION",
    "DomainCovarianceCalibrationApplicationV1",
    "DomainCovarianceCalibrationCertificateV1",
    "DomainCovarianceCalibrationConfigV1",
    "DomainCovarianceCalibrationDecisionV1",
    "apply_domain_covariance_calibration",
    "fit_domain_covariance_calibration",
]
