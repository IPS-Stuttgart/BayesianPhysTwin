"""Auditable MatPhys spring directions centered on a released PhysTwin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


MATPHYS_MATERIAL_NAMES = (
    "fabric",
    "leather",
    "rubber",
    "plastic",
    "metal",
    "wood",
    "paper",
    "silk",
    "denim",
    "fur",
)
MATPHYS_SHARED_PRIOR_CONTRACT = "matphys-teacher-direction-v1"


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


@dataclass(frozen=True)
class MatPhysPredictionCompetence:
    """Source-independent gate for a useful spatial spring proposal."""

    competent_direction: bool
    failure_reasons: tuple[str, ...]
    predicted_log_minimum: float
    predicted_log_maximum: float
    predicted_log_mean: float
    predicted_log_std: float
    lower_bound_fraction: float
    upper_bound_fraction: float
    predicted_teacher_correlation: float | None
    direction_teacher_correlation: float | None
    log_stiffness_minimum: float
    log_stiffness_maximum: float
    maximum_bound_fraction: float
    minimum_spatial_log_std: float

    def diagnostics(self) -> dict[str, object]:
        return {
            "contract": "matphys-spatial-competence-gate-v1",
            "competent_direction": self.competent_direction,
            "failure_reasons": list(self.failure_reasons),
            "criteria": {
                "maximum_bound_fraction": self.maximum_bound_fraction,
                "minimum_spatial_log_std": self.minimum_spatial_log_std,
            },
            "bounds": {
                "log_minimum": self.log_stiffness_minimum,
                "log_maximum": self.log_stiffness_maximum,
            },
            "prediction": {
                "minimum": self.predicted_log_minimum,
                "maximum": self.predicted_log_maximum,
                "mean": self.predicted_log_mean,
                "std": self.predicted_log_std,
                "lower_bound_fraction": self.lower_bound_fraction,
                "upper_bound_fraction": self.upper_bound_fraction,
                "teacher_correlation": self.predicted_teacher_correlation,
                "direction_teacher_correlation": self.direction_teacher_correlation,
            },
        }


def assess_matphys_prediction_competence(
    *,
    teacher_object_log_y: object,
    predicted_object_log_y: object,
    stiffness_minimum: float = 1.0e3,
    stiffness_maximum: float = 1.0e5,
    maximum_bound_fraction: float = 0.99,
    minimum_spatial_log_std: float = 1.0e-4,
    bound_log_tolerance: float = 1.0e-5,
) -> MatPhysPredictionCompetence:
    """Reject collapsed checkpoint outputs before an expensive Warp gate.

    The gate deliberately checks only whether the checkpoint proposes a
    nondegenerate spatial field. It does not use target trajectory error and
    therefore cannot select a proposal using held-out dynamics.
    """

    teacher = np.asarray(teacher_object_log_y, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_object_log_y, dtype=np.float64).reshape(-1)
    if teacher.shape != predicted.shape or teacher.size < 2:
        raise ValueError("teacher and prediction must contain the same spring field")
    if not np.isfinite(teacher).all() or not np.isfinite(predicted).all():
        raise ValueError("teacher and prediction must be finite")
    if (
        not np.isfinite(stiffness_minimum)
        or not np.isfinite(stiffness_maximum)
        or stiffness_minimum <= 0.0
        or stiffness_maximum <= stiffness_minimum
    ):
        raise ValueError("stiffness bounds must be finite, positive, and ordered")
    if not 0.0 <= maximum_bound_fraction <= 1.0:
        raise ValueError("maximum_bound_fraction must lie in [0, 1]")
    if minimum_spatial_log_std < 0.0 or bound_log_tolerance < 0.0:
        raise ValueError("competence tolerances must be nonnegative")

    lower = float(np.log(stiffness_minimum))
    upper = float(np.log(stiffness_maximum))
    lower_fraction = float(np.mean(np.abs(predicted - lower) <= bound_log_tolerance))
    upper_fraction = float(np.mean(np.abs(predicted - upper) <= bound_log_tolerance))
    predicted_std = float(np.std(predicted))
    reasons = []
    if lower_fraction >= maximum_bound_fraction:
        reasons.append("lower-bound-saturation")
    if upper_fraction >= maximum_bound_fraction:
        reasons.append("upper-bound-saturation")
    if predicted_std < minimum_spatial_log_std:
        reasons.append("spatially-constant-output")
    difference = predicted - teacher
    return MatPhysPredictionCompetence(
        competent_direction=not reasons,
        failure_reasons=tuple(reasons),
        predicted_log_minimum=float(np.min(predicted)),
        predicted_log_maximum=float(np.max(predicted)),
        predicted_log_mean=float(np.mean(predicted)),
        predicted_log_std=predicted_std,
        lower_bound_fraction=lower_fraction,
        upper_bound_fraction=upper_fraction,
        predicted_teacher_correlation=_finite_correlation(predicted, teacher),
        direction_teacher_correlation=_finite_correlation(difference, teacher),
        log_stiffness_minimum=lower,
        log_stiffness_maximum=upper,
        maximum_bound_fraction=float(maximum_bound_fraction),
        minimum_spatial_log_std=float(minimum_spatial_log_std),
    )


def material_distribution_from_weights(
    weights: Mapping[str, float],
    *,
    material_names: Sequence[str] = MATPHYS_MATERIAL_NAMES,
) -> np.ndarray:
    """Convert named nonnegative weights to one normalized MatPhys row."""

    names = tuple(str(name) for name in material_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("material_names must be nonempty and unique")
    unknown = sorted(set(weights) - set(names))
    if unknown:
        raise ValueError(f"unknown material names: {', '.join(unknown)}")
    row = np.asarray([float(weights.get(name, 0.0)) for name in names])
    return validate_material_distributions(row[None, :], len(names))[0]


def validate_material_distributions(
    distributions: object,
    num_materials: int = len(MATPHYS_MATERIAL_NAMES),
    *,
    expected_parts: int | None = None,
) -> np.ndarray:
    """Validate and normalize a per-part material-probability matrix."""

    values = np.asarray(distributions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != int(num_materials):
        raise ValueError(
            f"material distributions must have shape (K, {num_materials})"
        )
    if expected_parts is not None and values.shape[0] != int(expected_parts):
        raise ValueError(
            "material distribution part count does not match the graph partition"
        )
    if values.shape[0] < 1:
        raise ValueError("material distributions must contain a part")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("material distributions must be finite and nonnegative")
    mass = values.sum(axis=1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("every material distribution must have positive mass")
    return (values / mass).astype(np.float32)


@dataclass(frozen=True)
class MatPhysSpringDirection:
    """One-dimensional proposal that preserves the exact teacher at zero."""

    weights: np.ndarray
    prior_coefficient: float
    raw_log_difference: np.ndarray
    object_spring_count: int

    def reconstruct(self, teacher_log_y: object, coefficient: float) -> np.ndarray:
        teacher = np.asarray(teacher_log_y, dtype=np.float64).reshape(-1)
        if teacher.shape != self.raw_log_difference.shape:
            raise ValueError("teacher spring count changed")
        if not np.isfinite(coefficient):
            raise ValueError("coefficient must be finite")
        return teacher + self.weights[:, 0] * float(coefficient)

    def diagnostics(self) -> dict[str, object]:
        obj = self.raw_log_difference[: self.object_spring_count]
        ctrl = self.raw_log_difference[self.object_spring_count :]
        return {
            "contract": MATPHYS_SHARED_PRIOR_CONTRACT,
            "spring_count": int(self.raw_log_difference.size),
            "object_spring_count": int(self.object_spring_count),
            "controller_spring_count": int(ctrl.size),
            "prior_coefficient": float(self.prior_coefficient),
            "raw_log_difference": {
                "minimum": float(np.min(self.raw_log_difference)),
                "maximum": float(np.max(self.raw_log_difference)),
                "rms": float(np.sqrt(np.mean(self.raw_log_difference**2))),
                "object_mean": float(np.mean(obj)),
                "object_std": float(np.std(obj)),
                "controller_mean": None if not ctrl.size else float(np.mean(ctrl)),
                "controller_std": None if not ctrl.size else float(np.std(ctrl)),
            },
        }


def build_matphys_spring_direction(
    *,
    teacher_log_y: object,
    predicted_object_log_y: object,
    predicted_controller_log_y: object | None = None,
    object_spring_count: int | None = None,
    minimum_scale: float = 1.0e-8,
) -> MatPhysSpringDirection:
    """Build a normalized proposal direction from MatPhys to PhysTwin.

    The MatPhys point estimate is recovered at ``prior_coefficient``. A zero
    coefficient is exactly the released PhysTwin spring field, so validation
    can always reject the learned proposal without changing simulator bytes.
    """

    teacher = np.asarray(teacher_log_y, dtype=np.float64).reshape(-1)
    predicted_object = np.asarray(
        predicted_object_log_y, dtype=np.float64
    ).reshape(-1)
    predicted_controller = (
        np.empty(0, dtype=np.float64)
        if predicted_controller_log_y is None
        else np.asarray(predicted_controller_log_y, dtype=np.float64).reshape(-1)
    )
    inferred_object_count = int(predicted_object.size)
    if object_spring_count is None:
        object_spring_count = inferred_object_count
    if int(object_spring_count) != inferred_object_count:
        raise ValueError("object_spring_count disagrees with the MatPhys output")
    predicted = np.concatenate((predicted_object, predicted_controller))
    if teacher.shape != predicted.shape:
        raise ValueError(
            "released PhysTwin and MatPhys predictions have different spring counts"
        )
    if (
        teacher.size < 1
        or not np.isfinite(teacher).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("spring log stiffnesses must be finite and nonempty")
    if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be positive and finite")

    difference = predicted - teacher
    scale = float(np.max(np.abs(difference)))
    if scale < float(minimum_scale):
        raise ValueError("MatPhys prediction is indistinguishable from the teacher")
    weights = (difference / scale).astype(np.float32)[:, None]
    return MatPhysSpringDirection(
        weights=weights,
        prior_coefficient=scale,
        raw_log_difference=difference,
        object_spring_count=int(object_spring_count),
    )
