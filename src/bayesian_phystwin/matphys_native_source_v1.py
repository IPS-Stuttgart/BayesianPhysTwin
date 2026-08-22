"""Source-only scoring for native PhysTwin MatPhys covariance donors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

CHI_SQUARE_DF3_90: Final = 6.251388631170325


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class NativeMatPhysCaseEvidence:
    """Identity-aligned future residuals and raw physical covariance."""

    case_id: str
    residual_m: npt.NDArray[np.float64]
    raw_covariance_m2: npt.NDArray[np.float64]


def frame_zero_farthest_point_indices(
    points_m: np.ndarray,
    *,
    count: int,
) -> npt.NDArray[np.int64]:
    """Select deterministic material identities with node zero as the seed."""

    points = np.asarray(points_m, dtype=np.float64)
    _require(
        points.ndim == 2 and points.shape[1] == 3 and len(points) > 0,
        "points_m must have shape (N,3)",
    )
    _require(np.all(np.isfinite(points)), "points_m must be finite")
    _require(type(count) is int and 1 <= count <= len(points), "count is invalid")
    selected = [0]
    minimum_squared = np.sum(np.square(points - points[0]), axis=1)
    for _ in range(1, count):
        minimum_squared[np.asarray(selected, dtype=np.int64)] = -1.0
        index = int(np.argmax(minimum_squared))
        selected.append(index)
        minimum_squared = np.minimum(
            minimum_squared,
            np.sum(np.square(points - points[index]), axis=1),
        )
    return np.asarray(selected, dtype=np.int64)


def native_case_evidence(
    *,
    case_id: str,
    observed_m: np.ndarray,
    baseline_mean_m: np.ndarray,
    valid_mask: np.ndarray,
    raw_covariance_m2: np.ndarray,
    future_start: int,
    future_stop: int,
    identity_count: int = 128,
) -> NativeMatPhysCaseEvidence:
    """Extract the registered future events without outcome-based identity choice."""

    observed = np.asarray(observed_m, dtype=np.float64)
    baseline = np.asarray(baseline_mean_m, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    covariance = np.asarray(raw_covariance_m2, dtype=np.float64)
    _require(
        observed.ndim == 3 and observed.shape[-1] == 3,
        "observed_m must have shape (T,N,3)",
    )
    _require(baseline.shape == observed.shape, "baseline mean shape changed")
    _require(valid.shape == observed.shape[:2], "valid_mask shape changed")
    _require(
        covariance.shape == (*observed.shape[:2], 3, 3),
        "raw covariance shape changed",
    )
    _require(
        0 <= future_start < future_stop <= len(observed),
        "future interval is invalid",
    )
    _require(
        np.all(np.isfinite(observed))
        and np.all(np.isfinite(baseline))
        and np.all(np.isfinite(covariance)),
        "source evidence must be finite",
    )
    selected = frame_zero_farthest_point_indices(
        observed[0], count=min(identity_count, observed.shape[1])
    )
    selected_valid = valid[future_start:future_stop][:, selected]
    residual = (
        observed[future_start:future_stop][:, selected]
        - baseline[future_start:future_stop][:, selected]
    )[selected_valid]
    selected_covariance = covariance[future_start:future_stop][:, selected][
        selected_valid
    ]
    _require(len(residual) > 0, f"{case_id}: no registered future event is valid")
    eigenvalues = np.linalg.eigvalsh(
        0.5
        * (selected_covariance + np.swapaxes(selected_covariance, axis1=-1, axis2=-2))
    )
    _require(
        float(np.min(eigenvalues)) >= -1e-10,
        f"{case_id}: raw covariance is not PSD",
    )
    return NativeMatPhysCaseEvidence(
        case_id=case_id,
        residual_m=residual,
        raw_covariance_m2=selected_covariance,
    )


def calibrated_covariance(
    raw_covariance_m2: np.ndarray,
    *,
    scale: float,
    isotropic_std_m: float,
) -> npt.NDArray[np.float64]:
    """Apply source-selected scalar calibration to a raw MatPhys donor."""

    raw = np.asarray(raw_covariance_m2, dtype=np.float64)
    _require(
        raw.ndim == 3 and raw.shape[1:] == (3, 3),
        "raw covariance must have shape (E,3,3)",
    )
    _require(np.isfinite(scale) and scale >= 0.0, "scale must be nonnegative")
    _require(
        np.isfinite(isotropic_std_m) and isotropic_std_m > 0.0,
        "isotropic standard deviation must be positive",
    )
    result = scale * scale * raw + isotropic_std_m * isotropic_std_m * np.eye(3)[None]
    result = 0.5 * (result + np.swapaxes(result, axis1=-1, axis2=-2))
    _require(
        float(np.min(np.linalg.eigvalsh(result))) > 0.0,
        "calibrated covariance must be positive definite",
    )
    return result


def gaussian_case_metrics(
    residual_m: np.ndarray,
    covariance_m2: np.ndarray,
) -> dict[str, float]:
    """Score equal-weight 3-D marginal material-point events."""

    residual = np.asarray(residual_m, dtype=np.float64)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    _require(
        residual.ndim == 2 and residual.shape[1] == 3,
        "residual_m must have shape (E,3)",
    )
    _require(
        covariance.shape == (len(residual), 3, 3),
        "covariance and residual event counts differ",
    )
    sign, logdet = np.linalg.slogdet(covariance)
    _require(np.all(sign > 0.0), "covariance must be positive definite")
    solved = np.linalg.solve(covariance, residual[..., None])[..., 0]
    nees = np.einsum("ni,ni->n", residual, solved)
    nll = 0.5 * (3.0 * np.log(2.0 * np.pi) + logdet + nees)
    determinant = np.linalg.det(covariance)
    ellipsoid_volume = 4.0 * np.pi / 3.0 * CHI_SQUARE_DF3_90**1.5 * np.sqrt(determinant)
    return {
        "event_count": float(len(residual)),
        "coordinate_rmse_m": float(np.sqrt(np.mean(np.square(residual)))),
        "nll_nats_per_event": float(np.mean(nll)),
        "coverage_90": float(np.mean(nees <= CHI_SQUARE_DF3_90)),
        "mean_ellipsoid_volume_m3": float(np.mean(ellipsoid_volume)),
        "nees": float(np.mean(nees)),
    }


def select_candidate_calibration(
    cases: tuple[NativeMatPhysCaseEvidence, ...],
    *,
    scale_grid: tuple[float, ...],
    isotropic_std_grid_m: tuple[float, ...],
) -> tuple[float, float]:
    """Choose one candidate scale/floor by equal-case source NLL."""

    _require(len(cases) > 0, "at least one calibration case is required")
    choices: list[tuple[float, float, float, float]] = []
    for scale in scale_grid:
        for isotropic_std in isotropic_std_grid_m:
            metrics = [
                gaussian_case_metrics(
                    case.residual_m,
                    calibrated_covariance(
                        case.raw_covariance_m2,
                        scale=scale,
                        isotropic_std_m=isotropic_std,
                    ),
                )
                for case in cases
            ]
            choices.append(
                (
                    float(np.mean([item["nll_nats_per_event"] for item in metrics])),
                    float(
                        np.mean([item["mean_ellipsoid_volume_m3"] for item in metrics])
                    ),
                    float(scale),
                    float(isotropic_std),
                )
            )
    _, _, selected_scale, selected_std = min(choices)
    return selected_scale, selected_std


def select_isotropic_calibration(
    cases: tuple[NativeMatPhysCaseEvidence, ...],
    *,
    isotropic_std_grid_m: tuple[float, ...],
) -> float:
    """Choose a separately fitted isotropic comparator by equal-case NLL."""

    _require(len(cases) > 0, "at least one calibration case is required")
    choices: list[tuple[float, float]] = []
    for isotropic_std in isotropic_std_grid_m:
        case_nll = []
        for case in cases:
            covariance = np.broadcast_to(
                np.eye(3) * isotropic_std * isotropic_std,
                (len(case.residual_m), 3, 3),
            )
            case_nll.append(
                gaussian_case_metrics(case.residual_m, covariance)["nll_nats_per_event"]
            )
        choices.append((float(np.mean(case_nll)), float(isotropic_std)))
    return min(choices)[1]


__all__ = [
    "CHI_SQUARE_DF3_90",
    "NativeMatPhysCaseEvidence",
    "calibrated_covariance",
    "frame_zero_farthest_point_indices",
    "gaussian_case_metrics",
    "native_case_evidence",
    "select_candidate_calibration",
    "select_isotropic_calibration",
]
