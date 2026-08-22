"""Surface-space uncertainty scoring for target-excluded MatPhys ensembles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.stats import chi2

FloatArray: TypeAlias = npt.NDArray[np.floating]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

MATPHYS_SURFACE_UQ_SCHEMA: Final = "bayesian-phystwin.matphys-surface-uq-v1"
CHI2_90_DF3: Final = float(chi2.ppf(0.9, df=3))


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _finite_array(value: npt.ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    _require(np.all(np.isfinite(array)), f"{name} contains non-finite values")
    return array


def deterministic_camera_partition(
    camera_ids: list[str] | tuple[str, ...],
    *,
    scoring_camera_ids: list[str] | tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return canonical disjoint provider and scoring panels."""

    _require(bool(camera_ids), "camera roster is empty")
    all_cameras = tuple(sorted(camera_ids))
    scoring = tuple(sorted(scoring_camera_ids))
    _require(len(all_cameras) == len(set(all_cameras)), "camera roster repeats an ID")
    _require(len(scoring) == len(set(scoring)), "scoring panel repeats an ID")
    _require(set(scoring) <= set(all_cameras), "scoring camera is unavailable")
    provider = tuple(camera for camera in all_cameras if camera not in set(scoring))
    _require(len(provider) >= 2, "provider panel has fewer than two cameras")
    _require(len(scoring) >= 2, "scoring panel has fewer than two cameras")
    _require(not set(provider) & set(scoring), "camera panels overlap")
    return provider, scoring


def deterministic_subsample_indices(
    count: int,
    maximum: int,
    *,
    key: str,
) -> npt.NDArray[np.int64]:
    """Select a deterministic, key-bound subset without replacement."""

    _require(type(count) is int and count >= 0, "count must be nonnegative")
    _require(type(maximum) is int and maximum > 0, "maximum must be positive")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    seed = int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:8],
        byteorder="little",
        signed=False,
    )
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(count, size=maximum, replace=False)).astype(
        np.int64
    )


def backproject_masked_depth(
    depth_m: npt.ArrayLike,
    object_mask: npt.ArrayLike,
    intrinsics: npt.ArrayLike,
    camera_to_world: npt.ArrayLike,
    *,
    maximum_points: int,
    subsample_key: str,
) -> npt.NDArray[np.float64]:
    """Backproject one masked metric-depth frame into world coordinates."""

    depth: npt.NDArray[np.float64] = _finite_array(depth_m, name="depth").astype(
        np.float64, copy=False
    )
    mask = np.asarray(object_mask)
    calibration: npt.NDArray[np.float64] = _finite_array(
        intrinsics, name="intrinsics"
    ).astype(np.float64, copy=False)
    transform: npt.NDArray[np.float64] = _finite_array(
        camera_to_world, name="camera_to_world"
    ).astype(np.float64, copy=False)
    _require(depth.ndim == 2, "depth must have shape (H,W)")
    _require(mask.shape == depth.shape, "mask and depth shapes differ")
    _require(calibration.shape == (3, 3), "intrinsics must have shape (3,3)")
    _require(transform.shape == (4, 4), "camera_to_world must have shape (4,4)")
    valid = np.asarray(mask, dtype=np.bool_) & (depth > 0.0)
    rows, columns = np.nonzero(valid)
    selected = deterministic_subsample_indices(
        len(rows), maximum_points, key=subsample_key
    )
    rows = rows[selected]
    columns = columns[selected]
    z = depth[rows, columns]
    fx = float(calibration[0, 0])
    fy = float(calibration[1, 1])
    cx = float(calibration[0, 2])
    cy = float(calibration[1, 2])
    _require(fx > 0.0 and fy > 0.0, "intrinsic focal lengths must be positive")
    camera_points = np.column_stack(
        (
            (columns.astype(np.float64) - cx) * z / fx,
            (rows.astype(np.float64) - cy) * z / fy,
            z,
            np.ones_like(z),
        )
    )
    world = camera_points @ transform.T
    _require(
        np.allclose(world[:, 3], 1.0, rtol=0.0, atol=1e-8),
        "camera transform changed homogeneous scale",
    )
    return np.asarray(world[:, :3], dtype=np.float64)


@dataclass(frozen=True)
class SurfaceEvents:
    """Accepted point-to-surface residual events and MatPhys covariance."""

    residual_m: npt.NDArray[np.float64]
    covariance_m2: npt.NDArray[np.float64]
    frame_index: npt.NDArray[np.int64]
    node_index: npt.NDArray[np.int64]
    nearest_distance_m: npt.NDArray[np.float64]
    attempted_event_count: int

    @property
    def accepted_event_count(self) -> int:
        return len(self.residual_m)


def nearest_surface_events(
    mean_trajectory_m: npt.ArrayLike,
    covariance_m2: npt.ArrayLike,
    surface_clouds_m: list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
    *,
    maximum_distance_m: float,
) -> SurfaceEvents:
    """Associate each predicted node with the nearest disjoint-view surface point."""

    mean: npt.NDArray[np.float64] = _finite_array(
        mean_trajectory_m, name="mean trajectory"
    ).astype(np.float64, copy=False)
    covariance: npt.NDArray[np.float64] = _finite_array(
        covariance_m2, name="covariance"
    ).astype(np.float64, copy=False)
    _require(mean.ndim == 3 and mean.shape[-1] == 3, "mean shape changed")
    _require(covariance.shape == (*mean.shape[:2], 3, 3), "covariance shape changed")
    _require(len(surface_clouds_m) == len(mean), "surface frame count changed")
    _require(
        np.isfinite(maximum_distance_m) and maximum_distance_m > 0.0,
        "maximum distance must be positive",
    )

    residual_rows: list[np.ndarray] = []
    covariance_rows: list[np.ndarray] = []
    frame_rows: list[np.ndarray] = []
    node_rows: list[np.ndarray] = []
    distance_rows: list[np.ndarray] = []
    for frame, cloud_value in enumerate(surface_clouds_m):
        cloud: npt.NDArray[np.float64] = _finite_array(
            cloud_value, name=f"surface cloud {frame}"
        ).astype(np.float64, copy=False)
        _require(cloud.ndim == 2 and cloud.shape[1] == 3, "surface cloud shape changed")
        if len(cloud) == 0:
            continue
        distance, nearest = cKDTree(cloud).query(mean[frame], k=1)
        admitted = np.asarray(distance <= maximum_distance_m, dtype=np.bool_)
        indices = np.nonzero(admitted)[0]
        if not len(indices):
            continue
        residual_rows.append(cloud[nearest[indices]] - mean[frame, indices])
        covariance_rows.append(covariance[frame, indices])
        frame_rows.append(np.full(len(indices), frame, dtype=np.int64))
        node_rows.append(indices.astype(np.int64))
        distance_rows.append(np.asarray(distance[indices], dtype=np.float64))
    if residual_rows:
        residual = np.concatenate(residual_rows, axis=0)
        selected_covariance = np.concatenate(covariance_rows, axis=0)
        frame_index = np.concatenate(frame_rows)
        node_index = np.concatenate(node_rows)
        nearest_distance = np.concatenate(distance_rows)
    else:
        residual = np.empty((0, 3), dtype=np.float64)
        selected_covariance = np.empty((0, 3, 3), dtype=np.float64)
        frame_index = np.empty(0, dtype=np.int64)
        node_index = np.empty(0, dtype=np.int64)
        nearest_distance = np.empty(0, dtype=np.float64)
    return SurfaceEvents(
        residual_m=residual,
        covariance_m2=selected_covariance,
        frame_index=frame_index,
        node_index=node_index,
        nearest_distance_m=nearest_distance,
        attempted_event_count=int(mean.shape[0] * mean.shape[1]),
    )


def _regularized_covariance(
    covariance_m2: npt.ArrayLike,
    *,
    scale: float,
    observation_floor_m: float,
) -> npt.NDArray[np.float64]:
    covariance: npt.NDArray[np.float64] = _finite_array(
        covariance_m2, name="covariance"
    ).astype(np.float64, copy=False)
    _require(
        covariance.ndim == 3 and covariance.shape[1:] == (3, 3),
        "event covariance shape changed",
    )
    _require(np.isfinite(scale) and scale >= 0.0, "scale must be nonnegative")
    _require(
        np.isfinite(observation_floor_m) and observation_floor_m > 0.0,
        "observation floor must be positive",
    )
    symmetric = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    result = scale * symmetric + np.eye(3)[None, :, :] * observation_floor_m**2
    eigenvalues = np.linalg.eigvalsh(result)
    _require(
        np.all(eigenvalues > 0.0), "regularized covariance is not positive definite"
    )
    return result


def gaussian_nll(
    residual_m: npt.ArrayLike,
    covariance_m2: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return event-wise three-dimensional Gaussian negative log likelihood."""

    residual: npt.NDArray[np.float64] = _finite_array(
        residual_m, name="residual"
    ).astype(np.float64, copy=False)
    covariance: npt.NDArray[np.float64] = _finite_array(
        covariance_m2, name="covariance"
    ).astype(np.float64, copy=False)
    _require(residual.ndim == 2 and residual.shape[1] == 3, "residual shape changed")
    _require(covariance.shape == (len(residual), 3, 3), "covariance shape changed")
    sign, logdet = np.linalg.slogdet(covariance)
    _require(np.all(sign > 0.0), "covariance is not positive definite")
    solved = np.linalg.solve(covariance, residual[..., None])[..., 0]
    mahalanobis = np.einsum("ni,ni->n", residual, solved)
    return 0.5 * (3.0 * np.log(2.0 * np.pi) + logdet + mahalanobis)


def fit_matphys_scale(
    residual_m: npt.ArrayLike,
    covariance_m2: npt.ArrayLike,
    *,
    observation_floor_m: float,
    minimum_scale: float = 1e-3,
    maximum_scale: float = 1e6,
) -> float:
    """Fit one nonnegative epistemic multiplier by Gaussian likelihood."""

    residual: npt.NDArray[np.float64] = _finite_array(
        residual_m, name="residual"
    ).astype(np.float64, copy=False)
    covariance: npt.NDArray[np.float64] = _finite_array(
        covariance_m2, name="covariance"
    ).astype(np.float64, copy=False)
    _require(len(residual) > 0, "scale fit has no events")
    _require(
        0.0 < minimum_scale < maximum_scale and np.isfinite(maximum_scale),
        "scale bounds are invalid",
    )

    def objective(log_scale: float) -> float:
        scale = float(np.exp(log_scale))
        total = _regularized_covariance(
            covariance,
            scale=scale,
            observation_floor_m=observation_floor_m,
        )
        return float(np.mean(gaussian_nll(residual, total)))

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(minimum_scale)), float(np.log(maximum_scale))),
        method="bounded",
        options={"xatol": 1e-8},
    )
    _require(result.success and np.isfinite(result.fun), "MatPhys scale fit failed")
    return float(np.exp(result.x))


def fit_grouped_matphys_scale(
    residual_groups_m: list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
    covariance_groups_m2: list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
    *,
    observation_floor_m: float,
    minimum_scale: float = 1e-3,
    maximum_scale: float = 1e6,
) -> float:
    """Fit one MatPhys scale with equal weight for every source case."""

    _require(bool(residual_groups_m), "scale fit has no case groups")
    _require(
        len(residual_groups_m) == len(covariance_groups_m2),
        "residual and covariance group counts differ",
    )
    residual_groups: list[npt.NDArray[np.float64]] = []
    covariance_groups: list[npt.NDArray[np.float64]] = []
    for index, (residual_value, covariance_value) in enumerate(
        zip(residual_groups_m, covariance_groups_m2, strict=True)
    ):
        residual: npt.NDArray[np.float64] = _finite_array(
            residual_value, name=f"residual group {index}"
        ).astype(np.float64, copy=False)
        covariance: npt.NDArray[np.float64] = _finite_array(
            covariance_value, name=f"covariance group {index}"
        ).astype(np.float64, copy=False)
        _require(
            residual.ndim == 2 and residual.shape[1] == 3 and len(residual) > 0,
            "residual group shape changed",
        )
        _require(
            covariance.shape == (len(residual), 3, 3),
            "covariance group shape changed",
        )
        residual_groups.append(residual)
        covariance_groups.append(covariance)
    _require(
        0.0 < minimum_scale < maximum_scale and np.isfinite(maximum_scale),
        "scale bounds are invalid",
    )

    def objective(log_scale: float) -> float:
        scale = float(np.exp(log_scale))
        case_nll = []
        for residual, covariance in zip(
            residual_groups, covariance_groups, strict=True
        ):
            total = _regularized_covariance(
                covariance,
                scale=scale,
                observation_floor_m=observation_floor_m,
            )
            case_nll.append(float(np.mean(gaussian_nll(residual, total))))
        return float(np.mean(case_nll))

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(minimum_scale)), float(np.log(maximum_scale))),
        method="bounded",
        options={"xatol": 1e-8},
    )
    _require(
        result.success and np.isfinite(result.fun),
        "grouped MatPhys scale fit failed",
    )
    return float(np.exp(result.x))


def fit_isotropic_variance(
    residual_m: npt.ArrayLike,
    *,
    observation_floor_m: float,
) -> float:
    """Fit the maximum-likelihood isotropic variance with a fixed floor."""

    residual: npt.NDArray[np.float64] = _finite_array(
        residual_m, name="residual"
    ).astype(np.float64, copy=False)
    _require(residual.ndim == 2 and residual.shape[1] == 3, "residual shape changed")
    _require(len(residual) > 0, "isotropic fit has no events")
    raw = float(np.mean(np.square(residual, dtype=np.float64)))
    return max(raw, observation_floor_m**2)


def fit_grouped_isotropic_variance(
    residual_groups_m: list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
    *,
    observation_floor_m: float,
) -> float:
    """Fit isotropic variance while giving every source case equal weight."""

    _require(bool(residual_groups_m), "isotropic fit has no case groups")
    case_variances: list[float] = []
    for index, value in enumerate(residual_groups_m):
        residual: npt.NDArray[np.float64] = _finite_array(
            value, name=f"residual group {index}"
        ).astype(np.float64, copy=False)
        _require(
            residual.ndim == 2 and residual.shape[1] == 3 and len(residual) > 0,
            "residual group shape changed",
        )
        case_variances.append(float(np.mean(np.square(residual, dtype=np.float64))))
    return max(float(np.mean(case_variances)), observation_floor_m**2)


def equal_group_radial_quantile(
    residual_groups_m: list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
    *,
    probability: float,
) -> float:
    """Return a weighted radial quantile with equal mass assigned to each case."""

    _require(bool(residual_groups_m), "radial calibration has no case groups")
    _require(
        np.isfinite(probability) and 0.0 < probability < 1.0,
        "quantile probability is invalid",
    )
    radii: list[npt.NDArray[np.float64]] = []
    weights: list[npt.NDArray[np.float64]] = []
    group_weight = 1.0 / len(residual_groups_m)
    for index, value in enumerate(residual_groups_m):
        residual: npt.NDArray[np.float64] = _finite_array(
            value, name=f"residual group {index}"
        ).astype(np.float64, copy=False)
        _require(
            residual.ndim == 2 and residual.shape[1] == 3 and len(residual) > 0,
            "residual group shape changed",
        )
        radii.append(np.linalg.norm(residual, axis=1))
        weights.append(
            np.full(len(residual), group_weight / len(residual), dtype=np.float64)
        )
    values = np.concatenate(radii)
    mass = np.concatenate(weights)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(mass[order])
    selected = min(
        int(np.searchsorted(cumulative, probability, side="left")), len(order) - 1
    )
    return float(values[order[selected]])


def evaluate_gaussian_events(
    residual_m: npt.ArrayLike,
    covariance_m2: npt.ArrayLike,
) -> dict[str, float | int]:
    """Evaluate NLL, 90% coverage, and mean 90% ellipsoid volume."""

    residual: npt.NDArray[np.float64] = _finite_array(
        residual_m, name="residual"
    ).astype(np.float64, copy=False)
    covariance: npt.NDArray[np.float64] = _finite_array(
        covariance_m2, name="covariance"
    ).astype(np.float64, copy=False)
    _require(len(residual) > 0, "evaluation has no events")
    solved = np.linalg.solve(covariance, residual[..., None])[..., 0]
    mahalanobis = np.einsum("ni,ni->n", residual, solved)
    determinants = np.linalg.det(covariance)
    _require(np.all(determinants > 0.0), "covariance determinant is not positive")
    volumes = (4.0 / 3.0) * np.pi * CHI2_90_DF3**1.5 * np.sqrt(determinants)
    return {
        "event_count": len(residual),
        "mean_nll": float(np.mean(gaussian_nll(residual, covariance))),
        "coverage_90": float(np.mean(mahalanobis <= CHI2_90_DF3)),
        "mean_ellipsoid_volume_m3": float(np.mean(volumes)),
        "mean_mahalanobis_squared": float(np.mean(mahalanobis)),
    }


def matphys_total_covariance(
    covariance_m2: npt.ArrayLike,
    *,
    scale: float,
    observation_floor_m: float,
) -> npt.NDArray[np.float64]:
    """Public wrapper for the registered MatPhys covariance construction."""

    return _regularized_covariance(
        covariance_m2,
        scale=scale,
        observation_floor_m=observation_floor_m,
    )


def isotropic_total_covariance(
    event_count: int,
    *,
    variance_m2: float,
) -> npt.NDArray[np.float64]:
    """Construct an event-wise isotropic Gaussian comparator."""

    _require(
        type(event_count) is int and event_count > 0, "event count must be positive"
    )
    _require(
        np.isfinite(variance_m2) and variance_m2 > 0.0, "variance must be positive"
    )
    return np.broadcast_to(
        np.eye(3, dtype=np.float64) * variance_m2,
        (event_count, 3, 3),
    ).copy()


def evaluate_leave_one_group_out(
    case_ids: list[str] | tuple[str, ...],
    residual_groups_m: list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
    covariance_groups_m2: list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
    *,
    observation_floor_m: float,
) -> dict[str, object]:
    """Evaluate equal-case leave-one-out MatPhys and isotropic uncertainty."""

    _require(len(case_ids) >= 2, "leave-one-out evaluation needs two cases")
    _require(len(case_ids) == len(set(case_ids)), "leave-one-out case IDs repeat")
    _require(
        len(case_ids) == len(residual_groups_m) == len(covariance_groups_m2),
        "leave-one-out group counts differ",
    )
    residual_groups: list[npt.NDArray[np.float64]] = []
    covariance_groups: list[npt.NDArray[np.float64]] = []
    for index, (residual_value, covariance_value) in enumerate(
        zip(residual_groups_m, covariance_groups_m2, strict=True)
    ):
        residual: npt.NDArray[np.float64] = _finite_array(
            residual_value, name=f"residual group {index}"
        ).astype(np.float64, copy=False)
        covariance: npt.NDArray[np.float64] = _finite_array(
            covariance_value, name=f"covariance group {index}"
        ).astype(np.float64, copy=False)
        _require(
            residual.ndim == 2 and residual.shape[1] == 3 and len(residual) > 0,
            "residual group shape changed",
        )
        _require(
            covariance.shape == (len(residual), 3, 3),
            "covariance group shape changed",
        )
        residual_groups.append(residual)
        covariance_groups.append(covariance)

    rows: list[dict[str, object]] = []
    for held_index, case_id in enumerate(case_ids):
        train_residual = [
            value for index, value in enumerate(residual_groups) if index != held_index
        ]
        train_covariance = [
            value
            for index, value in enumerate(covariance_groups)
            if index != held_index
        ]
        scale = fit_grouped_matphys_scale(
            train_residual,
            train_covariance,
            observation_floor_m=observation_floor_m,
        )
        isotropic_variance = fit_grouped_isotropic_variance(
            train_residual,
            observation_floor_m=observation_floor_m,
        )
        conformal_radius = equal_group_radial_quantile(
            train_residual,
            probability=0.9,
        )
        held_residual = residual_groups[held_index]
        candidate_covariance = matphys_total_covariance(
            covariance_groups[held_index],
            scale=scale,
            observation_floor_m=observation_floor_m,
        )
        isotropic_covariance = isotropic_total_covariance(
            len(held_residual),
            variance_m2=isotropic_variance,
        )
        candidate = evaluate_gaussian_events(held_residual, candidate_covariance)
        isotropic = evaluate_gaussian_events(held_residual, isotropic_covariance)
        radius = np.linalg.norm(held_residual, axis=1)
        rows.append(
            {
                "case_id": case_id,
                "event_count": len(held_residual),
                "matphys_scale": scale,
                "isotropic_variance_m2": isotropic_variance,
                "conformal_radius_m": conformal_radius,
                "candidate": candidate,
                "isotropic": isotropic,
                "conformal": {
                    "coverage_90": float(np.mean(radius <= conformal_radius)),
                    "sphere_volume_m3": float(
                        (4.0 / 3.0) * np.pi * conformal_radius**3
                    ),
                },
                "candidate_nll_win": bool(
                    float(candidate["mean_nll"]) < float(isotropic["mean_nll"])
                ),
            }
        )

    candidate_nll = float(
        np.mean([float(row["candidate"]["mean_nll"]) for row in rows])  # type: ignore[index]
    )
    isotropic_nll = float(
        np.mean([float(row["isotropic"]["mean_nll"]) for row in rows])  # type: ignore[index]
    )
    candidate_volume = float(
        np.mean(
            [
                float(row["candidate"]["mean_ellipsoid_volume_m3"])  # type: ignore[index]
                for row in rows
            ]
        )
    )
    conformal_volume = float(
        np.mean(
            [
                float(row["conformal"]["sphere_volume_m3"])  # type: ignore[index]
                for row in rows
            ]
        )
    )
    return {
        "case_count": len(rows),
        "case_rows": rows,
        "equal_case_metrics": {
            "candidate_mean_nll": candidate_nll,
            "isotropic_mean_nll": isotropic_nll,
            "candidate_nll_improvement_nats": isotropic_nll - candidate_nll,
            "candidate_coverage_90": float(
                np.mean(
                    [
                        float(row["candidate"]["coverage_90"])  # type: ignore[index]
                        for row in rows
                    ]
                )
            ),
            "isotropic_coverage_90": float(
                np.mean(
                    [
                        float(row["isotropic"]["coverage_90"])  # type: ignore[index]
                        for row in rows
                    ]
                )
            ),
            "conformal_coverage_90": float(
                np.mean(
                    [
                        float(row["conformal"]["coverage_90"])  # type: ignore[index]
                        for row in rows
                    ]
                )
            ),
            "candidate_mean_ellipsoid_volume_m3": candidate_volume,
            "conformal_mean_sphere_volume_m3": conformal_volume,
            "candidate_volume_reduction_vs_conformal": (
                1.0 - candidate_volume / conformal_volume
            ),
            "candidate_nll_win_count": sum(
                bool(row["candidate_nll_win"]) for row in rows
            ),
        },
    }


def evaluate_guarded_leave_one_group_out(
    case_ids: list[str] | tuple[str, ...],
    residual_groups_m: list[npt.ArrayLike] | tuple[npt.ArrayLike, ...],
    covariance_groups_m2: list[npt.ArrayLike | None]
    | tuple[npt.ArrayLike | None, ...],
    *,
    observation_floor_m: float,
) -> dict[str, object]:
    """Evaluate MatPhys covariance with an exact isotropic abstention fallback."""

    _require(len(case_ids) >= 2, "leave-one-out evaluation needs two cases")
    _require(len(case_ids) == len(set(case_ids)), "leave-one-out case IDs repeat")
    _require(
        len(case_ids) == len(residual_groups_m) == len(covariance_groups_m2),
        "leave-one-out group counts differ",
    )
    residual_groups: list[npt.NDArray[np.float64]] = []
    covariance_groups: list[npt.NDArray[np.float64] | None] = []
    for index, (residual_value, covariance_value) in enumerate(
        zip(residual_groups_m, covariance_groups_m2, strict=True)
    ):
        residual: npt.NDArray[np.float64] = _finite_array(
            residual_value, name=f"residual group {index}"
        ).astype(np.float64, copy=False)
        _require(
            residual.ndim == 2 and residual.shape[1] == 3 and len(residual) > 0,
            "residual group shape changed",
        )
        residual_groups.append(residual)
        if covariance_value is None:
            covariance_groups.append(None)
            continue
        covariance: npt.NDArray[np.float64] = _finite_array(
            covariance_value, name=f"covariance group {index}"
        ).astype(np.float64, copy=False)
        _require(
            covariance.shape == (len(residual), 3, 3),
            "covariance group shape changed",
        )
        covariance_groups.append(covariance)
    _require(
        sum(value is not None for value in covariance_groups) >= 2,
        "guarded evaluation needs two admitted MatPhys groups",
    )

    rows: list[dict[str, object]] = []
    for held_index, case_id in enumerate(case_ids):
        train_residual = [
            value for index, value in enumerate(residual_groups) if index != held_index
        ]
        isotropic_variance = fit_grouped_isotropic_variance(
            train_residual,
            observation_floor_m=observation_floor_m,
        )
        conformal_radius = equal_group_radial_quantile(
            train_residual,
            probability=0.9,
        )
        held_residual = residual_groups[held_index]
        isotropic_covariance = isotropic_total_covariance(
            len(held_residual),
            variance_m2=isotropic_variance,
        )
        held_covariance = covariance_groups[held_index]
        scale: float | None
        if held_covariance is None:
            uncertainty_policy = "isotropic-fallback"
            scale = None
            candidate_covariance = isotropic_covariance.copy()
        else:
            uncertainty_policy = "matphys"
            paired_train = [
                (residual, covariance)
                for index, (residual, covariance) in enumerate(
                    zip(residual_groups, covariance_groups, strict=True)
                )
                if index != held_index and covariance is not None
            ]
            _require(
                len(paired_train) >= 2,
                "guarded MatPhys scale fit needs two admitted training cases",
            )
            scale = fit_grouped_matphys_scale(
                [residual for residual, _covariance in paired_train],
                [covariance for _residual, covariance in paired_train],
                observation_floor_m=observation_floor_m,
            )
            candidate_covariance = matphys_total_covariance(
                held_covariance,
                scale=scale,
                observation_floor_m=observation_floor_m,
            )
        candidate = evaluate_gaussian_events(held_residual, candidate_covariance)
        isotropic = evaluate_gaussian_events(held_residual, isotropic_covariance)
        radius = np.linalg.norm(held_residual, axis=1)
        rows.append(
            {
                "case_id": case_id,
                "event_count": len(held_residual),
                "uncertainty_policy": uncertainty_policy,
                "matphys_scale": scale,
                "isotropic_variance_m2": isotropic_variance,
                "conformal_radius_m": conformal_radius,
                "candidate": candidate,
                "isotropic": isotropic,
                "conformal": {
                    "coverage_90": float(np.mean(radius <= conformal_radius)),
                    "sphere_volume_m3": float(
                        (4.0 / 3.0) * np.pi * conformal_radius**3
                    ),
                },
                "candidate_nll_win": bool(
                    uncertainty_policy == "matphys"
                    and float(candidate["mean_nll"]) < float(isotropic["mean_nll"])
                ),
            }
        )

    candidate_nll = float(
        np.mean([float(row["candidate"]["mean_nll"]) for row in rows])  # type: ignore[index]
    )
    isotropic_nll = float(
        np.mean([float(row["isotropic"]["mean_nll"]) for row in rows])  # type: ignore[index]
    )
    candidate_volume = float(
        np.mean(
            [
                float(row["candidate"]["mean_ellipsoid_volume_m3"])  # type: ignore[index]
                for row in rows
            ]
        )
    )
    conformal_volume = float(
        np.mean(
            [
                float(row["conformal"]["sphere_volume_m3"])  # type: ignore[index]
                for row in rows
            ]
        )
    )
    return {
        "case_count": len(rows),
        "matphys_case_count": sum(
            row["uncertainty_policy"] == "matphys" for row in rows
        ),
        "isotropic_fallback_case_count": sum(
            row["uncertainty_policy"] == "isotropic-fallback" for row in rows
        ),
        "case_rows": rows,
        "equal_case_metrics": {
            "candidate_mean_nll": candidate_nll,
            "isotropic_mean_nll": isotropic_nll,
            "candidate_nll_improvement_nats": isotropic_nll - candidate_nll,
            "candidate_coverage_90": float(
                np.mean(
                    [
                        float(row["candidate"]["coverage_90"])  # type: ignore[index]
                        for row in rows
                    ]
                )
            ),
            "isotropic_coverage_90": float(
                np.mean(
                    [
                        float(row["isotropic"]["coverage_90"])  # type: ignore[index]
                        for row in rows
                    ]
                )
            ),
            "conformal_coverage_90": float(
                np.mean(
                    [
                        float(row["conformal"]["coverage_90"])  # type: ignore[index]
                        for row in rows
                    ]
                )
            ),
            "candidate_mean_ellipsoid_volume_m3": candidate_volume,
            "conformal_mean_sphere_volume_m3": conformal_volume,
            "candidate_volume_reduction_vs_conformal": (
                1.0 - candidate_volume / conformal_volume
            ),
            "candidate_nll_win_count": sum(
                bool(row["candidate_nll_win"]) for row in rows
            ),
        },
    }


__all__ = [
    "CHI2_90_DF3",
    "MATPHYS_SURFACE_UQ_SCHEMA",
    "SurfaceEvents",
    "backproject_masked_depth",
    "deterministic_camera_partition",
    "deterministic_subsample_indices",
    "evaluate_gaussian_events",
    "evaluate_guarded_leave_one_group_out",
    "evaluate_leave_one_group_out",
    "equal_group_radial_quantile",
    "fit_grouped_isotropic_variance",
    "fit_grouped_matphys_scale",
    "fit_isotropic_variance",
    "fit_matphys_scale",
    "gaussian_nll",
    "isotropic_total_covariance",
    "matphys_total_covariance",
    "nearest_surface_events",
]
