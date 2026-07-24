"""Conservative bias handling and admission for multiview ray updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phystwin_bayesian_anchor import RobustEndpointPosterior
from .phystwin_cotracker3_cues import CoTracker3RayDiscrepancyPosterior


@dataclass(frozen=True)
class AffineNuisanceDiagnostics:
    """Summary of the shared affine component removed from a ray update."""

    coefficients: np.ndarray
    observed_count: int
    raw_magnitude_median_m: float
    affine_magnitude_median_m: float
    retained_magnitude_median_m: float
    affine_explained_energy_fraction: float


@dataclass(frozen=True)
class BiasAwareRayEndpoint:
    """Ray endpoint after treating a shared affine field as nuisance bias."""

    posterior: RobustEndpointPosterior
    diagnostics: AffineNuisanceDiagnostics


@dataclass(frozen=True)
class PrefixAdmissionDecision:
    """Baseline-relative decision made without future observations."""

    accepted: bool
    observed_fraction: float
    median_inlier_probability: float
    absolute_improvement_m: float
    relative_improvement: float
    early_difference_m: float
    late_difference_m: float
    gates: dict[str, bool]


def remove_affine_ray_nuisance(
    ray: CoTracker3RayDiscrepancyPosterior,
    baseline_endpoint_m: np.ndarray,
    *,
    unobserved_variance_m2: float,
) -> BiasAwareRayEndpoint:
    """Remove an unidentifiable affine camera field and retain its uncertainty."""

    baseline = np.asarray(baseline_endpoint_m, dtype=float)
    if baseline.shape != ray.mean_m.shape:
        raise ValueError("baseline endpoint must match the ray posterior")
    if unobserved_variance_m2 <= 0.0:
        raise ValueError("unobserved variance must be positive")
    observed = np.asarray(ray.observed, dtype=bool)
    if observed.shape != (len(baseline),):
        raise ValueError("ray observation mask must match the baseline")
    if not np.any(observed):
        posterior = RobustEndpointPosterior(
            mean=np.zeros_like(ray.mean_m, dtype=float),
            variance=np.full(
                len(baseline),
                unobserved_variance_m2,
                dtype=float,
            ),
            final_inlier_probability=np.asarray(
                ray.final_inlier_probability,
                dtype=float,
            ).copy(),
            update_count=np.asarray(ray.update_count, dtype=np.int64).copy(),
        )
        return BiasAwareRayEndpoint(
            posterior=posterior,
            diagnostics=AffineNuisanceDiagnostics(
                coefficients=np.zeros((4, 3), dtype=float),
                observed_count=0,
                raw_magnitude_median_m=0.0,
                affine_magnitude_median_m=0.0,
                retained_magnitude_median_m=0.0,
                affine_explained_energy_fraction=0.0,
            ),
        )

    positions = baseline[observed]
    raw = np.asarray(ray.mean_m, dtype=float)[observed]
    design = np.column_stack((positions, np.ones(len(positions))))
    coefficients = np.linalg.lstsq(design, raw, rcond=None)[0]
    affine = design @ coefficients
    retained = raw - affine

    mean = np.zeros_like(ray.mean_m, dtype=float)
    mean[observed] = retained
    variance = np.full(len(mean), unobserved_variance_m2, dtype=float)
    affine_variance = np.sum(np.square(affine), axis=1) / 3.0
    variance[observed] = (
        np.asarray(ray.variance_m2, dtype=float)[observed] + affine_variance
    )

    raw_energy = float(np.sum(np.square(raw)))
    retained_energy = float(np.sum(np.square(retained)))
    explained = 0.0 if raw_energy <= 0.0 else 1.0 - retained_energy / raw_energy
    posterior = RobustEndpointPosterior(
        mean=mean,
        variance=variance,
        final_inlier_probability=np.asarray(
            ray.final_inlier_probability,
            dtype=float,
        ).copy(),
        update_count=np.asarray(ray.update_count, dtype=np.int64).copy(),
    )
    diagnostics = AffineNuisanceDiagnostics(
        coefficients=coefficients,
        observed_count=int(np.sum(observed)),
        raw_magnitude_median_m=float(np.median(np.linalg.norm(raw, axis=1))),
        affine_magnitude_median_m=float(np.median(np.linalg.norm(affine, axis=1))),
        retained_magnitude_median_m=float(np.median(np.linalg.norm(retained, axis=1))),
        affine_explained_energy_fraction=explained,
    )
    return BiasAwareRayEndpoint(
        posterior=posterior,
        diagnostics=diagnostics,
    )


def decide_prefix_admission(
    *,
    baseline_all_m: float,
    candidate_all_m: float,
    baseline_early_m: float,
    candidate_early_m: float,
    baseline_late_m: float,
    candidate_late_m: float,
    observed_fraction: float,
    median_inlier_probability: float,
    minimum_observed_fraction: float,
    minimum_inlier_probability: float,
    minimum_absolute_improvement_m: float,
    minimum_relative_improvement: float,
) -> PrefixAdmissionDecision:
    """Admit an update only when independent prefix evidence beats fallback."""

    positive = (
        baseline_all_m,
        candidate_all_m,
        baseline_early_m,
        candidate_early_m,
        baseline_late_m,
        candidate_late_m,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("prefix metrics must be finite and positive")
    if not 0.0 <= observed_fraction <= 1.0:
        raise ValueError("observed fraction must lie in [0, 1]")
    if not 0.0 <= median_inlier_probability <= 1.0:
        raise ValueError("inlier probability must lie in [0, 1]")
    if not 0.0 <= minimum_observed_fraction <= 1.0:
        raise ValueError("minimum observed fraction must lie in [0, 1]")
    if not 0.0 <= minimum_inlier_probability <= 1.0:
        raise ValueError("minimum inlier probability must lie in [0, 1]")
    if minimum_absolute_improvement_m < 0.0:
        raise ValueError("minimum absolute improvement must be nonnegative")
    if not 0.0 <= minimum_relative_improvement < 1.0:
        raise ValueError("minimum relative improvement must lie in [0, 1)")

    absolute_improvement = baseline_all_m - candidate_all_m
    relative_improvement = absolute_improvement / baseline_all_m
    gates = {
        "observation_support": (observed_fraction >= minimum_observed_fraction),
        "robust_inlier_support": (
            median_inlier_probability >= minimum_inlier_probability
        ),
        "absolute_prefix_improvement": (
            absolute_improvement >= minimum_absolute_improvement_m
        ),
        "relative_prefix_improvement": (
            relative_improvement >= minimum_relative_improvement
        ),
        "early_prefix_nonregression": (candidate_early_m <= baseline_early_m),
        "late_prefix_nonregression": (candidate_late_m <= baseline_late_m),
    }
    return PrefixAdmissionDecision(
        accepted=all(gates.values()),
        observed_fraction=observed_fraction,
        median_inlier_probability=median_inlier_probability,
        absolute_improvement_m=absolute_improvement,
        relative_improvement=relative_improvement,
        early_difference_m=candidate_early_m - baseline_early_m,
        late_difference_m=candidate_late_m - baseline_late_m,
        gates=gates,
    )
