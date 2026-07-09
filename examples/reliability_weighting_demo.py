"""Synthetic demo for reliability-aware pseudo-measurements."""

from __future__ import annotations

import numpy as np

from bayesian_phystwin import (
    ParameterEnsemble,
    PseudoMeasurementBatch,
    ReliabilityConfig,
    reliability_weighted_loss,
    robust_mixture_likelihood,
    score_reliability,
)


def main() -> None:
    rng = np.random.default_rng(7)
    n_tracks = 60
    t = np.linspace(0.0, 1.0, n_tracks)
    predicted = np.column_stack([t, 0.2 * np.sin(2 * np.pi * t), 0.05 * t])
    observed = predicted + rng.normal(scale=0.01, size=predicted.shape)

    drifting = np.arange(n_tracks) > 42
    observed[drifting] += np.array([0.09, -0.04, 0.02])

    confidence = np.where(drifting, 0.25, 0.9)
    occluded = (t > 0.45) & (t < 0.58)
    boundary_distance = np.where(occluded, 0.005, 0.08)
    flow_inconsistency = np.where(drifting, 0.2, 0.01)

    batch = PseudoMeasurementBatch(
        observed=observed,
        predicted=predicted,
        variance=0.01**2,
        confidence=confidence,
        occluded=occluded,
        boundary_distance=boundary_distance,
        flow_inconsistency=flow_inconsistency,
    )
    cfg = ReliabilityConfig()
    reliability = score_reliability(batch, cfg)
    likelihood = robust_mixture_likelihood(
        batch,
        prior_reliability=reliability.weights,
    )
    loss = reliability_weighted_loss(batch, cfg)

    stiffness_particles = rng.normal(loc=1.0, scale=0.25, size=(200, 1))
    ensemble = ParameterEnsemble.from_prior_samples(stiffness_particles)
    residuals = np.square(stiffness_particles[:, 0] - 0.85) * 40.0
    ensemble.update_from_residuals(
        residuals,
        variance=0.05,
        reliability=np.full(
            stiffness_particles.shape[0],
            likelihood.posterior_inlier_probability.mean(),
        ),
    )

    print(f"mean reliability: {reliability.weights.mean():.3f}")
    print(f"minimum reliability: {reliability.weights.min():.3f}")
    print(
        "mean posterior inlier probability: "
        f"{likelihood.posterior_inlier_probability.mean():.3f}"
    )
    print(f"effective track count: {reliability.effective_sample_size:.1f}/{n_tracks}")
    print(f"weighted loss: {loss:.3f}")
    print(f"robust mixture NLL: {likelihood.mean_negative_log_likelihood:.3f}")
    print(f"posterior stiffness mean: {ensemble.mean()[0]:.3f}")
    print(f"posterior stiffness std: {np.sqrt(ensemble.covariance()[0, 0]):.3f}")


if __name__ == "__main__":
    main()
