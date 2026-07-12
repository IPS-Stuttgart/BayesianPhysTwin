"""Accuracy and calibration checks for beta-zero physical posteriors."""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

from causal4d.contracts import PhysicalPosterior


def physical_posterior_moments(
    posterior: PhysicalPosterior,
) -> tuple[np.ndarray, np.ndarray]:
    """Moment-match epistemic rollout and conditional discrepancy uncertainty."""

    components = posterior.readout_trajectories_m.astype(float)
    mean = np.einsum("k,ktnc->tnc", posterior.weights, components)
    centered = components - mean[None]
    epistemic = np.einsum(
        "k,ktnc->tnc",
        posterior.weights,
        np.square(centered),
    )
    conditional = np.einsum(
        "k,knc->nc",
        posterior.weights,
        posterior.readout_variance_m2,
    )
    variance = epistemic + conditional[None]
    return mean, np.maximum(variance, np.finfo(float).tiny)


def evaluate_beta_zero_physical_posterior(
    posterior: PhysicalPosterior,
    truth_m: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    start_frame: int = 1,
    confidence_level: float = 0.90,
) -> dict[str, Any]:
    """Evaluate physical prediction with the semantic factor explicitly absent."""

    truth = np.asarray(truth_m, dtype=float)
    mean, variance = physical_posterior_moments(posterior)
    if truth.shape != mean.shape:
        raise ValueError("truth must match the physical posterior trajectory")
    if not 0 <= start_frame < len(truth):
        raise ValueError("start_frame must lie inside the posterior trajectory")
    valid = np.all(np.isfinite(truth), axis=2)
    if mask is not None:
        supplied = np.asarray(mask, dtype=bool)
        if supplied.shape == truth.shape:
            supplied = np.all(supplied, axis=2)
        if supplied.shape != truth.shape[:2]:
            raise ValueError("mask must have shape (T, N) or (T, N, 3)")
        valid &= supplied
    valid[:start_frame] = False
    if not np.any(valid):
        raise ValueError("physical validation has no valid held-out point frames")
    coordinate_valid = np.repeat(valid[:, :, None], truth.shape[2], axis=2)
    residual = mean - truth
    selected_residual = residual[coordinate_valid]
    selected_variance = variance[coordinate_valid]
    z_score = NormalDist().inv_cdf(0.5 * (1.0 + confidence_level))
    standard_deviation = np.sqrt(variance)
    lower = mean - z_score * standard_deviation
    upper = mean + z_score * standard_deviation
    selected_truth = truth[coordinate_valid]
    selected_lower = lower[coordinate_valid]
    selected_upper = upper[coordinate_valid]
    vectors = residual[valid]
    final_valid = valid[-1]
    fde = (
        float(np.mean(np.linalg.norm(residual[-1, final_valid], axis=1)))
        if np.any(final_valid)
        else None
    )
    return {
        "physical_posterior_id": posterior.artifact_id,
        "semantic_beta": 0.0,
        "semantic_evidence_consumed": False,
        "molmo_motion_consumed": False,
        "evaluation_frame_interval": [start_frame, len(truth)],
        "valid_point_frames": int(np.sum(valid)),
        "coordinate_rmse_m": float(np.sqrt(np.mean(np.square(selected_residual)))),
        "track_error_m": float(np.mean(np.linalg.norm(vectors, axis=1))),
        "fde_m": fde,
        "coverage": float(
            np.mean(
                (selected_truth >= selected_lower)
                & (selected_truth <= selected_upper)
            )
        ),
        "coverage_error": float(
            abs(
                np.mean(
                    (selected_truth >= selected_lower)
                    & (selected_truth <= selected_upper)
                )
                - confidence_level
            )
        ),
        "mean_interval_width_m": float(
            np.mean(selected_upper - selected_lower)
        ),
        "nees": float(np.mean(np.square(selected_residual) / selected_variance)),
        "gaussian_nll": float(
            np.mean(
                0.5
                * (
                    np.log(2.0 * np.pi * selected_variance)
                    + np.square(selected_residual) / selected_variance
                )
            )
        ),
    }
