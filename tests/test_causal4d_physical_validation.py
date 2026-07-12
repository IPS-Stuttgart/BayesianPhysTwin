import numpy as np

from causal4d.contracts import PhysicalPosterior, build_causal_context
from causal4d.physical_validation import (
    evaluate_beta_zero_physical_posterior,
    physical_posterior_moments,
)


def _posterior() -> PhysicalPosterior:
    observations = np.zeros((7, 1, 3), dtype=float)
    actions = np.zeros((7, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="physical_validation",
        case_id="synthetic",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )
    states = np.zeros((2, 5, 1, 3), dtype=float)
    states[0, :, 0, 0] = np.arange(5) * 0.01
    states[1, :, 0, 0] = np.arange(5) * 0.02
    return PhysicalPosterior(
        context=context,
        component_ids=("a", "b"),
        state_trajectories_m=states,
        readout_trajectories_m=states + np.asarray([0.001, 0.0, 0.0]),
        readout_variance_m2=np.full((2, 1, 3), 1e-5),
        weights=np.asarray([0.75, 0.25]),
        phi=np.asarray([[1.0], [1.0]]),
        kappa_cf=np.asarray([[0.0], [1.0]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 0]),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id="1" * 64,
        source_factual_intervention_id="2" * 64,
        source_query_id="3" * 64,
    )


def test_beta_zero_validation_reports_accuracy_and_calibration_without_semantics() -> None:
    posterior = _posterior()
    mean, variance = physical_posterior_moments(posterior)
    result = evaluate_beta_zero_physical_posterior(
        posterior,
        mean,
        mask=np.ones(mean.shape[:2], dtype=bool),
        start_frame=1,
    )
    assert result["semantic_beta"] == 0.0
    assert not result["semantic_evidence_consumed"]
    assert not result["molmo_motion_consumed"]
    assert result["coordinate_rmse_m"] == 0.0
    assert result["track_error_m"] == 0.0
    assert result["coverage"] == 1.0
    assert np.all(variance > 0.0)


def test_physical_moments_include_epistemic_and_discrepancy_variance() -> None:
    posterior = _posterior()
    mean, variance = physical_posterior_moments(posterior)
    expected_mean_x = (
        0.75 * posterior.readout_trajectories_m[0, :, 0, 0]
        + 0.25 * posterior.readout_trajectories_m[1, :, 0, 0]
    )
    assert np.allclose(mean[:, 0, 0], expected_mean_x)
    assert np.all(variance[:, 0, 0] >= 0.999e-5)
