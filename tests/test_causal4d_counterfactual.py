import numpy as np

from causal4d.contracts import (
    CounterfactualQuery,
    FactualIntervention,
    TwinBelief,
    build_causal_context,
)
from causal4d.counterfactual import (
    apply_counterfactual_operator,
    physical_posterior_mean,
)
from causal4d.rollout_bank import JointRolloutBank


def _setup(factual_weights=(0.1, 0.8, 0.1), *, contact_policy="new_contact"):
    full_frames = 9
    train_end = 4
    observations = np.zeros((full_frames, 1, 3), dtype=float)
    factual_actions = np.zeros((full_frames, 1, 3), dtype=float)
    original_cf = factual_actions.copy()
    factual_context = build_causal_context(
        protocol_id="counterfactual_unit",
        case_id="synthetic",
        observations=observations,
        observed_actions=factual_actions,
        counterfactual_actions=original_cf,
        intervention_frame=train_end,
        counterfactual_action_id="factual_action",
    )
    belief = TwinBelief(
        context=factual_context,
        endpoint_frame=train_end - 1,
        particle_ids=("p0",),
        theta_names=("spring",),
        endpoint_position_m=np.zeros((1, 1, 3)),
        endpoint_velocity_mps=np.zeros((1, 1, 3)),
        theta=np.asarray([[0.0]]),
        discrepancy_mean_m=np.asarray([[[0.002, 0.0, 0.0]]]),
        discrepancy_variance_m2=np.full((1, 1, 3), 1e-6),
        weights=np.asarray([1.0]),
    )
    factual = FactualIntervention(
        context=factual_context,
        component_ids=("nominal::p0", "shift::p0", "gain::p0"),
        phi_names=("gain_multiplier", "delay_steps", "rotation_degrees"),
        kappa_names=("attachment_shift_hand_0", "slip_fraction"),
        phi=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.15, 0.0, 0.0]]),
        kappa_obs=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        hypothesis_indices=np.asarray([0, 1, 2]),
        twin_particle_indices=np.asarray([0, 0, 0]),
        weights=np.asarray(factual_weights),
        evidence_frame_stop=6,
        source_twin_belief_id=belief.artifact_id,
    )
    counterfactual_actions = factual_actions.copy()
    counterfactual_actions[train_end:, 0, 0] = np.arange(1, 6) * 0.01
    query_context = build_causal_context(
        protocol_id="counterfactual_unit",
        case_id="synthetic",
        observations=observations,
        observed_actions=factual_actions,
        counterfactual_actions=counterfactual_actions,
        intervention_frame=train_end,
        counterfactual_action_id="new_action",
    )
    query = CounterfactualQuery(
        context=query_context,
        controller_points_m=counterfactual_actions[train_end:],
        horizon_frames=full_frames - train_end,
        contact_policy=contact_policy,
        source_factual_intervention_id=factual.artifact_id,
    )

    def metadata(identifier, gain, shift):
        return {
            "hypothesis_id": identifier,
            "action": {
                "proposal_id": "new_action",
                "future_action_observed": False,
            },
            "contact": {
                "attachment_shifts": [shift],
                "gain_multiplier": gain,
                "delay_steps": 0,
                "slip_fraction": 0.0,
                "rotation_degrees": 0.0,
            },
        }

    trajectories = np.zeros((3, 1, 6, 1, 3), dtype=np.float32)
    trajectories[0, 0, :, 0, 0] = np.arange(6) * 0.01
    trajectories[1, 0, :, 0, 0] = np.arange(6) * 0.012
    trajectories[2, 0, :, 0, 0] = np.arange(6) * 0.014
    bank = JointRolloutBank(
        hypothesis_ids=("nominal", "shift", "gain"),
        hypothesis_metadata=(
            metadata("nominal", 1.0, 0),
            metadata("shift", 1.0, 1),
            metadata("gain", 1.15, 0),
        ),
        hypothesis_prior_weights=np.asarray([0.5, 0.25, 0.25]),
        parameter_particles=np.asarray([[0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
        variance_floor_m2=2e-6,
    )
    manifest = {
        "causal_context": query_context.as_dict(),
        "twin_belief_id": belief.artifact_id,
    }
    return bank, manifest, belief, factual, query


def test_new_grasp_carries_phi_but_resamples_kappa_cf() -> None:
    bank, manifest, belief, factual, query = _setup((0.1, 0.8, 0.1))
    first = apply_counterfactual_operator(bank, manifest, belief, factual, query)
    other = _setup((0.8, 0.1, 0.1))
    second = apply_counterfactual_operator(*other)
    assert np.array_equal(first.weights, second.weights)
    assert np.allclose(first.weights, [0.6, 0.3, 0.1])
    assert first.metadata["persistent_phi_transferred"]
    assert first.metadata["fresh_kappa_cf_sampled"]
    assert not first.metadata["factual_kappa_reused"]


def test_same_grasp_preserves_factual_kappa_posterior() -> None:
    first_args = _setup((0.1, 0.8, 0.1), contact_policy="same_grasp")
    second_args = _setup((0.8, 0.1, 0.1), contact_policy="same_grasp")
    first = apply_counterfactual_operator(*first_args)
    second = apply_counterfactual_operator(*second_args)
    assert np.allclose(first.weights, [0.1, 0.8, 0.1])
    assert np.allclose(second.weights, [0.8, 0.1, 0.1])
    assert first.metadata["factual_kappa_reused"]
    assert not first.metadata["fresh_kappa_cf_sampled"]


def test_counterfactual_keeps_state_and_discrepancy_readout_distinct() -> None:
    bank, manifest, belief, factual, query = _setup()
    unchanged = bank.trajectories.copy()
    posterior = apply_counterfactual_operator(
        bank,
        manifest,
        belief,
        factual,
        query,
    )
    assert np.array_equal(bank.trajectories, unchanged)
    assert np.allclose(
        posterior.readout_trajectories_m - posterior.state_trajectories_m,
        np.asarray([0.002, 0.0, 0.0]),
    )
    assert np.allclose(posterior.readout_variance_m2, 3e-6)
    assert not np.array_equal(
        physical_posterior_mean(posterior),
        physical_posterior_mean(posterior, readout=False),
    )
