import numpy as np

from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.intervention_abduction import (
    FactualAbductionConfig,
    abduct_factual_intervention,
    evaluate_factual_abduction,
    factual_joint_weights,
)
from causal4d.rollout_bank import JointRolloutBank


def _problem():
    bank_frames = 8
    trajectories = np.zeros((3, 1, bank_frames, 1, 3), dtype=float)
    time = np.arange(bank_frames, dtype=float)
    trajectories[1, 0, :, 0, 0] = 0.01 * time
    trajectories[2, 0, :, 0, 0] = -0.01 * time

    def metadata(identifier, gain, shift):
        return {
            "hypothesis_id": identifier,
            "action": {
                "proposal_id": "known",
                "future_action_observed": True,
                "provenance": "unit factual action",
            },
            "contact": {
                "attachment_shifts": [shift],
                "gain_multiplier": gain,
                "delay_steps": 0,
                "slip_fraction": 0.0,
                "rotation_degrees": 0.0,
            },
        }

    bank = JointRolloutBank(
        hypothesis_ids=("nominal", "high_gain", "shifted"),
        hypothesis_metadata=(
            metadata("nominal", 1.0, 0),
            metadata("high_gain", 1.15, 0),
            metadata("shifted", 1.0, 1),
        ),
        hypothesis_prior_weights=np.asarray([0.6, 0.2, 0.2]),
        parameter_particles=np.asarray([[0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
        variance_floor_m2=1e-8,
    )
    full_frames = 11
    intervention_frame = 4
    full_observations = np.zeros((full_frames, 1, 3), dtype=float)
    full_observations[intervention_frame - 1 :] = trajectories[1, 0]
    actions = np.zeros((full_frames, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="abduction_unit",
        case_id="synthetic",
        observations=full_observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=intervention_frame,
    )
    belief = TwinBelief(
        context=context,
        endpoint_frame=intervention_frame - 1,
        particle_ids=("p0",),
        theta_names=("spring",),
        endpoint_position_m=np.zeros((1, 1, 3)),
        endpoint_velocity_mps=np.zeros((1, 1, 3)),
        theta=np.asarray([[0.0]]),
        discrepancy_mean_m=np.zeros((1, 1, 3)),
        discrepancy_variance_m2=np.zeros((1, 1, 3)),
        weights=np.asarray([1.0]),
    )
    observations = trajectories[1, 0].copy()
    mask = np.ones((bank_frames, 1), dtype=bool)
    config = FactualAbductionConfig(
        observation_scale_m=0.001,
        likelihood_power=60.0,
        dynamic_likelihood_weight=0.5,
    )
    return bank, belief, observations, mask, config


def test_abduction_recovers_realized_intervention_and_beats_no_z() -> None:
    bank, belief, observations, mask, config = _problem()
    factual = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
    )
    joint = factual_joint_weights(
        factual,
        hypothesis_count=3,
        particle_count=1,
    )
    assert np.argmax(joint[:, 0]) == 1
    assert factual.phi_names == (
        "gain_multiplier",
        "delay_steps",
        "rotation_degrees",
    )
    assert factual.kappa_names == ("attachment_shift_hand_0", "slip_fraction")
    result = evaluate_factual_abduction(
        bank,
        belief,
        factual,
        observations,
        observation_mask=mask,
        prefix_frame_count=4,
        config=config,
    )
    assert result["map_hypothesis_id"] == "high_gain"
    assert result["relative_track_error_improvement"] > 0.95


def test_abduction_cannot_see_observations_after_its_o_plus_prefix() -> None:
    bank, belief, observations, mask, config = _problem()
    first = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
    )
    changed = observations.copy()
    changed[4:] += 1000.0
    changed_mask = mask.copy()
    changed_mask[4:] = False
    second = abduct_factual_intervention(
        bank,
        belief,
        changed,
        prefix_frame_count=4,
        observation_mask=changed_mask,
        config=config,
    )
    assert first.artifact_id == second.artifact_id
    assert np.array_equal(first.weights, second.weights)


def test_abduction_requires_the_factual_observed_action() -> None:
    bank, belief, observations, mask, config = _problem()
    metadata = list(bank.hypothesis_metadata)
    metadata[0] = {
        **metadata[0],
        "action": {**metadata[0]["action"], "future_action_observed": False},
    }
    invalid = JointRolloutBank(
        hypothesis_ids=bank.hypothesis_ids,
        hypothesis_metadata=tuple(metadata),
        hypothesis_prior_weights=bank.hypothesis_prior_weights,
        parameter_particles=bank.parameter_particles,
        parameter_weights=bank.parameter_weights,
        trajectories=bank.trajectories,
    )
    with np.testing.assert_raises_regex(ValueError, "observed u_obs"):
        abduct_factual_intervention(
            invalid,
            belief,
            observations,
            prefix_frame_count=4,
            observation_mask=mask,
            config=config,
        )
