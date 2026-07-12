import numpy as np

from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.parameter_support_audit import (
    ParameterSupportAuditConfig,
    audit_parameter_support,
)
from causal4d.rollout_bank import JointRolloutBank


def _metadata(identifier: str, gain: float) -> dict:
    return {
        "hypothesis_id": identifier,
        "action": {
            "proposal_id": "known",
            "future_action_observed": True,
            "provenance": "unit",
        },
        "contact": {
            "attachment_shifts": [0],
            "gain_multiplier": gain,
            "delay_steps": 0,
            "slip_fraction": 0.0,
            "rotation_degrees": 0.0,
        },
    }


def _inputs() -> tuple[JointRolloutBank, TwinBelief, np.ndarray, np.ndarray]:
    frame_count = 6
    node_count = 2
    particle_count = 4
    trajectories = np.zeros(
        (2, particle_count, frame_count, node_count, 3),
        dtype=float,
    )
    for hypothesis in range(2):
        for particle in range(particle_count):
            trajectories[hypothesis, particle, :, :, 0] = np.arange(frame_count)[
                :, None
            ] * (0.001 + 0.0003 * hypothesis + 0.0002 * particle)
    particles = np.asarray([[0.0, 0.0], [0.2, 0.0], [-0.2, 0.0], [0.0, 0.3]])
    weights = np.asarray([0.40, 0.30, 0.20, 0.10])
    bank = JointRolloutBank(
        hypothesis_ids=("nominal", "gain"),
        hypothesis_metadata=(
            _metadata("nominal", 1.0),
            _metadata("gain", 1.1),
        ),
        hypothesis_prior_weights=np.asarray([0.7, 0.3]),
        parameter_particles=particles,
        parameter_weights=weights,
        trajectories=trajectories,
        variance_floor_m2=1e-6,
    )
    observations = trajectories[1, 2].astype(float)
    actions = np.zeros((8, 1, 3), dtype=float)
    context_observations = np.zeros((8, node_count, 3), dtype=float)
    context = build_causal_context(
        protocol_id="support_audit_unit",
        case_id="synthetic",
        observations=context_observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )
    belief = TwinBelief(
        context=context,
        endpoint_frame=2,
        particle_ids=tuple(f"p{index}" for index in range(particle_count)),
        theta_names=("object", "controller"),
        endpoint_position_m=np.zeros((particle_count, node_count, 3)),
        endpoint_velocity_mps=np.zeros((particle_count, node_count, 3)),
        theta=particles,
        discrepancy_mean_m=np.zeros((particle_count, node_count, 3)),
        discrepancy_variance_m2=np.full(
            (particle_count, node_count, 3),
            2e-6,
        ),
        weights=bank.parameter_weights,
    )
    return bank, belief, observations, np.ones((frame_count, node_count), dtype=bool)


def test_support_audit_compares_reductions_to_full_moments() -> None:
    bank, belief, observations, mask = _inputs()
    result = audit_parameter_support(
        bank,
        belief,
        observations,
        mask,
        config=ParameterSupportAuditConfig(
            counts=(2, 4),
            prefix_frame_count=3,
            energy_samples=4,
        ),
    )
    assert len(result["candidates"]) == 4
    assert result["stable_counts"]["top_mass"] == 4
    assert result["stable_counts"]["weighted_coreset"] == 2
    full = next(
        row
        for row in result["candidates"]
        if row["method"] == "top_mass" and row["count"] == 4
    )
    assert full["predictive_mean_rmse_vs_full_m"] == 0.0
    assert full["predictive_variance_relative_l2_vs_full"] == 0.0
    assert set(full["predictive"]["by_horizon"]) == {"early", "middle", "late"}


def test_future_labels_cannot_change_support_or_stability_selection() -> None:
    bank, belief, observations, mask = _inputs()
    config = ParameterSupportAuditConfig(
        counts=(2, 4),
        prefix_frame_count=3,
        energy_samples=4,
    )
    first = audit_parameter_support(
        bank,
        belief,
        observations,
        mask,
        config=config,
    )
    changed = observations.copy()
    changed[3:] += 10.0
    second = audit_parameter_support(
        bank,
        belief,
        changed,
        mask,
        config=config,
    )
    assert first["stable_counts"] == second["stable_counts"]
    for left, right in zip(first["candidates"], second["candidates"], strict=True):
        assert left["indices"] == right["indices"]
        assert left["weights"] == right["weights"]
        assert (
            left["predictive_mean_rmse_vs_full_m"]
            == right["predictive_mean_rmse_vs_full_m"]
        )
    assert (
        first["candidates"][0]["predictive"]["coordinate_rmse_m"]
        != second["candidates"][0]["predictive"]["coordinate_rmse_m"]
    )
