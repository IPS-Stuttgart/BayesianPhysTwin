import itertools
from pathlib import Path

import numpy as np

from causal4d.cli.audit_real_oracle_gap import _write_component_csv
from causal4d.contracts import (
    PhysicalPosterior,
    TwinBelief,
    build_causal_context,
)
from causal4d.intervention_abduction import FactualAbductionConfig
from causal4d.real_oracle_audit import (
    HoldoutOracleProtocol,
    audit_oracle_bank,
    bpt_nominal_prediction,
    oracle_gap_report,
    variance_decomposition,
    verify_nested_rollout_banks,
)
from causal4d.rollout_bank import JointRolloutBank


def _context(frame_count: int = 8):
    observations = np.zeros((frame_count, 2, 3), dtype=float)
    actions = np.zeros((frame_count, 1, 3), dtype=float)
    return build_causal_context(
        protocol_id="real_oracle_audit_unit",
        case_id="synthetic",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )


def _metadata(identifier: str, shift: int) -> dict:
    return {
        "hypothesis_id": identifier,
        "action": {
            "proposal_id": "known",
            "future_action_observed": True,
            "provenance": "unit known action",
        },
        "contact": {
            "attachment_shifts": [shift],
            "gain_multiplier": 1.0,
            "delay_steps": 0,
            "slip_fraction": 0.0,
            "rotation_degrees": 0.0,
        },
    }


def _bank_and_belief() -> tuple[JointRolloutBank, TwinBelief]:
    trajectories = np.zeros((2, 1, 6, 2, 3), dtype=float)
    trajectories[0, 0, :, :, 0] = np.arange(6)[:, None] * 0.001
    trajectories[1, 0, :, :, 0] = np.arange(6)[:, None] * 0.004
    bank = JointRolloutBank(
        hypothesis_ids=("nominal", "shifted"),
        hypothesis_metadata=(_metadata("nominal", 0), _metadata("shifted", 1)),
        hypothesis_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=np.asarray([[1.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
        variance_floor_m2=1e-6,
    )
    belief = TwinBelief(
        context=_context(),
        endpoint_frame=2,
        particle_ids=("theta_0",),
        theta_names=("spring",),
        endpoint_position_m=np.zeros((1, 2, 3)),
        endpoint_velocity_mps=np.zeros((1, 2, 3)),
        theta=np.asarray([[1.0]]),
        discrepancy_mean_m=np.zeros((1, 2, 3)),
        discrepancy_variance_m2=np.zeros((1, 2, 3)),
        weights=np.asarray([1.0]),
    )
    return bank, belief


def test_oracle_audit_separates_component_and_discrepancy_headroom() -> None:
    bank, belief = _bank_and_belief()
    truth = bank.trajectories[1, 0].astype(float)
    truth[:, 0, 1] += 0.02
    truth[:, 1, 1] -= 0.005
    protocol = HoldoutOracleProtocol(start_frame=2, stop_frame=6)
    report, rows = audit_oracle_bank(
        bank,
        belief,
        truth,
        np.ones(truth.shape[:2], dtype=bool),
        protocol,
        bank_name="unit",
        discrepancy_cap_m=0.01,
    )
    assert len(rows) == 2
    assert report["label_use"] == "diagnostic_only"
    assert not report["deployable"]
    assert report["best"]["discrepancy_aware"]["hypothesis_id"] == "shifted"
    assert report["best"]["per_node_constant_uncapped"]["hypothesis_id"] == "shifted"
    assert report["best"]["per_node_constant_uncapped"]["metrics"][
        "track_error_m"
    ] < 1e-12
    assert report["best"]["per_node_constant_capped"]["metrics"][
        "track_error_m"
    ] > 0.0


def test_nominal_bpt_update_accepts_only_the_declared_prefix() -> None:
    bank, belief = _bank_and_belief()
    prefix = bank.trajectories[0, 0, :3].astype(float)
    prediction, weights = bpt_nominal_prediction(
        bank,
        belief,
        prefix,
        prefix_mask=np.ones(prefix.shape[:2], dtype=bool),
        config=FactualAbductionConfig(),
    )
    assert prediction.shape == bank.trajectories.shape[2:]
    assert np.argmax(weights[:, 0]) == 0
    with np.testing.assert_raises_regex(ValueError, "leave a holdout"):
        bpt_nominal_prediction(
            bank,
            belief,
            bank.trajectories[0, 0],
            prefix_mask=np.ones(bank.trajectories.shape[2:4], dtype=bool),
            config=FactualAbductionConfig(),
        )


def test_expanded_bank_must_preserve_current_rollouts() -> None:
    current, _ = _bank_and_belief()
    expanded_metadata = tuple(
        {
            **metadata,
            "prior_weight": 0.1,
            "action": {**metadata["action"], "prior_weight": 0.1},
            "contact": {
                **metadata["contact"],
                "contact_prior_weight": 0.1,
            },
        }
        for metadata in current.hypothesis_metadata
    )
    expanded = JointRolloutBank(
        hypothesis_ids=("nominal", "shifted", "extra"),
        hypothesis_metadata=(
            *expanded_metadata,
            _metadata("extra", 2),
        ),
        hypothesis_prior_weights=np.ones(3),
        parameter_particles=current.parameter_particles,
        parameter_weights=current.parameter_weights,
        trajectories=np.concatenate(
            (current.trajectories, current.trajectories[:1] + 0.01),
            axis=0,
        ),
        variance_floor_m2=current.variance_floor_m2,
    )
    result = verify_nested_rollout_banks(current, expanded)
    assert result["verified"]
    assert result["maximum_absolute_trajectory_difference_m"] == 0.0


def _factorial_posterior() -> PhysicalPosterior:
    rows = list(itertools.product((-1.0, 1.0), repeat=3))
    state = np.zeros((len(rows), 5, 2, 3), dtype=float)
    readout = np.zeros_like(state)
    phi = []
    kappa = []
    particles = []
    for index, (theta_value, phi_value, kappa_value) in enumerate(rows):
        state_value = (
            0.001 * theta_value
            + 0.002 * phi_value
            + 0.003 * kappa_value
        )
        delta_value = 0.0005 * theta_value
        state[index] = state_value
        readout[index] = state_value + delta_value
        phi.append([phi_value])
        kappa.append([kappa_value])
        particles.append(0 if theta_value < 0 else 1)
    return PhysicalPosterior(
        context=_context(),
        component_ids=tuple(f"component_{index}" for index in range(len(rows))),
        state_trajectories_m=state,
        readout_trajectories_m=readout,
        readout_variance_m2=np.full((len(rows), 2, 3), 3e-6),
        weights=np.full(len(rows), 1.0 / len(rows)),
        phi=np.asarray(phi),
        kappa_cf=np.asarray(kappa),
        hypothesis_indices=np.arange(len(rows)),
        twin_particle_indices=np.asarray(particles),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id="1" * 64,
        source_factual_intervention_id="2" * 64,
        source_query_id="3" * 64,
    )


def test_variance_decomposition_has_exact_shapley_and_algebra_closure() -> None:
    posterior = _factorial_posterior()
    truth = np.zeros(posterior.readout_trajectories_m.shape[1:], dtype=float)
    result = variance_decomposition(
        posterior,
        truth,
        np.ones(truth.shape[:2], dtype=bool),
        HoldoutOracleProtocol(start_frame=1, stop_frame=5),
        variance_floor_m2=1e-6,
    )["all_holdout"]
    contributions = result["contributions"]
    assert np.isclose(contributions["theta_shapley"]["variance_m2"], 1e-6)
    assert np.isclose(contributions["phi_shapley"]["variance_m2"], 4e-6)
    assert np.isclose(contributions["kappa_shapley"]["variance_m2"], 9e-6)
    assert np.isclose(
        contributions["discrepancy_mean_epistemic"]["variance_m2"],
        0.25e-6,
    )
    assert np.isclose(
        contributions["state_discrepancy_cross"]["variance_m2"],
        1e-6,
    )
    assert np.isclose(
        contributions["discrepancy_conditional"]["variance_m2"],
        2e-6,
    )
    assert np.isclose(result["total_predictive_variance_m2"], 18.25e-6)
    assert max(result["closure"].values()) < 1e-15


def test_oracle_protocol_rejects_deployable_label_use() -> None:
    with np.testing.assert_raises_regex(ValueError, "diagnostic oracles"):
        HoldoutOracleProtocol(
            start_frame=2,
            stop_frame=4,
            label_use="training",
        )


def test_gap_report_identifies_the_dominant_subsystem() -> None:
    current = {
        "track_error_m": 0.030,
        "coordinate_rmse_m": 0.020,
    }

    def oracle(raw_track: float, raw_coordinate: float, ceiling_track: float):
        raw = {
            "metrics": {
                "track_error_m": raw_track,
                "coordinate_rmse_m": raw_coordinate,
            }
        }
        ceiling = {
            "metrics": {
                "track_error_m": ceiling_track,
                "coordinate_rmse_m": ceiling_track,
            }
        }
        return {
            "best": {
                "discrepancy_aware": raw,
                "per_node_constant_uncapped": ceiling,
                "per_node_constant_capped": ceiling,
            }
        }

    result = oracle_gap_report(
        current,
        oracle(0.028, 0.018, 0.010),
        oracle(0.027, 0.017, 0.005),
    )
    track = result["track_error_m"]
    assert track["dominant_gap"] == "model_gap"
    assert np.isclose(
        sum(track["fraction_of_total_diagnostic_headroom"].values()),
        1.0,
    )


def test_component_csv_uses_repository_native_lf(tmp_path: Path) -> None:
    output = tmp_path / "components.csv"
    _write_component_csv(
        output,
        [
            {
                "bank": "current",
                "component_id": "component",
                "hypothesis_id": "hypothesis",
                "particle_id": "particle",
                "hypothesis_index": 0,
                "particle_index": 0,
                "contact": {"shift": 0},
                "action": {"known": True},
            }
        ],
    )
    assert b"\r" not in output.read_bytes()
