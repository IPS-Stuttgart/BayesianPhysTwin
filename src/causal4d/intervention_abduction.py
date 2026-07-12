"""Abduce realized PhysTwin interventions from a causal O+ prefix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from causal4d.contracts import FactualIntervention, TwinBelief, array_sha256
from causal4d.rollout_bank import JointRolloutBank


@dataclass(frozen=True)
class FactualAbductionConfig:
    """Robust likelihood settings for factual intervention inference."""

    observation_scale_m: float = 0.01
    likelihood_power: float = 12.0
    dynamic_likelihood_weight: float = 0.25
    degrees_of_freedom: float = 4.0

    def __post_init__(self) -> None:
        if self.observation_scale_m <= 0.0 or self.likelihood_power <= 0.0:
            raise ValueError("observation scale and likelihood power must be positive")
        if self.dynamic_likelihood_weight < 0.0:
            raise ValueError("dynamic_likelihood_weight must be nonnegative")
        if self.degrees_of_freedom <= 0.0:
            raise ValueError("degrees_of_freedom must be positive")


def _belief_readout(
    bank: JointRolloutBank,
    belief: TwinBelief,
) -> tuple[np.ndarray, np.ndarray]:
    expected = (
        len(bank.parameter_weights),
        bank.node_count,
        bank.coordinate_count,
    )
    discrepancy = belief.discrepancy_mean_m[:, : bank.node_count]
    variance = belief.discrepancy_variance_m2[:, : bank.node_count]
    if discrepancy.shape != expected or variance.shape != expected:
        raise ValueError("TwinBelief discrepancy does not match the rollout bank")
    if not np.array_equal(belief.theta, bank.parameter_particles):
        raise ValueError("TwinBelief theta does not match the rollout bank")
    if not np.allclose(
        belief.weights,
        bank.parameter_weights,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("TwinBelief weights do not match the rollout bank")
    return discrepancy, variance


def physical_readout_components(
    bank: JointRolloutBank,
    belief: TwinBelief,
) -> np.ndarray:
    """Return state rollouts plus delta without modifying simulator trajectories."""

    discrepancy, _ = _belief_readout(bank, belief)
    return bank.trajectories.astype(float) + discrepancy[None, :, None]


def abduct_factual_intervention(
    bank: JointRolloutBank,
    belief: TwinBelief,
    observations_from_endpoint_m: np.ndarray,
    *,
    prefix_frame_count: int,
    observation_mask: np.ndarray | None = None,
    config: FactualAbductionConfig | None = None,
) -> FactualIntervention:
    """Infer persistent ``phi`` and factual event ``kappa_obs`` from O+ only."""

    settings = config or FactualAbductionConfig()
    if not 2 <= prefix_frame_count < bank.frame_count:
        raise ValueError("prefix_frame_count must reveal O+ and leave held-out frames")
    expected_stop = belief.context.o_plus.frame_start + prefix_frame_count - 1
    if expected_stop > belief.context.o_plus.frame_stop:
        raise ValueError("abduction prefix extends beyond O+")
    discrepancy, discrepancy_variance = _belief_readout(bank, belief)
    joint_weights = bank.update_from_observations(
        observations_from_endpoint_m,
        prefix_frame_count=prefix_frame_count,
        scale_m=settings.observation_scale_m,
        likelihood_power=settings.likelihood_power,
        dynamic_likelihood_weight=settings.dynamic_likelihood_weight,
        degrees_of_freedom=settings.degrees_of_freedom,
        mask=observation_mask,
        particle_discrepancy_m=discrepancy,
        particle_discrepancy_variance_m2=discrepancy_variance,
    )
    hand_count = len(bank.hypothesis_metadata[0]["contact"]["attachment_shifts"])
    phi_names = ("gain_multiplier", "delay_steps", "rotation_degrees")
    kappa_names = tuple(
        f"attachment_shift_hand_{index}" for index in range(hand_count)
    ) + ("slip_fraction",)
    component_ids = []
    phi = []
    kappa = []
    hypothesis_indices = []
    particle_indices = []
    for hypothesis_index, (hypothesis_id, metadata) in enumerate(
        zip(bank.hypothesis_ids, bank.hypothesis_metadata, strict=True)
    ):
        action = metadata["action"]
        if not bool(action["future_action_observed"]):
            raise ValueError("factual abduction requires the observed u_obs action")
        contact = metadata["contact"]
        persistent = (
            float(contact["gain_multiplier"]),
            float(contact["delay_steps"]),
            float(contact["rotation_degrees"]),
        )
        event = tuple(map(float, contact["attachment_shifts"])) + (
            float(contact["slip_fraction"]),
        )
        for particle_index, particle_id in enumerate(belief.particle_ids):
            component_ids.append(f"{hypothesis_id}::{particle_id}")
            phi.append(persistent)
            kappa.append(event)
            hypothesis_indices.append(hypothesis_index)
            particle_indices.append(particle_index)
    return FactualIntervention(
        context=belief.context,
        component_ids=tuple(component_ids),
        phi_names=phi_names,
        kappa_names=kappa_names,
        phi=np.asarray(phi, dtype=float),
        kappa_obs=np.asarray(kappa, dtype=float),
        hypothesis_indices=np.asarray(hypothesis_indices, dtype=np.int64),
        twin_particle_indices=np.asarray(particle_indices, dtype=np.int64),
        weights=joint_weights.reshape(-1),
        evidence_frame_stop=expected_stop,
        source_twin_belief_id=belief.artifact_id,
        metadata={
            "abduction_likelihood": asdict(settings),
            "observation_prefix_frame_count_including_endpoint": prefix_frame_count,
            "o_plus_frames_used": prefix_frame_count - 1,
            "future_frames_read_by_abduction": 0,
            "rollout_bank_trajectories_sha256": array_sha256(bank.trajectories),
            "discrepancy_scored_as_separate_readout": True,
            "discrepancy_injected_into_simulator_state": False,
        },
    )


def factual_joint_weights(
    factual: FactualIntervention,
    *,
    hypothesis_count: int,
    particle_count: int,
) -> np.ndarray:
    """Restore the rollout-bank matrix represented by a factual posterior."""

    if np.any(factual.hypothesis_indices >= hypothesis_count) or np.any(
        factual.twin_particle_indices >= particle_count
    ):
        raise ValueError("factual support exceeds the requested rollout bank")
    result = np.zeros((hypothesis_count, particle_count), dtype=float)
    np.add.at(
        result,
        (factual.hypothesis_indices, factual.twin_particle_indices),
        factual.weights,
    )
    if not np.isclose(np.sum(result), 1.0):
        raise RuntimeError("factual posterior lost probability mass")
    return result


def nominal_contact_hypotheses(bank: JointRolloutBank) -> np.ndarray:
    """Identify no-shift, unit-gain, no-delay, no-slip, no-rotation controls."""

    selected = []
    for index, metadata in enumerate(bank.hypothesis_metadata):
        contact = metadata["contact"]
        if (
            all(int(value) == 0 for value in contact["attachment_shifts"])
            and float(contact["gain_multiplier"]) == 1.0
            and int(contact["delay_steps"]) == 0
            and float(contact["slip_fraction"]) == 0.0
            and float(contact["rotation_degrees"]) == 0.0
        ):
            selected.append(index)
    if not selected:
        raise ValueError("rollout bank contains no nominal-contact hypothesis")
    return np.asarray(selected, dtype=np.int64)


def _prediction_metrics(
    prediction: np.ndarray,
    observations: np.ndarray,
    mask: np.ndarray,
    *,
    start_frame: int,
) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=float)[start_frame:]
    target = np.asarray(observations, dtype=float)[start_frame:]
    valid = np.asarray(mask, dtype=bool)[start_frame:] & np.all(
        np.isfinite(target), axis=2
    )
    if not np.any(valid):
        raise ValueError("held-out evaluation contains no valid points")
    residual = predicted - target
    vectors = residual[valid]
    return {
        "coordinate_rmse_m": float(np.sqrt(np.mean(np.square(vectors)))),
        "track_error_m": float(np.mean(np.linalg.norm(vectors, axis=1))),
        "valid_point_frames": int(np.sum(valid)),
    }


def evaluate_factual_abduction(
    bank: JointRolloutBank,
    belief: TwinBelief,
    factual: FactualIntervention,
    observations_from_endpoint_m: np.ndarray,
    *,
    observation_mask: np.ndarray,
    prefix_frame_count: int,
    config: FactualAbductionConfig | None = None,
) -> dict[str, Any]:
    """Compare BPT+z with a same-evidence BPT posterior fixed to nominal z."""

    settings = config or FactualAbductionConfig()
    discrepancy, discrepancy_variance = _belief_readout(bank, belief)
    z_weights = factual_joint_weights(
        factual,
        hypothesis_count=len(bank.hypothesis_ids),
        particle_count=len(bank.parameter_weights),
    )
    nominal = nominal_contact_hypotheses(bank)
    nominal_base = np.zeros_like(bank.prior_joint_weights)
    action_mass = bank.hypothesis_prior_weights[nominal]
    action_mass = action_mass / np.sum(action_mass)
    nominal_base[nominal] = action_mass[:, None] * bank.parameter_weights[None]
    nominal_weights = bank.update_from_observations(
        observations_from_endpoint_m,
        prefix_frame_count=prefix_frame_count,
        scale_m=settings.observation_scale_m,
        likelihood_power=settings.likelihood_power,
        dynamic_likelihood_weight=settings.dynamic_likelihood_weight,
        degrees_of_freedom=settings.degrees_of_freedom,
        mask=observation_mask,
        base_weights=nominal_base,
        particle_discrepancy_m=discrepancy,
        particle_discrepancy_variance_m2=discrepancy_variance,
    )
    components = physical_readout_components(bank, belief)
    z_prediction = np.einsum("hp,hptnc->tnc", z_weights, components)
    nominal_prediction = np.einsum("hp,hptnc->tnc", nominal_weights, components)
    z_metrics = _prediction_metrics(
        z_prediction,
        observations_from_endpoint_m,
        observation_mask,
        start_frame=prefix_frame_count,
    )
    nominal_metrics = _prediction_metrics(
        nominal_prediction,
        observations_from_endpoint_m,
        observation_mask,
        start_frame=prefix_frame_count,
    )
    improvement = 1.0 - z_metrics["track_error_m"] / nominal_metrics["track_error_m"]
    hypothesis_marginal = np.sum(z_weights, axis=1)
    return {
        "abduction_prefix_frame_count_including_endpoint": prefix_frame_count,
        "held_out_rollout_interval": [prefix_frame_count, bank.frame_count],
        "bpt_without_z": nominal_metrics,
        "bpt_plus_causal4d_z": z_metrics,
        "relative_track_error_improvement": float(improvement),
        "map_hypothesis_id": bank.hypothesis_ids[int(np.argmax(hypothesis_marginal))],
        "map_hypothesis_probability": float(np.max(hypothesis_marginal)),
        "nominal_hypothesis_probability": float(np.sum(hypothesis_marginal[nominal])),
        "parameter_marginal": np.sum(z_weights, axis=0).tolist(),
    }
