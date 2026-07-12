"""Abduction-action-prediction operator for Causal4D PhysTwin rollouts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from causal4d.contracts import (
    CausalContext,
    CounterfactualQuery,
    FactualIntervention,
    PhysicalPosterior,
    TwinBelief,
)
from causal4d.intervention_abduction import physical_readout_components
from causal4d.rollout_bank import JointRolloutBank


def _phi_from_metadata(metadata: Mapping[str, Any]) -> tuple[float, ...]:
    contact = metadata["contact"]
    return (
        float(contact["gain_multiplier"]),
        float(contact["delay_steps"]),
        float(contact["rotation_degrees"]),
    )


def _kappa_from_metadata(metadata: Mapping[str, Any]) -> tuple[float, ...]:
    contact = metadata["contact"]
    return tuple(map(float, contact["attachment_shifts"])) + (
        float(contact["slip_fraction"]),
    )


def _validate_factual_context(
    belief: TwinBelief,
    factual: FactualIntervention,
    query: CounterfactualQuery,
) -> None:
    if factual.source_twin_belief_id != belief.artifact_id:
        raise ValueError("factual intervention does not descend from TwinBelief")
    if query.source_factual_intervention_id != factual.artifact_id:
        raise ValueError("counterfactual query does not descend from factual abduction")
    for name in ("protocol_id", "o_minus", "o_plus", "u_obs"):
        expected = getattr(belief.context, name)
        if getattr(factual.context, name) != expected or getattr(query.context, name) != expected:
            raise ValueError(f"counterfactual artifacts disagree on factual {name}")


def _validate_query_bank(
    bank: JointRolloutBank,
    query: CounterfactualQuery,
    manifest: Mapping[str, Any],
) -> None:
    if "causal_context" not in manifest:
        raise ValueError("rollout manifest has no causal context")
    bank_context = CausalContext.from_dict(manifest["causal_context"])
    if bank_context != query.context:
        raise ValueError("rollout bank context does not match do(u_cf)")
    action_ids = {
        str(metadata["action"]["proposal_id"])
        for metadata in bank.hypothesis_metadata
    }
    if action_ids != {query.context.u_cf.action_id}:
        raise ValueError("rollout bank does not contain exactly the queried action")


def _new_contact_weights(
    bank: JointRolloutBank,
    factual: FactualIntervention,
) -> tuple[np.ndarray, float]:
    """Carry ``p(theta, phi)`` and sample a fresh event ``kappa_cf``."""

    phi_theta: defaultdict[tuple[int, tuple[float, ...]], float] = defaultdict(float)
    for index, weight in enumerate(factual.weights):
        key = (
            int(factual.twin_particle_indices[index]),
            tuple(map(float, factual.phi[index])),
        )
        phi_theta[key] += float(weight)

    conditional_denominator: defaultdict[tuple[float, ...], float] = defaultdict(float)
    query_phi = []
    for hypothesis_index, metadata in enumerate(bank.hypothesis_metadata):
        phi = _phi_from_metadata(metadata)
        query_phi.append(phi)
        conditional_denominator[phi] += float(
            bank.hypothesis_prior_weights[hypothesis_index]
        )
    weights = np.zeros_like(bank.prior_joint_weights)
    for hypothesis_index, phi in enumerate(query_phi):
        denominator = conditional_denominator[phi]
        conditional_kappa = (
            float(bank.hypothesis_prior_weights[hypothesis_index]) / denominator
        )
        for particle_index in range(len(bank.parameter_weights)):
            weights[hypothesis_index, particle_index] = (
                phi_theta[(particle_index, phi)] * conditional_kappa
            )
    retained_mass = float(np.sum(weights))
    if retained_mass <= 0.0:
        raise ValueError("query contact beam has no support for factual phi posterior")
    return weights / retained_mass, retained_mass


def _same_grasp_weights(
    bank: JointRolloutBank,
    factual: FactualIntervention,
) -> tuple[np.ndarray, float]:
    """Carry the complete factual ``(theta, phi, kappa_obs)`` joint posterior."""

    query_lookup: defaultdict[
        tuple[tuple[float, ...], tuple[float, ...]], list[int]
    ] = defaultdict(list)
    for hypothesis_index, metadata in enumerate(bank.hypothesis_metadata):
        query_lookup[
            (_phi_from_metadata(metadata), _kappa_from_metadata(metadata))
        ].append(hypothesis_index)
    weights = np.zeros_like(bank.prior_joint_weights)
    for component_index, weight in enumerate(factual.weights):
        key = (
            tuple(map(float, factual.phi[component_index])),
            tuple(map(float, factual.kappa_obs[component_index])),
        )
        matches = query_lookup.get(key, [])
        if not matches:
            continue
        share = float(weight) / len(matches)
        particle = int(factual.twin_particle_indices[component_index])
        for hypothesis_index in matches:
            weights[hypothesis_index, particle] += share
    retained_mass = float(np.sum(weights))
    if retained_mass <= 0.0:
        raise ValueError("query contact beam cannot represent the factual grasp")
    return weights / retained_mass, retained_mass


def apply_counterfactual_operator(
    bank: JointRolloutBank,
    manifest: Mapping[str, Any],
    belief: TwinBelief,
    factual: FactualIntervention,
    query: CounterfactualQuery,
) -> PhysicalPosterior:
    """Apply ``do(u_cf)`` while transferring phi and handling kappa explicitly."""

    _validate_factual_context(belief, factual, query)
    _validate_query_bank(bank, query, manifest)
    if not np.array_equal(bank.parameter_particles, belief.theta):
        raise ValueError("counterfactual bank theta differs from TwinBelief")
    expected_phi_names = (
        "gain_multiplier",
        "delay_steps",
        "rotation_degrees",
    )
    hand_count = len(bank.hypothesis_metadata[0]["contact"]["attachment_shifts"])
    expected_kappa_names = tuple(
        f"attachment_shift_hand_{index}" for index in range(hand_count)
    ) + ("slip_fraction",)
    if factual.phi_names != expected_phi_names or factual.kappa_names != expected_kappa_names:
        raise ValueError("factual intervention variable schema differs from query bank")

    if query.contact_policy == "new_contact":
        joint_weights, retained_mass = _new_contact_weights(bank, factual)
        reused_factual_kappa = False
    else:
        joint_weights, retained_mass = _same_grasp_weights(bank, factual)
        reused_factual_kappa = True

    state = bank.trajectories.reshape(
        -1,
        bank.frame_count,
        bank.node_count,
        bank.coordinate_count,
    )
    readout = physical_readout_components(bank, belief).reshape(state.shape)
    hypothesis_indices = np.repeat(
        np.arange(len(bank.hypothesis_ids), dtype=np.int64),
        len(bank.parameter_weights),
    )
    particle_indices = np.tile(
        np.arange(len(bank.parameter_weights), dtype=np.int64),
        len(bank.hypothesis_ids),
    )
    phi_by_hypothesis = np.asarray(
        [_phi_from_metadata(value) for value in bank.hypothesis_metadata],
        dtype=float,
    )
    kappa_by_hypothesis = np.asarray(
        [_kappa_from_metadata(value) for value in bank.hypothesis_metadata],
        dtype=float,
    )
    phi = phi_by_hypothesis[hypothesis_indices]
    kappa = kappa_by_hypothesis[hypothesis_indices]
    discrepancy_variance = (
        belief.discrepancy_variance_m2[
            particle_indices,
            : bank.node_count,
        ]
        + bank.variance_floor_m2
    )
    component_ids = tuple(
        f"{bank.hypothesis_ids[hypothesis]}::{belief.particle_ids[particle]}"
        for hypothesis, particle in zip(
            hypothesis_indices,
            particle_indices,
            strict=True,
        )
    )
    return PhysicalPosterior(
        context=query.context,
        component_ids=component_ids,
        state_trajectories_m=state,
        readout_trajectories_m=readout,
        readout_variance_m2=discrepancy_variance,
        weights=joint_weights.reshape(-1),
        phi=phi,
        kappa_cf=kappa,
        hypothesis_indices=hypothesis_indices,
        twin_particle_indices=particle_indices,
        phi_names=expected_phi_names,
        kappa_names=expected_kappa_names,
        source_twin_belief_id=belief.artifact_id,
        source_factual_intervention_id=factual.artifact_id,
        source_query_id=query.artifact_id,
        metadata={
            "operator": "abduction-action-prediction",
            "intervention": f"do({query.context.u_cf.action_id})",
            "contact_policy": query.contact_policy,
            "persistent_phi_transferred": True,
            "factual_kappa_reused": reused_factual_kappa,
            "fresh_kappa_cf_sampled": not reused_factual_kappa,
            "represented_factual_mass_before_renormalization": retained_mass,
            "rollout_includes_pre_intervention_endpoint": True,
            "discrepancy_injected_into_simulator_state": False,
            "discrepancy_applied_to_readout": True,
        },
    )


def physical_posterior_mean(
    posterior: PhysicalPosterior,
    *,
    readout: bool = True,
) -> np.ndarray:
    """Return the weighted state or discrepancy-aware readout trajectory."""

    values = (
        posterior.readout_trajectories_m
        if readout
        else posterior.state_trajectories_m
    )
    return np.einsum("k,ktnc->tnc", posterior.weights, values)
