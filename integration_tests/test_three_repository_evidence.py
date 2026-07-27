"""Evidence-bound causal integration supplement for the installed-wheel gate."""

from __future__ import annotations

import json
import os
import platform
from copy import deepcopy
from dataclasses import replace
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pytest
from causal4d.bpt_belief import (
    BPTBeliefExportConfig,
    build_twin_belief_from_replays,
)
from causal4d.contracts import (
    CounterfactualQuery,
    FactualIntervention,
    TwinBelief,
    build_causal_context,
    load_contract,
    save_contract,
)
from causal4d.counterfactual import apply_counterfactual_operator
from causal4d.observation_lineage import (
    bind_twin_belief_observation_lineage,
    load_observation_lineage,
    validate_twin_belief_observation_lineage,
)
from causal4d.phystwin_backend import BayesianPhysTwinParticles
from causal4d.provider_contract import require_bayesian_phystwin_provider
from causal4d.rollout_bank import JointRolloutBank
from prob4d.provider_v1 import save_observation_belief_export
from test_three_repository_golden_path import (
    DeterministicReplayProvider,
    _producer_artifact,
    _profile_particles,
    _run_bpt_update,
)

from bayesian_phystwin import (
    load_observation_belief,
    validate_prob4d_causal_observation_belief,
)
from bayesian_phystwin.evidence_policy import require_promotable_run_manifest
from bayesian_phystwin.repository_provenance import RepositoryState
from bayesian_phystwin.run_manifest import artifact_digest
from bayesian_phystwin.run_manifest_v2 import (
    RunManifestV2,
    load_run_manifest_v2,
    write_run_manifest,
)

CAUSAL_FRAME_STOP = 6
INTERVENTION_FRAME = CAUSAL_FRAME_STOP
FULL_FRAME_COUNT = 9


def _revision(name: str) -> str:
    value = os.environ[name]
    assert len(value) == 40
    assert all(character in "0123456789abcdef" for character in value)
    return value


def _wheel_digest(name: str) -> str:
    value = os.environ[name]
    assert len(value) == 64
    assert all(character in "0123456789abcdef" for character in value)
    return value


def _exact_producer_artifact():
    return replace(
        _producer_artifact(),
        source_revision=_revision("PROB4D_REVISION"),
    )


def _bound_counterfactual_artifacts(
    tmp_path: Path,
    lineage,
    bpt_result,
):
    provider = DeterministicReplayProvider(FULL_FRAME_COUNT)
    profile_path = tmp_path / "profile-bound.npz"
    particles, mass_accounting = _profile_particles(profile_path)

    replay_positions = []
    replay_velocities = []
    for particle in particles.log_scales:
        provider.set_group_log_scales(particle)
        positions, velocities = provider.replay_initial(
            frame_count=INTERVENTION_FRAME
        )
        replay_positions.append(positions)
        replay_velocities.append(velocities)
    positions = np.stack(replay_positions)
    velocities = np.stack(replay_velocities)

    observations = np.zeros((FULL_FRAME_COUNT, 1, 3), dtype=np.float64)
    observations[:INTERVENTION_FRAME] = positions[0]
    factual_actions = np.zeros_like(observations)
    context = build_causal_context(
        protocol_id="three-repository-installed-wheel-v1",
        case_id="three-repository-golden-path",
        observations=observations,
        observed_actions=factual_actions,
        counterfactual_actions=factual_actions,
        intervention_frame=INTERVENTION_FRAME,
        counterfactual_action_id="factual-action",
    )

    provider_manifest = require_bayesian_phystwin_provider()
    assert provider_manifest.provider_revision == _revision(
        "BAYESIAN_PHYSTWIN_REVISION"
    )
    unbound = build_twin_belief_from_replays(
        context=context,
        replay_positions_m=positions,
        replay_velocities_mps=velocities,
        observed_positions_m=observations,
        observed_valid=np.ones(observations.shape[:2], dtype=bool),
        theta=particles.log_scales,
        theta_names=(
            "object_spring_log_scale",
            "controller_spring_log_scale",
        ),
        weights=particles.weights,
        particle_ids=tuple(
            f"grid-{first}-{second}"
            for first, second in particles.grid_indices
        ),
        metadata={
            "provider_manifest_id": provider_manifest.manifest_id,
            "probability_mass_accounting": mass_accounting,
            "gauge_update": {
                "inference_admissible": bool(bpt_result.inference_admissible),
                "reason": bpt_result.reason,
                "state_coefficients": bpt_result.state_coefficients.tolist(),
            },
        },
        config=BPTBeliefExportConfig(interpolation_neighbors=1),
    )
    twin_belief = bind_twin_belief_observation_lineage(unbound, lineage)
    validation = validate_twin_belief_observation_lineage(twin_belief, lineage)
    assert validation["lineage_bound"] is True
    assert validation["observation_causal_frame_stop"] == INTERVENTION_FRAME

    twin_path = tmp_path / "lineage-bound-twin-belief.npz"
    save_contract(twin_path, twin_belief)
    loaded_twin = load_contract(twin_path)
    assert isinstance(loaded_twin, TwinBelief)
    assert loaded_twin.artifact_id == twin_belief.artifact_id

    hypothesis_phi = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.15, 0.0, 0.0]]
    )
    hypothesis_kappa = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    )
    particle_count = len(particles.weights)
    factual = FactualIntervention(
        context=context,
        component_ids=tuple(
            f"factual-{hypothesis}-{particle}"
            for hypothesis in range(3)
            for particle in range(particle_count)
        ),
        phi_names=(
            "gain_multiplier",
            "delay_steps",
            "rotation_degrees",
        ),
        kappa_names=("attachment_shift_hand_0", "slip_fraction"),
        phi=np.repeat(hypothesis_phi, particle_count, axis=0),
        kappa_obs=np.repeat(hypothesis_kappa, particle_count, axis=0),
        hypothesis_indices=np.repeat(
            np.arange(3, dtype=np.int64),
            particle_count,
        ),
        twin_particle_indices=np.tile(
            np.arange(particle_count, dtype=np.int64),
            3,
        ),
        weights=np.outer(
            np.asarray([0.1, 0.8, 0.1]),
            particles.weights,
        ).reshape(-1),
        evidence_frame_stop=INTERVENTION_FRAME + 1,
        source_twin_belief_id=twin_belief.artifact_id,
    )

    counterfactual_actions = factual_actions.copy()
    counterfactual_actions[INTERVENTION_FRAME:, 0, 0] = np.asarray(
        [0.01, 0.02, 0.03]
    )
    query_context = build_causal_context(
        protocol_id="three-repository-installed-wheel-v1",
        case_id="three-repository-golden-path",
        observations=observations,
        observed_actions=factual_actions,
        counterfactual_actions=counterfactual_actions,
        intervention_frame=INTERVENTION_FRAME,
        counterfactual_action_id="new-action",
    )
    query = CounterfactualQuery(
        context=query_context,
        controller_points_m=counterfactual_actions[INTERVENTION_FRAME:],
        horizon_frames=FULL_FRAME_COUNT - INTERVENTION_FRAME,
        contact_policy="new_contact",
        source_factual_intervention_id=factual.artifact_id,
    )

    hypotheses = (
        ("nominal", 1.0, 0),
        ("shift", 1.0, 1),
        ("gain", 1.15, 0),
    )
    trajectories = np.empty(
        (
            len(hypotheses),
            particle_count,
            FULL_FRAME_COUNT - INTERVENTION_FRAME + 1,
            1,
            3,
        ),
        dtype=np.float32,
    )
    hypothesis_metadata = []
    for hypothesis_index, (identifier, gain, shift) in enumerate(hypotheses):
        controls = counterfactual_actions.copy()
        controls[INTERVENTION_FRAME:, 0, 0] *= gain
        controls[INTERVENTION_FRAME:, 0, 1] += 0.001 * shift
        provider.set_controller_points(controls)
        hypothesis_metadata.append(
            {
                "hypothesis_id": identifier,
                "action": {
                    "proposal_id": "new-action",
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
        )
        for particle_index, particle in enumerate(particles.log_scales):
            provider.set_group_log_scales(particle)
            trajectories[hypothesis_index, particle_index] = (
                provider.replay_restart(
                    twin_belief.endpoint_position_m[particle_index],
                    twin_belief.endpoint_velocity_mps[particle_index],
                    start_frame=INTERVENTION_FRAME - 1,
                    stop_frame=FULL_FRAME_COUNT - 1,
                )
            )

    bank = JointRolloutBank(
        hypothesis_ids=tuple(value[0] for value in hypotheses),
        hypothesis_metadata=tuple(hypothesis_metadata),
        hypothesis_prior_weights=np.asarray([0.5, 0.25, 0.25]),
        parameter_particles=particles.log_scales,
        parameter_weights=particles.weights,
        trajectories=trajectories,
        variance_floor_m2=2e-6,
    )
    context_metadata = {
        "causal_context": query_context.as_dict(),
        "twin_belief_id": twin_belief.artifact_id,
        "provider_manifest_id": provider_manifest.manifest_id,
    }
    first = apply_counterfactual_operator(
        bank,
        context_metadata,
        twin_belief,
        factual,
        query,
    )
    second = apply_counterfactual_operator(
        bank,
        context_metadata,
        twin_belief,
        factual,
        query,
    )
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(
        first.state_trajectories_m,
        second.state_trajectories_m,
    )
    np.testing.assert_array_equal(
        first.readout_trajectories_m,
        second.readout_trajectories_m,
    )
    np.testing.assert_array_equal(
        first.readout_variance_m2,
        second.readout_variance_m2,
    )
    assert first.artifact_id == second.artifact_id
    assert first.source_twin_belief_id == twin_belief.artifact_id
    assert np.isclose(np.sum(first.weights), 1.0)

    selected_scales = {
        tuple(map(float, values)) for values in particles.log_scales
    }
    assert set(provider.scale_history) == selected_scales
    assert (0.2, 0.1) not in set(provider.scale_history)
    provider.close()

    posterior_path = tmp_path / "lineage-bound-physical-posterior.npz"
    save_contract(posterior_path, first)
    assert load_contract(posterior_path).artifact_id == first.artifact_id
    return (
        twin_belief,
        first,
        profile_path,
        twin_path,
        posterior_path,
        provider_manifest,
    )


def _manifest(
    tmp_path: Path,
    observation_path: Path,
    profile_path: Path,
    twin_path: Path,
    posterior_path: Path,
    *,
    observation_id: str,
    twin_id: str,
    posterior_id: str,
    provider_manifest_id: str,
) -> RunManifestV2:
    manifest = RunManifestV2(
        run_id="three-repository-installed-wheel-golden-path",
        repository="FlorianPfaff/Bayesian-PhysTwin",
        revision=_revision("BAYESIAN_PHYSTWIN_REVISION"),
        dirty=False,
        related_repositories=(
            RepositoryState(
                repository="FlorianPfaff/Prob4D",
                revision=_revision("PROB4D_REVISION"),
                dirty=False,
                role="observation",
            ),
            RepositoryState(
                repository="FlorianPfaff/Causal4D",
                revision=_revision("CAUSAL4D_REVISION"),
                dirty=False,
                role="downstream",
            ),
        ),
        command=(
            "python",
            "-I",
            "-m",
            "pytest",
            "-q",
            "test_three_repository_evidence.py",
        ),
        classification="infrastructure",
        statistical_unit="deterministic three-repository fixture",
        information_boundary={
            "causal_frame_stop_exclusive": CAUSAL_FRAME_STOP,
            "counterfactual_intervention_frame": INTERVENTION_FRAME,
            "future_prediction_payloads_opened": 0,
        },
        configuration={
            "observation_artifact_id": observation_id,
            "twin_belief_id": twin_id,
            "physical_posterior_id": posterior_id,
            "provider_manifest_id": provider_manifest_id,
            "prob4d_causal_stream_contract_version": 2,
        },
        inputs=(
            artifact_digest(
                observation_path,
                name="prob4d-observation",
                role="input",
                root=tmp_path,
            ),
            artifact_digest(
                profile_path,
                name="bpt-parameter-support-profile",
                role="input",
                root=tmp_path,
            ),
        ),
        outputs=(
            artifact_digest(
                twin_path,
                name="twin-belief",
                role="output",
                root=tmp_path,
            ),
            artifact_digest(
                posterior_path,
                name="physical-posterior",
                role="output",
                root=tmp_path,
            ),
        ),
        package_versions={
            name: importlib_metadata.version(name)
            for name in ("bayesian-phystwin", "causal4d", "prob4d")
        },
        runtime_environment={
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "wheel_sha256": {
                "bayesian-phystwin": _wheel_digest(
                    "BAYESIAN_PHYSTWIN_WHEEL_SHA256"
                ),
                "causal4d": _wheel_digest("CAUSAL4D_WHEEL_SHA256"),
                "prob4d": _wheel_digest("PROB4D_WHEEL_SHA256"),
            },
        },
        claim_ids=("bpt.infrastructure.three_repository_golden_path",),
        method_freeze_id="three-repository-installed-wheel-v1",
        protocol_id="three-repository-installed-wheel-v1",
        split_id="deterministic-fixture-v1",
        baseline_id="exact-zero-update-fallback-v1",
    )
    path = tmp_path / "run-manifest-v2.json"
    write_run_manifest(path, manifest)
    loaded = load_run_manifest_v2(path)
    assert loaded == manifest
    return loaded


def test_bound_counterfactual_and_promotable_evidence(tmp_path: Path) -> None:
    producer = _exact_producer_artifact()
    duplicate = _exact_producer_artifact()
    assert producer.artifact_id == duplicate.artifact_id

    observation_path = tmp_path / "exact-prob4d-observation.npz"
    save_observation_belief_export(observation_path, producer)
    observation = load_observation_belief(observation_path)
    bpt_validation = validate_prob4d_causal_observation_belief(observation)
    assert bpt_validation["stream_contract_version"] == 2

    lineage = load_observation_lineage(observation_path)
    assert lineage.source_revision == _revision("PROB4D_REVISION")
    assert lineage.artifact_id == observation.artifact_id
    bpt_result = _run_bpt_update(observation)

    (
        twin,
        posterior,
        profile_path,
        twin_path,
        posterior_path,
        provider_manifest,
    ) = _bound_counterfactual_artifacts(tmp_path, lineage, bpt_result)
    manifest = _manifest(
        tmp_path,
        observation_path,
        profile_path,
        twin_path,
        posterior_path,
        observation_id=observation.artifact_id,
        twin_id=twin.artifact_id,
        posterior_id=posterior.artifact_id,
        provider_manifest_id=provider_manifest.manifest_id,
    )
    admission = require_promotable_run_manifest(manifest, root=tmp_path)
    assert admission["status"] == "promotable"
    assert admission["evidence_fingerprint"] == manifest.evidence_fingerprint
    assert admission["input_artifact_count"] == 2

    with pytest.raises(ValueError, match="clean repositories"):
        require_promotable_run_manifest(replace(manifest, dirty=True))
    with pytest.raises(ValueError, match="at least one claim"):
        require_promotable_run_manifest(replace(manifest, claim_ids=()))

    incomplete = manifest.as_dict()
    incomplete.pop("baseline_id")
    incomplete_path = tmp_path / "incomplete-run-manifest.json"
    incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match schema"):
        load_run_manifest_v2(incomplete_path)

    posterior_path.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="artifact (size|digest) mismatch"):
        require_promotable_run_manifest(manifest, root=tmp_path)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("future_payload", "opening future payloads"),
        ("fixed_lag_strict_v2", "fixed-lag|approximate"),
        ("anchor_source_digest", "metric anchor is not bound"),
        ("duplicated_gauge_semantics", "factor definition changed"),
    ],
)
def test_both_consumers_reject_extended_semantic_drift(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    producer = _exact_producer_artifact()
    metadata = deepcopy(dict(producer.metadata))
    if case == "future_payload":
        metadata["causal_source_lineage"][
            "future_prediction_payloads_opened"
        ] = 1
    elif case == "fixed_lag_strict_v2":
        gauge_posterior = metadata["gauge_posterior"]
        gauge_posterior["model"] = "fixed_lag_block_diagonal_approximation_v1"
        gauge_posterior["cross_window_covariance_preserved"] = False
        gauge_posterior["fixed_lag_boundary_covariance_is_approximate"] = True
        metadata["joint_cross_window_gauge_covariance_represented"] = False
    elif case == "anchor_source_digest":
        metadata["metric_gauge_anchor"]["source_artifact_sha256"] = "e" * 64
    elif case == "duplicated_gauge_semantics":
        metadata["factor_definition"] = (
            "gauge covariance duplicated in local covariance and explicit factors"
        )
    else:  # pragma: no cover
        raise AssertionError(case)

    drifted = replace(producer, metadata=metadata)
    path = tmp_path / f"{case}.npz"
    save_observation_belief_export(path, drifted)
    with pytest.raises(ValueError, match=message):
        validate_prob4d_causal_observation_belief(
            load_observation_belief(path)
        )
    with pytest.raises(ValueError, match=message):
        load_observation_lineage(path)


def test_causal4d_rejects_incorrect_composed_mass() -> None:
    with pytest.raises(ValueError, match="composed retained mass"):
        BayesianPhysTwinParticles(
            log_scales=np.asarray([[0.0, 0.0]]),
            weights=np.asarray([1.0]),
            grid_indices=np.asarray([[0, 0]]),
            source_weight_key="prediction_weights",
            retained_probability_mass=0.70,
            bpt_retained_probability_mass=0.90,
            causal4d_retained_probability_mass=0.80,
        )
