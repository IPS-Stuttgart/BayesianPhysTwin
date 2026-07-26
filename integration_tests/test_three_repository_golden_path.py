"""Installed-wheel golden path across Prob4D, Bayesian-PhysTwin, and Causal4D.

The runner copies this file outside all three source trees before executing it.
Producer and consumer validation remain independent; only immutable artifacts and
expected decisions cross repository boundaries.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from importlib import import_module, metadata as importlib_metadata
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from prob4d.provider_v1 import (
    MetricGaugeAnchor,
    ObservationBeliefExportV1,
    bind_causal_stream_contract_v2,
    save_observation_belief_export,
)
from prob4d.sim3 import Sim3

from bayesian_phystwin import (
    GaugeAwareBeliefConfig,
    build_gauge_aware_batch_from_observation_belief,
    load_observation_belief,
    update_gauge_aware_belief,
    validate_prob4d_causal_observation_belief,
)
from bayesian_phystwin.causal4d_provider_v1 import PhysTwinReplayProvider

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
from causal4d.phystwin_backend import load_bayesian_phystwin_particles
from causal4d.provider_contract import require_bayesian_phystwin_provider
from causal4d.rollout_bank import JointRolloutBank


EXPECTED_OBSERVATION_ARTIFACT_ID = (
    "142e5fd52a5d7d99247f6bcf89b1521cec233f9d280f28cdf1060e004343522c"
)


def _producer_metadata(anchor: MetricGaugeAnchor) -> dict[str, Any]:
    return {
        "association_probability_definition": (
            "same decoded pixel identity within one independently decoded "
            "window; not downstream physical-node association"
        ),
        "causal_source_lineage": {
            "admissibility_rule": (
                "source_frame_max < causal_frame_stop_exclusive"
            ),
            "causal_frame_stop_exclusive": 6,
            "future_prediction_payloads_opened": 0,
            "motioncrafter_lineage_schema_version": 1,
            "motioncrafter_windowing_model": (
                "motioncrafter_sliding_window_v1"
            ),
            "producer": "Prob4D",
            "schema_version": 1,
            "selected_windows": [
                {
                    "frame_indices_sha256": "2" * 64,
                    "payload_sha256": "1" * 64,
                    "source_frame_max": 2,
                    "source_frame_start": 0,
                    "source_frame_stop_exclusive": 3,
                    "window_id": "window-0",
                },
                {
                    "frame_indices_sha256": "3" * 64,
                    "payload_sha256": "4" * 64,
                    "source_frame_max": 4,
                    "source_frame_start": 2,
                    "source_frame_stop_exclusive": 5,
                    "window_id": "window-1",
                },
                {
                    "frame_indices_sha256": "5" * 64,
                    "payload_sha256": "6" * 64,
                    "source_frame_max": 5,
                    "source_frame_start": 3,
                    "source_frame_stop_exclusive": 6,
                    "window_id": "window-2",
                },
            ],
            "source_artifact_sha256": "c" * 64,
            "source_product": "independently_decoded_overlap_windows",
        },
        "coordinate_frame": "phystwin-world",
        "effective_samples_per_group": 64.0,
        "factor_definition": "one shared joint gauge latent vector",
        "factor_group_semantics": (
            "all rows use one factor group; each window contributes its "
            "block of the same joint gauge covariance root"
        ),
        "fixed_lag": None,
        "gauge_mode": "sequential",
        "gauge_posterior": {
            "cross_window_covariance_preserved": True,
            "exported_factor_rank": 5,
            "fixed_lag_boundary_covariance_is_approximate": False,
            "full_dimension": 21,
            "max_gauge_rank": 64,
            "minimum_retained_gauge_trace": 0.999,
            "model": "sequential_joint_spanning_tree_v1",
            "parent_window_ids": [None, "window-0", "window-1"],
            "retained_covariance_trace_fraction": 1.0,
            "window_count": 3,
        },
        "group_definition": "absolute source frame across overlap windows",
        "group_prior_nominal_probability_definition": (
            "neutral one; no independently calibrated group nominal prior "
            "supplied"
        ),
        "joint_cross_window_gauge_covariance_represented": True,
        "metric_coordinates": True,
        "metric_gauge_anchor": {
            "artifact_id": anchor.artifact_id,
            "source_artifact_sha256": anchor.source_artifact_sha256,
            "source_kind": anchor.source_kind,
            "window_id": anchor.window_id,
        },
        "metric_units": "m",
        "minimum_prior_reliability": 0.05,
        "prior_reliability_definition": (
            "overlap disagreement only; independent of downstream physical "
            "innovation"
        ),
    }


def _producer_artifact() -> ObservationBeliefExportV1:
    anchor = MetricGaugeAnchor(
        window_id="window-0",
        global_from_local=Sim3.identity(),
        covariance=np.eye(7, dtype=np.float64) * 1e-6,
        coordinate_frame="phystwin-world",
        source_kind="prefix_registration",
        source_artifact_sha256="1" * 64,
        metadata={
            "calibration_artifact_sha256": "b" * 64,
            "purpose": "three-repository-installed-wheel-golden-path",
        },
    )
    covariance = np.repeat(
        (np.eye(3, dtype=np.float64) * 1e-5)[None],
        6,
        axis=0,
    )
    factors = np.zeros((6, 3, 5), dtype=np.float64)
    factors[0, 0, 0] = 0.0020
    factors[1, 1, 1] = 0.0015
    factors[2, 0, 0] = 0.0010
    factors[2, 2, 2] = 0.0025
    factors[3, 0, 3] = 0.0008
    factors[3, 1, 1] = 0.0010
    factors[4, 0, 0] = 0.0005
    factors[4, 2, 2] = 0.0015
    factors[5, 1, 1] = 0.0007
    factors[5, 2, 4] = 0.0012

    artifact = ObservationBeliefExportV1(
        case_id="three-repository-golden-path",
        stream_id="prob4d:causal-overlap-window-points",
        causal_frame_stop=6,
        view_names=("camera-0",),
        window_names=("window-0", "window-1", "window-2"),
        factor_names=tuple(
            f"joint_gauge_latent_{index:04d}" for index in range(5)
        ),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="d" * 40,
        source_artifact_sha256="c" * 64,
        declared_frame_ids=np.asarray([1, 2, 3, 4, 5], dtype=np.int64),
        mean_xyz_m=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.0, 0.1, 1.0],
                [0.1, 0.1, 1.0],
                [0.0, 0.2, 1.0],
                [0.1, 0.2, 1.0],
            ],
            dtype=np.float64,
        ),
        frame_ids=np.asarray([1, 2, 3, 4, 4, 5], dtype=np.int64),
        entity_ids=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        view_indices=np.zeros(6, dtype=np.int64),
        window_indices=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        correlation_group_ids=np.asarray(
            [0, 1, 2, 3, 3, 4],
            dtype=np.int64,
        ),
        factor_group_ids=np.zeros(6, dtype=np.int64),
        prior_reliability=np.asarray(
            [0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
            dtype=np.float64,
        ),
        association_probability=np.ones(6, dtype=np.float64),
        local_covariance_m2=covariance,
        low_rank_factor_m=factors,
        group_ids=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        group_prior_nominal_probability=np.ones(5, dtype=np.float64),
        group_composite_weight=np.asarray(
            [1.0, 1.0, 1.0, 0.5, 1.0],
            dtype=np.float64,
        ),
        metadata=_producer_metadata(anchor),
    )
    return bind_causal_stream_contract_v2(
        artifact,
        metric_anchor=anchor,
    )


def _round_trip_observation(path: Path):
    producer = _producer_artifact()
    save_observation_belief_export(path, producer)
    consumer = load_observation_belief(path)
    assert producer.artifact_id == EXPECTED_OBSERVATION_ARTIFACT_ID
    assert consumer.artifact_id == producer.artifact_id
    return consumer


def _run_bpt_update(observation):
    state = np.zeros((observation.observation_count, 3, 2))
    state[:, 0, 0] = np.where(observation.entity_ids == 0, 1.0, -1.0)
    state[:, 1, 1] = np.linspace(-1.0, 1.0, observation.observation_count)
    injected_coefficients = np.asarray([0.0015, -0.0008])
    innovation = np.einsum("ncs,s->nc", state, injected_coefficients)
    adapted = build_gauge_aware_batch_from_observation_belief(
        observation,
        physical_prediction_xyz_m=observation.mean_xyz_m - innovation,
        state_jacobian=state,
        query_state_jacobian=state[:2],
        physical_response_scale_m=0.05,
    )
    config = GaugeAwareBeliefConfig(maximum_iterations=8)
    first = update_gauge_aware_belief(adapted.batch, config=config)
    second = update_gauge_aware_belief(adapted.batch, config=config)

    assert first.inference_admissible == second.inference_admissible
    assert first.reason == second.reason
    np.testing.assert_array_equal(
        first.state_coefficients,
        second.state_coefficients,
    )
    np.testing.assert_array_equal(
        first.posterior_covariance,
        second.posterior_covariance,
    )
    assert first.input_lineage["observation_artifact_id"] == (
        observation.artifact_id
    )
    assert np.all(np.isfinite(first.posterior_covariance))
    np.testing.assert_allclose(
        first.posterior_covariance,
        first.posterior_covariance.T,
        atol=1e-12,
        rtol=0.0,
    )
    if not first.inference_admissible:
        np.testing.assert_array_equal(
            first.state_coefficients,
            np.zeros_like(first.state_coefficients),
        )
    return first


class DeterministicReplayProvider:
    """Small CPU fake satisfying the installed BPT replay protocol."""

    def __init__(self, frame_count: int, node_count: int = 1) -> None:
        self._frame_count = int(frame_count)
        self._node_count = int(node_count)
        self._log_scales = np.zeros(2, dtype=np.float64)
        self._controller_points = np.zeros(
            (frame_count, 1, 3),
            dtype=np.float64,
        )
        self._closed = False
        self.initial_calls = 0
        self.restart_calls = 0
        self.scale_history: list[tuple[float, float]] = []

    @property
    def device(self) -> str:
        return "cpu"

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("fake provider is closed")

    def set_group_log_scales(self, values: np.ndarray) -> None:
        self._require_open()
        candidate = np.asarray(values, dtype=np.float64)
        if candidate.shape != (2,) or not np.all(np.isfinite(candidate)):
            raise ValueError("fake provider expects two finite log-scales")
        self._log_scales = candidate.copy()
        self.scale_history.append(tuple(map(float, candidate)))

    def set_controller_points(self, values: np.ndarray) -> None:
        self._require_open()
        candidate = np.asarray(values, dtype=np.float64)
        if candidate.shape != self._controller_points.shape:
            raise ValueError("controller trajectory shape changed")
        if not np.all(np.isfinite(candidate)):
            raise ValueError("controller trajectory is non-finite")
        self._controller_points = candidate.copy()

    def replay_initial(
        self,
        *,
        frame_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_open()
        if not 1 <= frame_count <= self._frame_count:
            raise ValueError("invalid initial replay frame count")
        self.initial_calls += 1
        times = np.arange(frame_count, dtype=np.float64)
        positions = np.zeros(
            (frame_count, self._node_count, 3),
            dtype=np.float64,
        )
        positions[:, :, 0] = (
            0.01 * times[:, None]
            + 0.001 * float(np.sum(self._log_scales))
        )
        velocities = np.zeros_like(positions)
        velocities[:, :, 0] = 0.01
        return positions, velocities

    def replay_restart(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> np.ndarray:
        self._require_open()
        position = np.asarray(position_m, dtype=np.float64)
        velocity = np.asarray(velocity_mps, dtype=np.float64)
        if (
            position.shape != (self._node_count, 3)
            or velocity.shape != position.shape
            or not 0 <= start_frame < stop_frame < self._frame_count
        ):
            raise ValueError("invalid fake restart request")
        self.restart_calls += 1
        count = stop_frame - start_frame + 1
        trajectory = np.empty(
            (count, self._node_count, 3),
            dtype=np.float64,
        )
        trajectory[0] = position
        stiffness = float(np.exp(0.1 * np.sum(self._log_scales)))
        for offset in range(1, count):
            frame = start_frame + offset
            control_delta = (
                self._controller_points[frame]
                - self._controller_points[start_frame]
            )
            trajectory[offset] = (
                position
                + offset * velocity
                + stiffness * control_delta
            )
        return trajectory

    def close(self) -> None:
        self._closed = True


def _profile_particles(path: Path):
    source_weights = np.asarray([[0.5, 0.3], [0.1, 0.1]])
    prediction_weights = np.asarray([[5 / 9, 3 / 9], [1 / 9, 0.0]])
    np.savez(
        path,
        object_log_scales=np.asarray([-0.2, 0.2]),
        controller_log_scales=np.asarray([-0.1, 0.1]),
        posterior_weights=np.full((2, 2), 0.25),
        source_prediction_weights=source_weights,
        prediction_weights=prediction_weights,
    )
    particles = load_bayesian_phystwin_particles(
        path,
        maximum_count=2,
    )
    accounting = particles.probability_mass_accounting()
    assert np.isclose(particles.bpt_retained_probability_mass, 0.9)
    assert np.isclose(
        particles.causal4d_retained_probability_mass,
        8 / 9,
    )
    assert np.isclose(particles.retained_probability_mass, 0.8)
    assert [1, 1] not in particles.grid_indices.tolist()
    assert np.isclose(
        accounting["composed_relative_to_original_posterior"][
            "directly_retained_probability_mass"
        ],
        0.8,
    )
    return particles, accounting


def _counterfactual_artifacts(
    tmp_path: Path,
    observation_id: str,
    bpt_result,
):
    full_frames = 7
    intervention_frame = 4
    provider = DeterministicReplayProvider(full_frames)
    assert isinstance(provider, PhysTwinReplayProvider)

    particles, mass_accounting = _profile_particles(
        tmp_path / "profile.npz"
    )
    replay_positions = []
    replay_velocities = []
    for particle in particles.log_scales:
        provider.set_group_log_scales(particle)
        positions, velocities = provider.replay_initial(
            frame_count=intervention_frame
        )
        replay_positions.append(positions)
        replay_velocities.append(velocities)
    replay_positions_array = np.stack(replay_positions)
    replay_velocities_array = np.stack(replay_velocities)

    observations = np.zeros((full_frames, 1, 3), dtype=np.float64)
    observations[:intervention_frame] = replay_positions_array[0]
    factual_actions = np.zeros((full_frames, 1, 3), dtype=np.float64)
    factual_context = build_causal_context(
        protocol_id="three-repository-installed-wheel-v1",
        case_id="three-repository-golden-path",
        observations=observations,
        observed_actions=factual_actions,
        counterfactual_actions=factual_actions,
        intervention_frame=intervention_frame,
        counterfactual_action_id="factual-action",
    )

    provider_manifest = require_bayesian_phystwin_provider()
    assert provider_manifest.provider_revision == os.environ[
        "BAYESIAN_PHYSTWIN_REVISION"
    ]
    twin_belief = build_twin_belief_from_replays(
        context=factual_context,
        replay_positions_m=replay_positions_array,
        replay_velocities_mps=replay_velocities_array,
        observed_positions_m=observations,
        observed_valid=np.ones(
            observations.shape[:2],
            dtype=bool,
        ),
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
            "observation_artifact_id": observation_id,
            "provider_manifest_id": provider_manifest.manifest_id,
            "probability_mass_accounting": mass_accounting,
            "gauge_update": {
                "inference_admissible": bool(
                    bpt_result.inference_admissible
                ),
                "reason": bpt_result.reason,
                "state_coefficients": (
                    bpt_result.state_coefficients.tolist()
                ),
            },
        },
        config=BPTBeliefExportConfig(interpolation_neighbors=1),
    )
    assert twin_belief.metadata["observation_artifact_id"] == observation_id
    assert twin_belief.metadata["provider_manifest_id"] == (
        provider_manifest.manifest_id
    )

    twin_path = tmp_path / "twin-belief.npz"
    save_contract(twin_path, twin_belief)
    loaded_twin = load_contract(twin_path)
    assert isinstance(loaded_twin, TwinBelief)
    assert loaded_twin.artifact_id == twin_belief.artifact_id

    hypothesis_phi = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.15, 0.0, 0.0],
        ]
    )
    hypothesis_kappa = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )
    factual_hypothesis_weights = np.asarray([0.1, 0.8, 0.1])
    particle_count = len(particles.weights)
    factual = FactualIntervention(
        context=factual_context,
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
        kappa_names=(
            "attachment_shift_hand_0",
            "slip_fraction",
        ),
        phi=np.repeat(
            hypothesis_phi,
            particle_count,
            axis=0,
        ),
        kappa_obs=np.repeat(
            hypothesis_kappa,
            particle_count,
            axis=0,
        ),
        hypothesis_indices=np.repeat(
            np.arange(3, dtype=np.int64),
            particle_count,
        ),
        twin_particle_indices=np.tile(
            np.arange(particle_count, dtype=np.int64),
            3,
        ),
        weights=np.outer(
            factual_hypothesis_weights,
            particles.weights,
        ).reshape(-1),
        evidence_frame_stop=intervention_frame + 1,
        source_twin_belief_id=twin_belief.artifact_id,
    )

    counterfactual_actions = factual_actions.copy()
    counterfactual_actions[intervention_frame:, 0, 0] = np.asarray(
        [0.01, 0.02, 0.03]
    )
    query_context = build_causal_context(
        protocol_id="three-repository-installed-wheel-v1",
        case_id="three-repository-golden-path",
        observations=observations,
        observed_actions=factual_actions,
        counterfactual_actions=counterfactual_actions,
        intervention_frame=intervention_frame,
        counterfactual_action_id="new-action",
    )
    query = CounterfactualQuery(
        context=query_context,
        controller_points_m=counterfactual_actions[
            intervention_frame:
        ],
        horizon_frames=full_frames - intervention_frame,
        contact_policy="new_contact",
        source_factual_intervention_id=factual.artifact_id,
    )

    def hypothesis_metadata(
        identifier: str,
        gain: float,
        shift: int,
    ) -> dict[str, Any]:
        return {
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

    hypotheses = (
        ("nominal", 1.0, 0),
        ("shift", 1.0, 1),
        ("gain", 1.15, 0),
    )
    trajectories = np.empty(
        (
            len(hypotheses),
            particle_count,
            full_frames - intervention_frame + 1,
            1,
            3,
        ),
        dtype=np.float32,
    )
    for hypothesis_index, (_, gain, shift) in enumerate(hypotheses):
        controls = counterfactual_actions.copy()
        controls[intervention_frame:, 0, 0] *= gain
        controls[intervention_frame:, 0, 1] += 0.001 * shift
        provider.set_controller_points(controls)
        for particle_index, particle in enumerate(particles.log_scales):
            provider.set_group_log_scales(particle)
            trajectories[hypothesis_index, particle_index] = (
                provider.replay_restart(
                    twin_belief.endpoint_position_m[particle_index],
                    twin_belief.endpoint_velocity_mps[particle_index],
                    start_frame=intervention_frame - 1,
                    stop_frame=full_frames - 1,
                )
            )

    bank = JointRolloutBank(
        hypothesis_ids=tuple(value[0] for value in hypotheses),
        hypothesis_metadata=tuple(
            hypothesis_metadata(identifier, gain, shift)
            for identifier, gain, shift in hypotheses
        ),
        hypothesis_prior_weights=np.asarray([0.5, 0.25, 0.25]),
        parameter_particles=particles.log_scales,
        parameter_weights=particles.weights,
        trajectories=trajectories,
        variance_floor_m2=2e-6,
    )
    manifest = {
        "causal_context": query_context.as_dict(),
        "twin_belief_id": twin_belief.artifact_id,
        "provider_manifest_id": provider_manifest.manifest_id,
    }
    posterior = apply_counterfactual_operator(
        bank,
        manifest,
        twin_belief,
        factual,
        query,
    )
    expected_weights = np.concatenate(
        [
            0.6 * particles.weights,
            0.3 * particles.weights,
            0.1 * particles.weights,
        ]
    )
    np.testing.assert_allclose(
        posterior.weights,
        expected_weights,
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.isclose(
        posterior.metadata[
            "represented_factual_mass_before_renormalization"
        ],
        1.0,
    )
    assert posterior.source_twin_belief_id == twin_belief.artifact_id
    assert posterior.source_factual_intervention_id == factual.artifact_id
    assert posterior.source_query_id == query.artifact_id
    assert provider.initial_calls == particle_count
    assert provider.restart_calls == len(hypotheses) * particle_count

    selected_scales = {
        tuple(map(float, values))
        for values in particles.log_scales
    }
    replayed_scales = set(provider.scale_history)
    assert replayed_scales == selected_scales
    assert (0.2, 0.1) not in replayed_scales
    provider.close()
    return posterior


def test_packages_are_loaded_from_installed_wheels() -> None:
    source_roots = tuple(
        Path(value).resolve()
        for value in os.environ["THREE_REPO_SOURCE_ROOTS"].split(
            os.pathsep
        )
        if value
    )
    assert len(source_roots) == 3
    for import_name, distribution_name in (
        ("prob4d", "prob4d"),
        ("bayesian_phystwin", "bayesian-phystwin"),
        ("causal4d", "causal4d"),
    ):
        module_path = Path(import_module(import_name).__file__).resolve()
        assert all(root not in module_path.parents for root in source_roots)
        direct_url = importlib_metadata.distribution(
            distribution_name
        ).read_text("direct_url.json")
        if direct_url:
            payload = json.loads(direct_url)
            assert payload.get("dir_info", {}).get("editable") is not True


def test_three_repository_installed_wheel_golden_path(
    tmp_path: Path,
) -> None:
    observation = _round_trip_observation(
        tmp_path / "prob4d-observation.npz"
    )
    validation = validate_prob4d_causal_observation_belief(observation)
    assert validation["validated"] is True
    assert validation["stream_contract_version"] == 2
    assert validation["stream_contract_version_inferred"] is False
    assert validation["cross_window_covariance_preserved"] is True

    bpt_result = _run_bpt_update(observation)
    posterior = _counterfactual_artifacts(
        tmp_path,
        observation.artifact_id,
        bpt_result,
    )
    posterior_path = tmp_path / "physical-posterior.npz"
    save_contract(posterior_path, posterior)
    loaded = load_contract(posterior_path)
    assert loaded.artifact_id == posterior.artifact_id


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("future_window", "crosses its causal boundary"),
        ("stream_version", "version disagrees"),
        ("missing_calibration", "calibration_artifact_sha256"),
        (
            "untracked_anchor_covariance",
            "include metric-anchor covariance",
        ),
        ("per_window_gauge", "one shared factor group"),
        ("retained_trace", "retained-trace threshold"),
    ],
)
def test_bpt_consumer_rejects_semantic_drift(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    observation = _round_trip_observation(
        tmp_path / f"{case}-observation.npz"
    )
    metadata = deepcopy(dict(observation.metadata))
    replacement: dict[str, Any] = {"metadata": metadata}

    if case == "future_window":
        metadata["causal_source_lineage"]["selected_windows"][-1][
            "source_frame_max"
        ] = observation.causal_frame_stop
    elif case == "stream_version":
        metadata["prob4d_causal_stream_contract_version"] = 1
    elif case == "missing_calibration":
        del metadata["metric_gauge_anchor"][
            "calibration_artifact_sha256"
        ]
    elif case == "untracked_anchor_covariance":
        metadata["metric_anchor_covariance_in_joint_factor"] = False
    elif case == "per_window_gauge":
        replacement["factor_group_ids"] = observation.window_indices
    elif case == "retained_trace":
        metadata["gauge_posterior"][
            "retained_covariance_trace_fraction"
        ] = 0.8
    else:  # pragma: no cover - parameterization is exhaustive.
        raise AssertionError(f"unknown rejection case: {case}")

    drifted = replace(observation, **replacement)
    with pytest.raises(ValueError, match=message):
        validate_prob4d_causal_observation_belief(drifted)
