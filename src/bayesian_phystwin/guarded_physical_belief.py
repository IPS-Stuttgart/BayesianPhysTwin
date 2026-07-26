"""Content-addressed physical beliefs and prospective guard selection.

This module layers complete-belief artifacts and exact fallback on the stable
``causal4d_provider_v1`` facade. It does not replace the provider facade or
expose implementation-private simulator helpers.
"""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .causal4d_provider_v1 import (
    PhysicalBeliefProviderManifest,
    provider_manifest,
)

PHYSICAL_BELIEF_SCHEMA = "bayesian_phystwin.physical_belief"
PHYSICAL_BELIEF_VERSION = 1
GUARD_DECISION_SCHEMA = "bayesian_phystwin.guard_decision"
GUARD_DECISION_VERSION = 1
BELIEF_SELECTION_SCHEMA = "bayesian_phystwin.physical_belief_selection"
BELIEF_SELECTION_VERSION = 1

_GUARDED_CAPABILITIES = (
    "exact_baseline_fallback",
    "readout_discrepancy_moments",
)
_GUARDED_ARTIFACT_VERSIONS = {
    "PhysicalBeliefV1": PHYSICAL_BELIEF_VERSION,
    "GuardDecisionV1": GUARD_DECISION_VERSION,
    "PhysicalBeliefSelectionV1": BELIEF_SELECTION_VERSION,
}


def _validated_json(values: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(values), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain finite JSON values") from error


def _canonical_json(values: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(values),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _readonly(values: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def array_sha256(values: np.ndarray) -> str:
    """Hash an array including dtype, shape, and byte payload."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _artifact_id(
    descriptor: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(descriptor))
    for name, values in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(values).encode("ascii"))
    return digest.hexdigest()


def guarded_provider_manifest(
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Return the base provider manifest extended by guarded-belief semantics."""

    base = provider_manifest(provider_revision)
    versions = dict(base.artifact_schema_versions)
    versions.update(_GUARDED_ARTIFACT_VERSIONS)
    metadata = dict(base.metadata)
    metadata.update(
        {
            "guarded_belief_module": __name__,
            "guarded_belief_contract_version": 1,
        }
    )
    return PhysicalBeliefProviderManifest(
        provider_name=base.provider_name,
        provider_version=base.provider_version,
        provider_revision=base.provider_revision,
        schema_version=base.schema_version,
        capabilities=(*base.capabilities, *_GUARDED_CAPABILITIES),
        artifact_schema_versions=versions,
        metadata=metadata,
    )


@dataclass(frozen=True)
class PhysicalBeliefV1:
    """Portable particle belief over endpoint state, physics, and discrepancy."""

    provider_manifest_id: str
    endpoint_frame: int
    particle_ids: tuple[str, ...]
    theta_names: tuple[str, ...]
    endpoint_position_m: np.ndarray
    endpoint_velocity_mps: np.ndarray
    theta: np.ndarray
    discrepancy_mean_m: np.ndarray
    discrepancy_variance_m2: np.ndarray
    weights: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_sha256(self.provider_manifest_id, name="provider_manifest_id")
        if self.endpoint_frame < 0:
            raise ValueError("endpoint_frame must be nonnegative")
        if not self.theta_names or any(not value for value in self.theta_names):
            raise ValueError("theta_names must contain nonempty names")
        position = _readonly(self.endpoint_position_m, dtype=np.float64)
        velocity = _readonly(self.endpoint_velocity_mps, dtype=np.float64)
        theta = _readonly(self.theta, dtype=np.float64)
        discrepancy = _readonly(self.discrepancy_mean_m, dtype=np.float64)
        variance = _readonly(self.discrepancy_variance_m2, dtype=np.float64)
        weights = _readonly(self.weights, dtype=np.float64)
        if weights.ndim != 1 or len(weights) == 0:
            raise ValueError("weights must be a nonempty vector")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and nonnegative")
        if not np.isclose(np.sum(weights), 1.0, atol=1e-10, rtol=1e-10):
            raise ValueError("weights must sum to one")
        particle_count = len(weights)
        particle_ids = tuple(map(str, self.particle_ids))
        if (
            len(particle_ids) != particle_count
            or len(set(particle_ids)) != particle_count
        ):
            raise ValueError("particle_ids must uniquely identify every particle")
        if (
            position.ndim != 3
            or position.shape[0] != particle_count
            or position.shape[2] != 3
        ):
            raise ValueError("endpoint_position_m must have shape (P, N, 3)")
        if velocity.shape != position.shape or discrepancy.shape != position.shape:
            raise ValueError(
                "velocity and discrepancy means must match endpoint positions"
            )
        if variance.shape != position.shape:
            raise ValueError("discrepancy_variance_m2 must match endpoint positions")
        if theta.shape != (particle_count, len(self.theta_names)):
            raise ValueError("theta must have shape (P, len(theta_names))")
        if any(
            not np.all(np.isfinite(values))
            for values in (position, velocity, theta, discrepancy, variance)
        ):
            raise ValueError("physical belief arrays must be finite")
        if np.any(variance < 0.0):
            raise ValueError("discrepancy variances must be nonnegative")
        object.__setattr__(self, "particle_ids", particle_ids)
        object.__setattr__(self, "theta_names", tuple(map(str, self.theta_names)))
        object.__setattr__(self, "endpoint_position_m", position)
        object.__setattr__(self, "endpoint_velocity_mps", velocity)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "discrepancy_mean_m", discrepancy)
        object.__setattr__(self, "discrepancy_variance_m2", variance)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "metadata", _validated_json(self.metadata))

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": PHYSICAL_BELIEF_SCHEMA,
            "schema_version": PHYSICAL_BELIEF_VERSION,
            "provider_manifest_id": self.provider_manifest_id,
            "endpoint_frame": self.endpoint_frame,
            "particle_ids": list(self.particle_ids),
            "theta_names": list(self.theta_names),
            "metadata": dict(self.metadata),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "endpoint_position_m": self.endpoint_position_m,
            "endpoint_velocity_mps": self.endpoint_velocity_mps,
            "theta": self.theta,
            "discrepancy_mean_m": self.discrepancy_mean_m,
            "discrepancy_variance_m2": self.discrepancy_variance_m2,
            "weights": self.weights,
        }

    @property
    def artifact_id(self) -> str:
        return _artifact_id(self.descriptor(), self.arrays())


@dataclass(frozen=True)
class GuardDecisionV1:
    """Prospective decision kept separate from candidate inference."""

    candidate_valid: bool
    guard_accepted: bool
    reason: str
    certificate_id: str
    development_partition_sha256: str
    observation_artifact_id: str
    linearization_artifact_id: str
    primary_losses: Mapping[str, float]
    nonlinear_closure_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("guard decision reason must be nonempty")
        for name, value in (
            ("certificate_id", self.certificate_id),
            ("development_partition_sha256", self.development_partition_sha256),
            ("observation_artifact_id", self.observation_artifact_id),
            ("linearization_artifact_id", self.linearization_artifact_id),
        ):
            _validate_sha256(value, name=name)
        if self.nonlinear_closure_id is not None:
            _validate_sha256(
                self.nonlinear_closure_id, name="nonlinear_closure_id"
            )
        if self.guard_accepted and not self.candidate_valid:
            raise ValueError("guard_accepted requires candidate_valid")
        losses = {
            str(name): float(value)
            for name, value in dict(self.primary_losses).items()
        }
        if not losses or any(
            not name or not np.isfinite(value) for name, value in losses.items()
        ):
            raise ValueError("primary_losses must contain finite named values")
        object.__setattr__(self, "primary_losses", losses)
        object.__setattr__(self, "metadata", _validated_json(self.metadata))

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": GUARD_DECISION_SCHEMA,
            "schema_version": GUARD_DECISION_VERSION,
            "candidate_valid": self.candidate_valid,
            "guard_accepted": self.guard_accepted,
            "reason": self.reason,
            "certificate_id": self.certificate_id,
            "development_partition_sha256": self.development_partition_sha256,
            "observation_artifact_id": self.observation_artifact_id,
            "linearization_artifact_id": self.linearization_artifact_id,
            "nonlinear_closure_id": self.nonlinear_closure_id,
            "primary_losses": dict(self.primary_losses),
            "metadata": dict(self.metadata),
        }

    @property
    def decision_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.descriptor())).hexdigest()


@dataclass(frozen=True)
class PhysicalBeliefSelectionV1:
    """Content-addressed selection manifest over complete physical beliefs."""

    baseline_belief_id: str
    candidate_belief_id: str
    guard_decision_id: str
    selected_belief_id: str
    candidate_valid: bool
    guard_accepted: bool
    selected_candidate: bool
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("baseline_belief_id", self.baseline_belief_id),
            ("candidate_belief_id", self.candidate_belief_id),
            ("guard_decision_id", self.guard_decision_id),
            ("selected_belief_id", self.selected_belief_id),
        ):
            _validate_sha256(value, name=name)
        if not self.reason:
            raise ValueError("selection reason must be nonempty")
        if self.selected_candidate != (
            self.candidate_valid and self.guard_accepted
        ):
            raise ValueError(
                "selected_candidate must equal candidate_valid and guard_accepted"
            )
        expected = (
            self.candidate_belief_id
            if self.selected_candidate
            else self.baseline_belief_id
        )
        if self.selected_belief_id != expected:
            raise ValueError(
                "selected_belief_id does not match the selection decision"
            )
        object.__setattr__(self, "metadata", _validated_json(self.metadata))

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": BELIEF_SELECTION_SCHEMA,
            "schema_version": BELIEF_SELECTION_VERSION,
            "baseline_belief_id": self.baseline_belief_id,
            "candidate_belief_id": self.candidate_belief_id,
            "guard_decision_id": self.guard_decision_id,
            "selected_belief_id": self.selected_belief_id,
            "candidate_valid": self.candidate_valid,
            "guard_accepted": self.guard_accepted,
            "selected_candidate": self.selected_candidate,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    @property
    def selection_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.descriptor())).hexdigest()


def _validate_common_domain(
    baseline: PhysicalBeliefV1, candidate: PhysicalBeliefV1
) -> None:
    if baseline.endpoint_frame != candidate.endpoint_frame:
        raise ValueError("baseline and candidate endpoint frames differ")
    if baseline.particle_ids != candidate.particle_ids:
        raise ValueError("baseline and candidate particle identities differ")
    if baseline.theta_names != candidate.theta_names:
        raise ValueError("baseline and candidate parameter names differ")
    if baseline.provider_manifest_id != candidate.provider_manifest_id:
        raise ValueError("baseline and candidate provider manifests differ")
    for name in (
        "endpoint_position_m",
        "endpoint_velocity_mps",
        "theta",
        "discrepancy_mean_m",
        "discrepancy_variance_m2",
        "weights",
    ):
        if getattr(baseline, name).shape != getattr(candidate, name).shape:
            raise ValueError(f"baseline and candidate {name} shapes differ")


def select_physical_belief(
    baseline: PhysicalBeliefV1,
    candidate: PhysicalBeliefV1,
    guard: GuardDecisionV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[PhysicalBeliefV1, PhysicalBeliefSelectionV1]:
    """Select a complete candidate belief or reuse the exact baseline object."""

    _validate_common_domain(baseline, candidate)
    for key, expected in (
        ("observation_artifact_id", guard.observation_artifact_id),
        ("linearization_artifact_id", guard.linearization_artifact_id),
    ):
        actual = candidate.metadata.get(key)
        if actual is not None and actual != expected:
            raise ValueError(f"candidate {key} does not match the guard decision")
    selected_candidate = guard.candidate_valid and guard.guard_accepted
    selected = candidate if selected_candidate else baseline
    reason = (
        "guard-accepted"
        if selected_candidate
        else (
            "candidate-invalid"
            if not guard.candidate_valid
            else "guard-rejected"
        )
    )
    selection = PhysicalBeliefSelectionV1(
        baseline_belief_id=baseline.artifact_id,
        candidate_belief_id=candidate.artifact_id,
        guard_decision_id=guard.decision_id,
        selected_belief_id=selected.artifact_id,
        candidate_valid=guard.candidate_valid,
        guard_accepted=guard.guard_accepted,
        selected_candidate=selected_candidate,
        reason=reason,
        metadata=metadata or {},
    )
    if not selected_candidate and selected is not baseline:
        raise AssertionError(
            "rejected selection did not reuse the exact baseline object"
        )
    return selected, selection


def save_physical_belief(path: str | Path, belief: PhysicalBeliefV1) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = belief.descriptor()
    descriptor["artifact_id"] = belief.artifact_id
    np.savez_compressed(
        target,
        descriptor_json=np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        **belief.arrays(),
    )


def load_physical_belief(path: str | Path) -> PhysicalBeliefV1:
    with np.load(path, allow_pickle=False) as archive:
        if "descriptor_json" not in archive:
            raise ValueError("physical belief has no descriptor_json")
        descriptor = json.loads(str(archive["descriptor_json"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "descriptor_json"
        }
    if descriptor.get("schema_name") != PHYSICAL_BELIEF_SCHEMA:
        raise ValueError("unsupported physical-belief schema")
    if int(descriptor.get("schema_version", -1)) != PHYSICAL_BELIEF_VERSION:
        raise ValueError("unsupported physical-belief version")
    required = {
        "endpoint_position_m",
        "endpoint_velocity_mps",
        "theta",
        "discrepancy_mean_m",
        "discrepancy_variance_m2",
        "weights",
    }
    if set(arrays) != required:
        raise ValueError("physical-belief array set changed")
    belief = PhysicalBeliefV1(
        provider_manifest_id=str(descriptor["provider_manifest_id"]),
        endpoint_frame=int(descriptor["endpoint_frame"]),
        particle_ids=tuple(map(str, descriptor["particle_ids"])),
        theta_names=tuple(map(str, descriptor["theta_names"])),
        metadata=descriptor.get("metadata", {}),
        **arrays,
    )
    expected = str(descriptor.get("artifact_id", ""))
    _validate_sha256(expected, name="artifact_id")
    if belief.artifact_id != expected:
        raise ValueError("physical-belief digest does not match its payload")
    return belief


@dataclass(frozen=True)
class BPTBeliefExportConfigV1:
    """Frozen settings for endpoint-belief construction."""

    process_std_m: float = 0.005
    observation_std_m: float = 0.001
    initial_std_m: float = 0.01
    inlier_prior: float = 0.95
    outlier_variance_multiplier: float = 100.0
    interpolation_neighbors: int = 4
    maximum_discrepancy_m: float = 0.01

    def __post_init__(self) -> None:
        if self.process_std_m < 0.0:
            raise ValueError("process_std_m must be nonnegative")
        if self.observation_std_m <= 0.0 or self.initial_std_m <= 0.0:
            raise ValueError("observation and initial scales must be positive")
        if not 0.0 < self.inlier_prior < 1.0:
            raise ValueError("inlier_prior must lie in (0, 1)")
        if self.outlier_variance_multiplier <= 1.0:
            raise ValueError("outlier_variance_multiplier must exceed one")
        if self.interpolation_neighbors < 1 or self.maximum_discrepancy_m <= 0.0:
            raise ValueError("lifting settings must be positive")


def _lift_isotropic_discrepancy_variance(
    tracked_variance_m2: np.ndarray,
    state_count: int,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    tracked = np.asarray(tracked_variance_m2, dtype=float)
    neighbor_indices = np.asarray(indices, dtype=np.int64)
    neighbor_weights = np.asarray(weights, dtype=float)
    if (
        tracked.ndim != 1
        or not np.all(np.isfinite(tracked))
        or np.any(tracked < 0.0)
    ):
        raise ValueError("tracked variance must be a finite nonnegative vector")
    if state_count < len(tracked):
        raise ValueError("state_count cannot be smaller than the tracked state")
    extra_count = state_count - len(tracked)
    if (
        neighbor_indices.shape != neighbor_weights.shape
        or neighbor_indices.shape[0] != extra_count
    ):
        raise ValueError("lift map must identify every untracked state node")
    if np.any(neighbor_indices < 0) or np.any(
        neighbor_indices >= len(tracked)
    ):
        raise ValueError("lift map references an unavailable tracked node")
    if extra_count and not np.allclose(
        np.sum(neighbor_weights, axis=1), 1.0
    ):
        raise ValueError("lift weights must sum to one")
    scalar = np.empty(state_count, dtype=float)
    scalar[: len(tracked)] = tracked
    if extra_count:
        scalar[len(tracked) :] = np.sum(
            np.square(neighbor_weights) * tracked[neighbor_indices], axis=1
        )
    return np.repeat(scalar[:, None], 3, axis=1)


def build_physical_belief_from_replays(
    *,
    manifest: PhysicalBeliefProviderManifest,
    causal_frame_stop: int,
    replay_positions_m: np.ndarray,
    replay_velocities_mps: np.ndarray,
    observed_positions_m: np.ndarray,
    observed_valid: np.ndarray,
    theta: np.ndarray,
    theta_names: tuple[str, ...],
    weights: np.ndarray,
    particle_ids: tuple[str, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
    config: BPTBeliefExportConfigV1 | None = None,
) -> PhysicalBeliefV1:
    """Build a portable endpoint belief using only the declared causal prefix."""

    from .causal4d_provider_v1 import (
        build_lift_map,
        lift_residual,
        robust_random_walk_endpoint,
    )

    required = set(_GUARDED_CAPABILITIES)
    if not required.issubset(manifest.capabilities):
        raise ValueError("manifest lacks guarded-belief capabilities")
    if manifest.artifact_schema_versions.get("PhysicalBeliefV1") != 1:
        raise ValueError("manifest lacks PhysicalBeliefV1 schema version 1")
    settings = config or BPTBeliefExportConfigV1()
    positions = np.asarray(replay_positions_m, dtype=float)
    velocities = np.asarray(replay_velocities_mps, dtype=float)
    observed = np.asarray(observed_positions_m, dtype=float)
    valid = np.asarray(observed_valid, dtype=bool)
    particle_values = np.asarray(theta, dtype=float)
    particle_weights = np.asarray(weights, dtype=float)
    if causal_frame_stop < 1:
        raise ValueError("causal_frame_stop must be positive")
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("replay_positions_m must have shape (P, T, N, 3)")
    if velocities.shape != positions.shape:
        raise ValueError("replay velocities must match replay positions")
    particle_count, frame_count, state_count, _ = positions.shape
    if frame_count < causal_frame_stop:
        raise ValueError("replays do not cover the causal prefix")
    if (
        observed.ndim != 3
        or observed.shape[2] != 3
        or len(observed) < causal_frame_stop
    ):
        raise ValueError("observed_positions_m must cover the causal prefix")
    tracked_count = observed.shape[1]
    if tracked_count > state_count or valid.shape != observed.shape[:2]:
        raise ValueError(
            "observed validity or tracked state size is inconsistent"
        )
    if particle_values.shape != (particle_count, len(theta_names)):
        raise ValueError("theta does not identify every replay particle")
    if particle_weights.shape != (particle_count,):
        raise ValueError("weights do not identify every replay particle")
    if not 1 <= settings.interpolation_neighbors <= tracked_count:
        raise ValueError(
            "interpolation_neighbors exceeds the tracked point count"
        )

    lift_indices, lift_weights = build_lift_map(
        positions[0, 0], tracked_count, settings.interpolation_neighbors
    )
    discrepancy_means = np.empty(
        (particle_count, state_count, 3), dtype=float
    )
    discrepancy_variances = np.empty_like(discrepancy_means)
    update_counts: list[int] = []
    final_inlier_probabilities: list[float] = []
    for particle_index in range(particle_count):
        residual = (
            observed[:causal_frame_stop]
            - positions[
                particle_index, :causal_frame_stop, :tracked_count
            ]
        )
        posterior = robust_random_walk_endpoint(
            residual,
            valid[:causal_frame_stop],
            end_frame=causal_frame_stop,
            process_variance=settings.process_std_m**2,
            observation_variance=settings.observation_std_m**2,
            initial_variance=settings.initial_std_m**2,
            inlier_prior=settings.inlier_prior,
            outlier_variance_multiplier=(
                settings.outlier_variance_multiplier
            ),
        )
        discrepancy_means[particle_index] = lift_residual(
            posterior.mean[None],
            state_count,
            lift_indices,
            lift_weights,
            maximum_norm=settings.maximum_discrepancy_m,
        )[0]
        discrepancy_variances[particle_index] = (
            _lift_isotropic_discrepancy_variance(
                posterior.variance,
                state_count,
                lift_indices,
                lift_weights,
            )
        )
        update_counts.append(int(np.sum(posterior.update_count)))
        supported = posterior.update_count > 0
        final_inlier_probabilities.append(
            float(np.mean(posterior.final_inlier_probability[supported]))
            if np.any(supported)
            else 0.0
        )

    endpoint = causal_frame_stop - 1
    diagnostics = {
        "causal_fit_window": [0, causal_frame_stop],
        "future_frames_read_by_estimator": 0,
        "particle_state_source": "provider replay through causal prefix",
        "discrepancy_role": (
            "separate readout/process discrepancy; not injected into state"
        ),
        "discrepancy_filter": asdict(settings),
        "particle_update_counts": update_counts,
        "particle_mean_final_inlier_probability": (
            final_inlier_probabilities
        ),
    }
    diagnostics.update(metadata or {})
    identifiers = particle_ids or tuple(
        f"theta_{index:04d}" for index in range(particle_count)
    )
    return PhysicalBeliefV1(
        provider_manifest_id=manifest.manifest_id,
        endpoint_frame=endpoint,
        particle_ids=identifiers,
        theta_names=theta_names,
        endpoint_position_m=positions[:, endpoint],
        endpoint_velocity_mps=velocities[:, endpoint],
        theta=particle_values,
        discrepancy_mean_m=discrepancy_means,
        discrepancy_variance_m2=discrepancy_variances,
        weights=particle_weights,
        metadata=diagnostics,
    )


def replay_official_phystwin_particles(
    *,
    official_repo: str | Path,
    data: Mapping[str, Any],
    optimal: Mapping[str, Any],
    checkpoint_path: str | Path,
    graph: Any,
    log_scales: np.ndarray,
    original_count: int,
    surface_point_count: int,
    frame_count: int,
    dt: float,
    num_substeps: int,
    self_collision: bool,
    deterministic_spring_forces: bool,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay selected grouped-spring particles through the public facade."""

    from .causal4d_provider_v1 import initialize_simulator, rollout_initial

    particles = np.asarray(log_scales, dtype=float)
    if particles.ndim != 2 or len(particles) == 0:
        raise ValueError("log_scales must have shape (P, D)")
    simulator, torch, wp, _ = initialize_simulator(
        Path(official_repo),
        data,
        optimal,
        Path(checkpoint_path),
        graph,
        num_surface_points=original_count + surface_point_count,
        original_count=original_count,
        dt=dt,
        num_substeps=num_substeps,
        self_collision=self_collision,
        deterministic_spring_forces=deterministic_spring_forces,
        spring_parameterization="grouped",
        device=device,
    )
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    try:
        for particle in particles:
            with torch.no_grad():
                simulator.group_log_scale_tensor.copy_(
                    torch.as_tensor(
                        particle, dtype=torch.float32, device=device
                    )
                )
            position, velocity = rollout_initial(
                simulator, wp, frame_count=frame_count
            )
            positions.append(np.asarray(position))
            velocities.append(np.asarray(velocity))
    finally:
        del simulator
        gc.collect()
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
    return np.stack(positions), np.stack(velocities)


__all__ = [
    "BELIEF_SELECTION_SCHEMA",
    "BELIEF_SELECTION_VERSION",
    "BPTBeliefExportConfigV1",
    "GUARD_DECISION_SCHEMA",
    "GUARD_DECISION_VERSION",
    "GuardDecisionV1",
    "PHYSICAL_BELIEF_SCHEMA",
    "PHYSICAL_BELIEF_VERSION",
    "PhysicalBeliefSelectionV1",
    "PhysicalBeliefV1",
    "array_sha256",
    "build_physical_belief_from_replays",
    "guarded_provider_manifest",
    "load_physical_belief",
    "replay_official_phystwin_particles",
    "save_physical_belief",
    "select_physical_belief",
]
