"""Probabilistic residual primitives for reusable Deform360 twins.

This module is deliberately separate from the frozen Causal4D protocol.  It
contains the source-development machinery for a residual model that can return
exactly to its physical prediction when its prospective trust gate rejects it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # Torch is an optional dependency for the learning track.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal installs.
    torch = None
    nn = None


BAYESIAN_RESIDUAL_SCHEMA_VERSION = 1
BAYESIAN_RESIDUAL_PROTOCOL_ID = "deform360-bayesian-residual-source-v1"
CANONICAL_BAYESIAN_RESIDUAL_CONFIG_SHA256 = (
    "df0a5c3c9b79257aee126526629791a3a15d81e35fbbf6df3abee6073d413743"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bayesian_residual_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_bayesian_residual_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the source-only SOTA development lock."""

    _require(
        payload.get("schema_version") == BAYESIAN_RESIDUAL_SCHEMA_VERSION,
        "unsupported Bayesian-residual schema",
    )
    observed = bayesian_residual_config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "Bayesian-residual config checksum mismatch",
    )
    _require(
        observed == CANONICAL_BAYESIAN_RESIDUAL_CONFIG_SHA256,
        "Bayesian-residual config differs from the canonical lock",
    )
    config = payload.get("config", {})
    _require(
        config.get("protocol_id") == BAYESIAN_RESIDUAL_PROTOCOL_ID,
        "Bayesian-residual protocol id changed",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("development_outcomes")
        == "27 already-open independent-source episodes only"
        and boundary.get("penguin_held_episode_ids") == [0, 2, 5, 8]
        and boundary.get("penguin_held_media_or_outcomes_may_open") is False
        and boundary.get("causal4d_frozen_claim_may_change") is False,
        "Bayesian-residual information boundary changed",
    )
    evaluation = config.get("evaluation", {})
    _require(
        evaluation.get("outer_split") == "leave-one-object-out"
        and evaluation.get("unit_of_replication") == "episode"
        and evaluation.get("future_object_frames_allowed_as_model_input") is False,
        "Bayesian-residual evaluation boundary changed",
    )
    fallback = config.get("fallback", {})
    _require(
        fallback.get("rejected_residual_is_byte_identical_to_physics") is True
        and fallback.get("thresholds_selected_on_outer_training_objects_only") is True,
        "Bayesian-residual fallback guarantee changed",
    )
    gates = config.get("development_gates", {})
    _require(
        gates.get("minimum_future_track_improvement_fraction") == 0.05
        and gates.get("minimum_future_chamfer_improvement_fraction") == 0.05
        and gates.get("minimum_late_track_improvement_fraction") == 0.05
        and gates.get("minimum_late_chamfer_improvement_fraction") == 0.05
        and gates.get("maximum_episode_degradation_fraction") == 0.10
        and gates.get("all_gates_conjunctive") is True,
        "Bayesian-residual development gates changed",
    )
    targets = config.get("external_score_targets", {})
    _require(
        targets.get("deform360_multi_episode_future_cd_m") == 0.051
        and targets.get("deform360_multi_episode_future_track_error_m") == 0.079
        and targets.get("deform360_multi_object_future_cd_m") == 0.038
        and targets.get("deform360_multi_object_future_track_error_m") == 0.048,
        "external score targets changed",
    )
    return {
        "passed": True,
        "protocol_id": BAYESIAN_RESIDUAL_PROTOCOL_ID,
        "config_sha256": observed,
        "held_episode_ids": [0, 2, 5, 8],
    }


def load_bayesian_residual_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "Bayesian-residual config must be an object")
    validate_bayesian_residual_config(payload)
    return payload


def clustered_effective_weights(
    reliability: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """Normalize node weights without counting correlated duplicates twice."""

    values = np.asarray(reliability, dtype=np.float64)
    clusters = np.asarray(cluster_ids)
    _require(values.shape == clusters.shape, "reliability and clusters differ")
    _require(values.ndim in (1, 2), "clustered weights require (N,) or (B,N)")
    _require(np.all(np.isfinite(values)) and np.all(values >= 0.0), "invalid reliability")
    batched_values = values[None] if values.ndim == 1 else values
    batched_clusters = clusters[None] if clusters.ndim == 1 else clusters
    output = np.zeros_like(batched_values, dtype=np.float64)
    for batch_index, (row, row_clusters) in enumerate(
        zip(batched_values, batched_clusters, strict=True)
    ):
        _, inverse, counts = np.unique(
            row_clusters, return_inverse=True, return_counts=True
        )
        effective = row / counts[inverse]
        total = float(np.sum(effective))
        _require(total > 0.0, "effective reliability has zero mass")
        output[batch_index] = effective / total
    return output[0] if values.ndim == 1 else output


def moment_match_residual_ensemble(
    means: np.ndarray,
    aleatoric_variances: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine aleatoric variance and between-model epistemic spread."""

    member_means = np.asarray(means, dtype=np.float64)
    member_variances = np.asarray(aleatoric_variances, dtype=np.float64)
    _require(member_means.ndim >= 2, "ensemble means need a member axis")
    _require(member_means.shape[-1] == 3, "ensemble means must be 3D vectors")
    if member_variances.shape == member_means.shape[:-1]:
        member_variances = np.repeat(member_variances[..., None], 3, axis=-1)
    _require(
        member_variances.shape == member_means.shape,
        "ensemble variance shape differs from means",
    )
    member_count = member_means.shape[0]
    if weights is None:
        member_weights = np.full(member_count, 1.0 / member_count)
    else:
        member_weights = np.asarray(weights, dtype=np.float64)
        _require(member_weights.shape == (member_count,), "ensemble weights differ")
        _require(
            np.all(member_weights >= 0.0)
            and np.isclose(np.sum(member_weights), 1.0),
            "ensemble weights are invalid",
        )
    reshape = (member_count,) + (1,) * (member_means.ndim - 1)
    expanded_weights = member_weights.reshape(reshape)
    mean = np.sum(expanded_weights * member_means, axis=0)
    variance = np.sum(
        expanded_weights
        * (member_variances + np.square(member_means - mean[None])),
        axis=0,
    )
    return mean, variance


def inflate_variance_for_action_shift(
    variance: np.ndarray,
    action_distance: np.ndarray | float,
    inflation_rate: float,
) -> np.ndarray:
    """Widen a residual posterior as a counterfactual action leaves support."""

    values = np.asarray(variance, dtype=np.float64)
    distance = np.asarray(action_distance, dtype=np.float64)
    _require(np.all(values >= 0.0), "variance is negative")
    _require(np.all(distance >= 0.0), "action distance is negative")
    _require(np.isfinite(inflation_rate) and inflation_rate >= 0.0, "invalid inflation")
    return values * (1.0 + inflation_rate * np.square(distance))


def residual_acceptance_mask(
    utility_probability: np.ndarray,
    predictive_variance_m2: np.ndarray,
    action_distance: np.ndarray,
    *,
    minimum_utility_probability: float,
    maximum_variance_m2: float,
    maximum_action_distance: float,
) -> np.ndarray:
    """Apply source-calibrated residual admission criteria."""

    probability = np.asarray(utility_probability, dtype=np.float64)
    variance = np.asarray(predictive_variance_m2, dtype=np.float64)
    distance = np.asarray(action_distance, dtype=np.float64)
    _require(probability.shape == variance.shape == distance.shape, "gate shapes differ")
    return (
        (probability >= minimum_utility_probability)
        & (variance <= maximum_variance_m2)
        & (distance <= maximum_action_distance)
    )


def apply_exact_residual_fallback(
    physics_prediction: np.ndarray,
    residual_prediction: np.ndarray,
    accepted: np.ndarray,
) -> np.ndarray:
    """Return exact physics values wherever the residual gate abstains."""

    physics = np.asarray(physics_prediction)
    residual = np.asarray(residual_prediction)
    mask = np.asarray(accepted, dtype=bool)
    _require(physics.shape == residual.shape, "prediction shapes differ")
    while mask.ndim < physics.ndim:
        mask = mask[..., None]
    _require(mask.shape[:-1] == physics.shape[:-1], "acceptance shape differs")
    return np.where(mask, residual, physics)


@dataclass(frozen=True)
class BayesianResidualModelConfig:
    hidden_dim: int = 64
    message_steps: int = 3
    maximum_residual_speed_mps: float = 0.30
    minimum_variance_m2ps2: float = 1.0e-8
    maximum_variance_m2ps2: float = 0.25


if nn is not None:

    @dataclass(frozen=True)
    class ResidualVelocityDistribution:
        mean_mps: torch.Tensor
        aleatoric_variance_m2ps2: torch.Tensor
        utility_probability: torch.Tensor


    def _unit_vector(value: torch.Tensor, epsilon: float = 1.0e-8) -> torch.Tensor:
        norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
        return value / torch.clamp(norm, min=epsilon)


    class EquivariantBayesianResidual(nn.Module):
        """Small O(3)-equivariant residual head with heteroscedastic output."""

        def __init__(self, config: BayesianResidualModelConfig | None = None) -> None:
            super().__init__()
            self.config = config or BayesianResidualModelConfig()
            hidden = self.config.hidden_dim
            _require(hidden >= 8, "residual hidden dimension is too small")
            _require(self.config.message_steps >= 1, "message steps must be positive")
            self.node_encoder = nn.Sequential(
                nn.Linear(7, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            self.message_mlp = nn.Sequential(
                nn.Linear(2 * hidden + 5, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            self.vector_coefficients = nn.Linear(hidden, 3)
            self.node_update = nn.Sequential(
                nn.Linear(2 * hidden + 1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.output_coefficients = nn.Linear(hidden, 6)
            self.log_variance = nn.Linear(hidden, 1)
            self.utility_logit = nn.Linear(hidden, 1)

        def forward(
            self,
            *,
            positions_m: torch.Tensor,
            velocities_mps: torch.Tensor,
            physics_positions_m: torch.Tensor,
            physics_velocities_mps: torch.Tensor,
            controller_positions_m: torch.Tensor,
            controller_velocities_mps: torch.Tensor,
            contact_probabilities: torch.Tensor,
            prior_reliability: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> ResidualVelocityDistribution:
            positions = positions_m
            velocities = velocities_mps
            physics_positions = physics_positions_m
            physics_velocities = physics_velocities_mps
            _require(positions.ndim == 3 and positions.shape[-1] == 3, "invalid positions")
            _require(velocities.shape == positions.shape, "invalid velocities")
            _require(physics_positions.shape == positions.shape, "invalid physics positions")
            _require(physics_velocities.shape == positions.shape, "invalid physics velocities")
            batch, node_count, _ = positions.shape
            _require(
                controller_positions_m.ndim == 3
                and controller_positions_m.shape[0] == batch
                and controller_positions_m.shape[-1] == 3,
                "invalid controller positions",
            )
            _require(
                controller_velocities_mps.shape == controller_positions_m.shape,
                "invalid controller velocities",
            )
            controller_count = controller_positions_m.shape[1]
            _require(
                contact_probabilities.shape == (batch, node_count, controller_count),
                "invalid contact probabilities",
            )
            _require(
                prior_reliability.shape == (batch, node_count),
                "invalid prior reliability",
            )
            _require(edge_index.shape[0] == 2, "edge index must have shape (2,E)")

            physics_delta = physics_positions - positions
            controller_offsets = (
                controller_positions_m[:, None] - positions[:, :, None]
            )
            controller_velocities = controller_velocities_mps[:, None].expand(
                -1, node_count, -1, -1
            )
            contact_mass = torch.amax(contact_probabilities, dim=-1, keepdim=True)
            denominator = torch.clamp(
                torch.sum(contact_probabilities, dim=-1, keepdim=True), min=1.0e-8
            )
            weighted_offset = torch.sum(
                contact_probabilities[..., None] * controller_offsets, dim=2
            ) / denominator
            weighted_controller_velocity = torch.sum(
                contact_probabilities[..., None] * controller_velocities, dim=2
            ) / denominator
            active = contact_mass > 0.0
            weighted_offset = torch.where(active, weighted_offset, torch.zeros_like(weighted_offset))
            weighted_controller_velocity = torch.where(
                active,
                weighted_controller_velocity,
                torch.zeros_like(weighted_controller_velocity),
            )

            scalar_features = torch.cat(
                (
                    torch.linalg.vector_norm(velocities, dim=-1, keepdim=True),
                    torch.linalg.vector_norm(physics_velocities, dim=-1, keepdim=True),
                    torch.linalg.vector_norm(physics_delta, dim=-1, keepdim=True),
                    contact_mass,
                    torch.linalg.vector_norm(weighted_offset, dim=-1, keepdim=True),
                    torch.linalg.vector_norm(
                        weighted_controller_velocity, dim=-1, keepdim=True
                    ),
                    prior_reliability[..., None],
                ),
                dim=-1,
            )
            hidden = self.node_encoder(scalar_features)
            source = edge_index[0].long()
            target = edge_index[1].long()
            _require(
                bool(torch.all(source >= 0))
                and bool(torch.all(target >= 0))
                and bool(torch.all(source < node_count))
                and bool(torch.all(target < node_count)),
                "edge index is out of bounds",
            )
            aggregated_vector = torch.zeros_like(positions)
            for _ in range(self.config.message_steps):
                relative_position = positions[:, source] - positions[:, target]
                relative_velocity = velocities[:, source] - velocities[:, target]
                relative_physics = physics_delta[:, source] - physics_delta[:, target]
                direction = _unit_vector(relative_position)
                edge_invariants = torch.cat(
                    (
                        torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True),
                        torch.linalg.vector_norm(relative_velocity, dim=-1, keepdim=True),
                        torch.sum(relative_velocity * direction, dim=-1, keepdim=True),
                        torch.linalg.vector_norm(relative_physics, dim=-1, keepdim=True),
                        torch.sum(relative_physics * direction, dim=-1, keepdim=True),
                    ),
                    dim=-1,
                )
                message = self.message_mlp(
                    torch.cat((hidden[:, target], hidden[:, source], edge_invariants), dim=-1)
                )
                coefficients = torch.tanh(self.vector_coefficients(message))
                vector_message = (
                    coefficients[..., 0:1] * direction
                    + coefficients[..., 1:2] * _unit_vector(relative_velocity)
                    + coefficients[..., 2:3] * _unit_vector(relative_physics)
                )
                aggregated_scalar = torch.zeros_like(hidden)
                aggregated_vector = torch.zeros_like(positions)
                aggregated_scalar.index_add_(1, target, message)
                aggregated_vector.index_add_(1, target, vector_message)
                degree = torch.zeros(
                    node_count, dtype=positions.dtype, device=positions.device
                )
                degree.index_add_(0, target, torch.ones_like(target, dtype=positions.dtype))
                degree = torch.clamp(degree, min=1.0)
                aggregated_scalar = aggregated_scalar / degree[None, :, None]
                aggregated_vector = aggregated_vector / degree[None, :, None]
                update = self.node_update(
                    torch.cat(
                        (
                            hidden,
                            aggregated_scalar,
                            torch.linalg.vector_norm(
                                aggregated_vector, dim=-1, keepdim=True
                            ),
                        ),
                        dim=-1,
                    )
                )
                hidden = hidden + update

            bases = torch.stack(
                (
                    _unit_vector(physics_delta),
                    _unit_vector(velocities),
                    _unit_vector(physics_velocities),
                    _unit_vector(weighted_offset),
                    _unit_vector(weighted_controller_velocity),
                    _unit_vector(aggregated_vector),
                ),
                dim=-2,
            )
            output_coefficients = torch.tanh(self.output_coefficients(hidden))
            mean = self.config.maximum_residual_speed_mps * torch.sum(
                output_coefficients[..., None] * bases, dim=-2
            ) / 6.0
            raw_log_variance = self.log_variance(hidden).squeeze(-1)
            minimum = self.config.minimum_variance_m2ps2
            maximum = self.config.maximum_variance_m2ps2
            variance = minimum + (maximum - minimum) * torch.sigmoid(
                raw_log_variance
            )
            utility = torch.sigmoid(self.utility_logit(hidden).squeeze(-1))
            return ResidualVelocityDistribution(
                mean_mps=mean,
                aleatoric_variance_m2ps2=variance,
                utility_probability=utility,
            )


    @dataclass(frozen=True)
    class TemporalResidualState:
        """Per-node history carried by the causal temporal residual."""

        hidden: torch.Tensor
        previous_mean_mps: torch.Tensor


    @dataclass(frozen=True)
    class TemporalBayesianResidualModelConfig:
        hidden_dim: int = 64
        message_steps: int = 3
        temporal_hidden_dim: int = 64
        maximum_residual_speed_mps: float = 0.30
        maximum_temporal_correction_mps: float = 0.15
        minimum_variance_m2ps2: float = 1.0e-8
        maximum_variance_m2ps2: float = 0.25


    class TemporalEquivariantBayesianResidual(nn.Module):
        """Causal temporal refinement of an equivariant instantaneous residual.

        The recurrent state contains only invariant scalar features and a previous
        equivariant vector prediction. Rotating every physical vector, including
        gravity, therefore rotates the output without changing its scalar trust
        or uncertainty. The temporal correction heads start at zero so this model
        initially behaves like a smoothed instantaneous residual.
        """

        _TEMPORAL_FEATURE_COUNT = 11

        def __init__(
            self,
            config: TemporalBayesianResidualModelConfig | None = None,
        ) -> None:
            super().__init__()
            self.config = config or TemporalBayesianResidualModelConfig()
            _require(
                self.config.temporal_hidden_dim >= 8,
                "temporal hidden dimension is too small",
            )
            _require(
                0.0 < self.config.maximum_temporal_correction_mps
                <= self.config.maximum_residual_speed_mps,
                "temporal correction speed is invalid",
            )
            instantaneous_config = BayesianResidualModelConfig(
                hidden_dim=self.config.hidden_dim,
                message_steps=self.config.message_steps,
                maximum_residual_speed_mps=self.config.maximum_residual_speed_mps,
                minimum_variance_m2ps2=self.config.minimum_variance_m2ps2,
                maximum_variance_m2ps2=self.config.maximum_variance_m2ps2,
            )
            self.instantaneous = EquivariantBayesianResidual(instantaneous_config)
            temporal_hidden = self.config.temporal_hidden_dim
            self.temporal_encoder = nn.Sequential(
                nn.Linear(self._TEMPORAL_FEATURE_COUNT, temporal_hidden),
                nn.SiLU(),
                nn.Linear(temporal_hidden, temporal_hidden),
                nn.SiLU(),
            )
            self.temporal_cell = nn.GRUCell(temporal_hidden, temporal_hidden)
            self.instantaneous_gate = nn.Linear(temporal_hidden, 1)
            self.vector_coefficients = nn.Linear(temporal_hidden, 4)
            self.log_variance_scale = nn.Linear(temporal_hidden, 1)
            self.utility_logit_delta = nn.Linear(temporal_hidden, 1)
            self._initialize_temporal_heads()

        def _initialize_temporal_heads(self) -> None:
            nn.init.zeros_(self.instantaneous_gate.weight)
            nn.init.constant_(self.instantaneous_gate.bias, 2.0)
            for layer in (
                self.vector_coefficients,
                self.log_variance_scale,
                self.utility_logit_delta,
            ):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        def initial_state(
            self,
            *,
            batch_size: int,
            node_count: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> TemporalResidualState:
            _require(batch_size >= 1 and node_count >= 1, "invalid temporal state size")
            hidden = torch.zeros(
                batch_size,
                node_count,
                self.config.temporal_hidden_dim,
                device=device,
                dtype=dtype,
            )
            previous = torch.zeros(
                batch_size,
                node_count,
                3,
                device=device,
                dtype=dtype,
            )
            return TemporalResidualState(hidden=hidden, previous_mean_mps=previous)

        @staticmethod
        def detach_state(state: TemporalResidualState) -> TemporalResidualState:
            return TemporalResidualState(
                hidden=state.hidden.detach(),
                previous_mean_mps=state.previous_mean_mps.detach(),
            )

        def forward(
            self,
            *,
            positions_m: torch.Tensor,
            velocities_mps: torch.Tensor,
            physics_positions_m: torch.Tensor,
            physics_velocities_mps: torch.Tensor,
            controller_positions_m: torch.Tensor,
            controller_velocities_mps: torch.Tensor,
            contact_probabilities: torch.Tensor,
            prior_reliability: torch.Tensor,
            edge_index: torch.Tensor,
            temporal_state: TemporalResidualState | None = None,
            gravity_direction: torch.Tensor | None = None,
        ) -> tuple[ResidualVelocityDistribution, TemporalResidualState]:
            instantaneous = self.instantaneous(
                positions_m=positions_m,
                velocities_mps=velocities_mps,
                physics_positions_m=physics_positions_m,
                physics_velocities_mps=physics_velocities_mps,
                controller_positions_m=controller_positions_m,
                controller_velocities_mps=controller_velocities_mps,
                contact_probabilities=contact_probabilities,
                prior_reliability=prior_reliability,
                edge_index=edge_index,
            )
            batch_size, node_count, _ = positions_m.shape
            if temporal_state is None:
                temporal_state = self.initial_state(
                    batch_size=batch_size,
                    node_count=node_count,
                    device=positions_m.device,
                    dtype=positions_m.dtype,
                )
            _require(
                temporal_state.hidden.shape
                == (batch_size, node_count, self.config.temporal_hidden_dim),
                "temporal hidden state shape differs",
            )
            _require(
                temporal_state.previous_mean_mps.shape == positions_m.shape,
                "previous residual shape differs",
            )
            if gravity_direction is None:
                gravity = positions_m.new_tensor((0.0, 0.0, -1.0)).view(1, 1, 3)
            else:
                gravity = gravity_direction.to(
                    device=positions_m.device,
                    dtype=positions_m.dtype,
                )
                if gravity.ndim == 2:
                    gravity = gravity[:, None]
                _require(
                    gravity.shape in ((batch_size, 1, 3), positions_m.shape),
                    "gravity direction shape differs",
                )
            gravity = _unit_vector(gravity).expand(batch_size, node_count, 3)
            physics_delta = physics_positions_m - positions_m
            contact_mass = torch.amax(contact_probabilities, dim=-1, keepdim=True)
            variance = instantaneous.aleatoric_variance_m2ps2
            temporal_features = torch.cat(
                (
                    torch.linalg.vector_norm(
                        instantaneous.mean_mps, dim=-1, keepdim=True
                    ),
                    torch.linalg.vector_norm(velocities_mps, dim=-1, keepdim=True),
                    torch.linalg.vector_norm(
                        physics_velocities_mps, dim=-1, keepdim=True
                    ),
                    torch.linalg.vector_norm(physics_delta, dim=-1, keepdim=True),
                    torch.sum(velocities_mps * gravity, dim=-1, keepdim=True),
                    torch.sum(
                        physics_velocities_mps * gravity, dim=-1, keepdim=True
                    ),
                    contact_mass,
                    prior_reliability[..., None],
                    instantaneous.utility_probability[..., None],
                    torch.log(torch.clamp(variance, min=1.0e-12))[..., None],
                    torch.linalg.vector_norm(
                        temporal_state.previous_mean_mps,
                        dim=-1,
                        keepdim=True,
                    ),
                ),
                dim=-1,
            )
            encoded = self.temporal_encoder(temporal_features)
            hidden = self.temporal_cell(
                encoded.reshape(batch_size * node_count, -1),
                temporal_state.hidden.reshape(batch_size * node_count, -1),
            ).reshape(batch_size, node_count, -1)

            gate = torch.sigmoid(self.instantaneous_gate(hidden))
            smoothed = (
                gate * instantaneous.mean_mps
                + (1.0 - gate) * temporal_state.previous_mean_mps
            )
            vector_bases = torch.stack(
                (
                    _unit_vector(physics_delta),
                    _unit_vector(velocities_mps),
                    _unit_vector(physics_velocities_mps),
                    gravity,
                ),
                dim=-2,
            )
            coefficients = torch.tanh(self.vector_coefficients(hidden))
            temporal_correction = (
                self.config.maximum_temporal_correction_mps
                * torch.sum(coefficients[..., None] * vector_bases, dim=-2)
                / vector_bases.shape[-2]
            )
            mean = smoothed + temporal_correction
            speed = torch.linalg.vector_norm(mean, dim=-1, keepdim=True)
            mean = mean * torch.clamp(
                self.config.maximum_residual_speed_mps
                / torch.clamp(speed, min=1.0e-12),
                max=1.0,
            )

            log_scale = torch.clamp(
                self.log_variance_scale(hidden).squeeze(-1), min=-4.0, max=4.0
            )
            refined_variance = torch.clamp(
                variance * torch.exp(log_scale),
                min=self.config.minimum_variance_m2ps2,
                max=self.config.maximum_variance_m2ps2,
            )
            probability = torch.clamp(
                instantaneous.utility_probability, min=1.0e-6, max=1.0 - 1.0e-6
            )
            utility_logit = torch.logit(probability) + self.utility_logit_delta(
                hidden
            ).squeeze(-1)
            refined = ResidualVelocityDistribution(
                mean_mps=mean,
                aleatoric_variance_m2ps2=refined_variance,
                utility_probability=torch.sigmoid(utility_logit),
            )
            return refined, TemporalResidualState(
                hidden=hidden,
                previous_mean_mps=mean,
            )


    def clustered_student_t_nll(
        target_residual_mps: torch.Tensor,
        prediction: ResidualVelocityDistribution,
        prior_reliability: torch.Tensor,
        cluster_ids: torch.Tensor,
        *,
        degrees_of_freedom: float = 4.0,
    ) -> torch.Tensor:
        """Use the innovation once while discounting correlated observations."""

        _require(target_residual_mps.shape == prediction.mean_mps.shape, "target shape differs")
        _require(
            prior_reliability.shape == prediction.aleatoric_variance_m2ps2.shape,
            "reliability shape differs",
        )
        _require(cluster_ids.shape == prior_reliability.shape, "cluster shape differs")
        _require(degrees_of_freedom > 2.0, "Student-t degrees of freedom are invalid")
        weights = torch.zeros_like(prior_reliability)
        for batch_index in range(len(weights)):
            _, inverse, counts = torch.unique(
                cluster_ids[batch_index], return_inverse=True, return_counts=True
            )
            effective = prior_reliability[batch_index] / counts[inverse].to(
                prior_reliability.dtype
            )
            total = torch.sum(effective)
            if bool(total <= 0.0):
                raise ValueError("effective reliability has zero mass")
            weights[batch_index] = effective / total
        variance = prediction.aleatoric_variance_m2ps2
        innovation = target_residual_mps - prediction.mean_mps
        mahalanobis = torch.sum(torch.square(innovation), dim=-1) / variance
        dimension = target_residual_mps.shape[-1]
        nll = 0.5 * dimension * torch.log(variance) + 0.5 * (
            degrees_of_freedom + dimension
        ) * torch.log1p(mahalanobis / degrees_of_freedom)
        return torch.mean(torch.sum(weights * nll, dim=-1))


else:

    class EquivariantBayesianResidual:  # pragma: no cover - minimal install guard.
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError(
                "EquivariantBayesianResidual requires the optional torch dependency"
            )


    class TemporalEquivariantBayesianResidual:  # pragma: no cover
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError(
                "TemporalEquivariantBayesianResidual requires the optional torch "
                "dependency"
            )


__all__ = [
    "BAYESIAN_RESIDUAL_PROTOCOL_ID",
    "BayesianResidualModelConfig",
    "EquivariantBayesianResidual",
    "TemporalEquivariantBayesianResidual",
    "apply_exact_residual_fallback",
    "bayesian_residual_config_sha256",
    "clustered_effective_weights",
    "inflate_variance_for_action_shift",
    "load_bayesian_residual_config",
    "moment_match_residual_ensemble",
    "residual_acceptance_mask",
    "validate_bayesian_residual_config",
]

if nn is not None:
    __all__.extend(
        (
            "ResidualVelocityDistribution",
            "TemporalBayesianResidualModelConfig",
            "TemporalResidualState",
            "clustered_student_t_nll",
        )
    )
