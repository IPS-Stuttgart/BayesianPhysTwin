"""E(3)-equivariant generalized-force corrections for PhysTwin rollouts.

The model predicts scalar coefficients for vector bases that rotate with the
physical state. Internal edge messages are antisymmetric, while body/contact
terms are explicitly gated by supplied physical support. A zero admission
weight returns exact zero and therefore leaves the official simulator path
unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


EQUIVARIANT_FORCE_CONTRACT = "phystwin-equivariant-generalized-force-v2"


@dataclass(frozen=True)
class EquivariantForceConfig:
    """Architecture and physical bounds for the generalized-force model."""

    hidden_dim: int = 64
    hidden_layers: int = 2
    latent_dim: int = 8
    regime_dim: int = 5
    maximum_normalized_force: float = 1.0
    velocity_scale_mps: float = 0.10
    displacement_scale_m: float = 0.05
    minimum_length_m: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.hidden_dim < 1 or self.hidden_layers < 1:
            raise ValueError("hidden dimensions must be positive")
        if self.latent_dim < 0 or self.regime_dim < 1:
            raise ValueError("latent_dim and regime_dim are invalid")
        positive = (
            self.maximum_normalized_force,
            self.velocity_scale_mps,
            self.displacement_scale_m,
            self.minimum_length_m,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("physical force scales must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonicalize_force_edges(
    edges: np.ndarray,
    *,
    num_nodes: int,
) -> np.ndarray:
    """Return sorted unique undirected edges with the lower endpoint first."""

    values = np.asarray(edges, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("edges must have shape (E, 2)")
    if num_nodes < 1 or np.any(values < 0) or np.any(values >= num_nodes):
        raise ValueError("edge endpoints must index the supplied nodes")
    if np.any(values[:, 0] == values[:, 1]):
        raise ValueError("self edges are not valid force interactions")
    ordered = np.sort(values, axis=1)
    unique = np.unique(ordered, axis=0)
    order = np.lexsort((unique[:, 1], unique[:, 0]))
    result = np.ascontiguousarray(unique[order], dtype=np.int64)
    result.setflags(write=False)
    return result


def force_rest_lengths(
    rest_positions_m: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Compute finite positive rest lengths for a canonical force graph."""

    positions = np.asarray(rest_positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("rest_positions_m must have shape (N, 3)")
    canonical = canonicalize_force_edges(edges, num_nodes=len(positions))
    lengths = np.linalg.norm(
        positions[canonical[:, 1]] - positions[canonical[:, 0]],
        axis=1,
    )
    if not np.all(np.isfinite(positions)) or np.any(lengths <= 0.0):
        raise ValueError("rest geometry must be finite with positive edge lengths")
    result = np.ascontiguousarray(lengths)
    result.setflags(write=False)
    return result


def _make_mlp(torch: Any, input_dim: int, output_dim: int, config: EquivariantForceConfig):
    layers: list[Any] = []
    current = input_dim
    for _ in range(config.hidden_layers):
        layers.extend((torch.nn.Linear(current, config.hidden_dim), torch.nn.SiLU()))
        current = config.hidden_dim
    output = torch.nn.Linear(current, output_dim)
    torch.nn.init.zeros_(output.weight)
    torch.nn.init.zeros_(output.bias)
    layers.append(output)
    return torch.nn.Sequential(*layers)


def _batched_vector(torch: Any, value: Any, *, batch: int, nodes: int, name: str):
    tensor = torch.as_tensor(value)
    if tensor.ndim == 2 and tuple(tensor.shape) == (nodes, 3):
        tensor = tensor.unsqueeze(0).expand(batch, -1, -1)
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (nodes, 3):
        raise ValueError(f"{name} must have shape (N, 3) or (B, N, 3)")
    if tensor.shape[0] != batch:
        raise ValueError(f"{name} batch dimension is inconsistent")
    return tensor


def _batched_node_scalar(
    torch: Any,
    value: Any,
    *,
    batch: int,
    nodes: int,
    name: str,
):
    tensor = torch.as_tensor(value)
    if tensor.ndim == 1 and tensor.shape[0] == nodes:
        tensor = tensor.unsqueeze(0).expand(batch, -1)
    if tensor.ndim != 2 or tuple(tensor.shape) != (batch, nodes):
        raise ValueError(f"{name} must have shape (N,) or (B, N)")
    return tensor


def _batched_global(
    torch: Any,
    value: Any,
    *,
    batch: int,
    width: int,
    name: str,
):
    tensor = torch.as_tensor(value)
    if tensor.ndim == 1 and tensor.shape[0] == width:
        tensor = tensor.unsqueeze(0).expand(batch, -1)
    if tensor.ndim != 2 or tuple(tensor.shape) != (batch, width):
        raise ValueError(f"{name} must have shape ({width},) or (B, {width})")
    return tensor


def _unit(torch: Any, value: Any, epsilon: float):
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value / torch.clamp(norm, min=epsilon), norm


def build_equivariant_force_model(
    torch: Any,
    config: EquivariantForceConfig | None = None,
):
    """Build a Torch module whose output rotates equivariantly with its inputs."""

    selected = config or EquivariantForceConfig()
    edge_scalar_dim = 9 + selected.regime_dim + selected.latent_dim
    node_scalar_dim = 9 + selected.regime_dim + selected.latent_dim

    class EquivariantGeneralizedForce(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = selected
            self.edge_head = _make_mlp(torch, edge_scalar_dim, 3, selected)
            self.node_head = _make_mlp(torch, node_scalar_dim, 4, selected)

        def forward(
            self,
            positions_m,
            velocities_mps,
            rest_positions_m,
            edges,
            rest_lengths_m,
            control_displacement_m,
            control_velocity_mps,
            action_support,
            external_support,
            gravity_mps2,
            force_scale_sim,
            action_activity,
            regime_probabilities,
            latent,
            admission_weight=1.0,
        ):
            positions = torch.as_tensor(positions_m)
            unbatched = positions.ndim == 2
            if unbatched:
                positions = positions.unsqueeze(0)
            if positions.ndim != 3 or positions.shape[-1] != 3:
                raise ValueError("positions_m must have shape (N, 3) or (B, N, 3)")
            batch, nodes, _ = positions.shape
            dtype = positions.dtype
            device = positions.device

            def vector(value, name):
                return _batched_vector(
                    torch,
                    value,
                    batch=batch,
                    nodes=nodes,
                    name=name,
                ).to(device=device, dtype=dtype)

            velocities = vector(velocities_mps, "velocities_mps")
            rest = vector(rest_positions_m, "rest_positions_m")
            control_delta = vector(
                control_displacement_m, "control_displacement_m"
            )
            control_velocity = vector(control_velocity_mps, "control_velocity_mps")
            action_node_support = _batched_node_scalar(
                torch,
                action_support,
                batch=batch,
                nodes=nodes,
                name="action_support",
            ).to(device=device, dtype=dtype)
            external_node_support = _batched_node_scalar(
                torch,
                external_support,
                batch=batch,
                nodes=nodes,
                name="external_support",
            ).to(device=device, dtype=dtype)
            if bool(
                torch.any((action_node_support < 0.0) | (action_node_support > 1.0))
            ) or bool(
                torch.any(
                    (external_node_support < 0.0)
                    | (external_node_support > 1.0)
                )
            ):
                raise ValueError("physical support values must lie in [0, 1]")

            edge_index = torch.as_tensor(edges, dtype=torch.long, device=device)
            if edge_index.ndim != 2 or edge_index.shape[1] != 2:
                raise ValueError("edges must have shape (E, 2)")
            if bool(torch.any(edge_index < 0)) or bool(torch.any(edge_index >= nodes)):
                raise ValueError("edge endpoints are outside the node array")
            if bool(torch.any(edge_index[:, 0] == edge_index[:, 1])):
                raise ValueError("self edges are not valid force interactions")
            first, second = edge_index[:, 0], edge_index[:, 1]
            lengths0 = torch.as_tensor(
                rest_lengths_m, dtype=dtype, device=device
            ).reshape(-1)
            if lengths0.shape[0] != edge_index.shape[0] or bool(
                torch.any(lengths0 <= 0.0)
            ):
                raise ValueError("rest_lengths_m must be a positive edge vector")

            gravity = _batched_global(
                torch,
                gravity_mps2,
                batch=batch,
                width=3,
                name="gravity_mps2",
            ).to(device=device, dtype=dtype)
            force_scale = torch.as_tensor(
                force_scale_sim,
                dtype=dtype,
                device=device,
            ).reshape(-1)
            if force_scale.numel() == 1:
                force_scale = force_scale.expand(batch)
            if force_scale.shape[0] != batch or bool(
                torch.any(~torch.isfinite(force_scale))
                | torch.any(force_scale <= 0.0)
            ):
                raise ValueError(
                    "force_scale_sim must provide B finite positive values"
                )
            regime = _batched_global(
                torch,
                regime_probabilities,
                batch=batch,
                width=selected.regime_dim,
                name="regime_probabilities",
            ).to(device=device, dtype=dtype)
            latent_values = _batched_global(
                torch,
                latent,
                batch=batch,
                width=selected.latent_dim,
                name="latent",
            ).to(device=device, dtype=dtype)
            if bool(torch.any(regime < 0.0)) or not bool(
                torch.allclose(
                    torch.sum(regime, dim=1),
                    torch.ones(batch, dtype=dtype, device=device),
                    atol=1.0e-5,
                    rtol=1.0e-5,
                )
            ):
                raise ValueError("regime probabilities must be simplex-valued")
            activity = torch.as_tensor(
                action_activity, dtype=dtype, device=device
            ).reshape(-1)
            if activity.numel() == 1:
                activity = activity.expand(batch)
            if activity.shape[0] != batch or bool(
                torch.any((activity < 0.0) | (activity > 1.0))
            ):
                raise ValueError("action_activity must provide B values in [0, 1]")

            displacement = positions[:, second] - positions[:, first]
            rest_displacement = rest[:, second] - rest[:, first]
            relative_velocity = velocities[:, second] - velocities[:, first]
            direction, length = _unit(
                torch, displacement, selected.minimum_length_m
            )
            rest_direction, _ = _unit(
                torch, rest_displacement, selected.minimum_length_m
            )
            radial_speed = torch.sum(relative_velocity * direction, dim=-1)
            tangential_velocity = (
                relative_velocity - radial_speed.unsqueeze(-1) * direction
            )
            _, tangential_speed = _unit(
                torch, tangential_velocity, selected.minimum_length_m
            )
            alignment = torch.sum(direction * rest_direction, dim=-1)
            strain = length.squeeze(-1) / lengths0.unsqueeze(0) - 1.0
            edge_activity = activity[:, None].expand(-1, len(first))
            edge_context = torch.cat(
                (
                    regime[:, None].expand(-1, len(first), -1),
                    latent_values[:, None].expand(-1, len(first), -1),
                ),
                dim=-1,
            )
            edge_features = torch.cat(
                (
                    strain.unsqueeze(-1),
                    (radial_speed / selected.velocity_scale_mps).unsqueeze(-1),
                    (
                        tangential_speed.squeeze(-1)
                        / selected.velocity_scale_mps
                    ).unsqueeze(-1),
                    alignment.unsqueeze(-1),
                    action_node_support[:, first].unsqueeze(-1),
                    action_node_support[:, second].unsqueeze(-1),
                    external_node_support[:, first].unsqueeze(-1),
                    external_node_support[:, second].unsqueeze(-1),
                    edge_activity.unsqueeze(-1),
                    edge_context,
                ),
                dim=-1,
            )
            edge_coefficients = torch.tanh(self.edge_head(edge_features))
            edge_bases = torch.stack(
                (
                    direction,
                    tangential_velocity / selected.velocity_scale_mps,
                    direction - rest_direction,
                ),
                dim=-2,
            )
            edge_force = (
                force_scale[:, None, None]
                * selected.maximum_normalized_force
                * torch.sum(edge_coefficients[..., None] * edge_bases, dim=-2)
            )
            internal_force = torch.zeros_like(positions)
            internal_force.index_add_(1, first, edge_force)
            internal_force.index_add_(1, second, -edge_force)

            gravity_direction, gravity_norm = _unit(
                torch, gravity, selected.minimum_length_m
            )
            velocity_direction, speed = _unit(
                torch, velocities, selected.minimum_length_m
            )
            control_direction, control_distance = _unit(
                torch, control_delta, selected.minimum_length_m
            )
            control_velocity_direction, control_speed = _unit(
                torch, control_velocity, selected.minimum_length_m
            )
            gravity_nodes = gravity_direction[:, None].expand(-1, nodes, -1)
            node_context = torch.cat(
                (
                    regime[:, None].expand(-1, nodes, -1),
                    latent_values[:, None].expand(-1, nodes, -1),
                ),
                dim=-1,
            )
            node_features = torch.cat(
                (
                    (speed / selected.velocity_scale_mps),
                    (
                        torch.linalg.vector_norm(positions - rest, dim=-1, keepdim=True)
                        / selected.displacement_scale_m
                    ),
                    (control_distance / selected.displacement_scale_m),
                    (control_speed / selected.velocity_scale_mps),
                    torch.sum(
                        velocity_direction * gravity_nodes, dim=-1, keepdim=True
                    ),
                    torch.sum(
                        control_direction * gravity_nodes, dim=-1, keepdim=True
                    ),
                    action_node_support.unsqueeze(-1),
                    external_node_support.unsqueeze(-1),
                    activity[:, None, None].expand(-1, nodes, -1),
                    node_context,
                ),
                dim=-1,
            )
            node_coefficients = torch.tanh(self.node_head(node_features))
            control_gate = (
                action_node_support * activity[:, None]
            ).unsqueeze(-1)
            external_gate = external_node_support.unsqueeze(-1)
            node_bases = torch.stack(
                (
                    gravity_nodes * external_gate,
                    control_direction * control_gate,
                    control_velocity_direction * control_gate,
                    velocity_direction * activity[:, None, None],
                ),
                dim=-2,
            )
            external_force = (
                force_scale[:, None, None]
                * selected.maximum_normalized_force
                * torch.sum(node_coefficients[..., None] * node_bases, dim=-2)
            )
            force = internal_force + external_force
            force_norm = torch.linalg.vector_norm(force, dim=-1, keepdim=True)
            cap = (
                force_scale[:, None, None]
                * selected.maximum_normalized_force
            )
            maximum_norm = torch.amax(force_norm, dim=1, keepdim=True)
            force = force * torch.clamp(
                cap
                / torch.clamp(maximum_norm, min=selected.minimum_length_m),
                max=1.0,
            )

            admission = torch.as_tensor(
                admission_weight, dtype=dtype, device=device
            ).reshape(-1)
            if admission.numel() == 1:
                admission = admission.expand(batch)
            if admission.shape[0] != batch or bool(
                torch.any((admission < 0.0) | (admission > 1.0))
            ):
                raise ValueError("admission_weight must provide B values in [0, 1]")
            force = force * admission[:, None, None]
            return force[0] if unbatched else force

    return EquivariantGeneralizedForce()


def maximum_node_force_sim(force_sim: np.ndarray) -> float:
    """Return the largest nodal force norm in native simulator units."""

    values = np.asarray(force_sim, dtype=float)
    if values.ndim < 2 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("force_sim must be a finite (..., N, 3) array")
    return float(np.max(np.linalg.norm(values, axis=-1), initial=0.0))
