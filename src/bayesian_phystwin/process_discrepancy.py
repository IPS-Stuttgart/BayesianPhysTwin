"""Dynamics-consistent latent process discrepancy for PhysTwin rollouts.

The module is deliberately opt-in.  A zero coefficient schedule is dispatched
through the unchanged baseline replay method, so adding this module cannot alter
released trajectories unless a caller explicitly enables a non-zero force field.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

PROCESS_DISCREPANCY_SCHEMA_VERSION = 1
ContactPolicy = Literal["all_supported", "contact_only", "exclude_contact"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(
    value: np.ndarray,
    *,
    dtype: Any = np.float64,
    finite: bool = True,
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    if finite:
        _require(np.all(np.isfinite(result)), "array contains non-finite values")
    result.setflags(write=False)
    return result


def _readonly_bool(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=bool).copy()
    result.setflags(write=False)
    return result


def _json_data(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _canonicalize_column_signs(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _symmetric_psd(
    value: np.ndarray,
    *,
    name: str,
    tolerance: float = 1e-10,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(
        matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1],
        f"{name} must be square",
    )
    _require(np.all(np.isfinite(matrix)), f"{name} contains non-finite values")
    symmetric = 0.5 * (matrix + matrix.T)
    scale = max(float(np.linalg.norm(symmetric, ord=2)), 1.0)
    asymmetry = float(np.max(np.abs(matrix - matrix.T), initial=0.0))
    _require(asymmetry <= tolerance * scale, f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    _require(
        float(np.min(eigenvalues, initial=0.0)) >= -tolerance * scale,
        f"{name} must be positive semidefinite",
    )
    if np.any(eigenvalues < 0.0):
        vectors = np.linalg.eigh(symmetric)[1]
        symmetric = (vectors * np.maximum(eigenvalues, 0.0)) @ vectors.T
        symmetric = 0.5 * (symmetric + symmetric.T)
    return _readonly(symmetric)


def _constraint_matrix(
    reference_positions_m: np.ndarray,
    *,
    origin_m: np.ndarray,
    enforce_zero_net_force: bool,
    enforce_zero_net_torque: bool,
) -> np.ndarray:
    node_count = len(reference_positions_m)
    rows: list[np.ndarray] = []
    if enforce_zero_net_force:
        for coordinate in range(3):
            row = np.zeros(3 * node_count, dtype=np.float64)
            row[coordinate::3] = 1.0
            rows.append(row)
    if enforce_zero_net_torque:
        torque = np.zeros((3, 3 * node_count), dtype=np.float64)
        for node, position in enumerate(reference_positions_m - origin_m):
            x_value, y_value, z_value = position
            cross = np.asarray(
                (
                    (0.0, -z_value, y_value),
                    (z_value, 0.0, -x_value),
                    (-y_value, x_value, 0.0),
                ),
                dtype=np.float64,
            )
            torque[:, 3 * node : 3 * node + 3] = cross
        rows.extend(torque)
    if not rows:
        return np.empty((0, 3 * node_count), dtype=np.float64)
    return np.stack(rows)


def _nullspace(matrix: np.ndarray, *, tolerance: float) -> tuple[np.ndarray, int]:
    values = np.asarray(matrix, dtype=np.float64)
    column_count = values.shape[1]
    if values.shape[0] == 0:
        return np.eye(column_count, dtype=np.float64), 0
    _, singular_values, right = np.linalg.svd(values, full_matrices=True)
    scale = float(singular_values[0]) if len(singular_values) else 1.0
    threshold = tolerance * max(values.shape) * max(scale, 1.0)
    rank = int(np.sum(singular_values > threshold))
    return right[rank:].T.copy(), rank


@dataclass(frozen=True)
class DynamicsConsistentForceBasis:
    """Low-rank nodal-force basis with explicit support and conservation rules.

    ``force_basis_per_coefficient`` has shape ``(node, 3, coefficient)``.  Its
    flattened columns are orthonormal, so the latent coefficient unit is the
    Newton and coefficient covariance has units of Newton squared.
    """

    graph_basis: np.ndarray
    force_basis_per_coefficient: np.ndarray
    reference_positions_m: np.ndarray
    active_node_mask: np.ndarray
    contact_node_mask: np.ndarray
    attached_node_mask: np.ndarray
    constraint_origin_m: np.ndarray
    contact_policy: ContactPolicy
    enforce_zero_net_force: bool
    enforce_zero_net_torque: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        graph_basis = _readonly(self.graph_basis)
        force_basis = _readonly(self.force_basis_per_coefficient)
        positions = _readonly(self.reference_positions_m)
        active = _readonly_bool(self.active_node_mask)
        contact = _readonly_bool(self.contact_node_mask)
        attached = _readonly_bool(self.attached_node_mask)
        origin = _readonly(self.constraint_origin_m)
        _require(
            graph_basis.ndim == 2 and graph_basis.shape[1] >= 1,
            "graph_basis must have shape (node, rank>=1)",
        )
        node_count = graph_basis.shape[0]
        _require(
            positions.shape == (node_count, 3),
            "reference_positions_m must have shape (node, 3)",
        )
        _require(
            np.allclose(
                graph_basis.T @ graph_basis,
                np.eye(graph_basis.shape[1]),
                atol=1e-8,
                rtol=1e-8,
            ),
            "graph_basis must be orthonormal",
        )
        _require(
            active.shape == contact.shape == attached.shape == (node_count,),
            "node masks must match graph_basis",
        )
        _require(origin.shape == (3,), "constraint_origin_m must have shape (3,)")
        _require(np.any(active), "force support must contain at least one active node")
        _require(not np.any(active & attached), "attached nodes cannot be active")
        _require(
            self.contact_policy in {
                "all_supported",
                "contact_only",
                "exclude_contact",
            },
            "unsupported contact policy",
        )
        if self.contact_policy == "contact_only":
            _require(
                not np.any(active & ~contact),
                "contact_only support includes a non-contact node",
            )
        if self.contact_policy == "exclude_contact":
            _require(
                not np.any(active & contact),
                "exclude_contact support includes a contact node",
            )
        _require(
            force_basis.ndim == 3
            and force_basis.shape[:2] == (node_count, 3)
            and force_basis.shape[2] >= 1,
            "force_basis_per_coefficient must have shape (node, 3, coefficient>=1)",
        )
        _require(
            np.array_equal(
                force_basis[~active],
                np.zeros_like(force_basis[~active]),
            ),
            "force basis must be exactly zero outside active support",
        )
        flattened = force_basis.reshape(3 * node_count, force_basis.shape[2])
        _require(
            np.allclose(
                flattened.T @ flattened,
                np.eye(force_basis.shape[2]),
                atol=2e-10,
                rtol=2e-10,
            ),
            "force basis columns must be orthonormal",
        )
        residuals = self._constraint_residuals_for(force_basis)
        scale = max(float(np.max(np.abs(force_basis), initial=0.0)), 1.0)
        if self.enforce_zero_net_force:
            _require(
                float(
                    np.max(
                        np.abs(residuals["net_force_per_coefficient"]),
                        initial=0.0,
                    )
                )
                <= 5e-10 * scale,
                "force basis violates zero-net-force constraint",
            )
        if self.enforce_zero_net_torque:
            torque_scale = max(
                float(np.max(np.linalg.norm(positions - origin, axis=1), initial=0.0)),
                1.0,
            )
            _require(
                float(
                    np.max(
                        np.abs(residuals["net_torque_per_coefficient"]),
                        initial=0.0,
                    )
                )
                <= 5e-10 * scale * torque_scale,
                "force basis violates zero-net-torque constraint",
            )
        object.__setattr__(self, "graph_basis", graph_basis)
        object.__setattr__(self, "force_basis_per_coefficient", force_basis)
        object.__setattr__(self, "reference_positions_m", positions)
        object.__setattr__(self, "active_node_mask", active)
        object.__setattr__(self, "contact_node_mask", contact)
        object.__setattr__(self, "attached_node_mask", attached)
        object.__setattr__(self, "constraint_origin_m", origin)
        object.__setattr__(
            self,
            "diagnostics",
            _json_data(self.diagnostics, name="diagnostics"),
        )

    @property
    def node_count(self) -> int:
        return int(self.graph_basis.shape[0])

    @property
    def graph_rank(self) -> int:
        return int(self.graph_basis.shape[1])

    @property
    def coefficient_count(self) -> int:
        return int(self.force_basis_per_coefficient.shape[2])

    @property
    def basis_id(self) -> str:
        payload = {
            "schema_version": PROCESS_DISCREPANCY_SCHEMA_VERSION,
            "artifact_kind": "DynamicsConsistentForceBasis",
            "contact_policy": self.contact_policy,
            "enforce_zero_net_force": self.enforce_zero_net_force,
            "enforce_zero_net_torque": self.enforce_zero_net_torque,
            "diagnostics": self.diagnostics,
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        arrays = {
            "graph_basis": self.graph_basis,
            "force_basis_per_coefficient": self.force_basis_per_coefficient,
            "reference_positions_m": self.reference_positions_m,
            "active_node_mask": self.active_node_mask,
            "contact_node_mask": self.contact_node_mask,
            "attached_node_mask": self.attached_node_mask,
            "constraint_origin_m": self.constraint_origin_m,
        }
        for name, value in sorted(arrays.items()):
            digest.update(name.encode("ascii"))
            digest.update(_array_sha256(value).encode("ascii"))
        return digest.hexdigest()

    def decode(self, coefficients_n: np.ndarray) -> np.ndarray:
        """Decode latent coefficients into a nodal force field in Newtons."""

        coefficients = np.asarray(coefficients_n, dtype=np.float64)
        _require(
            coefficients.shape == (self.coefficient_count,),
            "coefficients_n must match the force-basis dimension",
        )
        _require(np.all(np.isfinite(coefficients)), "coefficients_n must be finite")
        if not np.any(coefficients != 0.0):
            return np.zeros((self.node_count, 3), dtype=np.float64)
        return np.einsum(
            "nck,k->nc",
            self.force_basis_per_coefficient,
            coefficients,
            optimize=True,
        )

    def mechanical_power_jacobian(self, velocity_mps: np.ndarray) -> np.ndarray:
        """Return ``d power / d coefficient`` in metres per second."""

        velocity = np.asarray(velocity_mps, dtype=np.float64)
        _require(
            velocity.shape == (self.node_count, 3),
            "velocity_mps must have shape (node, 3)",
        )
        _require(np.all(np.isfinite(velocity)), "velocity_mps must be finite")
        return np.einsum(
            "nc,nck->k",
            velocity,
            self.force_basis_per_coefficient,
            optimize=True,
        )

    def node_force_covariance_n2(
        self,
        coefficient_covariance_n2: np.ndarray,
    ) -> np.ndarray:
        covariance = _symmetric_psd(
            coefficient_covariance_n2,
            name="coefficient_covariance_n2",
        )
        _require(
            covariance.shape == (self.coefficient_count, self.coefficient_count),
            "coefficient covariance must match the force-basis dimension",
        )
        return np.einsum(
            "nci,ij,ndj->ncd",
            self.force_basis_per_coefficient,
            covariance,
            self.force_basis_per_coefficient,
            optimize=True,
        )

    def constraint_residuals(self, coefficients_n: np.ndarray) -> dict[str, np.ndarray]:
        return self._constraint_residuals_for(self.decode(coefficients_n)[:, :, None])

    def _constraint_residuals_for(
        self,
        force_basis: np.ndarray,
    ) -> dict[str, np.ndarray]:
        net_force = np.sum(force_basis, axis=0)
        relative = self.reference_positions_m - self.constraint_origin_m
        torque = np.cross(
            relative[:, :, None],
            force_basis,
            axisa=1,
            axisb=1,
            axisc=1,
        )
        return {
            "net_force_per_coefficient": net_force,
            "net_torque_per_coefficient": np.sum(torque, axis=0),
        }


def build_dynamics_consistent_force_basis(
    graph_basis: np.ndarray,
    reference_positions_m: np.ndarray,
    *,
    eligible_node_mask: np.ndarray | None = None,
    contact_node_mask: np.ndarray | None = None,
    attached_node_mask: np.ndarray | None = None,
    contact_policy: ContactPolicy = "all_supported",
    enforce_zero_net_force: bool = True,
    enforce_zero_net_torque: bool = True,
    constraint_origin_m: np.ndarray | None = None,
    maximum_force_rank: int | None = None,
    svd_tolerance: float = 1e-10,
) -> DynamicsConsistentForceBasis:
    """Build a support-aware constrained force basis from scalar graph modes."""

    basis = np.asarray(graph_basis, dtype=np.float64)
    positions = np.asarray(reference_positions_m, dtype=np.float64)
    _require(
        basis.ndim == 2 and basis.shape[1] >= 1,
        "graph_basis must have shape (node, rank>=1)",
    )
    node_count, graph_rank = basis.shape
    _require(
        positions.shape == (node_count, 3),
        "reference_positions_m must have shape (node, 3)",
    )
    _require(
        np.all(np.isfinite(basis)) and np.all(np.isfinite(positions)),
        "basis and positions must be finite",
    )
    _require(
        np.allclose(
            basis.T @ basis,
            np.eye(graph_rank),
            atol=1e-8,
            rtol=1e-8,
        ),
        "graph_basis must be orthonormal",
    )
    _require(svd_tolerance > 0.0, "svd_tolerance must be positive")
    if maximum_force_rank is not None:
        _require(maximum_force_rank >= 1, "maximum_force_rank must be positive")
    _require(
        contact_policy in {"all_supported", "contact_only", "exclude_contact"},
        "unsupported contact policy",
    )

    eligible = (
        np.ones(node_count, dtype=bool)
        if eligible_node_mask is None
        else np.asarray(eligible_node_mask, dtype=bool)
    )
    contact = (
        np.zeros(node_count, dtype=bool)
        if contact_node_mask is None
        else np.asarray(contact_node_mask, dtype=bool)
    )
    attached = (
        np.zeros(node_count, dtype=bool)
        if attached_node_mask is None
        else np.asarray(attached_node_mask, dtype=bool)
    )
    _require(
        eligible.shape == contact.shape == attached.shape == (node_count,),
        "node masks must match graph_basis",
    )
    active = eligible & ~attached
    if contact_policy == "contact_only":
        active &= contact
    elif contact_policy == "exclude_contact":
        active &= ~contact
    _require(np.any(active), "support policy removed every force node")

    if constraint_origin_m is None:
        origin = np.mean(positions[active], axis=0)
    else:
        origin = np.asarray(constraint_origin_m, dtype=np.float64)
        _require(
            origin.shape == (3,) and np.all(np.isfinite(origin)),
            "constraint_origin_m must be a finite 3-vector",
        )

    raw = np.zeros((3 * node_count, 3 * graph_rank), dtype=np.float64)
    for mode in range(graph_rank):
        for coordinate in range(3):
            raw[coordinate::3, 3 * mode + coordinate] = basis[:, mode]
    row_active = np.repeat(active, 3)
    raw[~row_active] = 0.0

    constraints = _constraint_matrix(
        positions,
        origin_m=origin,
        enforce_zero_net_force=enforce_zero_net_force,
        enforce_zero_net_torque=enforce_zero_net_torque,
    )
    coefficient_constraints = constraints @ raw
    nullspace, constraint_rank = _nullspace(
        coefficient_constraints,
        tolerance=svd_tolerance,
    )
    projected = raw @ nullspace
    left, singular_values, _ = np.linalg.svd(projected, full_matrices=False)
    scale = float(singular_values[0]) if len(singular_values) else 1.0
    threshold = svd_tolerance * max(projected.shape) * max(scale, 1.0)
    force_rank = int(np.sum(singular_values > threshold))
    _require(force_rank >= 1, "constraints removed the complete graph-force basis")
    if maximum_force_rank is not None:
        force_rank = min(force_rank, maximum_force_rank)
    flattened = _canonicalize_column_signs(left[:, :force_rank])
    flattened[~row_active] = 0.0
    force_basis = flattened.reshape(node_count, 3, force_rank)

    net_force = np.sum(force_basis, axis=0)
    relative = positions - origin
    net_torque = np.sum(
        np.cross(
            relative[:, :, None],
            force_basis,
            axisa=1,
            axisb=1,
            axisc=1,
        ),
        axis=0,
    )
    diagnostics = {
        "active_node_count": int(np.sum(active)),
        "attached_node_count": int(np.sum(attached)),
        "contact_node_count": int(np.sum(contact)),
        "raw_coefficient_count": int(3 * graph_rank),
        "constraint_row_count": int(len(constraints)),
        "constraint_rank": constraint_rank,
        "force_coefficient_count": force_rank,
        "maximum_basis_net_force": float(
            np.max(np.abs(net_force), initial=0.0)
        ),
        "maximum_basis_net_torque_nm": float(
            np.max(np.abs(net_torque), initial=0.0)
        ),
        "svd_tolerance": float(svd_tolerance),
    }
    return DynamicsConsistentForceBasis(
        graph_basis=basis,
        force_basis_per_coefficient=force_basis,
        reference_positions_m=positions,
        active_node_mask=active,
        contact_node_mask=contact,
        attached_node_mask=attached,
        constraint_origin_m=origin,
        contact_policy=contact_policy,
        enforce_zero_net_force=enforce_zero_net_force,
        enforce_zero_net_torque=enforce_zero_net_torque,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class StableLatentForceProcess:
    """Stable linear Gaussian coefficient process ``c[k+1] = A c[k] + w``."""

    transition_matrix: np.ndarray
    process_covariance_n2: np.ndarray
    frame_dt_s: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        transition = _readonly(self.transition_matrix)
        _require(
            transition.ndim == 2
            and transition.shape[0] == transition.shape[1]
            and transition.shape[0] >= 1,
            "transition_matrix must be square and nonempty",
        )
        covariance = _symmetric_psd(
            self.process_covariance_n2,
            name="process_covariance_n2",
        )
        _require(
            covariance.shape == transition.shape,
            "process covariance must match transition_matrix",
        )
        _require(
            self.frame_dt_s > 0.0 and np.isfinite(self.frame_dt_s),
            "frame_dt_s must be positive and finite",
        )
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
        _require(
            spectral_radius < 1.0 - 1e-12,
            "transition_matrix must be strictly stable",
        )
        object.__setattr__(self, "transition_matrix", transition)
        object.__setattr__(self, "process_covariance_n2", covariance)
        object.__setattr__(
            self,
            "metadata",
            _json_data(self.metadata, name="metadata"),
        )

    @property
    def coefficient_count(self) -> int:
        return int(self.transition_matrix.shape[0])

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.transition_matrix))))

    @property
    def process_id(self) -> str:
        payload = {
            "schema_version": PROCESS_DISCREPANCY_SCHEMA_VERSION,
            "artifact_kind": "StableLatentForceProcess",
            "frame_dt_s": self.frame_dt_s,
            "metadata": self.metadata,
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name, value in (
            ("process_covariance_n2", self.process_covariance_n2),
            ("transition_matrix", self.transition_matrix),
        ):
            digest.update(name.encode("ascii"))
            digest.update(_array_sha256(value).encode("ascii"))
        return digest.hexdigest()

    @classmethod
    def isotropic_ornstein_uhlenbeck(
        cls,
        coefficient_count: int,
        *,
        frame_dt_s: float,
        half_life_s: float,
        stationary_std_n: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> StableLatentForceProcess:
        _require(coefficient_count >= 1, "coefficient_count must be positive")
        _require(
            frame_dt_s > 0.0 and np.isfinite(frame_dt_s),
            "frame_dt_s must be positive and finite",
        )
        _require(
            half_life_s > 0.0 and np.isfinite(half_life_s),
            "half_life_s must be positive and finite",
        )
        _require(
            stationary_std_n >= 0.0 and np.isfinite(stationary_std_n),
            "stationary_std_n must be finite and nonnegative",
        )
        decay = float(np.exp(-np.log(2.0) * frame_dt_s / half_life_s))
        transition = decay * np.eye(coefficient_count, dtype=np.float64)
        variance = stationary_std_n**2
        process_covariance = variance * (1.0 - decay**2) * np.eye(
            coefficient_count,
            dtype=np.float64,
        )
        process_metadata = {} if metadata is None else dict(metadata)
        process_metadata.update(
            {
                "model": "isotropic_ornstein_uhlenbeck",
                "half_life_s": float(half_life_s),
                "stationary_std_n": float(stationary_std_n),
            }
        )
        return cls(
            transition_matrix=transition,
            process_covariance_n2=process_covariance,
            frame_dt_s=frame_dt_s,
            metadata=process_metadata,
        )

    def stationary_covariance_n2(self) -> np.ndarray:
        dimension = self.coefficient_count
        identity = np.eye(dimension * dimension, dtype=np.float64)
        system = identity - np.kron(self.transition_matrix, self.transition_matrix)
        vector = np.linalg.solve(system, self.process_covariance_n2.reshape(-1))
        covariance = vector.reshape(dimension, dimension)
        return _symmetric_psd(
            0.5 * (covariance + covariance.T),
            name="stationary_covariance_n2",
        )


@dataclass(frozen=True)
class LatentForceBelief:
    """Gaussian belief over dynamics-discrepancy force coefficients."""

    mean_n: np.ndarray
    covariance_n2: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mean = _readonly(self.mean_n)
        _require(mean.ndim == 1 and len(mean) >= 1, "mean_n must be a nonempty vector")
        covariance = _symmetric_psd(self.covariance_n2, name="covariance_n2")
        _require(
            covariance.shape == (len(mean), len(mean)),
            "covariance_n2 must match mean_n",
        )
        object.__setattr__(self, "mean_n", mean)
        object.__setattr__(self, "covariance_n2", covariance)
        object.__setattr__(
            self,
            "metadata",
            _json_data(self.metadata, name="metadata"),
        )

    @property
    def coefficient_count(self) -> int:
        return int(len(self.mean_n))

    @classmethod
    def zero(cls, coefficient_count: int) -> LatentForceBelief:
        _require(coefficient_count >= 1, "coefficient_count must be positive")
        return cls(
            mean_n=np.zeros(coefficient_count, dtype=np.float64),
            covariance_n2=np.zeros(
                (coefficient_count, coefficient_count),
                dtype=np.float64,
            ),
            metadata={"role": "exact_zero_force"},
        )

    def force_mean_n(self, basis: DynamicsConsistentForceBasis) -> np.ndarray:
        _require(
            basis.coefficient_count == self.coefficient_count,
            "belief and force basis dimensions differ",
        )
        return basis.decode(self.mean_n)

    def node_force_covariance_n2(
        self,
        basis: DynamicsConsistentForceBasis,
    ) -> np.ndarray:
        _require(
            basis.coefficient_count == self.coefficient_count,
            "belief and force basis dimensions differ",
        )
        return basis.node_force_covariance_n2(self.covariance_n2)


@dataclass(frozen=True)
class LatentForceConditioningResult:
    posterior: LatentForceBelief
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            _json_data(self.diagnostics, name="diagnostics"),
        )


def predict_latent_force_belief(
    belief: LatentForceBelief,
    process: StableLatentForceProcess,
    *,
    steps: int = 1,
) -> LatentForceBelief:
    """Propagate coefficient uncertainty under a strictly stable process."""

    _require(steps >= 0, "steps must be nonnegative")
    _require(
        belief.coefficient_count == process.coefficient_count,
        "belief and process dimensions differ",
    )
    mean = np.asarray(belief.mean_n, dtype=np.float64).copy()
    covariance = np.asarray(belief.covariance_n2, dtype=np.float64).copy()
    for _ in range(steps):
        mean = process.transition_matrix @ mean
        covariance = (
            process.transition_matrix @ covariance @ process.transition_matrix.T
            + process.process_covariance_n2
        )
        covariance = 0.5 * (covariance + covariance.T)
    if not np.any(mean != 0.0):
        mean = np.zeros_like(mean)
    return LatentForceBelief(
        mean_n=mean,
        covariance_n2=covariance,
        metadata={
            **belief.metadata,
            "prediction_steps": int(steps),
            "process_model": process.metadata.get("model", "linear_gaussian"),
        },
    )


def _whiten_observation_block(
    response: np.ndarray,
    observed: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten one observation block without constructing diagonal matrices."""

    value = np.asarray(covariance, dtype=np.float64)
    observation_count = response.shape[0]
    if value.ndim == 1:
        _require(
            value.shape == (observation_count,),
            "observation variance vector has the wrong length",
        )
        _require(
            np.all(np.isfinite(value)) and np.all(value > 0.0),
            "observation variances must be positive and finite",
        )
        inverse_standard_deviation = 1.0 / np.sqrt(value)
        return (
            inverse_standard_deviation[:, None] * response,
            inverse_standard_deviation * observed,
        )
    value = _symmetric_psd(value, name="observation_covariance")
    _require(
        value.shape == (observation_count, observation_count),
        "observation covariance has the wrong shape",
    )
    try:
        factor = np.linalg.cholesky(value)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "observation covariance must be positive definite"
        ) from error
    return (
        np.linalg.solve(factor, response),
        np.linalg.solve(factor, observed),
    )


def condition_latent_force_belief(
    prior: LatentForceBelief,
    response_per_coefficient: np.ndarray,
    innovation: np.ndarray,
    observation_covariance: np.ndarray,
    *,
    force_basis: DynamicsConsistentForceBasis | None = None,
    velocity_mps: np.ndarray | None = None,
    work_precision_per_watt2: float = 0.0,
    coefficient_precision_per_n2: float = 0.0,
) -> LatentForceConditioningResult:
    """Condition a low-rank latent-force belief in coefficient space.

    The work term is the quadratic penalty
    ``0.5 * work_precision_per_watt2 * power(c)^2``.  It is implemented as a
    zero-valued pseudo-observation of instantaneous mechanical power.  The
    posterior solve has the dimension of the prior covariance rank rather than
    the number of trajectory-response rows.
    """

    response = np.asarray(response_per_coefficient, dtype=np.float64)
    observed = np.asarray(innovation, dtype=np.float64)
    _require(
        response.ndim == 2
        and response.shape[0] >= 1
        and response.shape[1] == prior.coefficient_count,
        "response_per_coefficient must have shape (observation>=1, coefficient)",
    )
    _require(
        observed.shape == (response.shape[0],),
        "innovation must match the response rows",
    )
    _require(
        np.all(np.isfinite(response)) and np.all(np.isfinite(observed)),
        "response and innovation must be finite",
    )
    _require(
        work_precision_per_watt2 >= 0.0
        and np.isfinite(work_precision_per_watt2),
        "work precision must be finite and nonnegative",
    )
    _require(
        coefficient_precision_per_n2 >= 0.0
        and np.isfinite(coefficient_precision_per_n2),
        "coefficient precision must be finite and nonnegative",
    )
    whitened_response, whitened_observation = _whiten_observation_block(
        response,
        observed,
        observation_covariance,
    )
    response_blocks = [whitened_response]
    observation_blocks = [whitened_observation]
    power_jacobian = None
    if work_precision_per_watt2 > 0.0:
        _require(
            force_basis is not None and velocity_mps is not None,
            "work regularization requires force_basis and velocity_mps",
        )
        _require(
            force_basis.coefficient_count == prior.coefficient_count,
            "force basis and prior dimensions differ",
        )
        power_jacobian = force_basis.mechanical_power_jacobian(velocity_mps)
        power_scale = float(np.sqrt(work_precision_per_watt2))
        response_blocks.append(power_scale * power_jacobian[None])
        observation_blocks.append(np.zeros(1, dtype=np.float64))
    if coefficient_precision_per_n2 > 0.0:
        coefficient_scale = float(np.sqrt(coefficient_precision_per_n2))
        response_blocks.append(
            coefficient_scale
            * np.eye(prior.coefficient_count, dtype=np.float64)
        )
        observation_blocks.append(
            np.zeros(prior.coefficient_count, dtype=np.float64)
        )

    augmented_response = np.concatenate(response_blocks, axis=0)
    augmented_observation = np.concatenate(observation_blocks)
    prior_mean = np.asarray(prior.mean_n)
    prior_covariance = np.asarray(prior.covariance_n2)
    eigenvalues, eigenvectors = np.linalg.eigh(prior_covariance)
    scale = max(float(np.max(eigenvalues, initial=0.0)), 1.0)
    threshold = np.finfo(np.float64).eps * max(prior.coefficient_count, 1) * scale
    active = eigenvalues > threshold
    if np.any(active):
        prior_factor = eigenvectors[:, active] * np.sqrt(eigenvalues[active])
        reduced_response = augmented_response @ prior_factor
        reduced_residual = (
            augmented_observation - augmented_response @ prior_mean
        )
        reduced_precision = (
            np.eye(prior_factor.shape[1], dtype=np.float64)
            + reduced_response.T @ reduced_response
        )
        reduced_mean = np.linalg.solve(
            reduced_precision,
            reduced_response.T @ reduced_residual,
        )
        posterior_mean = prior_mean + prior_factor @ reduced_mean
        posterior_covariance = prior_factor @ np.linalg.solve(
            reduced_precision,
            prior_factor.T,
        )
        posterior_covariance = 0.5 * (
            posterior_covariance + posterior_covariance.T
        )
    else:
        posterior_mean = prior_mean.copy()
        posterior_covariance = prior_covariance.copy()

    posterior = LatentForceBelief(
        mean_n=posterior_mean,
        covariance_n2=posterior_covariance,
        metadata={
            **prior.metadata,
            "conditioned_observation_count": int(response.shape[0]),
            "work_regularized": bool(work_precision_per_watt2 > 0.0),
            "coefficient_regularized": bool(coefficient_precision_per_n2 > 0.0),
        },
    )
    predicted_before = response @ prior_mean
    predicted_after = response @ posterior.mean_n
    diagnostics: dict[str, Any] = {
        "prior_covariance_rank": int(np.sum(active)),
        "prior_covariance_trace_n2": float(np.trace(prior_covariance)),
        "posterior_covariance_trace_n2": float(np.trace(posterior.covariance_n2)),
        "innovation_rmse_before": float(
            np.sqrt(np.mean(np.square(observed - predicted_before)))
        ),
        "innovation_rmse_after": float(
            np.sqrt(np.mean(np.square(observed - predicted_after)))
        ),
        "work_precision_per_watt2": float(work_precision_per_watt2),
        "coefficient_precision_per_n2": float(coefficient_precision_per_n2),
    }
    if power_jacobian is not None:
        diagnostics.update(
            {
                "prior_mechanical_power_w": float(power_jacobian @ prior_mean),
                "posterior_mechanical_power_w": float(
                    power_jacobian @ posterior.mean_n
                ),
            }
        )
    return LatentForceConditioningResult(
        posterior=posterior,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class LatentForceForecast:
    coefficient_mean_n: np.ndarray
    coefficient_covariance_n2: np.ndarray
    force_mean_n: np.ndarray
    node_force_covariance_n2: np.ndarray

    def __post_init__(self) -> None:
        coefficient_mean = _readonly(self.coefficient_mean_n)
        coefficient_covariance = _readonly(self.coefficient_covariance_n2)
        force_mean = _readonly(self.force_mean_n)
        node_covariance = _readonly(self.node_force_covariance_n2)
        _require(
            coefficient_mean.ndim == 2,
            "coefficient_mean_n must have shape (frame, coefficient)",
        )
        frame_count, coefficient_count = coefficient_mean.shape
        _require(
            coefficient_covariance.shape
            == (frame_count, coefficient_count, coefficient_count),
            "coefficient covariance forecast has the wrong shape",
        )
        _require(
            force_mean.ndim == 3
            and force_mean.shape[0] == frame_count
            and force_mean.shape[2] == 3,
            "force_mean_n must have shape (frame, node, 3)",
        )
        _require(
            node_covariance.shape
            == (frame_count, force_mean.shape[1], 3, 3),
            "node force covariance forecast has the wrong shape",
        )
        object.__setattr__(self, "coefficient_mean_n", coefficient_mean)
        object.__setattr__(self, "coefficient_covariance_n2", coefficient_covariance)
        object.__setattr__(self, "force_mean_n", force_mean)
        object.__setattr__(self, "node_force_covariance_n2", node_covariance)


def forecast_latent_force_belief(
    initial: LatentForceBelief,
    process: StableLatentForceProcess,
    force_basis: DynamicsConsistentForceBasis,
    *,
    frame_count: int,
) -> LatentForceForecast:
    """Forecast mean forces and coefficient/node uncertainty for each frame."""

    _require(frame_count >= 1, "frame_count must be positive")
    _require(
        initial.coefficient_count
        == process.coefficient_count
        == force_basis.coefficient_count,
        "belief, process, and force-basis dimensions differ",
    )
    means = []
    covariances = []
    force_means = []
    node_covariances = []
    current = initial
    for frame in range(frame_count):
        if frame:
            current = predict_latent_force_belief(current, process)
        means.append(current.mean_n)
        covariances.append(current.covariance_n2)
        force_means.append(current.force_mean_n(force_basis))
        node_covariances.append(current.node_force_covariance_n2(force_basis))
    return LatentForceForecast(
        coefficient_mean_n=np.stack(means),
        coefficient_covariance_n2=np.stack(covariances),
        force_mean_n=np.stack(force_means),
        node_force_covariance_n2=np.stack(node_covariances),
    )


def mechanical_work_summary(
    force_schedule_n: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    frame_dt_s: float,
) -> dict[str, Any]:
    """Summarize signed and absolute work of a force schedule."""

    forces = np.asarray(force_schedule_n, dtype=np.float64)
    velocity = np.asarray(velocity_mps, dtype=np.float64)
    _require(
        forces.ndim == 3 and forces.shape[2] == 3 and velocity.shape == forces.shape,
        "force and velocity schedules must have shape (frame, node, 3)",
    )
    _require(
        np.all(np.isfinite(forces)) and np.all(np.isfinite(velocity)),
        "force and velocity schedules must be finite",
    )
    _require(
        frame_dt_s > 0.0 and np.isfinite(frame_dt_s),
        "frame_dt_s must be positive and finite",
    )
    power = np.sum(forces * velocity, axis=(1, 2))
    signed_work = frame_dt_s * np.cumsum(power)
    return {
        "power_by_frame_w": power.tolist(),
        "signed_work_by_frame_j": signed_work.tolist(),
        "total_signed_work_j": float(signed_work[-1]) if len(signed_work) else 0.0,
        "total_absolute_work_j": float(frame_dt_s * np.sum(np.abs(power))),
        "maximum_absolute_power_w": float(np.max(np.abs(power), initial=0.0)),
    }
