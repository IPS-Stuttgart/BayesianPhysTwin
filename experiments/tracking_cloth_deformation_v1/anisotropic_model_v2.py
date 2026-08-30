"""Anisotropic spring-mesh model bank for source-only active-probe development.

This module is a separately versioned development model.  It does not modify the
historical isotropic Tracking Cloth evaluator.  The bank is fixed from generic
cloth mechanics before any new target-outcome access and is used only to test
whether query-directed and parameter-directed logged-probe criteria can make
different decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import Inputs
from .model import Predictions, velocity


@dataclass(frozen=True)
class AnisotropicParameters:
    weft_stiffness_per_mass: float
    warp_stiffness_per_mass: float
    shear_stiffness_per_mass: float
    bend_stiffness_per_mass: float
    damping_per_mass: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            self.weft_stiffness_per_mass,
            self.warp_stiffness_per_mass,
            self.shear_stiffness_per_mass,
            self.bend_stiffness_per_mass,
            self.damping_per_mass,
        )


def nominal_parameters(protocol: dict[str, Any]) -> AnisotropicParameters:
    raw = protocol["anisotropic_nominal"]
    result = AnisotropicParameters(
        float(raw["weft_stiffness_per_mass"]),
        float(raw["warp_stiffness_per_mass"]),
        float(raw["shear_stiffness_per_mass"]),
        float(raw["bend_stiffness_per_mass"]),
        float(raw["damping_per_mass"]),
    )
    _validate_parameters(result)
    return result


def _validate_parameters(parameters: AnisotropicParameters) -> None:
    values = np.asarray(parameters.as_tuple(), dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("anisotropic parameters must be finite and positive")


def parameter_bank(protocol: dict[str, Any]) -> list[AnisotropicParameters]:
    """Return the frozen 55-member anisotropic development bank.

    The geometric mean of the two structural stiffnesses equals the registered
    base stiffness.  Three constitutive profiles vary shear versus bending, and
    two damping levels separate dissipation from elastic anisotropy.  The exact
    nominal member is appended once when it is not already in the factorial.
    """
    bank: list[AnisotropicParameters] = []
    for base in protocol["anisotropic_base_stiffness_per_mass"]:
        base_value = float(base)
        for ratio in protocol["anisotropy_ratios"]:
            ratio_value = float(ratio)
            if not np.isfinite(ratio_value) or ratio_value <= 0.0:
                raise ValueError("anisotropy ratio must be finite and positive")
            root = float(np.sqrt(ratio_value))
            weft = base_value * root
            warp = base_value / root
            for profile in protocol["constitutive_profiles"]:
                shear = base_value * float(profile["shear_ratio"])
                bend = base_value * float(profile["bend_ratio"])
                for damping in protocol["anisotropic_damping_per_mass"]:
                    candidate = AnisotropicParameters(
                        weft, warp, shear, bend, float(damping)
                    )
                    _validate_parameters(candidate)
                    bank.append(candidate)
    nominal = nominal_parameters(protocol)
    if nominal.as_tuple() not in {member.as_tuple() for member in bank}:
        bank.append(nominal)
    if len(bank) != int(protocol["anisotropic_model_count"]):
        raise ValueError(
            "anisotropic model bank size differs from the registered protocol"
        )
    if len({member.as_tuple() for member in bank}) != len(bank):
        raise ValueError("anisotropic model bank contains duplicate members")
    return bank


def edges(markers: int) -> tuple[np.ndarray, np.ndarray]:
    """Return grid edges and constitutive class indices.

    Class 0 is one-hop weft/horizontal structure, class 1 one-hop warp/vertical
    structure, class 2 diagonal shear, and class 3 two-hop bending.
    """
    rows, cols = (5, 4) if markers == 20 else (4, 3)
    if markers not in (12, 20):
        raise ValueError("anisotropic cloth model supports only 12 or 20 markers")
    links: list[tuple[int, int]] = []
    kinds: list[int] = []
    for row in range(rows):
        for col in range(cols):
            for dr, dc, kind in (
                (0, 1, 0),
                (1, 0, 1),
                (1, 1, 2),
                (1, -1, 2),
                (0, 2, 3),
                (2, 0, 3),
            ):
                rr, cc = row + dr, col + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    links.append((row * cols + col, rr * cols + cc))
                    kinds.append(kind)
    return np.asarray(links, dtype=int), np.asarray(kinds, dtype=int)


def rollout(
    inputs: Inputs,
    parameters: AnisotropicParameters,
    protocol: dict[str, Any],
    *,
    inject: bool,
) -> np.ndarray:
    """Symplectic anisotropic spring rollout with recorded corner boundary."""
    _validate_parameters(parameters)
    links, kinds = edges(len(inputs.order))
    left, right = links.T
    rest = np.linalg.norm(inputs.prefix[0, right] - inputs.prefix[0, left], axis=1)
    if np.min(rest) <= 1e-6:
        raise ValueError("degenerate initial anisotropic spring")
    stiffness = np.choose(
        kinds,
        [
            parameters.weft_stiffness_per_mass,
            parameters.warp_stiffness_per_mass,
            parameters.shear_stiffness_per_mass,
            parameters.bend_stiffness_per_mass,
        ],
    )
    start = inputs.cutoff if inject else 0
    x = inputs.prefix[start].copy()
    if inject:
        if start < 4:
            raise ValueError("state injection requires five causal prefix frames")
        v = velocity(
            inputs.times[start - 4 : start + 1], inputs.prefix[start - 4 : start + 1]
        )
    else:
        v = velocity(inputs.times[:5], inputs.prefix[:5])
    result = np.empty((len(inputs.times), len(inputs.order), 3))
    result[: start + 1] = inputs.prefix[: start + 1]
    origin = inputs.prefix[0].mean(axis=0)
    substeps = int(protocol["integration_substeps"])
    if substeps < 1:
        raise ValueError("integration_substeps must be positive")
    for time_index in range(start + 1, len(inputs.times)):
        full_dt = inputs.times[time_index] - inputs.times[time_index - 1]
        if not np.isfinite(full_dt) or full_dt <= 0.0:
            raise ValueError("nonpositive rollout time step")
        dt = full_dt / substeps
        boundary_velocity = (
            inputs.boundary[time_index] - inputs.boundary[time_index - 1]
        ) / full_dt
        for substep in range(1, substeps + 1):
            delta = x[right] - x[left]
            lengths = np.linalg.norm(delta, axis=1)
            force = (
                stiffness * (lengths - rest) / np.maximum(lengths, 1e-9)
            )[:, None] * delta
            acceleration = -parameters.damping_per_mass * v
            acceleration[:, 2] -= float(protocol["gravity_m_s2"])
            np.add.at(acceleration, left, force)
            np.add.at(acceleration, right, -force)
            v += dt * acceleration
            x += dt * v
            fraction = substep / substeps
            x[inputs.corners] = (
                (1.0 - fraction) * inputs.boundary[time_index - 1]
                + fraction * inputs.boundary[time_index]
            )
            v[inputs.corners] = boundary_velocity
        if not np.isfinite(x).all() or np.max(np.linalg.norm(x - origin, axis=1)) > 10:
            raise ValueError("numerically invalid anisotropic rollout")
        result[time_index] = x
    return result


def predict(inputs: Inputs, protocol: dict[str, Any]) -> Predictions:
    if protocol.get("model_family") != "anisotropic-spring-v2":
        raise ValueError("anisotropic v2 predictor requires its registered model family")
    nominal = rollout(inputs, nominal_parameters(protocol), protocol, inject=False)
    bank = np.stack(
        [rollout(inputs, member, protocol, inject=True) for member in parameter_bank(protocol)]
    )
    return Predictions(inputs, nominal, bank)


__all__ = [
    "AnisotropicParameters",
    "edges",
    "nominal_parameters",
    "parameter_bank",
    "predict",
    "rollout",
]
