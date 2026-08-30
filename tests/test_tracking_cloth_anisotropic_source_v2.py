import json
from pathlib import Path

import numpy as np

from experiments.tracking_cloth_deformation_v1.anisotropic_model_v2 import (
    edges,
    parameter_bank,
    rollout,
)
from experiments.tracking_cloth_deformation_v1.data import Inputs


HERE = Path(__file__).resolve().parents[1] / "experiments" / "tracking_cloth_deformation_v1"


def protocol():
    return json.loads((HERE / "active_probe_anisotropic_protocol_v2.json").read_text())


def synthetic_inputs():
    rows, cols = 4, 3
    grid = np.array(
        [[0.1 * col, 0.0, -0.1 * row] for row in range(rows) for col in range(cols)],
        dtype=float,
    )
    times = np.arange(12, dtype=float) * 0.02
    prefix = np.repeat(grid[None, :, :], 5, axis=0)
    corners = np.array([0, 2], dtype=int)
    boundary = np.repeat(grid[corners][None, :, :], len(times), axis=0)
    for index in range(5, len(times)):
        fraction = (index - 4) / (len(times) - 4)
        boundary[index, 0, 0] -= 0.03 * fraction
        boundary[index, 1, 0] += 0.01 * fraction
        boundary[index, 1, 1] += 0.02 * fraction
    return Inputs(
        times,
        prefix,
        boundary,
        np.arange(rows * cols),
        corners,
        4,
        0.0,
        1.0,
    )


def test_registered_anisotropic_bank_is_unique_and_nominal_is_extra_member():
    registered = protocol()
    bank = parameter_bank(registered)
    assert len(bank) == 55
    assert len({member.as_tuple() for member in bank}) == 55
    nominal = tuple(registered["anisotropic_nominal"].values())
    assert nominal in {member.as_tuple() for member in bank}
    assert registered["target_scoring_authorized"] is False
    assert registered["paper_claim_authorized"] is False


def test_edge_classes_cover_structure_shear_and_bending():
    links, kinds = edges(12)
    assert links.ndim == 2 and links.shape[1] == 2
    assert set(kinds.tolist()) == {0, 1, 2, 3}
    assert len(links) == len(kinds)


def test_swapping_warp_weft_stiffness_changes_asymmetric_rollout():
    registered = protocol()
    registered["integration_substeps"] = 2
    bank = parameter_bank(registered)
    left = next(
        member
        for member in bank
        if np.isclose(member.weft_stiffness_per_mass, 200.0)
        and np.isclose(member.warp_stiffness_per_mass, 800.0)
        and np.isclose(member.shear_stiffness_per_mass, 200.0)
        and np.isclose(member.bend_stiffness_per_mass, 40.0)
        and np.isclose(member.damping_per_mass, 1.0)
    )
    right = next(
        member
        for member in bank
        if np.isclose(member.weft_stiffness_per_mass, 800.0)
        and np.isclose(member.warp_stiffness_per_mass, 200.0)
        and np.isclose(member.shear_stiffness_per_mass, 200.0)
        and np.isclose(member.bend_stiffness_per_mass, 40.0)
        and np.isclose(member.damping_per_mass, 1.0)
    )
    inputs = synthetic_inputs()
    first = rollout(inputs, left, registered, inject=True)
    second = rollout(inputs, right, registered, inject=True)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert np.max(np.abs(first - second)) > 1e-8
