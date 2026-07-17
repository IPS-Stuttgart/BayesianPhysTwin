from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_bayesian_residual import (
    BayesianResidualModelConfig,
    EquivariantBayesianResidual,
    apply_exact_residual_fallback,
    clustered_effective_weights,
    inflate_variance_for_action_shift,
    load_bayesian_residual_config,
    moment_match_residual_ensemble,
    residual_acceptance_mask,
)
from causal4d_public.deform360_bayesian_residual_experiment import (
    load_cross_fitted_trust_scales,
)


torch = pytest.importorskip("torch")


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "deform360_bayesian_residual_source_v1.json"
    )


def test_source_protocol_is_canonically_locked(tmp_path: Path) -> None:
    payload = load_bayesian_residual_config(_config_path())

    assert payload["config"]["information_boundary"][
        "penguin_held_media_or_outcomes_may_open"
    ] is False
    changed = json.loads(json.dumps(payload))
    changed["config"]["information_boundary"][
        "penguin_held_media_or_outcomes_may_open"
    ] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_bayesian_residual_config(path)


def test_frozen_cross_fitted_trust_decisions_are_reconstructed() -> None:
    root = Path(__file__).resolve().parents[1]

    scales = load_cross_fitted_trust_scales(
        root
        / "milestones/deform360-reusable-trust-source-v1/artifacts/"
        "same_object_trust_diagnosis.json",
        root
        / "milestones/deform360-graph-action-support-independent-source-v1/"
        "artifacts/failure_diagnosis.json",
    )

    assert len(scales) == 27
    assert sum(value > 0.0 for value in scales.values()) == 10
    assert scales["002-rope-silk/2"] == 0.0
    assert scales["002-rope-silk/6"] == pytest.approx(0.7953490210931101)


def test_clustered_weights_do_not_count_a_duplicated_block_twice() -> None:
    reliability = np.array([0.9, 0.5, 0.8])
    clusters = np.array([0, 0, 1])
    base = clustered_effective_weights(reliability, clusters)

    duplicated = clustered_effective_weights(
        np.repeat(reliability, 2), np.repeat(clusters, 2)
    )

    np.testing.assert_allclose(
        np.array([duplicated[0:2].sum(), duplicated[2:4].sum(), duplicated[4:6].sum()]),
        base,
    )


def test_ensemble_variance_contains_member_spread() -> None:
    means = np.array([[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
    variances = np.zeros((2, 1))

    mean, variance = moment_match_residual_ensemble(means, variances)

    np.testing.assert_allclose(mean, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(variance, [[1.0, 0.0, 0.0]])


def test_action_shift_only_widens_variance() -> None:
    variance = np.array([0.01, 0.02])
    inflated = inflate_variance_for_action_shift(
        variance, np.array([0.0, 2.0]), inflation_rate=0.5
    )

    np.testing.assert_allclose(inflated, [0.01, 0.06])


def test_rejected_residual_is_byte_identical_to_physics() -> None:
    physics = np.arange(24, dtype=np.float64).reshape(2, 4, 3)
    residual = physics + 10.0
    accepted = np.zeros((2, 4), dtype=bool)

    output = apply_exact_residual_fallback(physics, residual, accepted)

    assert output.tobytes() == physics.tobytes()


def test_gate_is_conjunctive() -> None:
    accepted = residual_acceptance_mask(
        np.array([0.9, 0.9, 0.4]),
        np.array([0.01, 0.20, 0.01]),
        np.array([0.1, 0.1, 0.1]),
        minimum_utility_probability=0.8,
        maximum_variance_m2=0.05,
        maximum_action_distance=0.5,
    )

    np.testing.assert_array_equal(accepted, [True, False, False])


def _rotation_z() -> torch.Tensor:
    return torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def test_residual_model_is_rotation_equivariant() -> None:
    torch.manual_seed(7)
    model = EquivariantBayesianResidual(
        BayesianResidualModelConfig(hidden_dim=16, message_steps=2)
    ).eval()
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.1, 0.0]]]
    )
    velocities = torch.tensor(
        [[[0.01, 0.02, 0.0], [0.00, 0.01, 0.0], [-0.01, 0.0, 0.0]]]
    )
    physics_positions = positions + 0.01 * torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]]
    )
    physics_velocities = velocities * 0.8
    controllers = torch.tensor([[[0.25, 0.0, 0.0]]])
    controller_velocities = torch.tensor([[[0.0, 0.03, 0.0]]])
    contact = torch.tensor([[[0.0], [0.4], [0.8]]])
    reliability = torch.tensor([[0.9, 0.7, 0.8]])
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    kwargs = {
        "positions_m": positions,
        "velocities_mps": velocities,
        "physics_positions_m": physics_positions,
        "physics_velocities_mps": physics_velocities,
        "controller_positions_m": controllers,
        "controller_velocities_mps": controller_velocities,
        "contact_probabilities": contact,
        "prior_reliability": reliability,
        "edge_index": edges,
    }
    with torch.no_grad():
        original = model(**kwargs)
        rotation = _rotation_z()
        rotated = model(
            **{
                **kwargs,
                "positions_m": positions @ rotation.T,
                "velocities_mps": velocities @ rotation.T,
                "physics_positions_m": physics_positions @ rotation.T,
                "physics_velocities_mps": physics_velocities @ rotation.T,
                "controller_positions_m": controllers @ rotation.T,
                "controller_velocities_mps": controller_velocities @ rotation.T,
            }
        )

    torch.testing.assert_close(rotated.mean_mps, original.mean_mps @ rotation.T)
    torch.testing.assert_close(
        rotated.aleatoric_variance_m2ps2,
        original.aleatoric_variance_m2ps2,
    )
    torch.testing.assert_close(rotated.utility_probability, original.utility_probability)


def test_duplicate_controller_surface_points_do_not_change_prediction() -> None:
    torch.manual_seed(11)
    model = EquivariantBayesianResidual(
        BayesianResidualModelConfig(hidden_dim=16, message_steps=1)
    ).eval()
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    controllers = torch.tensor([[[0.01, 0.0, 0.0], [0.12, 0.0, 0.0]]])
    controller_velocity = torch.tensor([[[0.0, 0.02, 0.0], [0.0, 0.02, 0.0]]])
    contact = torch.tensor([[[0.8, 0.1], [0.2, 0.7]]])
    kwargs = {
        "positions_m": positions,
        "velocities_mps": torch.zeros_like(positions),
        "physics_positions_m": positions,
        "physics_velocities_mps": torch.zeros_like(positions),
        "prior_reliability": torch.ones(1, 2),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
    }
    with torch.no_grad():
        original = model(
            **kwargs,
            controller_positions_m=controllers,
            controller_velocities_mps=controller_velocity,
            contact_probabilities=contact,
        )
        duplicated = model(
            **kwargs,
            controller_positions_m=torch.repeat_interleave(controllers, 2, dim=1),
            controller_velocities_mps=torch.repeat_interleave(
                controller_velocity, 2, dim=1
            ),
            contact_probabilities=torch.repeat_interleave(contact, 2, dim=2),
        )

    torch.testing.assert_close(duplicated.mean_mps, original.mean_mps)
    torch.testing.assert_close(
        duplicated.aleatoric_variance_m2ps2,
        original.aleatoric_variance_m2ps2,
    )
    torch.testing.assert_close(
        duplicated.utility_probability, original.utility_probability
    )
