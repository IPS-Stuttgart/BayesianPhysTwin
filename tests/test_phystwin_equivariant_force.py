from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from bayesian_phystwin.phystwin_equivariant_force import (
    EquivariantForceConfig,
    build_equivariant_force_model,
    canonicalize_force_edges,
    force_rest_lengths,
    maximum_node_force_n,
)
from bayesian_phystwin.phystwin_equivariant_force_artifact import (
    EquivariantForceArtifact,
    load_equivariant_force_artifact,
    write_equivariant_force_artifact,
)
from bayesian_phystwin.phystwin_equivariant_force_data import (
    EquivariantForceEpisode,
    load_equivariant_force_episode,
    validate_force_episode_model_compatibility,
    write_equivariant_force_episode,
)
from bayesian_phystwin.phystwin_equivariant_force_gate import (
    evaluate_equivariant_force_official_warp_gate,
)
from bayesian_phystwin.phystwin_equivariant_force_warp import (
    controller_attachment_matrix,
    controller_conditioning_fields,
    predict_equivariant_force,
    rollout_equivariant_force_segment,
)
from bayesian_phystwin.phystwin_equivariant_force_training import (
    EquivariantForceTrainingConfig,
    adapt_equivariant_force_latent,
    crossfit_equivariant_force_competence,
    fit_shared_equivariant_force_model,
    force_target_metrics,
)
from bayesian_phystwin.phystwin_equivariant_force_source import (
    ForceTargetBuildConfig,
    build_released_equivariant_force_episode,
    load_equivariant_force_source_protocol,
)
from bayesian_phystwin.phystwin_force_targets import (
    acceleration_to_force_targets,
    estimate_residual_acceleration,
    graph_smooth_residual_acceleration,
)


def _inputs(torch, *, batch: int | None = None):
    rest = torch.tensor(
        [
            [-0.7, 0.0, 0.0],
            [-0.2, 0.1, 0.0],
            [0.3, -0.1, 0.1],
            [0.8, 0.0, 0.2],
        ],
        dtype=torch.float32,
    )
    positions = rest + torch.tensor(
        [
            [0.02, -0.01, 0.01],
            [-0.01, 0.03, 0.00],
            [0.00, -0.02, 0.02],
            [0.03, 0.01, -0.01],
        ]
    )
    velocities = torch.tensor(
        [
            [0.01, 0.02, 0.00],
            [-0.02, 0.01, 0.03],
            [0.02, -0.01, 0.01],
            [0.00, 0.02, -0.02],
        ]
    )
    control_delta = torch.tensor(
        [
            [0.02, 0.01, 0.00],
            [0.01, 0.00, 0.01],
            [0.00, 0.02, 0.00],
            [0.03, 0.01, 0.02],
        ]
    )
    control_velocity = 0.5 * velocities
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 2]])
    rest_lengths = torch.linalg.vector_norm(
        rest[edges[:, 1]] - rest[edges[:, 0]], dim=1
    )
    values = {
        "positions_m": positions,
        "velocities_mps": velocities,
        "rest_positions_m": rest,
        "edges": edges,
        "rest_lengths_m": rest_lengths,
        "control_displacement_m": control_delta,
        "control_velocity_mps": control_velocity,
        "action_support": torch.tensor([1.0, 0.8, 0.3, 0.0]),
        "external_support": torch.tensor([0.0, 0.2, 0.7, 1.0]),
        "gravity_mps2": torch.tensor([0.0, 0.0, -9.81]),
        "action_activity": torch.tensor(0.75),
        "regime_probabilities": torch.tensor([0.1, 0.2, 0.3, 0.25, 0.15]),
        "latent": torch.linspace(-0.2, 0.2, 8),
    }
    if batch is not None:
        for name in (
            "positions_m",
            "velocities_mps",
            "control_displacement_m",
            "control_velocity_mps",
        ):
            values[name] = values[name].unsqueeze(0).expand(batch, -1, -1).clone()
        values["regime_probabilities"] = (
            values["regime_probabilities"].unsqueeze(0).expand(batch, -1).clone()
        )
        values["latent"] = values["latent"].unsqueeze(0).expand(batch, -1).clone()
        values["action_activity"] = torch.full((batch,), 0.75)
    return values


def _randomize_model(torch, model) -> None:
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                0.15
                * torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )


def _rotation(torch):
    axis = torch.tensor([0.3, -0.4, 0.5])
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.tensor(0.73)
    skew = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        torch.eye(3)
        + torch.sin(angle) * skew
        + (1.0 - torch.cos(angle)) * (skew @ skew)
    )


def test_edge_canonicalization_and_rest_lengths_are_deterministic() -> None:
    edges = np.array([[2, 0], [1, 0], [0, 2], [2, 1]])
    expected = np.array([[0, 1], [0, 2], [1, 2]])
    canonical = canonicalize_force_edges(edges, num_nodes=3)
    np.testing.assert_array_equal(canonical, expected)
    lengths = force_rest_lengths(np.eye(3), canonical)
    np.testing.assert_allclose(lengths, np.sqrt(2.0))
    assert not canonical.flags.writeable
    assert not lengths.flags.writeable


def test_zero_initialized_model_and_zero_admission_are_exact() -> None:
    torch = pytest.importorskip("torch")
    model = build_equivariant_force_model(torch)
    inputs = _inputs(torch)
    force = model(**inputs)
    torch.testing.assert_close(force, torch.zeros_like(force), rtol=0.0, atol=0.0)

    _randomize_model(torch, model)
    force = model(**inputs, admission_weight=0.0)
    torch.testing.assert_close(force, torch.zeros_like(force), rtol=0.0, atol=0.0)


def test_model_is_translation_invariant_and_rotation_equivariant() -> None:
    torch = pytest.importorskip("torch")
    model = build_equivariant_force_model(torch)
    _randomize_model(torch, model)
    inputs = _inputs(torch)
    reference = model(**inputs)

    translated = dict(inputs)
    offset = torch.tensor([1.7, -2.3, 0.4])
    translated["positions_m"] = inputs["positions_m"] + offset
    translated["rest_positions_m"] = inputs["rest_positions_m"] + offset
    torch.testing.assert_close(
        model(**translated), reference, rtol=2.0e-5, atol=2.0e-6
    )

    rotation = _rotation(torch)
    rotated = dict(inputs)
    for name in (
        "positions_m",
        "velocities_mps",
        "rest_positions_m",
        "control_displacement_m",
        "control_velocity_mps",
    ):
        rotated[name] = inputs[name] @ rotation.T
    rotated["gravity_mps2"] = inputs["gravity_mps2"] @ rotation.T
    torch.testing.assert_close(
        model(**rotated),
        reference @ rotation.T,
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_internal_messages_conserve_total_force() -> None:
    torch = pytest.importorskip("torch")
    model = build_equivariant_force_model(torch)
    _randomize_model(torch, model)
    with torch.no_grad():
        final = model.node_head[-1]
        final.weight.zero_()
        final.bias.zero_()
    force = model(**_inputs(torch))
    torch.testing.assert_close(
        torch.sum(force, dim=0),
        torch.zeros(3),
        rtol=0.0,
        atol=2.0e-7,
    )


def test_external_action_terms_are_support_gated() -> None:
    torch = pytest.importorskip("torch")
    model = build_equivariant_force_model(torch)
    with torch.no_grad():
        final = model.node_head[-1]
        final.bias[1] = 0.5
    inputs = _inputs(torch)
    inputs["action_support"] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    inputs["external_support"] = torch.zeros(4)
    force = model(**inputs)
    assert torch.linalg.vector_norm(force[0]) > 0.0
    torch.testing.assert_close(
        force[1:], torch.zeros_like(force[1:]), rtol=0.0, atol=0.0
    )


def test_force_bound_is_enforced_for_the_complete_graph() -> None:
    torch = pytest.importorskip("torch")
    config = EquivariantForceConfig(maximum_force_per_node_n=0.07)
    model = build_equivariant_force_model(torch, config)
    _randomize_model(torch, model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(30.0)
    force = model(**_inputs(torch))
    assert maximum_node_force_n(force.detach().numpy()) <= 0.07 + 1.0e-7


def test_synthetic_state_dependent_internal_force_is_learnable() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    config = EquivariantForceConfig(
        hidden_dim=32,
        hidden_layers=2,
        maximum_force_per_node_n=0.30,
    )
    model = build_equivariant_force_model(torch, config)
    batch = 48
    inputs = _inputs(torch, batch=batch)
    rest = inputs["rest_positions_m"]
    inputs["positions_m"] = (
        rest.unsqueeze(0)
        + 0.035 * torch.randn(batch, len(rest), 3)
    )
    inputs["velocities_mps"] = 0.04 * torch.randn(batch, len(rest), 3)
    inputs["control_displacement_m"].zero_()
    inputs["control_velocity_mps"].zero_()
    inputs["action_support"] = torch.zeros(len(rest))
    inputs["external_support"] = torch.zeros(len(rest))

    edges = inputs["edges"]
    first, second = edges[:, 0], edges[:, 1]
    displacement = (
        inputs["positions_m"][:, second] - inputs["positions_m"][:, first]
    )
    direction = displacement / torch.clamp(
        torch.linalg.vector_norm(displacement, dim=-1, keepdim=True), min=1.0e-6
    )
    strain = (
        torch.linalg.vector_norm(displacement, dim=-1)
        / inputs["rest_lengths_m"][None]
        - 1.0
    )
    relative_velocity = (
        inputs["velocities_mps"][:, second]
        - inputs["velocities_mps"][:, first]
    )
    radial = torch.sum(relative_velocity * direction, dim=-1)
    edge_target = (0.10 * strain + 0.025 * radial)[:, :, None] * direction
    target = torch.zeros_like(inputs["positions_m"])
    target.index_add_(1, first, edge_target)
    target.index_add_(1, second, -edge_target)

    initial_mse = float(torch.mean(torch.square(target)))
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    for _ in range(350):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(**inputs)
        loss = torch.mean(torch.square(prediction - target))
        loss.backward()
        optimizer.step()
    final_mse = float(
        torch.mean(torch.square(model(**inputs) - target)).detach()
    )
    assert final_mse < 0.15 * initial_mse


def _artifact(model, config):
    return EquivariantForceArtifact.from_model(
        model,
        config=config,
        source_checksums={"source_protocol": "a" * 64},
        information_boundary={
            "target_future_used_for_fit_or_selection": False,
            "exact_zero_force_fallback": True,
            "force_location": "inside_official_warp",
            "source_complete_outcomes_may_supervise_shared_weights": True,
            "target_prefix_may_adapt_latent_only": True,
        },
        training_summary={"source_cases": 17, "selected_epoch": 12},
        admission_policy={
            "baseline": "unchanged_bayesian_phystwin",
            "fallback_is_bit_exact": True,
        },
    )


def test_typed_artifact_round_trip_preserves_predictions(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = EquivariantForceConfig(hidden_dim=12, hidden_layers=2)
    model = build_equivariant_force_model(torch, config)
    _randomize_model(torch, model)
    artifact = _artifact(model, config)
    record = write_equivariant_force_artifact(tmp_path / "force_model", artifact)
    loaded = load_equivariant_force_artifact(tmp_path / "force_model.json")
    assert loaded.artifact_id == record["artifact_id"]
    torch.testing.assert_close(
        loaded.instantiate(torch)(**_inputs(torch)),
        model(**_inputs(torch)),
        rtol=0.0,
        atol=0.0,
    )


def test_artifact_detects_weight_tampering(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    config = EquivariantForceConfig(hidden_dim=8)
    artifact = _artifact(build_equivariant_force_model(torch, config), config)
    write_equivariant_force_artifact(tmp_path / "force_model", artifact)
    weights = tmp_path / "force_model.npz"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_equivariant_force_artifact(tmp_path / "force_model")


def test_artifact_rejects_future_selected_models() -> None:
    torch = pytest.importorskip("torch")
    config = EquivariantForceConfig(hidden_dim=8)
    model = build_equivariant_force_model(torch, config)
    with pytest.raises(ValueError, match="causal or fallback boundary"):
        EquivariantForceArtifact.from_model(
            model,
            config=config,
            source_checksums={"source_protocol": "b" * 64},
            information_boundary={
                "target_future_used_for_fit_or_selection": True,
                "exact_zero_force_fallback": True,
                "force_location": "inside_official_warp",
            },
            training_summary={},
            admission_policy={},
        )


def test_controller_attachment_is_normalized_without_inventing_support() -> None:
    matrix, support = controller_attachment_matrix(
        np.array(
            [
                [0, 1],
                [0, 4],
                [0, 5],
                [2, 5],
                [1, 3],
            ]
        ),
        num_object_nodes=4,
        num_control_nodes=2,
    )
    np.testing.assert_allclose(matrix[0], [0.5, 0.5])
    np.testing.assert_allclose(matrix[2], [0.0, 1.0])
    np.testing.assert_array_equal(support, [1.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(matrix[1], [0.0, 0.0])


def test_controller_conditioning_uses_measured_action_and_support_prior() -> None:
    positions = np.zeros((3, 3))
    previous = np.zeros((2, 3))
    target = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
    attachment = np.array([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]])
    fields = controller_conditioning_fields(
        positions,
        previous,
        target,
        attachment,
        frame_dt_s=0.1,
        support_prior=np.array([0.0, 0.7, 0.2]),
        activity_speed_mps=2.0,
    )
    np.testing.assert_allclose(
        fields["control_displacement_m"],
        [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.05, 0.1, 0.0]],
    )
    np.testing.assert_array_equal(fields["action_support"], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(fields["external_support"], [1.0, 0.7, 1.0])
    assert fields["action_activity"] == pytest.approx(1.0)


def test_force_predictor_abstains_without_evaluating_the_model() -> None:
    class ForbiddenModel:
        def eval(self):
            raise AssertionError("zero admission must not evaluate the model")

    force = predict_equivariant_force(
        ForbiddenModel(),
        object(),
        positions_m=np.ones((4, 3)),
        velocities_mps=np.zeros((4, 3)),
        rest_positions_m=np.ones((4, 3)),
        object_edges=np.array([[0, 1]]),
        rest_lengths_m=np.ones(1),
        conditioning={},
        gravity_mps2=np.array([0.0, 0.0, -9.81]),
        regime_probabilities=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        latent=np.zeros(8),
        admission_weight=0.0,
        device="cpu",
    )
    np.testing.assert_array_equal(force, np.zeros((4, 3), dtype=np.float32))


def test_zero_admission_delegates_to_the_existing_warp_rollout(
    monkeypatch,
) -> None:
    import bayesian_phystwin.phystwin_equivariant_force_warp as warp_module

    expected_positions = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    expected_velocities = -expected_positions

    def baseline(*args, **kwargs):
        assert kwargs["start_frame"] == 1
        assert kwargs["stop_frame"] == 3
        return expected_positions, expected_velocities

    monkeypatch.setattr(warp_module, "_rollout_state_segment", baseline)

    class Simulator:
        def __init__(self):
            self.clear_count = 0

        def clear_external_forces(self):
            self.clear_count += 1

    simulator = Simulator()
    positions, velocities, forces, diagnostics = rollout_equivariant_force_segment(
        simulator,
        object(),
        object(),
        object(),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        start_frame=1,
        stop_frame=3,
        rest_positions_m=np.zeros((3, 3)),
        object_edges=np.array([[0, 1], [1, 2]]),
        rest_lengths_m=np.ones(2),
        controller_points_m=np.zeros((3, 1, 3)),
        attachment_matrix=np.zeros((3, 1)),
        support_prior=np.zeros(3),
        regime_probabilities=np.tile([1.0, 0.0, 0.0, 0.0, 0.0], (3, 1)),
        latent=np.zeros(8),
        gravity_mps2=np.array([0.0, 0.0, -9.81]),
        frame_dt_s=1.0 / 30.0,
        activity_speed_mps=0.1,
        admission_weight=0.0,
        device="cpu",
    )
    assert simulator.clear_count == 1
    assert positions is expected_positions
    assert velocities is expected_velocities
    np.testing.assert_array_equal(forces, np.zeros((2, 3, 3)))
    assert diagnostics.maximum_force_n == 0.0


def _quadratic_residual_case(*, noise: float = 0.0):
    rng = np.random.default_rng(23)
    frames, nodes, dt = 15, 4, 0.04
    time = np.arange(frames, dtype=float) * dt
    acceleration = np.array(
        [
            [0.12, -0.04, 0.03],
            [-0.05, 0.08, 0.01],
            [0.02, 0.03, -0.07],
            [0.06, -0.02, 0.04],
        ]
    )
    velocity = np.array(
        [
            [0.01, 0.00, -0.01],
            [0.00, -0.02, 0.01],
            [0.02, 0.01, 0.00],
            [-0.01, 0.00, 0.02],
        ]
    )
    residual = (
        0.5 * np.square(time)[:, None, None] * acceleration[None]
        + time[:, None, None] * velocity[None]
    )
    residual += noise * rng.normal(size=residual.shape)
    baseline = np.zeros_like(residual)
    return baseline + residual, baseline, np.ones((frames, nodes), dtype=bool), acceleration, dt


def test_local_polynomial_targets_recover_quadratic_acceleration_robustly() -> None:
    observed, baseline, valid, acceleration, dt = _quadratic_residual_case()
    observed[6, 0] += np.array([0.04, -0.03, 0.02])
    robust = estimate_residual_acceleration(
        observed,
        baseline,
        valid,
        frame_dt_s=dt,
        window_radius=4,
        huber_delta_m=0.001,
    )
    ordinary = estimate_residual_acceleration(
        observed,
        baseline,
        valid,
        frame_dt_s=dt,
        window_radius=4,
        huber_delta_m=1.0,
    )
    robust_error = np.linalg.norm(robust.mean_mps2[7, 0] - acceleration[0])
    ordinary_error = np.linalg.norm(ordinary.mean_mps2[7, 0] - acceleration[0])
    assert robust_error < ordinary_error
    np.testing.assert_allclose(
        robust.mean_mps2[7, 1:],
        acceleration[1:],
        atol=1.0e-7,
    )
    assert robust.diagnostics["innovation_used_once"] is True


def test_force_targets_do_not_read_mutated_future_frames() -> None:
    observed, baseline, valid, _, dt = _quadratic_residual_case(noise=1.0e-5)
    first = estimate_residual_acceleration(
        observed, baseline, valid, frame_dt_s=dt, end_frame=9
    )
    mutated = observed.copy()
    mutated[9:] += 1000.0
    second = estimate_residual_acceleration(
        mutated, baseline, valid, frame_dt_s=dt, end_frame=9
    )
    np.testing.assert_array_equal(first.mean_mps2, second.mean_mps2)
    np.testing.assert_array_equal(first.variance_m2ps4, second.variance_m2ps4)
    np.testing.assert_array_equal(first.observed, second.observed)


def test_causal_force_targets_do_not_read_later_frames_within_boundary() -> None:
    observed, baseline, valid, _, dt = _quadratic_residual_case(noise=1.0e-5)
    first = estimate_residual_acceleration(
        observed,
        baseline,
        valid,
        frame_dt_s=dt,
        end_frame=12,
        causal_window=True,
    )
    mutated = observed.copy()
    mutated[8:12] += 1000.0
    second = estimate_residual_acceleration(
        mutated,
        baseline,
        valid,
        frame_dt_s=dt,
        end_frame=12,
        causal_window=True,
    )
    np.testing.assert_array_equal(
        first.mean_mps2[:8],
        second.mean_mps2[:8],
    )
    np.testing.assert_array_equal(
        first.variance_m2ps4[:8],
        second.variance_m2ps4[:8],
    )
    assert first.diagnostics["future_frames_used_per_target"] is False


def test_target_uncertainty_increases_with_observation_noise() -> None:
    low = _quadratic_residual_case(noise=1.0e-5)
    high = _quadratic_residual_case(noise=4.0e-4)
    low_estimate = estimate_residual_acceleration(
        low[0], low[1], low[2], frame_dt_s=low[4]
    )
    high_estimate = estimate_residual_acceleration(
        high[0], high[1], high[2], frame_dt_s=high[4]
    )
    assert np.median(
        high_estimate.variance_m2ps4[high_estimate.observed]
    ) > np.median(low_estimate.variance_m2ps4[low_estimate.observed])


def test_graph_smoothed_force_targets_cover_unobserved_nodes_and_obey_cap() -> None:
    observed, baseline, valid, _, dt = _quadratic_residual_case(noise=2.0e-5)
    estimate = estimate_residual_acceleration(
        observed[:, :3],
        baseline[:, :3],
        valid[:, :3],
        frame_dt_s=dt,
    )
    adjacency = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    laplacian = np.eye(4) - adjacency / np.sum(adjacency, axis=1)[:, None]
    smoothed = graph_smooth_residual_acceleration(
        estimate,
        laplacian,
        prior_strength=0.5,
        covariance_probes=32,
    )
    assert smoothed.mean_mps2.shape == (15, 4, 3)
    assert np.any(smoothed.observed[:, 3])
    targets = acceleration_to_force_targets(
        smoothed,
        np.array([0.2, 0.3, 0.4, 0.5]),
        maximum_force_n=0.015,
    )
    assert maximum_node_force_n(targets.mean_n) <= 0.015 + 1.0e-12
    assert np.any(targets.training_weight[:, 3] > 0.0)
    assert np.all(np.isfinite(targets.variance_n2))
    assert np.all(targets.variance_n2 > 0.0)


def _force_episode() -> EquivariantForceEpisode:
    frames, nodes = 8, 4
    rest = np.stack(
        (np.linspace(0.0, 0.3, nodes), np.zeros(nodes), np.zeros(nodes)), axis=1
    )
    positions = np.repeat(rest[None], frames, axis=0)
    positions[:, :, 1] = np.linspace(0.0, 0.04, frames)[:, None]
    velocities = np.zeros_like(positions)
    velocities[1:] = np.diff(positions, axis=0) / 0.04
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    return EquivariantForceEpisode(
        case_id="synthetic",
        positions_m=positions,
        velocities_mps=velocities,
        rest_positions_m=rest,
        object_edges=edges,
        rest_lengths_m=np.full(3, 0.1),
        control_displacement_m=np.zeros_like(positions),
        control_velocity_mps=np.zeros_like(positions),
        action_support=np.array([1.0, 0.0, 0.0, 0.0]),
        external_support=np.zeros((frames, nodes)),
        gravity_mps2=np.array([0.0, 0.0, -9.81]),
        action_activity=np.linspace(0.0, 1.0, frames),
        regime_probabilities=np.tile(
            [1.0, 0.0, 0.0, 0.0, 0.0], (frames, 1)
        ),
        force_targets_n=np.zeros_like(positions),
        force_target_variance_n2=np.full((frames, nodes), 1.0e-5),
        force_target_weight=np.ones((frames, nodes)),
        fit_end_frame=5,
        validation_end_frame=8,
        frame_dt_s=0.04,
        source_checksums={"synthetic_source": "c" * 64},
        information_boundary={
            "target_future_used_for_episode_construction": False,
            "force_targets_use_state_innovation_once": True,
            "prior_reliability_uses_state_residual": False,
        },
        diagnostics={"synthetic": True},
    )


def test_force_episode_round_trip_and_model_compatibility(tmp_path) -> None:
    episode = _force_episode()
    record = write_equivariant_force_episode(tmp_path / "episode", episode)
    loaded = load_equivariant_force_episode(tmp_path / "episode.npz")
    assert loaded.artifact_id == record["artifact_id"]
    np.testing.assert_array_equal(loaded.positions_m, episode.positions_m)
    validate_force_episode_model_compatibility(
        loaded, EquivariantForceConfig(regime_dim=5)
    )
    with pytest.raises(ValueError, match="regime dimension"):
        validate_force_episode_model_compatibility(
            loaded, EquivariantForceConfig(regime_dim=4)
        )


def test_force_episode_rejects_residual_based_prior_reliability() -> None:
    episode = _force_episode()
    payload = {
        name: getattr(episode, name)
        for name in episode.__dataclass_fields__
    }
    payload["information_boundary"] = {
        **episode.information_boundary,
        "prior_reliability_uses_state_residual": True,
    }
    with pytest.raises(ValueError, match="information boundary"):
        EquivariantForceEpisode(**payload)


def _transfer_episode(case_id: str, angle: float) -> EquivariantForceEpisode:
    frames, nodes = 10, 4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rest = np.stack(
        (np.linspace(0.0, 0.3, nodes), np.zeros(nodes), np.zeros(nodes)),
        axis=1,
    ).astype(np.float32)
    rest = rest @ rotation.T
    positions = np.repeat(rest[None], frames, axis=0)
    velocities = np.zeros_like(positions)
    control_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32) @ rotation.T
    control_displacement = np.broadcast_to(
        0.02 * control_direction,
        (frames, nodes, 3),
    ).copy()
    control_velocity = np.broadcast_to(
        0.04 * control_direction,
        (frames, nodes, 3),
    ).copy()
    support = np.broadcast_to(
        np.array([1.0, 0.8, 0.6, 0.4], dtype=np.float32),
        (frames, nodes),
    ).copy()
    activity = np.linspace(0.3, 1.0, frames, dtype=np.float32)
    targets = (
        0.025
        * activity[:, None, None]
        * support[:, :, None]
        * control_direction[None, None]
    )
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    return EquivariantForceEpisode(
        case_id=case_id,
        positions_m=positions,
        velocities_mps=velocities,
        rest_positions_m=rest,
        object_edges=edges,
        rest_lengths_m=np.full(3, 0.1),
        control_displacement_m=control_displacement,
        control_velocity_mps=control_velocity,
        action_support=support,
        external_support=np.zeros((frames, nodes)),
        gravity_mps2=np.array([0.0, 0.0, -9.81]),
        action_activity=activity,
        regime_probabilities=np.tile(
            [1.0, 0.0, 0.0, 0.0, 0.0], (frames, 1)
        ),
        force_targets_n=targets,
        force_target_variance_n2=np.full((frames, nodes), 1.0e-6),
        force_target_weight=np.ones((frames, nodes)),
        fit_end_frame=5,
        validation_end_frame=frames,
        frame_dt_s=0.04,
        source_checksums={"synthetic_source": "d" * 64},
        information_boundary={
            "target_future_used_for_episode_construction": False,
            "force_targets_use_state_innovation_once": True,
            "prior_reliability_uses_state_residual": False,
        },
        diagnostics={"synthetic_transfer": True, "rotation_rad": angle},
    )


def _training_configs():
    model = EquivariantForceConfig(
        hidden_dim=8,
        hidden_layers=1,
        latent_dim=1,
        maximum_force_per_node_n=0.05,
    )
    training = EquivariantForceTrainingConfig(
        training_steps=120,
        adaptation_steps=20,
        learning_rate=0.02,
        adaptation_learning_rate=0.01,
        weight_decay=0.0,
        latent_regularization=0.01,
        adaptation_regularization=0.01,
        gradient_clip=5.0,
        huber_delta_n=0.005,
        seeds=(7,),
        minimum_force_target_improvement=0.10,
        minimum_both_win_folds=2,
        device="cpu",
    )
    return model, training


def test_shared_force_model_transfers_equivariantly_to_unseen_rotation() -> None:
    torch = pytest.importorskip("torch")
    source = [
        _transfer_episode("source_x", 0.0),
        _transfer_episode("source_y", np.pi / 2.0),
    ]
    held = _transfer_episode("held_diagonal", np.pi / 4.0)
    model_config, training_config = _training_configs()
    model, _, fit = fit_shared_equivariant_force_model(
        source,
        torch,
        model_config=model_config,
        training_config=training_config,
        seed=7,
    )
    latent, adaptation = adapt_equivariant_force_latent(
        model,
        held,
        torch,
        model_config=model_config,
        training_config=training_config,
        seed=7,
    )
    metrics = force_target_metrics(
        model,
        held,
        latent,
        torch,
        start_frame=held.fit_end_frame,
        stop_frame=held.validation_end_frame,
        device="cpu",
    )
    assert fit["terminal_step_selected"] is True
    assert adaptation["future_frames_used"] is False
    assert metrics["improvement"] > 0.70


def test_crossfit_force_competence_never_authorizes_warp_promotion(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    episodes = [
        _transfer_episode("case_a", 0.0),
        _transfer_episode("case_b", np.pi / 3.0),
        _transfer_episode("case_c", 2.0 * np.pi / 3.0),
    ]
    folds = [
        {"name": "fold_a", "held_out_cases": ["case_a"]},
        {"name": "fold_b", "held_out_cases": ["case_b"]},
        {"name": "fold_c", "held_out_cases": ["case_c"]},
    ]
    model_config, training_config = _training_configs()
    summary = crossfit_equivariant_force_competence(
        episodes,
        folds,
        tmp_path / "crossfit",
        torch,
        model_config=model_config,
        training_config=training_config,
    )
    assert summary["force_target_competence_passed"] is True
    assert summary["official_warp_promotion_authorized"] is False
    assert set(summary["case_mean_force_target_improvement"]) == {
        "case_a",
        "case_b",
        "case_c",
    }
    assert (tmp_path / "crossfit" / "summary.json").is_file()


@pytest.mark.parametrize(
    "folds",
    [
        [
            {"name": "fold_a", "held_out_cases": ["case_a"]},
            {"name": "fold_b", "held_out_cases": ["case_b"]},
        ],
        [
            {"name": "fold_a", "held_out_cases": ["case_a", "case_b"]},
            {"name": "fold_b", "held_out_cases": ["case_b", "case_c"]},
        ],
    ],
)
def test_crossfit_rejects_incomplete_or_overlapping_case_coverage(
    tmp_path,
    folds,
) -> None:
    torch = pytest.importorskip("torch")
    episodes = [
        _transfer_episode("case_a", 0.0),
        _transfer_episode("case_b", np.pi / 3.0),
        _transfer_episode("case_c", 2.0 * np.pi / 3.0),
    ]
    model_config, training_config = _training_configs()
    with pytest.raises(ValueError, match="disjoint complete"):
        crossfit_equivariant_force_competence(
            episodes,
            folds,
            tmp_path / "invalid",
            torch,
            model_config=model_config,
            training_config=training_config,
        )


def _dense_random_walk_laplacian(node_count, edges):
    adjacency = np.zeros((node_count, node_count), dtype=float)
    values = np.asarray(edges, dtype=int)
    adjacency[values[:, 0], values[:, 1]] = 1.0
    adjacency[values[:, 1], values[:, 0]] = 1.0
    degree = np.sum(adjacency, axis=1)
    inverse = np.zeros_like(degree)
    inverse[degree > 0.0] = 1.0 / degree[degree > 0.0]
    return np.diag(degree > 0.0) - inverse[:, None] * adjacency


def test_released_case_adapter_builds_causal_typed_force_episode(
    tmp_path,
    monkeypatch,
) -> None:
    import bayesian_phystwin.phystwin_equivariant_force_source as source_module

    monkeypatch.setattr(
        source_module,
        "normalized_spring_laplacian",
        _dense_random_walk_laplacian,
    )
    frames, original, dt = 9, 4, 0.04
    structure = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.02, 0.0],
            [0.3, 0.0, 0.0],
            [0.15, 0.04, 0.02],
            [0.15, -0.03, -0.02],
        ]
    )
    baseline = np.repeat(structure[None], frames, axis=0)
    time = np.arange(frames) * dt
    residual = np.zeros((frames, original, 3))
    residual[:, :, 1] = (
        0.5
        * np.square(time)[:, None]
        * np.array([0.10, 0.08, 0.06, 0.04])[None]
    )
    controller = np.zeros((frames, 1, 3))
    controller[:, 0, 0] = 0.02 + 0.01 * np.arange(frames)
    data = {
        "object_points": baseline[:, :original] + residual,
        "object_visibilities": np.ones((frames, original), dtype=bool),
        "object_motions_valid": np.ones((frames, original), dtype=bool),
        "controller_points": controller,
        "surface_points": structure[original : original + 1],
        "interior_points": structure[original + 1 :],
    }
    optimal = {
        "object_radius": 0.25,
        "object_max_neighbours": 6,
        "controller_radius": 0.25,
        "controller_max_neighbours": 4,
    }
    final_path = tmp_path / "final_data.pkl"
    baseline_path = tmp_path / "inference.pkl"
    optimal_path = tmp_path / "optimal_params.pkl"
    for path, value in (
        (final_path, data),
        (baseline_path, baseline),
        (optimal_path, optimal),
    ):
        with path.open("wb") as handle:
            pickle.dump(value, handle)

    episode = build_released_equivariant_force_episode(
        "adapter_case",
        final_path,
        baseline_path,
        optimal_path,
        fit_end_frame=5,
        validation_end_frame=frames,
        frame_dt_s=dt,
        target_config=ForceTargetBuildConfig(
            window_radius=2,
            graph_covariance_probes=2,
        ),
        activity_speed_mps=0.1,
    )
    assert episode.positions_m.shape == (frames, len(structure), 3)
    assert episode.information_boundary["force_targets_are_causal_per_frame"]
    assert episode.diagnostics["target"]["causal_window"] is True
    assert np.any(episode.force_target_weight[2:] > 0.0)
    assert maximum_node_force_n(episode.force_targets_n) <= 0.5 + 1.0e-12
    assert set(episode.source_checksums) == {
        "baseline_trajectory",
        "final_data",
        "optimal_params",
    }


def test_source_protocol_is_typed_and_rejects_overlapping_folds(
    tmp_path,
) -> None:
    protocol_path = (
        Path(__file__).parents[1]
        / "configs"
        / "sota"
        / "phystwin_equivariant_force_source_v1.json"
    )
    protocol = load_equivariant_force_source_protocol(
        protocol_path,
        device="cpu",
    )
    assert len(protocol.payload["source_cases"]) == 17
    assert protocol.training.device == "cpu"
    assert protocol.training.seeds == (17, 43, 101)

    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload["source_folds"][1]["held_out_cases"].append(
        payload["source_folds"][0]["held_out_cases"][0]
    )
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint complete"):
        load_equivariant_force_source_protocol(invalid)


def _official_gate_records(protocol, *, ratio: float = 0.94):
    return [
        {
            "case_id": case,
            "target_artifacts_opened": False,
            "zero_force_bitwise_parity": True,
            "readout_correction_shrinkage": 0.20,
            "reference": {
                "chamfer_distance_m": 0.010,
                "track_error_m": 0.020,
                "late_track_error_m": 0.024,
            },
            "candidate": {
                "chamfer_distance_m": ratio * 0.010,
                "track_error_m": ratio * 0.020,
                "late_track_error_m": ratio * 0.024,
            },
        }
        for case in protocol.payload["source_cases"]
    ]


def test_official_warp_gate_passes_only_the_locked_multimetric_result() -> None:
    protocol_path = (
        Path(__file__).parents[1]
        / "configs"
        / "sota"
        / "phystwin_equivariant_force_source_v1.json"
    )
    protocol = load_equivariant_force_source_protocol(protocol_path)
    result = evaluate_equivariant_force_official_warp_gate(
        _official_gate_records(protocol),
        protocol,
        force_target_competence_passed=True,
    )
    assert result["source_gate_passed"] is True
    assert result["independent_preregistered_evaluation_authorized"] is True
    assert result["historical_target_access_authorized"] is False
    assert all(result["checks"].values())


def test_official_warp_gate_fails_on_parity_or_worst_case_regression() -> None:
    protocol_path = (
        Path(__file__).parents[1]
        / "configs"
        / "sota"
        / "phystwin_equivariant_force_source_v1.json"
    )
    protocol = load_equivariant_force_source_protocol(protocol_path)
    records = _official_gate_records(protocol)
    records[0]["zero_force_bitwise_parity"] = False
    records[1]["candidate"]["track_error_m"] = 1.06 * records[1]["reference"][
        "track_error_m"
    ]
    result = evaluate_equivariant_force_official_warp_gate(
        records,
        protocol,
        force_target_competence_passed=True,
    )
    assert result["source_gate_passed"] is False
    assert result["independent_preregistered_evaluation_authorized"] is False
    assert result["checks"]["zero_force_bitwise_parity"] is False
    assert result["checks"]["maximum_case_metric_ratio"] is False


def test_official_warp_gate_cannot_override_failed_force_competence() -> None:
    protocol_path = (
        Path(__file__).parents[1]
        / "configs"
        / "sota"
        / "phystwin_equivariant_force_source_v1.json"
    )
    protocol = load_equivariant_force_source_protocol(protocol_path)
    result = evaluate_equivariant_force_official_warp_gate(
        _official_gate_records(protocol),
        protocol,
        force_target_competence_passed=False,
    )
    assert result["source_gate_passed"] is False
    assert result["checks"]["force_target_competence"] is False
