from __future__ import annotations

import json

import numpy as np
import pytest

from bayesian_phystwin.process_discrepancy import (
    LatentForceBelief,
    StableLatentForceProcess,
    build_dynamics_consistent_force_basis,
    condition_latent_force_belief,
    forecast_latent_force_belief,
    mechanical_work_summary,
    predict_latent_force_belief,
)
from bayesian_phystwin.process_discrepancy_evidence import (
    build_process_discrepancy_candidate_configuration,
    compare_process_discrepancy_rollouts,
    load_source_frozen_process_selection,
    select_source_frozen_process_candidate,
    write_source_frozen_process_selection,
)
from bayesian_phystwin.process_discrepancy_replay import (
    OfficialProcessDiscrepancyReplayAdapter,
    replay_with_process_force_schedule,
)


def _graph_basis(node_count: int = 10, rank: int = 4) -> np.ndarray:
    coordinate = np.linspace(-1.0, 1.0, node_count)
    columns = (
        np.ones(node_count),
        coordinate,
        coordinate**2,
        np.sin(np.pi * coordinate),
        np.cos(np.pi * coordinate),
    )
    return np.linalg.qr(np.column_stack(columns[:rank]), mode="reduced")[0]


def _positions(node_count: int = 10) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, node_count, endpoint=False)
    return np.column_stack((np.cos(angle), np.sin(angle), 0.2 * np.sin(2.0 * angle)))


def _force_basis():
    node_count = 10
    contact = np.zeros(node_count, dtype=bool)
    contact[:8] = True
    attached = np.zeros(node_count, dtype=bool)
    attached[0] = True
    return build_dynamics_consistent_force_basis(
        _graph_basis(node_count),
        _positions(node_count),
        contact_node_mask=contact,
        attached_node_mask=attached,
        contact_policy="contact_only",
        enforce_zero_net_force=True,
        enforce_zero_net_torque=True,
    )


def test_force_basis_respects_support_force_and_torque_constraints() -> None:
    basis = _force_basis()

    assert basis.coefficient_count >= 1
    assert not basis.active_node_mask[0]
    assert not np.any(basis.active_node_mask[8:])
    np.testing.assert_array_equal(
        basis.force_basis_per_coefficient[~basis.active_node_mask],
        0.0,
    )
    residuals = basis.constraint_residuals(
        np.linspace(0.1, 0.3, basis.coefficient_count)
    )
    np.testing.assert_allclose(residuals["net_force_per_coefficient"], 0.0, atol=1e-10)
    np.testing.assert_allclose(
        residuals["net_torque_per_coefficient"],
        0.0,
        atol=1e-10,
    )
    assert len(basis.basis_id) == 64


def test_force_basis_can_retain_external_contact_resultants_when_declared() -> None:
    basis = build_dynamics_consistent_force_basis(
        _graph_basis(),
        _positions(),
        contact_policy="all_supported",
        enforce_zero_net_force=False,
        enforce_zero_net_torque=False,
        maximum_force_rank=5,
    )

    assert basis.coefficient_count == 5
    assert basis.diagnostics["constraint_row_count"] == 0


def test_ou_process_is_strictly_stable_and_stationary() -> None:
    process = StableLatentForceProcess.isotropic_ornstein_uhlenbeck(
        5,
        frame_dt_s=0.05,
        half_life_s=0.4,
        stationary_std_n=0.3,
    )
    stationary = process.stationary_covariance_n2()
    predicted = (
        process.transition_matrix @ stationary @ process.transition_matrix.T
        + process.process_covariance_n2
    )

    assert process.spectral_radius < 1.0
    assert len(process.process_id) == 64
    np.testing.assert_allclose(stationary, 0.09 * np.eye(5), atol=1e-12)
    np.testing.assert_allclose(predicted, stationary, atol=1e-12)


def test_general_stable_process_stationary_covariance_solves_lyapunov() -> None:
    transition = np.asarray(((0.65, 0.15), (0.0, 0.45)))
    process = StableLatentForceProcess(
        transition_matrix=transition,
        process_covariance_n2=np.diag((0.03, 0.02)),
        frame_dt_s=0.05,
    )
    stationary = process.stationary_covariance_n2()

    np.testing.assert_allclose(
        stationary,
        transition @ stationary @ transition.T + process.process_covariance_n2,
        atol=1e-12,
    )
    assert np.min(np.linalg.eigvalsh(stationary)) >= 0.0


def test_prediction_propagates_uncertainty_and_preserves_exact_zero_mean() -> None:
    basis = _force_basis()
    process = StableLatentForceProcess.isotropic_ornstein_uhlenbeck(
        basis.coefficient_count,
        frame_dt_s=0.05,
        half_life_s=0.3,
        stationary_std_n=0.2,
    )
    initial = LatentForceBelief.zero(basis.coefficient_count)
    predicted = predict_latent_force_belief(initial, process, steps=3)

    np.testing.assert_array_equal(predicted.mean_n, 0.0)
    assert np.trace(predicted.covariance_n2) > 0.0
    np.testing.assert_array_equal(initial.force_mean_n(basis), 0.0)


def test_conditioning_recovers_coefficients_and_reduces_uncertainty() -> None:
    rng = np.random.default_rng(7)
    basis = _force_basis()
    coefficient_count = basis.coefficient_count
    response = rng.normal(size=(5 * coefficient_count, coefficient_count))
    expected = rng.normal(scale=0.2, size=coefficient_count)
    innovation = response @ expected
    prior = LatentForceBelief(
        mean_n=np.zeros(coefficient_count),
        covariance_n2=10.0 * np.eye(coefficient_count),
    )

    result = condition_latent_force_belief(
        prior,
        response,
        innovation,
        np.full(len(innovation), 1e-8),
    )

    np.testing.assert_allclose(result.posterior.mean_n, expected, atol=1e-5)
    assert result.diagnostics["posterior_covariance_trace_n2"] < result.diagnostics[
        "prior_covariance_trace_n2"
    ]
    assert result.diagnostics["innovation_rmse_after"] < 1e-5


def test_conditioning_preserves_zero_variance_coefficient_subspace() -> None:
    prior = LatentForceBelief(
        mean_n=np.asarray((0.1, -0.2, 0.3)),
        covariance_n2=np.diag((1.0, 0.0, 0.2)),
    )
    response = np.asarray(((0.0, 1.0, 0.0), (1.0, 0.0, 1.0)))
    result = condition_latent_force_belief(
        prior,
        response,
        np.asarray((5.0, 0.0)),
        np.asarray((1e-3, 1e-3)),
    )

    assert result.posterior.mean_n[1] == prior.mean_n[1]
    np.testing.assert_array_equal(result.posterior.covariance_n2[1], 0.0)
    np.testing.assert_array_equal(result.posterior.covariance_n2[:, 1], 0.0)
    assert result.diagnostics["prior_covariance_rank"] == 2


def test_work_regularization_reduces_instantaneous_power() -> None:
    basis = _force_basis()
    coefficient_count = basis.coefficient_count
    velocity = np.linspace(-0.4, 0.7, basis.node_count * 3).reshape(
        basis.node_count,
        3,
    )
    power = basis.mechanical_power_jacobian(velocity)
    assert np.linalg.norm(power) > 0.0
    response = np.eye(coefficient_count)
    target = power / np.linalg.norm(power)
    prior = LatentForceBelief(
        mean_n=np.zeros(coefficient_count),
        covariance_n2=2.0 * np.eye(coefficient_count),
    )
    unregularized = condition_latent_force_belief(
        prior,
        response,
        target,
        np.full(coefficient_count, 1e-3),
    ).posterior
    regularized_result = condition_latent_force_belief(
        prior,
        response,
        target,
        np.full(coefficient_count, 1e-3),
        force_basis=basis,
        velocity_mps=velocity,
        work_precision_per_watt2=1e4,
    )

    assert abs(power @ regularized_result.posterior.mean_n) < abs(
        power @ unregularized.mean_n
    )
    assert abs(regularized_result.diagnostics["posterior_mechanical_power_w"]) < abs(
        regularized_result.diagnostics["prior_mechanical_power_w"]
        - power @ target
    )


def test_forecast_exposes_coefficient_and_node_force_uncertainty() -> None:
    basis = _force_basis()
    process = StableLatentForceProcess.isotropic_ornstein_uhlenbeck(
        basis.coefficient_count,
        frame_dt_s=0.05,
        half_life_s=0.25,
        stationary_std_n=0.15,
    )
    initial = LatentForceBelief(
        mean_n=np.ones(basis.coefficient_count) * 0.05,
        covariance_n2=np.eye(basis.coefficient_count) * 0.01,
    )
    forecast = forecast_latent_force_belief(
        initial,
        process,
        basis,
        frame_count=6,
    )

    assert forecast.coefficient_mean_n.shape == (6, basis.coefficient_count)
    assert forecast.force_mean_n.shape == (6, basis.node_count, 3)
    assert forecast.node_force_covariance_n2.shape == (6, basis.node_count, 3, 3)
    assert np.linalg.norm(forecast.coefficient_mean_n[-1]) < np.linalg.norm(
        forecast.coefficient_mean_n[0]
    )


def test_mechanical_work_summary_uses_signed_and_absolute_work() -> None:
    forces = np.asarray(
        (
            ((1.0, 0.0, 0.0),),
            ((-1.0, 0.0, 0.0),),
        )
    )
    velocity = np.ones_like(forces)
    summary = mechanical_work_summary(forces, velocity, frame_dt_s=0.2)

    assert summary["total_signed_work_j"] == pytest.approx(0.0)
    assert summary["total_absolute_work_j"] == pytest.approx(0.4)


class _DispatchProvider:
    def __init__(self) -> None:
        self.baseline_calls = 0
        self.force_calls = 0
        self.clear_calls = 0
        self.baseline = np.arange(18, dtype=float).reshape(2, 3, 3)

    def clear_external_forces(self) -> None:
        self.clear_calls += 1

    def replay_restart(self, position_m, velocity_mps, *, start_frame, stop_frame):
        self.baseline_calls += 1
        return self.baseline

    def replay_restart_with_force_schedule(
        self,
        position_m,
        velocity_mps,
        force_schedule_n,
        *,
        start_frame,
        stop_frame,
    ):
        self.force_calls += 1
        return self.baseline + 1.0


def test_zero_force_schedule_uses_exact_baseline_dispatch() -> None:
    provider = _DispatchProvider()
    state = np.zeros((3, 3))
    result = replay_with_process_force_schedule(
        provider,
        state,
        state,
        np.zeros((2, 3, 3)),
        start_frame=4,
        stop_frame=6,
    )

    assert result is provider.baseline
    assert provider.clear_calls == 1
    assert provider.baseline_calls == 1
    assert provider.force_calls == 0


def test_nonzero_force_schedule_uses_opt_in_dispatch() -> None:
    provider = _DispatchProvider()
    state = np.zeros((3, 3))
    schedule = np.zeros((2, 3, 3))
    schedule[0, 1, 0] = 0.1
    result = replay_with_process_force_schedule(
        provider,
        state,
        state,
        schedule,
        start_frame=4,
        stop_frame=6,
    )

    np.testing.assert_array_equal(result, provider.baseline + 1.0)
    assert provider.baseline_calls == 0
    assert provider.force_calls == 1


def test_official_adapter_zero_schedule_never_enters_force_runtime() -> None:
    class Simulator:
        def __init__(self) -> None:
            self.clear_count = 0

        def clear_external_forces(self) -> None:
            self.clear_count += 1

    class Warp:
        def __init__(self) -> None:
            self.sync_count = 0

        def synchronize(self) -> None:
            self.sync_count += 1

    class Provider:
        def __init__(self) -> None:
            self._simulator = Simulator()
            self._torch = object()
            self._wp = Warp()
            self._device = "cpu"
            self.calls = 0
            self.result = np.zeros((2, 3, 3))

        def _require_open(self) -> None:
            return None

        def replay_restart(self, position_m, velocity_mps, *, start_frame, stop_frame):
            self.calls += 1
            return self.result

    provider = Provider()
    adapter = OfficialProcessDiscrepancyReplayAdapter(provider)
    state = np.zeros((3, 3))
    result = adapter.replay_restart_with_force_schedule(
        state,
        state,
        np.zeros((2, 3, 3)),
        start_frame=0,
        stop_frame=2,
    )

    assert result is provider.result
    assert provider.calls == 1
    assert provider._simulator.clear_count == 1


def test_official_adapter_applies_each_force_and_clears_after_replay() -> None:
    class Tensor:
        def __init__(self, values) -> None:
            self.values = np.asarray(values, dtype=np.float32).copy()

        def contiguous(self):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Torch:
        float32 = np.float32

        @staticmethod
        def as_tensor(values, *, dtype, device):
            assert dtype is np.float32
            assert device == "cpu"
            return Tensor(values)

    class State:
        def __init__(self, node_count: int) -> None:
            self.wp_x = Tensor(np.zeros((node_count, 3)))
            self.wp_v = Tensor(np.zeros((node_count, 3)))

    class Simulator:
        def __init__(self, node_count: int) -> None:
            self.object_collision_flag = False
            self.wp_states = [State(node_count), State(node_count)]
            self.forward_graph = object()
            self.applied = []
            self.clear_count = 0

        def clear_external_forces(self) -> None:
            self.clear_count += 1

        def set_external_forces(self, values) -> None:
            array = np.asarray(values, dtype=np.float32).copy()
            self.applied.append(array)
            self.wp_states[-1].wp_x = Tensor(array)

        def set_init_state(self, position, velocity) -> None:
            self.wp_states[0].wp_x = position
            self.wp_states[0].wp_v = velocity

        def set_controller_target(self, frame, *, pure_inference) -> None:
            assert pure_inference is True
            assert frame in {4, 5}

    class Warp:
        vec3 = object()

        def __init__(self) -> None:
            self.sync_count = 0
            self.launch_count = 0

        @staticmethod
        def from_torch(value, *, dtype, requires_grad):
            assert requires_grad is False
            return value

        def synchronize(self) -> None:
            self.sync_count += 1

        def capture_launch(self, graph) -> None:
            self.launch_count += 1

        @staticmethod
        def to_torch(value):
            return value

    class Provider:
        def __init__(self) -> None:
            self._simulator = Simulator(3)
            self._torch = Torch()
            self._wp = Warp()
            self._device = "cpu"

        def _require_open(self) -> None:
            return None

        def replay_restart(self, *args, **kwargs):
            raise AssertionError("nonzero schedule entered the baseline path")

    provider = Provider()
    adapter = OfficialProcessDiscrepancyReplayAdapter(provider)
    schedule = np.arange(18, dtype=float).reshape(2, 3, 3) * 0.01
    state = np.zeros((3, 3))
    result = adapter.replay_restart_with_force_schedule(
        state,
        state,
        schedule,
        start_frame=4,
        stop_frame=6,
    )

    np.testing.assert_allclose(result, schedule.astype(np.float32))
    assert len(provider._simulator.applied) == 2
    assert provider._simulator.clear_count == 2
    assert provider._wp.launch_count == 2


def _comparison(scale: float) -> dict:
    reference = np.zeros((3, 2, 3))
    baseline = np.ones_like(reference) * 2.0
    readout = np.ones_like(reference) * 1.5
    process = np.ones_like(reference) * scale
    return compare_process_discrepancy_rollouts(
        reference,
        baseline,
        readout,
        process,
    )


def test_comparison_reports_both_required_controls() -> None:
    summary = _comparison(1.0)

    assert summary["comparisons"][
        "process_reduction_vs_baseline_fraction"
    ] == pytest.approx(0.5)
    assert summary["comparisons"][
        "process_reduction_vs_readout_fraction"
    ] == pytest.approx(1.0 / 3.0)


def test_comparison_rejects_method_specific_missing_outputs() -> None:
    reference = np.zeros((2, 2, 3))
    baseline = np.ones_like(reference)
    readout = np.ones_like(reference)
    process = np.ones_like(reference)
    process[1, 1, 2] = np.nan

    with pytest.raises(ValueError, match="common support"):
        compare_process_discrepancy_rollouts(
            reference,
            baseline,
            readout,
            process,
        )


def test_source_frozen_selection_roundtrip_and_target_seal(tmp_path) -> None:
    basis = _force_basis()
    process = StableLatentForceProcess.isotropic_ornstein_uhlenbeck(
        basis.coefficient_count,
        frame_dt_s=0.05,
        half_life_s=0.3,
        stationary_std_n=0.2,
    )
    candidate = build_process_discrepancy_candidate_configuration(
        basis,
        process,
        response_model_id="c" * 64,
        work_precision_per_watt2=10.0,
        coefficient_precision_per_n2=0.5,
    )
    summaries = {"source_a": _comparison(1.0), "source_b": _comparison(1.1)}
    checksums = {"source_a": "a" * 64, "source_b": "b" * 64}
    selection = select_source_frozen_process_candidate(
        candidate,
        summaries,
        held_out_case_ids=("target_a", "target_b"),
        source_checksums=checksums,
        minimum_mean_improvement_fraction=0.2,
        maximum_case_regression_fraction=0.0,
    )
    written = write_source_frozen_process_selection(
        tmp_path / "selection.json",
        selection,
    )
    loaded = load_source_frozen_process_selection(written["path"])

    assert selection.selected is True
    assert loaded.selection_id == selection.selection_id
    assert len(selection.selection_id) == 64
    assert selection.candidate_configuration["force_basis_id"] == basis.basis_id
    assert selection.candidate_configuration["process_id"] == process.process_id

    payload = json.loads((tmp_path / "selection.json").read_text())
    payload["candidate_configuration"]["work_precision_per_watt2"] += 1.0
    (tmp_path / "selection.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="candidate_id"):
        load_source_frozen_process_selection(tmp_path / "selection.json")

    write_source_frozen_process_selection(tmp_path / "selection.json", selection)
    payload = json.loads((tmp_path / "selection.json").read_text())
    payload["source_metrics"]["mean_process_coordinate_rmse_m"] += 0.1
    (tmp_path / "selection.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_source_frozen_process_selection(tmp_path / "selection.json")

    with pytest.raises(ValueError, match="target outcomes"):
        select_source_frozen_process_candidate(
            candidate,
            summaries,
            held_out_case_ids=("target_a",),
            source_checksums=checksums,
            target_outcomes_used_for_selection=True,
        )
