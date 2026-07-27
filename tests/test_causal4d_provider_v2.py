from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.causal4d_provider_v2 as provider_api
import bayesian_phystwin.phystwin.replay as replay_impl
from bayesian_phystwin.causal4d_provider_v2 import (
    CAUSAL4D_PROVIDER_API_VERSION,
    InitialReplayRequestV1,
    OfficialPhysTwinReplayProviderV2,
    PhysTwinReplayProviderV2,
    ReplayTrajectoryV1,
    RestartReplayRequestV1,
    causal4d_provider_manifest,
    create_official_replay_provider_v2,
)


class _FakeArray:
    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float32).copy()

    @property
    def shape(self):
        return self.values.shape

    def contiguous(self):
        return self

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


class _FakeTarget(_FakeArray):
    def copy_(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float32).copy()


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    float32 = np.float32

    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    @staticmethod
    def no_grad():
        return nullcontext()

    @staticmethod
    def as_tensor(values, *, dtype, device):
        del device
        return _FakeArray(np.asarray(values, dtype=dtype))


class _FakeWarp:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _FakeSimulator:
    def __init__(self) -> None:
        self.group_log_scale_tensor = _FakeTarget(np.zeros(2, dtype=np.float32))
        self.controller_points = _FakeArray(np.zeros((4, 1, 3), dtype=np.float32))


def _provider() -> tuple[
    OfficialPhysTwinReplayProviderV2,
    _FakeSimulator,
    _FakeTorch,
    _FakeWarp,
]:
    simulator = _FakeSimulator()
    torch = _FakeTorch()
    warp = _FakeWarp()
    provider = OfficialPhysTwinReplayProviderV2(
        simulator,
        torch,
        warp,
        device="cuda:0",
        frame_dt_s=0.04,
        simulator_configuration_id="sim-config-001",
        released_initial_state_id="released-state-001",
    )
    return provider, simulator, torch, warp


def test_manifest_declares_owned_stateless_replay_contract() -> None:
    manifest = causal4d_provider_manifest(provider_revision="abc123")
    assert manifest["schema_version"] == CAUSAL4D_PROVIDER_API_VERSION == 2
    assert manifest["metadata"] == {
        "provider_api": "bayesian_phystwin.causal4d_provider_v2",
        "provider_api_version": 2,
        "legacy_provider_api": "bayesian_phystwin.causal4d_provider_v1",
    }
    assert {
        "immutable_replay_trajectories",
        "restart_velocity_history",
        "stateless_replay_requests",
        "typed_replay_requests",
    }.issubset(set(manifest["capabilities"]))
    assert manifest["artifact_schema_versions"]["ReplayTrajectory"] == 1


def test_v2_excludes_legacy_mutation_and_unchecked_artifact_helpers() -> None:
    for name in (
        "initialize_simulator",
        "load_pickle",
        "set_simulator_arrays",
        "state_numpy",
    ):
        assert not hasattr(provider_api, name)


def test_requests_copy_and_freeze_all_array_inputs() -> None:
    scales = np.asarray((0.2, -0.1), dtype=np.float32)
    controls = np.arange(12, dtype=np.float32).reshape(4, 1, 3)
    position = np.zeros((2, 3), dtype=np.float32)
    velocity = np.ones((2, 3), dtype=np.float32)
    request = RestartReplayRequestV1(
        request_id="restart-1",
        simulator_configuration_id="sim-config-001",
        initial_state_id="belief-particle-7",
        group_log_scales=scales,
        controller_points_m=controls,
        position_m=position,
        velocity_mps=velocity,
        start_frame=1,
        stop_frame=4,
    )

    scales[:] = 9.0
    controls[:] = 9.0
    position[:] = 9.0
    velocity[:] = 9.0
    np.testing.assert_allclose(request.group_log_scales, (0.2, -0.1))
    np.testing.assert_allclose(request.controller_points_m.reshape(-1), np.arange(12))
    np.testing.assert_allclose(request.position_m, 0.0)
    np.testing.assert_allclose(request.velocity_mps, 1.0)
    assert not request.group_log_scales.flags.writeable
    assert not request.controller_points_m.flags.writeable
    assert not request.position_m.flags.writeable
    assert not request.velocity_mps.flags.writeable
    with pytest.raises(ValueError):
        request.position_m[0, 0] = 1.0


def test_initial_replay_is_explicit_and_returns_immutable_provenance(monkeypatch) -> None:
    provider, simulator, torch, warp = _provider()
    assert isinstance(provider, PhysTwinReplayProviderV2)
    monkeypatch.setattr(
        replay_impl,
        "_rollout_initial_trajectory",
        lambda simulator_arg, wp_arg, *, frame_count: (
            np.zeros((frame_count, 2, 3), dtype=np.float32),
            np.ones((frame_count, 2, 3), dtype=np.float32),
        ),
    )
    controls = np.arange(12, dtype=np.float32).reshape(4, 1, 3)
    request = InitialReplayRequestV1(
        request_id="initial-1",
        simulator_configuration_id="sim-config-001",
        initial_state_id="released-state-001",
        group_log_scales=np.asarray((0.25, -0.2)),
        controller_points_m=controls,
        frame_count=3,
    )

    trajectory = provider.replay(request)

    assert isinstance(trajectory, ReplayTrajectoryV1)
    assert trajectory.positions_m.shape == trajectory.velocities_mps.shape == (3, 2, 3)
    np.testing.assert_array_equal(trajectory.frame_ids, (0, 1, 2))
    assert trajectory.dt_s == 0.04
    assert trajectory.request_id == "initial-1"
    assert trajectory.simulator_configuration_id == "sim-config-001"
    assert trajectory.initial_state_id == "released-state-001"
    assert not trajectory.positions_m.flags.writeable
    assert not trajectory.velocities_mps.flags.writeable
    assert not trajectory.frame_ids.flags.writeable
    np.testing.assert_allclose(simulator.group_log_scale_tensor.values, (0.25, -0.2))
    np.testing.assert_allclose(simulator.controller_points.values, controls)
    assert warp.synchronize_calls == 2

    provider.close()
    provider.close()
    assert torch.cuda.empty_cache_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        provider.replay(request)


def test_restart_replay_returns_position_and_velocity_history(monkeypatch) -> None:
    provider, _, _, _ = _provider()
    captured = {}

    def restart(
        simulator_arg,
        torch_arg,
        wp_arg,
        position,
        velocity,
        **kwargs,
    ):
        captured["position"] = position.copy()
        captured["velocity"] = velocity.copy()
        captured["kwargs"] = kwargs
        count = kwargs["stop_frame"] - kwargs["start_frame"]
        return (
            np.repeat(position[None], count, axis=0),
            np.repeat(velocity[None], count, axis=0),
        )

    monkeypatch.setattr(replay_impl, "_rollout_restart_trajectory", restart)
    request = RestartReplayRequestV1(
        request_id="restart-1",
        simulator_configuration_id="sim-config-001",
        initial_state_id="belief-particle-7",
        group_log_scales=np.asarray((0.1, 0.2)),
        controller_points_m=np.zeros((4, 1, 3), dtype=np.float32),
        position_m=np.zeros((2, 3), dtype=np.float32),
        velocity_mps=np.ones((2, 3), dtype=np.float32),
        start_frame=1,
        stop_frame=4,
    )

    trajectory = provider.replay(request)

    assert trajectory.positions_m.shape == trajectory.velocities_mps.shape == (3, 2, 3)
    np.testing.assert_array_equal(trajectory.frame_ids, (1, 2, 3))
    np.testing.assert_allclose(trajectory.positions_m, 0.0)
    np.testing.assert_allclose(trajectory.velocities_mps, 1.0)
    np.testing.assert_allclose(captured["position"], 0.0)
    np.testing.assert_allclose(captured["velocity"], 1.0)
    assert captured["kwargs"]["device"] == "cuda:0"
    assert trajectory.initial_state_id == "belief-particle-7"


def test_configuration_and_released_state_mismatches_fail_before_mutation() -> None:
    provider, _, _, warp = _provider()
    controls = np.zeros((4, 1, 3), dtype=np.float32)
    wrong_configuration = InitialReplayRequestV1(
        request_id="bad-config",
        simulator_configuration_id="other-config",
        initial_state_id="released-state-001",
        group_log_scales=np.zeros(2),
        controller_points_m=controls,
        frame_count=2,
    )
    wrong_state = InitialReplayRequestV1(
        request_id="bad-state",
        simulator_configuration_id="sim-config-001",
        initial_state_id="other-state",
        group_log_scales=np.zeros(2),
        controller_points_m=controls,
        frame_count=2,
    )

    with pytest.raises(ValueError, match="configuration"):
        provider.replay(wrong_configuration)
    with pytest.raises(ValueError, match="released state"):
        provider.replay(wrong_state)
    assert warp.synchronize_calls == 0


def test_provider_preflights_request_shapes_before_mutating_simulator() -> None:
    provider, simulator, _, warp = _provider()
    request = InitialReplayRequestV1(
        request_id="bad-controls",
        simulator_configuration_id="sim-config-001",
        initial_state_id="released-state-001",
        group_log_scales=np.asarray((0.5, -0.5)),
        controller_points_m=np.zeros((3, 1, 3), dtype=np.float32),
        frame_count=2,
    )

    with pytest.raises(ValueError, match="controller points"):
        provider.replay(request)

    np.testing.assert_allclose(simulator.group_log_scale_tensor.values, 0.0)
    assert warp.synchronize_calls == 0


def test_factory_binds_frame_interval_and_identifiers(monkeypatch, tmp_path: Path) -> None:
    simulator = _FakeSimulator()
    torch = _FakeTorch()
    warp = _FakeWarp()
    calls = {}

    def initialize(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return simulator, torch, warp, {"runtime": "fake"}

    monkeypatch.setattr(replay_impl, "_initialize_official_simulator", initialize)
    provider = create_official_replay_provider_v2(
        tmp_path,
        {"controller_points": np.zeros((4, 1, 3))},
        {"spring": 1.0},
        tmp_path / "checkpoint.pt",
        object(),
        num_surface_points=3,
        original_count=2,
        dt=0.005,
        num_substeps=8,
        self_collision=False,
        simulator_configuration_id="sim-config-001",
        released_initial_state_id="released-state-001",
        spring_parameterization="grouped",
        device="cpu",
    )

    assert isinstance(provider, PhysTwinReplayProviderV2)
    assert provider.frame_dt_s == pytest.approx(0.04)
    assert provider.simulator_configuration_id == "sim-config-001"
    assert provider.released_initial_state_id == "released-state-001"
    assert calls["kwargs"]["spring_parameterization"] == "grouped"
    assert calls["kwargs"]["device"] == "cpu"
    provider.close()
