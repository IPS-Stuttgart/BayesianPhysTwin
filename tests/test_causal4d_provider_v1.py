from __future__ import annotations

import builtins
import hashlib
import pickle
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.causal4d_provider_v1 as provider_api
from bayesian_phystwin.causal4d_provider_v1 import (
    CAUSAL4D_PROVIDER_API_VERSION,
    OfficialPhysTwinReplayProvider,
    PhysTwinReplayProvider,
    build_lift_map,
    causal4d_provider_manifest,
    create_official_replay_provider,
    lift_residual,
    load_pickle,
    sha256_file,
    target_validity,
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


def test_manifest_exposes_versioned_causal4d_contract() -> None:
    manifest = causal4d_provider_manifest(provider_revision="abc123")
    assert manifest["provider_name"] == "bayesian-phystwin"
    assert manifest["provider_revision"] == "abc123"
    assert manifest["schema_version"] == CAUSAL4D_PROVIDER_API_VERSION
    assert manifest["metadata"] == {
        "provider_api": "bayesian_phystwin.causal4d_provider_v1",
        "provider_api_version": 1,
    }
    assert {
        "artifact_checksums",
        "particle_endpoint_position",
        "particle_endpoint_velocity",
        "physical_parameter_particles",
        "phystwin_replay",
    }.issubset(set(manifest["capabilities"]))
    assert manifest["artifact_schema_versions"] == {
        "GraphBelief": 1,
        "TwinBelief": 1,
    }


def test_artifact_helpers_are_public_and_stable(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.pkl"
    value = {"x": np.arange(3)}
    with artifact_path.open("wb") as handle:
        pickle.dump(value, handle)

    loaded = load_pickle(artifact_path)
    np.testing.assert_array_equal(loaded["x"], value["x"])
    assert (
        sha256_file(artifact_path)
        == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    )


def test_validity_and_lifting_helpers_match_expected_geometry() -> None:
    visible = np.asarray(((True, False), (True, True), (False, True)))
    motion_valid = np.asarray(((False, True), (True, False)))
    np.testing.assert_array_equal(
        target_validity(visible, motion_valid),
        np.asarray(((True, False), (False, True), (True, False))),
    )

    vertices = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)))
    indices, weights = build_lift_map(vertices, original_count=2, neighbors=2)
    np.testing.assert_array_equal(indices.shape, (1, 2))
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)

    tracked = np.asarray((((0.1, 0.0, 0.0), (0.3, 0.0, 0.0)),))
    lifted = lift_residual(
        tracked,
        state_count=3,
        indices=indices,
        weights=weights,
        maximum_norm=1.0,
    )
    np.testing.assert_allclose(lifted[0, 2, 0], 0.2)


def test_official_adapter_implements_replay_protocol(monkeypatch) -> None:
    simulator = _FakeSimulator()
    torch = _FakeTorch()
    warp = _FakeWarp()
    adapter = OfficialPhysTwinReplayProvider(
        simulator,
        torch,
        warp,
        device="cuda:0",
    )
    assert isinstance(adapter, PhysTwinReplayProvider)

    monkeypatch.setattr(
        provider_api,
        "_rollout_initial",
        lambda simulator_arg, wp_arg, *, frame_count: (
            np.zeros((frame_count, 2, 3), dtype=np.float32),
            np.ones((frame_count, 2, 3), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        provider_api,
        "rollout_restart",
        lambda simulator_arg, torch_arg, wp_arg, position, velocity, **kwargs: (
            np.repeat(
                position[None], kwargs["stop_frame"] - kwargs["start_frame"], axis=0
            )
        ),
    )

    adapter.set_group_log_scales(np.asarray((0.2, -0.1)))
    np.testing.assert_allclose(simulator.group_log_scale_tensor.values, (0.2, -0.1))

    controls = np.arange(12, dtype=np.float32).reshape(4, 1, 3)
    adapter.set_controller_points(controls)
    np.testing.assert_allclose(simulator.controller_points.values, controls)

    positions, velocities = adapter.replay_initial(frame_count=3)
    assert positions.shape == velocities.shape == (3, 2, 3)
    restart = adapter.replay_restart(
        np.zeros((2, 3)),
        np.ones((2, 3)),
        start_frame=1,
        stop_frame=4,
    )
    assert restart.shape == (3, 2, 3)
    assert warp.synchronize_calls == 2

    adapter.close()
    adapter.close()
    assert torch.cuda.empty_cache_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        adapter.replay_initial(frame_count=1)


def test_factory_hides_simulator_initialization(monkeypatch, tmp_path: Path) -> None:
    simulator = _FakeSimulator()
    torch = _FakeTorch()
    warp = _FakeWarp()
    calls = {}

    def initialize(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return simulator, torch, warp, {"runtime": "fake"}

    monkeypatch.setattr(provider_api, "initialize_simulator", initialize)
    replay = create_official_replay_provider(
        tmp_path,
        {"controller_points": np.zeros((4, 1, 3))},
        {"spring": 1.0},
        tmp_path / "checkpoint.pt",
        object(),
        num_surface_points=3,
        original_count=2,
        dt=0.03,
        num_substeps=8,
        self_collision=False,
        spring_parameterization="grouped",
        device="cpu",
    )
    assert isinstance(replay, PhysTwinReplayProvider)
    assert calls["kwargs"]["spring_parameterization"] == "grouped"
    assert calls["kwargs"]["device"] == "cpu"
    replay.close()


def test_adapter_rejects_mismatched_public_inputs() -> None:
    adapter = OfficialPhysTwinReplayProvider(
        _FakeSimulator(),
        _FakeTorch(),
        _FakeWarp(),
        device="cuda:0",
    )
    with pytest.raises(ValueError, match="group log-scales"):
        adapter.set_group_log_scales(np.zeros(3))
    with pytest.raises(ValueError, match="controller points"):
        adapter.set_controller_points(np.zeros((3, 1, 3)))
    with pytest.raises(ValueError, match="restart position"):
        adapter.replay_restart(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            start_frame=0,
            stop_frame=1,
        )


def test_lift_map_uses_numpy_fallback_when_scipy_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy.spatial":
            raise ImportError("forced SciPy absence")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
        )
    )

    indices, weights = build_lift_map(vertices, original_count=2, neighbors=2)

    np.testing.assert_array_equal(indices.shape, (1, 2))
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)
