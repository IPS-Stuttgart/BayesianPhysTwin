from __future__ import annotations

import hashlib
import pickle
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.causal4d_provider_v1 as provider_api
import bayesian_phystwin.phystwin.replay as replay_v2
from bayesian_phystwin.causal4d_belief_provider_v1 import (
    CAUSAL4D_BELIEF_PROVIDER_API_VERSION,
    DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1,
    FixedBayesianAnchorConfigV1,
    RobustEndpointPosteriorV1,
    causal4d_belief_provider_manifest,
    infer_fixed_bayesian_anchor_endpoint,
)
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
from bayesian_phystwin.contracts.replay import InitialReplayRequestV1
from bayesian_phystwin.phystwin.replay import OfficialPhysTwinReplayProviderV2
from bayesian_phystwin.phystwin_bayesian_anchor import robust_random_walk_endpoint


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
                position[None],
                kwargs["stop_frame"] - kwargs["start_frame"],
                axis=0,
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


def test_v2_adapter_executes_explicit_initial_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulator = _FakeSimulator()
    torch = _FakeTorch()
    warp = _FakeWarp()
    adapter = OfficialPhysTwinReplayProviderV2(
        simulator,
        torch,
        warp,
        device="cpu",
        frame_dt_s=0.03,
        simulator_configuration_id="config-v1",
        released_initial_state_id="released-v1",
    )
    monkeypatch.setattr(
        replay_v2,
        "_rollout_initial_trajectory",
        lambda simulator_arg, wp_arg, *, frame_count: (
            np.zeros((frame_count, 2, 3), dtype=np.float32),
            np.ones((frame_count, 2, 3), dtype=np.float32),
        ),
    )
    controls = np.arange(12, dtype=np.float32).reshape(4, 1, 3)
    common = {
        "request_id": "request-v1",
        "simulator_configuration_id": "config-v1",
        "group_log_scales": np.asarray((0.2, -0.1)),
        "controller_points_m": controls,
        "frame_count": 3,
    }

    with pytest.raises(ValueError, match="released state"):
        adapter.replay(
            InitialReplayRequestV1(
                initial_state_id="wrong-state",
                **common,
            )
        )

    trajectory = adapter.replay(
        InitialReplayRequestV1(
            initial_state_id="released-v1",
            **common,
        )
    )
    np.testing.assert_allclose(
        simulator.group_log_scale_tensor.values,
        (0.2, -0.1),
    )
    np.testing.assert_allclose(simulator.controller_points.values, controls)
    np.testing.assert_array_equal(trajectory.frame_ids, np.arange(3))
    assert (
        trajectory.positions_m.shape
        == trajectory.velocities_mps.shape
        == (
            3,
            2,
            3,
        )
    )
    assert warp.synchronize_calls == 2
    adapter.close()


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


def _fixed_anchor_inputs() -> tuple[np.ndarray, np.ndarray]:
    residual = np.zeros((7, 3, 3), dtype=np.float64)
    residual[:, 0, 0] = np.linspace(0.0, 0.006, len(residual))
    residual[:, 1, 1] = 0.002
    residual[-1, 1] = 0.08
    residual[:, 2, 2] = -0.003
    valid = np.ones((7, 3), dtype=bool)
    valid[2:5, 2] = False
    return residual, valid


def test_fixed_anchor_provider_manifest_is_explicit_and_versioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = causal4d_belief_provider_manifest(provider_revision="belief-abc")
    assert manifest["provider_revision"] == "belief-abc"
    assert manifest["schema_version"] == CAUSAL4D_BELIEF_PROVIDER_API_VERSION == 1
    assert manifest["artifact_schema_versions"] == {
        "FixedBayesianAnchorConfig": 1,
        "RobustEndpointPosterior": 1,
    }
    assert {
        "causal_prefix_endpoint_inference",
        "fixed_bayesian_anchor_endpoint",
        "immutable_endpoint_posterior",
        "numpy_only_endpoint_inference",
        "residual_finite_preflight",
    }.issubset(set(manifest["capabilities"]))
    assert manifest["metadata"] == {
        "provider_api": "bayesian_phystwin.causal4d_belief_provider_v1",
        "provider_api_version": 1,
        "inference_role": "fixed robust readout-discrepancy endpoint",
    }

    monkeypatch.setenv("BAYESIAN_PHYSTWIN_REVISION", "belief-env")
    assert causal4d_belief_provider_manifest()["provider_revision"] == "belief-env"
    monkeypatch.delenv("BAYESIAN_PHYSTWIN_REVISION")
    assert causal4d_belief_provider_manifest()["provider_revision"]


def test_fixed_anchor_defaults_match_the_frozen_additional_protocol() -> None:
    assert DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1 == FixedBayesianAnchorConfigV1(
        process_std_m=0.005,
        observation_std_m=0.001,
        initial_std_m=0.01,
        inlier_prior=0.95,
        outlier_variance_multiplier=100.0,
    )


def test_fixed_anchor_provider_matches_historical_endpoint_inference() -> None:
    residual, valid = _fixed_anchor_inputs()
    config = FixedBayesianAnchorConfigV1(
        process_std_m=0.0015,
        observation_std_m=0.002,
        initial_std_m=0.008,
        inlier_prior=0.9,
        outlier_variance_multiplier=75.0,
    )

    actual = infer_fixed_bayesian_anchor_endpoint(
        residual,
        valid,
        end_frame=6,
        config=config,
    )
    expected = robust_random_walk_endpoint(
        residual,
        valid,
        end_frame=6,
        process_variance=config.process_std_m**2,
        observation_variance=config.observation_std_m**2,
        initial_variance=config.initial_std_m**2,
        inlier_prior=config.inlier_prior,
        outlier_variance_multiplier=config.outlier_variance_multiplier,
    )

    np.testing.assert_array_equal(actual.mean_m, expected.mean)
    np.testing.assert_array_equal(actual.variance_m2, expected.variance)
    np.testing.assert_array_equal(
        actual.final_nominal_probability,
        expected.final_inlier_probability,
    )
    np.testing.assert_array_equal(actual.update_count, expected.update_count)


def test_fixed_anchor_posterior_copies_freezes_and_keeps_compatibility_aliases() -> (
    None
):
    residual, valid = _fixed_anchor_inputs()
    posterior = infer_fixed_bayesian_anchor_endpoint(
        residual,
        valid,
        end_frame=len(residual),
    )
    snapshot = posterior.mean_m.copy()
    residual[:] = 10.0

    np.testing.assert_array_equal(posterior.mean_m, snapshot)
    assert posterior.mean is posterior.mean_m
    assert posterior.variance is posterior.variance_m2
    assert posterior.final_inlier_probability is posterior.final_nominal_probability
    np.testing.assert_array_equal(
        posterior.updated_mask,
        posterior.update_count > 0,
    )
    for values in (
        posterior.mean_m,
        posterior.variance_m2,
        posterior.final_nominal_probability,
        posterior.update_count,
        posterior.updated_mask,
    ):
        assert not values.flags.writeable
    with pytest.raises(ValueError):
        posterior.mean_m[0, 0] = 1.0


def test_fixed_anchor_posterior_marks_never_observed_tracks() -> None:
    residual, valid = _fixed_anchor_inputs()
    valid[:, 1] = False
    posterior = infer_fixed_bayesian_anchor_endpoint(
        residual,
        valid,
        end_frame=len(residual),
    )

    assert posterior.update_count[1] == 0
    assert not posterior.updated_mask[1]
    assert posterior.final_nominal_probability[1] == 0.0
    assert posterior.updated_mask.flags.writeable is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("process_std_m", -1.0, "process_std_m"),
        ("process_std_m", np.inf, "process_std_m"),
        ("observation_std_m", 0.0, "observation_std_m"),
        ("initial_std_m", np.nan, "initial_std_m"),
        ("inlier_prior", 1.0, "inlier_prior"),
        ("outlier_variance_multiplier", 1.0, "outlier_variance_multiplier"),
    ),
)
def test_fixed_anchor_config_rejects_invalid_values(
    field: str,
    value: float,
    message: str,
) -> None:
    values = {
        "process_std_m": 0.005,
        "observation_std_m": 0.001,
        "initial_std_m": 0.01,
        "inlier_prior": 0.95,
        "outlier_variance_multiplier": 100.0,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        FixedBayesianAnchorConfigV1(**values)


@pytest.mark.parametrize(
    ("residual", "valid", "end_frame", "message"),
    (
        (np.zeros((3, 2)), np.ones((3, 2), dtype=bool), 3, "residual_m"),
        (
            np.zeros((3, 2, 3)),
            np.ones((3, 1), dtype=bool),
            3,
            "valid",
        ),
        (
            np.full((3, 2, 3), np.nan),
            np.ones((3, 2), dtype=bool),
            3,
            "finite",
        ),
        (
            np.zeros((3, 2, 3)),
            np.ones((3, 2), dtype=bool),
            0,
            "inside",
        ),
        (
            np.zeros((3, 2, 3)),
            np.ones((3, 2), dtype=bool),
            1.5,
            "integer",
        ),
    ),
)
def test_fixed_anchor_provider_rejects_invalid_inputs(
    residual: np.ndarray,
    valid: np.ndarray,
    end_frame: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        infer_fixed_bayesian_anchor_endpoint(
            residual,
            valid,
            end_frame=end_frame,
        )


def test_fixed_anchor_provider_rejects_wrong_config_type() -> None:
    residual, valid = _fixed_anchor_inputs()
    with pytest.raises(TypeError, match="FixedBayesianAnchorConfigV1"):
        infer_fixed_bayesian_anchor_endpoint(
            residual,
            valid,
            end_frame=2,
            config=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"mean_m": np.zeros((0, 3))}, "mean_m"),
        ({"variance_m2": np.asarray((0.1,))}, "variance_m2"),
        (
            {"final_nominal_probability": np.asarray((0.5, 1.5))},
            "final_nominal_probability",
        ),
        ({"update_count": np.asarray((1.0, 2.0))}, "integers"),
        ({"update_count": np.asarray((1, -1))}, "nonnegative"),
    ),
)
def test_fixed_anchor_posterior_contract_rejects_invalid_arrays(
    kwargs: dict[str, np.ndarray],
    message: str,
) -> None:
    values = {
        "mean_m": np.zeros((2, 3)),
        "variance_m2": np.asarray((0.1, 0.2)),
        "final_nominal_probability": np.asarray((0.5, 0.8)),
        "update_count": np.asarray((1, 2), dtype=np.int64),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        RobustEndpointPosteriorV1(**values)


def test_fixed_anchor_provider_import_does_not_load_optional_stacks() -> None:
    code = """
import sys
import bayesian_phystwin.causal4d_belief_provider_v1
forbidden = ('cv2', 'scipy', 'torch', 'warp')
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit(f'optional modules loaded: {loaded}')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
