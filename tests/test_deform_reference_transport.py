"""Synthetic/source-only tests; no upstream or empirical data is opened."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_reference_transport import (
    ARMS,
    config_for_source,
    learned_reference_offsets,
    score_predictions,
    source_decision,
    transport_pair,
)
from bayesian_phystwin_experiments.deform_state_restart import (
    RodState,
    paired_physical_readout,
    sparse_state_increments,
    update_rod_state,
)

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "reference_transport_runner",
        ROOT / "scripts/remote/run_deform_reference_transport.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_matches_implementation_and_keeps_frozen_boundaries():
    plan = _runner()._plan()
    assert plan["names"] == sorted(plan["names"])
    assert plan["maximum_native_attempts"] == 1
    assert plan["source_cases_already_opened"]
    for key in (
        "fresh_transfer_authorized",
        "target_access",
        "held_v8_access",
        "gpu_execution",
    ):
        assert not plan[key]


def test_inputs_ignore_all_unpermitted_truth():
    runner = _runner()
    raw = np.random.default_rng(31).normal(size=(14, 500, 12, 3))
    poison = np.full_like(raw, np.nan)
    poison[:, :2] = raw[:, :2]
    poison[:, 2:172, (0, 1, 10, 11)] = raw[:, 2:172, (0, 1, 10, 11)]
    for t in (43, 51):
        poison[:, t, (2, 4, 6, 8)] = raw[:, t, (2, 4, 6, 8)]
    for expected, actual in zip(
        runner.permitted_inputs(raw), runner.permitted_inputs(poison), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_reference_velocity_is_backward_difference_of_forecast_offset():
    config = config_for_source()
    a = np.zeros((2, 170, 12, 3))
    b = a.copy()
    b[:, :, 2:10] = np.arange(170)[None, :, None, None] * 0.0001
    offset, velocity = learned_reference_offsets(b, a, config)
    assert offset.shape == (2, 121, 12, 3)
    np.testing.assert_allclose(offset[:, 0, 2:10], 0.0049)
    np.testing.assert_allclose(velocity[:, :, 2:10], 0.01)
    np.testing.assert_array_equal(velocity[:, :, config.clamped_nodes], 0)
    altered = b.copy()
    altered[:, 100] += 1
    altered[:, 100, config.clamped_nodes] = 0
    changed, _ = learned_reference_offsets(altered, a, config)
    np.testing.assert_array_equal(changed[:, :51], offset[:, :51])


@pytest.mark.parametrize("failure", ["clamp", "nan", "shape"])
def test_reference_rejects_invalid_inputs(failure):
    config = config_for_source()
    a = np.zeros((1, 170, 12, 3))
    b = a.copy()
    if failure == "clamp":
        b[:, 100, 0] = 0.001
    elif failure == "nan":
        b[:, 100, 2] = np.nan
    else:
        b = b[:, :169]
    with pytest.raises(ValueError):
        learned_reference_offsets(b, a, config)


def _toy(nonlinear=True):
    torch = pytest.importorskip("torch")
    p = torch.zeros((2, 12, 3), dtype=torch.float64)
    p[:, 2:10] = 0.4
    state = RodState(
        p,
        p * 0 + 0.02,
        p - 0.002,
        torch.ones(2, 3, dtype=p.dtype),
        torch.zeros(2, 11, dtype=p.dtype),
        49,
    )
    actions = np.zeros((2, 4, 4, 3))
    calls = []

    def step(s, action):
        calls.append(s.prediction_index)
        v = 0.94 * s.velocity - 0.2 * (s.positions ** (3 if nonlinear else 1))
        x = s.positions + 0.01 * v
        x[:, (0, 1, 10, 11)] = torch.as_tensor(action)
        v[:, (0, 1, 10, 11)] = 0
        return RodState(
            x,
            v,
            s.positions.clone(),
            s.material_u0.clone(),
            s.theta + 0.001 * s.positions[:, 1:, 0],
            s.prediction_index + 1,
        )

    states = [state.clone()]
    for t in range(4):
        states.append(step(states[-1], actions[:, t]))
    base = np.stack([s.positions.numpy().copy() for s in states[1:]], axis=1)
    offsets = torch.zeros((2, 5, 12, 3), dtype=p.dtype)
    offsets[:, :, 2:10] = 0.08
    dx, dv = p * 0, p * 0
    dx[:, 2:10], dv[:, 2:10] = 0.01, 0.03
    kwargs = dict(
        advance=step,
        nominal_states=states,
        future_actions=actions,
        incumbent=base,
        pose_increment=dx,
        velocity_increment=dv,
        position_offsets=offsets,
        velocity_offsets=offsets * 0,
        clamped_nodes=(0, 1, 10, 11),
    )
    calls.clear()
    return torch, kwargs, calls


def test_zero_innovation_is_exact_original_object_without_native_execution():
    torch, kwargs, calls = _toy()
    kwargs.update(
        pose_increment=torch.zeros_like(kwargs["pose_increment"]),
        velocity_increment=torch.zeros_like(kwargs["velocity_increment"]),
    )
    result, trace = transport_pair(**kwargs, mode="reference_centered")
    assert result is kwargs["incumbent"]
    assert not trace and not calls


def test_zero_reference_reduces_to_ordinary_paired_propagation_byte_exact():
    torch, kwargs, _ = _toy()
    kwargs["position_offsets"] = torch.zeros_like(kwargs["position_offsets"])
    state = update_rod_state(
        kwargs["nominal_states"][0],
        kwargs["pose_increment"],
        kwargs["velocity_increment"],
        gain=1,
        clamped_nodes=(0, 1, 10, 11),
    )
    future = []
    for t in range(4):
        state = kwargs["advance"](state, kwargs["future_actions"][:, t])
        future.append(state.positions.numpy().copy())
    expected = paired_physical_readout(
        kwargs["incumbent"], kwargs["incumbent"], np.stack(future, axis=1)
    )
    actual, _ = transport_pair(**kwargs, mode="reference_centered")
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def test_common_centering_preserves_differences_and_input_memory():
    _, kwargs, _ = _toy()
    original = [s.clone() for s in kwargs["nominal_states"]]
    result, trace = transport_pair(**kwargs, mode="reference_centered")
    for t in range(1, 4):
        target = (
            kwargs["nominal_states"][t].positions.numpy()
            + kwargs["position_offsets"][:, t].numpy()
        )
        np.testing.assert_allclose(
            trace["center_before"][:, t], target, rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            trace["updated_before"][:, t] - trace["center_before"][:, t],
            trace["updated_after"][:, t - 1] - trace["center_after"][:, t - 1],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            trace["updated_velocity_before"][:, t]
            - trace["center_velocity_before"][:, t],
            trace["updated_velocity_after"][:, t - 1]
            - trace["center_velocity_after"][:, t - 1],
            atol=1e-15,
        )
    np.testing.assert_allclose(
        result,
        kwargs["incumbent"] + trace["updated_after"] - trace["center_after"],
        atol=1e-15,
    )
    for before, after in zip(original, kwargs["nominal_states"], strict=True):
        for field in dataclasses.fields(before):
            a, b = getattr(before, field.name), getattr(after, field.name)
            assert a == b if field.name == "prediction_index" else a.equal(b)


def test_reference_choice_can_change_nonlinear_error_transport():
    torch, kwargs, _ = _toy()
    centered, _ = transport_pair(**kwargs, mode="reference_centered")
    initialized, _ = transport_pair(**kwargs, mode="reference_initialized")
    kwargs["position_offsets"] = torch.zeros_like(kwargs["position_offsets"])
    raw, _ = transport_pair(**kwargs, mode="reference_centered")
    assert not np.array_equal(centered, initialized)
    assert not np.array_equal(centered, raw)


def test_reference_choice_cancels_for_linear_toy_dynamics():
    torch, kwargs, _ = _toy(nonlinear=False)
    centered, _ = transport_pair(**kwargs, mode="reference_centered")
    kwargs["position_offsets"] = torch.zeros_like(kwargs["position_offsets"])
    raw, _ = transport_pair(**kwargs, mode="reference_centered")
    np.testing.assert_allclose(centered, raw, atol=1e-15)


@pytest.mark.parametrize(
    "failure", ["mode", "shape", "clamp", "nan", "indices", "increments"]
)
def test_transport_fails_closed(failure):
    _, kwargs, _ = _toy()
    mode = "reference_centered"
    if failure == "mode":
        mode = "choose_best"
    elif failure == "shape":
        kwargs["position_offsets"] = kwargs["position_offsets"][:, :4]
    elif failure == "clamp":
        kwargs["position_offsets"][:, 1, 0] = 0.01
    elif failure == "nan":
        kwargs["velocity_offsets"][:, 0, 3] = float("nan")
    elif failure == "indices":
        kwargs["nominal_states"][2] = dataclasses.replace(
            kwargs["nominal_states"][2], prediction_index=80
        )
    else:
        kwargs["pose_increment"][:, 0] = 0.01
    with pytest.raises(ValueError):
        transport_pair(**kwargs, mode=mode)


def _scores():
    names = _runner()._plan()["names"]
    truth = np.zeros((14, 120, 12, 3))
    predictions = {arm: np.ones_like(truth) * 0.01 for arm in ARMS}
    predictions["reference_centered"] *= 0.9
    return score_predictions(names, predictions, truth, config_for_source())


def test_primary_gate_passes_synthetic_gain_but_never_authorizes_transfer():
    scores = _scores()
    assert scores["decision"]["passed"]
    assert scores["decision"]["primary_joint_wins"] == 13
    assert not scores["decision"]["future_transfer_authorized"]
    assert "103.pkl" not in scores["case_metrics"]


def test_secondary_cannot_rescue_failed_primary():
    metrics = _scores()["case_metrics"]
    for case in metrics.values():
        case["reference_centered"] = case["paired"]
        for values in case["reference_initialized"].values():
            values["point_rmse_mm"] *= 0.01
    assert not source_decision(metrics)["passed"]


def test_worst_case_regression_fails_even_if_mean_good():
    metrics = _scores()["case_metrics"]
    case = metrics[sorted(metrics)[0]]
    case["reference_centered"]["all"]["point_rmse_mm"] = (
        1.06 * case["paired"]["all"]["point_rmse_mm"]
    )
    assert not source_decision(metrics)["checks"]["worst_rmse_ratio_at_most_1_05"]


def test_custody_artifacts_are_write_once_and_detect_tampering(tmp_path):
    runner = _runner()
    path = tmp_path / "item.json"
    value = runner._write(path, {"source_only": True})
    assert runner._read(path) == value
    with pytest.raises(FileExistsError):
        runner._write(path, {"source_only": True})
    path.write_text(json.dumps({**value, "source_only": False}))
    with pytest.raises(ValueError, match="canonical"):
        runner._read(path)


def test_sealed_array_shape_dtype_bytes_and_members_are_bound(tmp_path):
    runner = _runner()
    path = tmp_path / "arrays.npz"
    data = {"x": np.arange(12, dtype=np.float32).reshape(4, 3)}
    binding = runner._save_arrays(path, data)
    assert runner._load_arrays(path, binding)["x"].tobytes() == data["x"].tobytes()
    binding["arrays"]["x"]["dtype"] = "<f8"
    with pytest.raises(ValueError, match="identity"):
        runner._load_arrays(path, binding)


def test_complete_synthetic_prediction_stage_seals_before_any_truth(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    runner = _runner()
    config = config_for_source()

    class SyntheticRod:
        def __init__(self, *_args):
            pass

        def initialize(self, values):
            p = torch.tensor(values[:, 1], dtype=torch.float32)
            return RodState(
                p, p * 0, p.clone(), torch.zeros(14, 3), torch.zeros(14, 11), -1
            )

        def advance(self, state, action):
            v = 0.99 * state.velocity - 0.05 * state.positions
            x = state.positions + 0.01 * v
            x[:, config.clamped_nodes] = torch.tensor(action, dtype=torch.float32)
            v[:, config.clamped_nodes] = 0
            return RodState(
                x,
                v,
                state.positions.clone(),
                state.material_u0.clone(),
                state.theta.clone(),
                state.prediction_index + 1,
            )

        def rollout(self, state, actions):
            values = []
            for t in range(actions.shape[1]):
                state = self.advance(state, actions[:, t])
                values.append(state.positions.numpy().copy())
            return np.stack(values, axis=1), state

    plan = runner._plan()
    raw = np.zeros((14, 500, 12, 3), dtype=np.float32)
    raw[:, 43, config.observed_nodes] = 0.01
    raw[:, 51, config.observed_nodes] = 0.018
    incumbent = np.zeros((14, 170, 12, 3), dtype=np.float64)
    incumbent[:, :, 2:10] = 0.002
    initial, actions, obs = runner.permitted_inputs(raw)
    dx, dv = sparse_state_increments(incumbent[:, :50], obs, config)
    rod = SyntheticRod()
    _, snapshot = rod.rollout(rod.initialize(initial), actions[:, :50])
    updated = update_rod_state(
        snapshot,
        torch.tensor(dx, dtype=torch.float32),
        torch.tensor(dv, dtype=torch.float32),
        gain=1,
        clamped_nodes=config.clamped_nodes,
    )
    points, _ = rod.rollout(updated, actions[:, 50:])
    expected = paired_physical_readout(incumbent[:, 50:], np.zeros_like(points), points)
    archive_path, paired_path = tmp_path / "inputs.npz", tmp_path / "parent.npz"
    np.savez_compressed(
        archive_path,
        names=np.asarray(plan["names"]),
        baseline_predictions=np.zeros((14, 170, 12, 3), dtype=np.float32),
        candidate_predictions=incumbent,
        targets=np.zeros_like(incumbent),
    )
    np.savez_compressed(
        paired_path,
        names=np.asarray(plan["names"]),
        incumbent=incumbent[:, 50:],
        incumbent_propagated_pose_velocity=expected,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"ordered_names": plan["names"]}))
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": {}}, checkpoint)
    plan.update(
        runtime={
            "python": ".".join(platform.python_version_tuple()[:2]),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "seed": 260929,
        },
        manifest={"path": str(manifest)},
        checkpoint={"path": str(checkpoint)},
        archive={"path": str(archive_path)},
        paired_archive={"path": str(paired_path)},
    )
    source = types.SimpleNamespace(
        _assert_upstream=lambda *_a: {"synthetic": True},
        _load_upstream=lambda *_a: None,
        _load_named_trajectories=lambda *_a, **_k: {
            n: raw[i] for i, n in enumerate(plan["names"])
        },
    )
    monkeypatch.setitem(sys.modules, "run_deform_dlo_source", source)
    monkeypatch.setitem(
        sys.modules,
        "run_deform_sparse_state_restart",
        types.SimpleNamespace(NativeRod=SyntheticRod),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "run"
    output.mkdir()
    for label in ("manifest", "checkpoint", "archive", "paired_archive"):
        plan[label]["sha256"] = runner.file_digest(Path(plan[label]["path"]))
    lock = runner._write(
        output / "lock.json",
        {
            "plan": plan,
            "revision": "synthetic-only",
            "source_files": {
                "scripts/verify_deform_reference_transport.py": runner.file_digest(
                    ROOT / "scripts/verify_deform_reference_transport.py"
                )
            },
        },
    )
    with np.load(archive_path) as data:
        archive_type = type(data)
    original_get = archive_type.__getitem__

    def deny_truth(self, name):
        if name == "targets":
            raise AssertionError("source truth accessed before prediction seal")
        return original_get(self, name)

    with monkeypatch.context() as restricted:
        restricted.setattr(archive_type, "__getitem__", deny_truth)
        runner._predict(output, lock, plan, time.monotonic())
    seal = runner._read(output / "prediction_seal.json")
    assert seal["ordinary_successes"] == 14 and seal["complete"]
    assert not seal["future_truth_scored"]
    assert not (output / "result.json").exists()
    arrays = runner._load_arrays(output / "predictions.npz", seal["predictions"])
    assert all(np.isfinite(arrays[arm]).all() for arm in ARMS)
    assert arrays["paired"].tobytes() == expected.tobytes()
    monkeypatch.setattr(runner, "verify_lock", lambda *_args: (lock, plan))
    runner.score(
        output,
        runner.file_digest(output / "lock.json"),
        runner.file_digest(output / "prediction_seal.json"),
    )
    spec = importlib.util.spec_from_file_location(
        "reference_second_arithmetic",
        ROOT / "scripts/verify_deform_reference_transport.py",
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    report = verifier.verify(output, ROOT, archive_path, paired_path)
    assert report["passed"]
    assert report["trajectory_horizon_arm_metrics_verified"] == 624
    assert not report["independent_human_review"]
