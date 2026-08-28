"""Synthetic tests only; no empirical source/target data or upstream import."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_kinematic_boundary import (
    CLAMPS,
    config_for_source,
    hard_boundary_readout,
    hard_position_projection,
    install_hard_position_projection,
    score_predictions,
)
from bayesian_phystwin_experiments.deform_state_restart import (
    RodState,
    paired_physical_readout,
    sparse_state_increments,
    update_rod_state,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/remote"))
    return _load(ROOT / "scripts/remote/run_deform_kinematic_boundary.py")


def _projection_inputs():
    torch = pytest.importorskip("torch")
    points = torch.tensor([[[0.0, 0, 0], [2.0, 0, 0], [4.0, 0, 0]]])
    model = types.SimpleNamespace(device="cpu", n_vert=3)
    return (
        torch,
        model,
        points,
        torch.ones(1, 2),
        torch.ones(1, 3),
        torch.tensor([1.0, 1.0, 0.0]),
    )


def test_hard_boundaries_skip_only_edge_with_both_ends_prescribed():
    torch, model, points, lengths, mass, mask = _projection_inputs()
    original = points.clone()
    out = hard_position_projection(
        model, points, lengths, mass, mask, iterative_times=1
    )
    assert out is points
    assert torch.equal(out[:, :2], original[:, :2])
    torch.testing.assert_close(out[0, 2], torch.tensor([2.8, 0, 0]))
    # Prescribed edge length is 2, deliberately not its unit rest length.
    assert float(torch.linalg.norm(out[0, 1] - out[0, 0])) == 2


def test_unprescribed_edge_preserves_mass_weighted_center():
    torch, model, points, lengths, mass, mask = _projection_inputs()
    model.n_vert = 2
    points, lengths, mass = (
        points[:, :2].clone(),
        lengths[:, :1],
        torch.tensor([[1.0, 3.0]]),
    )
    before = (points * mass.unsqueeze(-1)).sum(1)
    out = hard_position_projection(
        model, points, lengths, mass, mask[:2] * 0, iterative_times=1
    )
    torch.testing.assert_close((out * mass.unsqueeze(-1)).sum(1), before)
    torch.testing.assert_close(out[0, :, 0], torch.tensor([0.9, 1.7]))


def test_native_broadcast_mass_and_expanded_mass_are_byte_identical():
    torch, model, points, lengths, mass, mask = _projection_inputs()
    points = points.repeat(3, 1, 1)
    lengths = lengths.repeat(3, 1)
    a = hard_position_projection(model, points.clone(), lengths, mass, mask)
    b = hard_position_projection(
        model, points.clone(), lengths, mass.repeat(3, 1), mask
    )
    assert a.numpy().tobytes() == b.numpy().tobytes()


def test_signed_zero_anchor_bytes_survive_projection():
    torch, model, points, lengths, mass, mask = _projection_inputs()
    points[0, 0, 1] = -0.0
    before = points[:, mask.bool()].numpy().tobytes()
    out = hard_position_projection(model, points, lengths, mass, mask)
    assert out[:, mask.bool()].numpy().tobytes() == before


@pytest.mark.parametrize(
    "failure",
    ["mode", "iterations", "shape", "mass", "length", "mask", "nan", "device"],
)
def test_projection_rejects_unregistered_inputs(failure):
    _, model, points, lengths, mass, mask = _projection_inputs()
    kwargs = {}
    if failure == "mode":
        kwargs["mode"] = "numpy"
    elif failure == "iterations":
        kwargs["iterative_times"] = 0
    elif failure == "shape":
        mass = mass[:, :2]
    elif failure == "mass":
        mass[0, 0] = 0
    elif failure == "length":
        lengths[0, 0] = -1
    elif failure == "mask":
        mask[0] = 0.5
    elif failure == "nan":
        points[0, 0, 0] = float("nan")
    else:
        model.device = "cuda"
    with pytest.raises(ValueError):
        hard_position_projection(model, points, lengths, mass, mask, **kwargs)


def test_installation_is_opt_in_instance_local_and_does_not_change_weights():
    torch = pytest.importorskip("torch")

    class Model:
        device = "cpu"

        def __init__(self):
            self.weight = torch.tensor([1.0, 2.0])

        def applyInternalConstraintsIteration(self, value):
            return value

    a, b = Model(), Model()
    original = a.applyInternalConstraintsIteration.__func__
    before = a.weight.numpy().tobytes()
    assert not install_hard_position_projection(a)
    assert a.applyInternalConstraintsIteration.__func__ is original
    assert install_hard_position_projection(a, enabled=True)
    assert a.applyInternalConstraintsIteration.__func__ is hard_position_projection
    assert b.applyInternalConstraintsIteration.__func__ is original
    assert a.weight.numpy().tobytes() == before
    with pytest.raises(ValueError, match="already"):
        install_hard_position_projection(a, enabled=True)
    with pytest.raises(ValueError, match="boolean"):
        install_hard_position_projection(b, enabled=1)


def test_readout_is_opt_in_and_preserves_frozen_offset():
    old = np.zeros((2, 6, 12, 3))
    old[:, :, 2:10] = 0.003
    native, hard = np.zeros_like(old), np.ones_like(old) * 0.004
    assert hard_boundary_readout(old, native, hard) is old
    actual = hard_boundary_readout(old, native, hard, enabled=True)
    np.testing.assert_array_equal(actual, hard + old - native)
    np.testing.assert_array_equal(actual[:, :, CLAMPS], hard[:, :, CLAMPS])
    assert old[:, :, CLAMPS].sum() == 0


@pytest.mark.parametrize("failure", ["shape", "nan", "clamp"])
def test_readout_rejects_invalid_or_boundary_changing_offset(failure):
    a = np.zeros((1, 170, 12, 3))
    b, c = a.copy(), a.copy()
    if failure == "shape":
        b = b[:, :169]
    elif failure == "nan":
        c[0, 40, 3, 0] = np.nan
    else:
        a[0, 40, 0, 0] = 0.001
    with pytest.raises(ValueError):
        hard_boundary_readout(a, b, c, enabled=True)


def _scores(runner):
    names = runner._plan()["names"]
    truth = np.zeros((14, 120, 12, 3))
    values = {
        "incumbent": 0.02,
        "paired": 0.01,
        "hard_baseline": 0.012,
        "hard_paired": 0.005,
    }
    predictions = {k: np.full_like(truth, v) for k, v in values.items()}
    return names, truth, predictions


def test_gate_requires_new_backend_and_sparse_update_value(runner):
    names, truth, predictions = _scores(runner)
    result = score_predictions(names, predictions, truth)
    assert result["decision"]["passed"]
    assert result["decision"]["primary_joint_wins"] == 13
    assert not result["decision"]["transfer_authorized"]
    predictions["hard_baseline"][:] = 0.001
    result = score_predictions(names, predictions, truth)
    assert not result["decision"]["passed"]
    assert not result["decision"]["checks"]["sparse_update_improves_hard_baseline"]


def test_diagnostic_cannot_rescue_primary_and_design_case_is_excluded(runner):
    names, truth, predictions = _scores(runner)
    predictions["hard_baseline"][:] = 0
    predictions["hard_paired"] = predictions["paired"].copy()
    predictions["hard_paired"][0] = 1e8
    result = score_predictions(names, predictions, truth)
    assert not result["decision"]["passed"]
    assert "103.pkl" not in result["case_metrics"]
    assert (
        result["decision"]["means"]["hard_paired"]
        == result["decision"]["means"]["paired"]
    )


def test_one_large_regression_fails_even_with_mean_gain(runner):
    names, truth, predictions = _scores(runner)
    predictions["hard_paired"][1] = 0.02
    result = score_predictions(names, predictions, truth)
    assert not result["decision"]["checks"]["worst_rmse_ratio_at_most_1_05"]


def test_exact_source_plan_and_no_retry_controls(runner, tmp_path, monkeypatch):
    plan = runner._plan()
    assert plan["maximum_native_attempts"] == 1
    assert plan["prescribed_segment_inextensibility_claimed"] is False
    assert plan["future_free_node_truth_in_prediction"] is False
    assert plan["publication"] == "local-private-only-no-push-no-main-merge"
    path = tmp_path / runner.PROTOCOL
    path.parent.mkdir(parents=True)
    plan["source_gate"]["minimum_l1_rmse_relative_gain"] = 0
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="complete frozen"):
        runner._plan()


def test_attempt_ledger_precedes_prediction_and_prevents_retry(
    runner, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        runner, "verify_lock", lambda *_: ({"artifact_id": "synthetic"}, {})
    )

    def fail(*_):
        assert (tmp_path / "attempt.json").exists()
        calls.append(1)
        raise RuntimeError("synthetic runtime failure")

    monkeypatch.setattr(runner, "_predict", fail)
    with pytest.raises(RuntimeError):
        runner.predict(tmp_path, "synthetic")
    with pytest.raises(FileExistsError):
        runner.predict(tmp_path, "synthetic")
    assert len(calls) == 1
    failure = runner.custody._read(tmp_path / "technical_failure.json")
    assert failure["ordinary_successes"] == 0
    assert not failure["future_scoring_authorized"]


def test_complete_synthetic_production_and_second_arithmetic(
    runner, tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    config = config_for_source()

    class SyntheticRod:
        def __init__(self, *_):
            self.model = types.SimpleNamespace(
                device="cpu",
                n_vert=12,
                applyInternalConstraintsIteration=lambda p, *_a: p,
                m_restWprev=torch.zeros(14, 11, 2),
                m_restWnext=torch.zeros(14, 11, 2),
                learned_pmass=torch.ones(14, 12),
            )

        def initialize(self, values):
            assert not getattr(self.model, "_bpt_hard_position_projection", False)
            p = torch.tensor(values[:, 1], dtype=torch.float32)
            return RodState(
                p, p * 0, p.clone(), torch.zeros(14, 3), torch.zeros(14, 11), -1
            )

        def advance(self, state, action):
            v = 0.99 * state.velocity - 0.05 * state.positions
            p = state.positions + 0.01 * v
            p[:, CLAMPS] = torch.tensor(action, dtype=torch.float32)
            p = self.model.applyInternalConstraintsIteration(
                p,
                torch.ones(14, 11) * 0.05,
                self.model.learned_pmass,
                torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
            )
            v = (p - state.positions) / 0.01
            return RodState(
                p,
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
    raw = np.full((14, 500, 12, 3), np.nan, dtype=np.float32)
    raw[:, :2] = 0
    raw[:, 2:172, CLAMPS] = 0
    raw[:, 43, config.observed_nodes] = 0.01
    raw[:, 51, config.observed_nodes] = 0.018
    initial, actions, observations = runner.custody.permitted_inputs(raw)
    incumbent = np.zeros((14, 170, 12, 3))
    incumbent[:, :, 2:10] = 0.002
    dx, dv = sparse_state_increments(incumbent[:, :50], observations, config)
    native = SyntheticRod()
    _, state = native.rollout(native.initialize(initial), actions[:, :50])
    updated = update_rod_state(
        state,
        torch.tensor(dx, dtype=torch.float32),
        torch.tensor(dv, dtype=torch.float32),
        gain=1,
        clamped_nodes=CLAMPS,
    )
    points, _ = native.rollout(updated, actions[:, 50:])
    old_paired = paired_physical_readout(
        incumbent[:, 50:], np.zeros_like(points), points
    )
    archive_path, parent_path = tmp_path / "source.npz", tmp_path / "parent.npz"
    np.savez_compressed(
        archive_path,
        names=np.array(plan["names"]),
        candidate_predictions=incumbent,
        baseline_predictions=np.zeros((14, 170, 12, 3), dtype=np.float32),
        targets=np.zeros_like(incumbent),
    )
    np.savez_compressed(
        parent_path,
        names=np.array(plan["names"]),
        incumbent=incumbent[:, 50:],
        incumbent_propagated_pose_velocity=old_paired,
    )
    manifest, checkpoint = tmp_path / "manifest.json", tmp_path / "checkpoint.pt"
    manifest.write_text(json.dumps({"ordered_names": plan["names"]}))
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
        paired_archive={"path": str(parent_path)},
    )
    for key in ("manifest", "checkpoint", "archive", "paired_archive"):
        plan[key]["sha256"] = runner.file_digest(Path(plan[key]["path"]))
    source = types.SimpleNamespace(
        _assert_upstream=lambda *_: {"synthetic": True},
        _load_upstream=lambda *_: None,
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
    verifier_path = "scripts/verify_deform_kinematic_boundary.py"
    lock = runner.custody._write(
        output / "lock.json",
        {
            "schema": "deform-kinematic-boundary-source-v1-lock",
            "revision": "synthetic",
            "plan": plan,
            "source_files": {verifier_path: runner.file_digest(ROOT / verifier_path)},
        },
    )
    with np.load(archive_path) as data:
        archive_type = type(data)
    original_get = archive_type.__getitem__

    def deny_truth(self, name):
        if name == "targets":
            raise AssertionError("truth read before prediction barrier")
        return original_get(self, name)

    with monkeypatch.context() as restricted:
        restricted.setattr(archive_type, "__getitem__", deny_truth)
        runner._predict(output, lock, plan)
    seal = runner.custody._read(output / "prediction_seal.json")
    assert seal["ordinary_successes"] == 14 and seal["complete"]
    assert not seal["future_truth_scored"]
    assert not (output / "result.json").exists()
    arrays = runner.custody._load_arrays(
        output / "predictions.npz", seal["predictions"]
    )
    runner.validate_prediction_invariants(arrays)
    bad = {k: v.copy() for k, v in arrays.items()}
    bad["hard_native"][0, 100, 1, 0] = 1
    with pytest.raises(ValueError, match="invariant"):
        runner.validate_prediction_invariants(bad)
    monkeypatch.setattr(runner, "verify_lock", lambda *_: (lock, plan))
    seal_hash = runner.file_digest(output / "prediction_seal.json")
    runner.score(output, "synthetic", seal_hash)
    verifier = _load(ROOT / verifier_path)
    report = verifier.verify(output, ROOT, archive_path, parent_path)
    assert report["passed"] and report["metrics_verified"] == 624
    assert not report["independent_human_review"]
    assert not report["new_native_execution"]
    with monkeypatch.context() as restricted:
        restricted.setattr(archive_type, "__getitem__", deny_truth)
        with pytest.raises(ValueError, match="already exists"):
            runner.score(output, "synthetic", seal_hash)
