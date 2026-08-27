"""Source-only contracts for the isolated native DEFORM state-update study."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_state_restart import (
    RestartConfig,
    RodState,
    aggregate_paired_metrics,
    array_digest,
    file_digest,
    interpolate_material_residual,
    paired_physical_readout,
    prediction_metrics,
    sparse_state_increments,
    update_rod_state,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "restart_runner",
        ROOT / "scripts/remote/run_deform_sparse_state_restart.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "other"},
        {"observation_frames": (49, 41)},
        {"observation_frames": (41, 50)},
        {"dt_s": float("nan")},
        {"dt_s": -1.0},
        {"forecast_end": 49},
        {"hidden_nodes": (2, 3)},
        {"observed_nodes": (2, 2)},
        {"observed_nodes": (4, 2)},
        {"hidden_nodes": (12,)},
        {"bootstrap_replicates": 0},
    ],
)
def test_invalid_time_or_identity_contract(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(RestartConfig(), **changes)


def test_material_interpolation_exact_knots_and_clamps():
    config = RestartConfig()
    values = np.arange(24, dtype=float).reshape(2, 4, 3) / 1000
    result = interpolate_material_residual(values, config)
    np.testing.assert_array_equal(result[:, config.observed_nodes], values)
    np.testing.assert_array_equal(result[:, config.clamped_nodes], 0.0)
    np.testing.assert_allclose(result[:, 3], (values[:, 0] + values[:, 1]) / 2)
    np.testing.assert_allclose(result[:, 9], values[:, 3] / 2)


@pytest.mark.parametrize("values", [np.zeros((2, 3)), np.full((2, 4, 3), np.nan)])
def test_invalid_interpolation_rows(values):
    with pytest.raises(ValueError):
        interpolate_material_residual(values, RestartConfig())


def test_sparse_velocity_is_residual_slope_not_observed_velocity():
    config = RestartConfig()
    time = np.arange(50)[None, :, None, None] * config.dt_s
    reference = np.broadcast_to(time * 0.5, (2, 50, 12, 3)).copy()
    observed = reference[:, config.observation_frames][
        :, :, config.observed_nodes
    ].copy()
    observed[:, 0] += 0.01
    observed[:, 1] += 0.018
    dx, dv = sparse_state_increments(reference, observed, config)
    np.testing.assert_allclose(dx[:, config.observed_nodes], 0.018)
    np.testing.assert_allclose(dv[:, config.observed_nodes], 0.1)
    np.testing.assert_array_equal(dv[:, config.clamped_nodes], 0.0)


def test_constant_residual_has_zero_velocity_update():
    config = RestartConfig()
    reference = np.zeros((1, 50, 12, 3))
    dx, dv = sparse_state_increments(reference, np.ones((1, 2, 4, 3)) * 0.02, config)
    np.testing.assert_allclose(dx[:, config.observed_nodes], 0.02)
    np.testing.assert_array_equal(dv, 0.0)


@pytest.mark.parametrize("length", [49, 51, 170])
def test_update_rejects_nonprefix_arrays(length):
    with pytest.raises(ValueError, match="permitted reference prefix"):
        sparse_state_increments(
            np.zeros((1, length, 12, 3)), np.zeros((1, 2, 4, 3)), RestartConfig()
        )


def test_model_inputs_are_invariant_to_future_nonclamped_truth():
    runner = _runner()
    config = RestartConfig()
    data = np.random.default_rng(123).normal(size=(2, 500, 12, 3))
    changed = data.copy()
    changed[:, 2:, 2:10] = np.nan
    before = runner.causal_model_inputs(data, config)
    after = runner.causal_model_inputs(changed, config)
    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left, right)


def _state():
    torch = pytest.importorskip("torch")
    return torch, RodState(
        torch.zeros(2, 12, 3),
        torch.ones(2, 12, 3),
        torch.full((2, 12, 3), -1.0),
        torch.ones(2, 3),
        torch.ones(2, 11),
        49,
    )


def test_zero_gain_and_zero_innovation_clone_every_internal_state():
    torch, state = _state()
    zero = torch.zeros_like(state.positions)
    for gain in (0.0, 1.0):
        copied = update_rod_state(
            state, zero, zero, gain=gain, clamped_nodes=(0, 1, 10, 11)
        )
        assert copied is not state
        for field in dataclasses.fields(state):
            a, b = getattr(state, field.name), getattr(copied, field.name)
            if field.name == "prediction_index":
                assert a == b
            else:
                assert torch.equal(a, b)
                assert a.data_ptr() != b.data_ptr()


def test_state_update_preserves_material_memory_and_is_not_in_place():
    torch, state = _state()
    dx = torch.zeros_like(state.positions)
    dv = torch.zeros_like(dx)
    dx[:, 2:10] = 0.01
    dv[:, 2:10] = 0.1
    updated = update_rod_state(state, dx, dv, gain=0.25, clamped_nodes=(0, 1, 10, 11))
    assert torch.equal(state.positions, torch.zeros_like(state.positions))
    torch.testing.assert_close(
        updated.positions[:, 2:10], torch.full((2, 8, 3), 0.0025)
    )
    torch.testing.assert_close(updated.velocity[:, 2:10], torch.full((2, 8, 3), 1.025))
    for name in ("previous_positions", "theta", "material_u0"):
        assert torch.equal(getattr(updated, name), getattr(state, name))


def test_state_update_rejects_actuator_changes_and_nonfinite_innovation():
    torch, state = _state()
    zero = torch.zeros_like(state.positions)
    wrong = zero.clone()
    wrong[:, 0] = 1
    with pytest.raises(ValueError, match="actuator"):
        update_rod_state(state, wrong, zero, gain=1.0, clamped_nodes=(0, 1, 10, 11))
    wrong[:, 2] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        update_rod_state(state, wrong, zero, gain=1.0, clamped_nodes=(0, 1, 10, 11))


def test_paired_response_exact_identity_and_no_double_counting():
    nominal = np.zeros((2, 120, 12, 3), dtype=np.float32)
    incumbent = np.ones_like(nominal, dtype=np.float64) * 0.1
    assert paired_physical_readout(incumbent, nominal, nominal.copy()) is incumbent
    changed = nominal + np.float32(0.01)
    output = paired_physical_readout(incumbent, nominal, changed)
    np.testing.assert_allclose(output, 0.11)
    np.testing.assert_array_equal(incumbent, 0.1)


def test_metric_conventions_are_coordinate_l1_and_euclidean_rmse():
    truth = np.zeros((3, 2, 3))
    prediction = np.broadcast_to([0.003, 0.004, 0.0], truth.shape)
    result = prediction_metrics(prediction, truth)
    assert result["coordinate_l1_mm"] == pytest.approx(7 / 3)
    assert result["point_rmse_mm"] == pytest.approx(5)
    assert result["fde_mm"] == pytest.approx(5)


def test_paired_aggregation_excludes_design_and_uses_whole_trajectories():
    config = dataclasses.replace(RestartConfig(), bootstrap_replicates=100)
    truth = np.zeros((3, 120, 12, 3))
    baseline = np.ones_like(truth) * 0.01
    improved = baseline * 0.5
    improved[0] = 100
    result = aggregate_paired_metrics(
        {"incumbent": baseline, "candidate": improved},
        truth,
        ["103.pkl", "a.pkl", "b.pkl"],
        config,
    )
    candidate = result["summaries"]["candidate"]
    assert candidate["case_count"] == 2
    assert candidate["coordinate_l1_mm_change_percent"] == pytest.approx(-50)
    assert candidate["joint_wins"] == 2
    np.testing.assert_allclose(candidate["coordinate_l1_mm_delta_ci95"], [-5, -5])
    assert result["bootstrap_unit"] == "whole-trajectory-not-coordinate-or-frame"


def test_bytes_identity_binds_dtype_shape_and_values():
    a = np.arange(6, dtype=np.float32)
    assert array_digest(a) == array_digest(a.copy())
    assert array_digest(a) != array_digest(a.reshape(2, 3))
    assert array_digest(a) != array_digest(a.astype(np.float64))


def test_write_once_and_source_receipt_verification(tmp_path, monkeypatch):
    runner = _runner()
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    receipt = tmp_path / "receipt.json"
    value = {
        "schema": "deform-state-restart-source-receipt-v1",
        "git_clean": True,
        "files": {"source.py": file_digest(source)},
    }
    write_json_once(receipt, value)
    with pytest.raises(FileExistsError):
        write_json_once(receipt, value)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    assert runner.verify_source(receipt, file_digest(receipt)) == value
    source.write_text("x = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        runner.verify_source(receipt, file_digest(receipt))


def test_registered_protocol_keeps_protected_scope_closed():
    config = json.loads(
        (ROOT / "configs/sota/deform_sparse_state_restart_dev_v1.json").read_text()
    )
    assert config["runtime"]["device"] == "cpu"
    assert len(config["expected_names"]) == 14
    assert config["sparse_budget_point_observations"] == 8
    assert config["future_nonclamped_positions_are_model_inputs"] is False
    assert config["original_model_or_results_modified"] is False
    assert config["fresh_cohort_or_official_sota_claim_authorized"] is False
