"""Synthetic-only projection, source-gate, and prediction-custody regressions."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_constraint_split import (
    ARMS,
    PRIMARY,
    ProjectionUnavailable,
    SplitConfig,
    config_record,
    constraint_rows,
    score_arrays,
    split_forecast,
    tangent_project,
)
from bayesian_phystwin_experiments.deform_state_restart import array_digest, file_digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return SplitConfig()


@pytest.fixture
def geometry():
    points = np.zeros((120, 12, 3))
    points[:, :, 0] = np.arange(12) * 0.05
    return points


@pytest.fixture
def runner():
    path = ROOT / "scripts/remote/run_deform_constraint_split_source.py"
    spec = importlib.util.spec_from_file_location("constraint_split_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_straight_rod_separates_axial_and_transverse(geometry, config):
    delta = np.zeros_like(geometry)
    delta[:, 2:10] = (0.01, 0.02, -0.03)
    projected, info = tangent_project(delta, geometry, config)
    np.testing.assert_allclose(projected[..., 0], 0, atol=1e-16)
    np.testing.assert_allclose(projected[..., 1:], delta[..., 1:], atol=1e-16)
    assert info["constraint_rank_min"] == 8
    assert info["maximum_linear_constraint_residual_m"] < 1e-16
    assert not np.any(projected[:, config.clamped_nodes])


@pytest.mark.parametrize("seed", range(8))
def test_projection_idempotent_orthogonal_and_rotation_equivariant(
    seed, geometry, config
):
    rng = np.random.default_rng(seed)
    geometry = geometry + rng.normal(0, 0.003, geometry.shape)
    delta = rng.normal(0, 0.01, geometry.shape)
    delta[:, config.clamped_nodes] = 0
    projected, info = tangent_project(delta, geometry, config)
    again, _ = tangent_project(projected, geometry, config)
    np.testing.assert_allclose(again, projected, atol=1e-15)
    np.testing.assert_allclose(
        np.sum(projected * (delta - projected), axis=(1, 2)), 0, atol=1e-16
    )
    assert info["tangent_squared_norm_m2"] <= info["input_squared_norm_m2"] + 1e-14
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated, _ = tangent_project(delta @ rotation, geometry @ rotation + 100, config)
    np.testing.assert_allclose(rotated, projected @ rotation, atol=3e-14)


def test_rows_are_derivative_of_edge_length_with_fixed_clamps(geometry, config):
    rng = np.random.default_rng(42)
    geometry = geometry + rng.normal(0, 0.01, geometry.shape)
    delta = rng.normal(0, 0.001, geometry.shape)
    delta[:, config.clamped_nodes] = 0
    rows, free = constraint_rows(geometry, config)
    epsilon = 1e-5
    finite_difference = (
        np.linalg.norm(np.diff(geometry + epsilon * delta, axis=1), axis=-1)
        - np.linalg.norm(np.diff(geometry - epsilon * delta, axis=1), axis=-1)
    ) / (2 * epsilon)
    analytic = np.einsum("tij,tj->ti", rows, delta[:, free].reshape(120, -1))
    np.testing.assert_allclose(analytic, finite_difference, atol=1e-11)


def test_complementary_construction_recovers_known_tangent_and_normal(geometry, config):
    dynamic = np.zeros_like(geometry)
    readout = np.zeros_like(geometry)
    dynamic[:, 2:10] = (0.01, 0.02, 0)
    readout[:, 2:10] = (0.03, 0.04, 0)
    arms, info = split_forecast(
        geometry, geometry + dynamic, geometry + readout, geometry, config
    )
    expected = np.zeros_like(geometry)
    expected[:, 2:10] = (0.03, 0.02, 0)
    np.testing.assert_allclose(arms[PRIMARY], geometry + expected, atol=1e-15)
    assert arms["incumbent"] is geometry
    assert info["ordinary_success"] and not info["exact_fallback"]
    np.testing.assert_allclose(arms["half_blend"], geometry + (dynamic + readout) / 2)


def test_equal_corrections_recompose_and_zero_returns_original(geometry, config):
    delta = np.zeros_like(geometry)
    delta[:, 2:10] = (0.003, 0.002, 0.001)
    arms, _ = split_forecast(
        geometry, geometry + delta, geometry + delta, geometry, config
    )
    np.testing.assert_allclose(arms[PRIMARY], geometry + delta, atol=1e-15)
    zero, _ = split_forecast(geometry, geometry, geometry, geometry, config)
    assert zero[PRIMARY] is geometry
    assert zero["tangent_only"] is geometry


def test_degenerate_geometry_exact_fallback_and_zero_does_not_hide_failure(
    geometry, config
):
    broken = geometry.copy()
    broken[:, 5] = broken[:, 4]
    arms, info = split_forecast(geometry, geometry, geometry, broken, config)
    assert not info["ordinary_success"] and info["exact_fallback"]
    assert info["reason"] == "degenerate_nominal_edge"
    assert arms[PRIMARY] is geometry
    assert array_digest(arms[PRIMARY]) == array_digest(geometry)


@pytest.mark.parametrize(
    "kind", ["nan", "clamp", "time", "dtype", "nonconstant_readout"]
)
def test_invalid_correction_contract_rejected(kind, geometry, config):
    incumbent, paired, readout, nominal = (geometry.copy() for _ in range(4))
    if kind == "nan":
        paired[0, 5, 0] = np.nan
    elif kind == "clamp":
        paired[0, 0, 0] += 0.01
    elif kind == "time":
        nominal = nominal[:-1]
    elif kind == "dtype":
        paired = paired.astype(np.float32)
    else:
        readout[-1, 5, 0] += 0.001
    with pytest.raises(ValueError):
        split_forecast(incumbent, paired, readout, nominal, config)


def test_nonfinite_nominal_retains_paired(geometry, config):
    nominal = geometry.copy()
    nominal[0, 3, 0] = np.nan
    arms, info = split_forecast(geometry, geometry, geometry, nominal, config)
    assert info["reason"] == "nonfinite_or_empty_nominal_geometry"
    assert arms[PRIMARY] is geometry


def test_only_own_time_model_geometry_changes_projection(geometry, config):
    delta = np.zeros_like(geometry)
    delta[:, 5, 0] = 0.01
    first, _ = tangent_project(delta, geometry, config)
    changed = geometry.copy()
    changed[-1, 5, 1] = 0.04
    second, _ = tangent_project(delta, changed, config)
    np.testing.assert_array_equal(first[:-1], second[:-1])
    assert not np.allclose(first[-1], second[-1])


def test_exact_gate_pass_and_fallback_failure(config):
    names = ["103.pkl", *[f"synthetic{i}.pkl" for i in range(13)]]
    truth = np.zeros((14, 120, 12, 3))
    predictions = {
        arm: np.full_like(truth, 0.002 if arm == PRIMARY else 0.01) for arm in ARMS
    }
    result = score_arrays(predictions, truth, names, [True] * 14, config)
    assert result["analysis_count"] == 13
    assert result["joint_wins_over_paired"] == 13
    assert result["gate"]["passed"]
    assert not result["new_transfer_or_target_execution_authorized"]
    result = score_arrays(predictions, truth, names, [False, *[True] * 13], config)
    assert not result["gate"]["passed"]


def test_stronger_blend_blocks_primary_and_ablation_cannot_rescue(config):
    names = ["103.pkl", *[f"synthetic{i}.pkl" for i in range(13)]]
    truth = np.zeros((14, 120, 12, 3))
    predictions = {arm: np.full_like(truth, 0.01) for arm in ARMS}
    predictions[PRIMARY][:] = 0.008
    predictions["half_blend"][:] = 0.005
    predictions["tangent_only"][:] = 0
    result = score_arrays(predictions, truth, names, [True] * 14, config)
    assert not result["gate"]["passed"]
    assert not result["gate"]["checks"][
        "at_least_2_percent_rmse_gain_over_every_control"
    ]


def test_equal_case_aggregation_excludes_only_design_case(config):
    names = ["103.pkl", *[f"synthetic{i}.pkl" for i in range(13)]]
    truth = np.zeros((14, 120, 12, 3))
    prediction = np.ones_like(truth) * 0.002
    prediction[0] = 999
    result = score_arrays(
        {arm: prediction for arm in ARMS}, truth, names, [True] * 14, config
    )
    assert result["metrics"][PRIMARY]["mean"]["all"][
        "coordinate_l1_mm"
    ] == pytest.approx(2)
    assert not result["gate"]["passed"]
    with pytest.raises(ValueError, match="entire fixed roster"):
        score_arrays(
            {arm: prediction[:-1] for arm in ARMS},
            truth[:-1],
            names[:-1],
            [True] * 13,
            config,
        )


def test_protocol_runtime_and_boundaries(runner):
    value = runner.load_protocol()
    assert value["config"] == config_record(SplitConfig())
    assert value["observations"]["count"] == 8
    assert value["runtime"]["cpu_only"]
    assert not value["native_replays_performed"]
    assert not value["transfer_objects_accessed"]


@pytest.mark.parametrize("change", ["method", "gate", "config", "boundary", "offset"])
def test_protocol_drift_is_rejected(runner, monkeypatch, change):
    value = runner.load_protocol()
    if change == "method":
        value["method"] = "learned blend"
    elif change == "gate":
        value["gate"]["minimum_rmse_gain_over_every_control_percent"] = 0
    elif change == "config":
        value["config"]["relative_svd_tolerance"] = 0.2
    elif change == "offset":
        value["inputs"]["source_truth"]["prediction_index_offset"] = 2
    else:
        value["protected_data_access"] = True
    monkeypatch.setattr(runner, "read_json", lambda path: value)
    with pytest.raises(ValueError):
        runner.load_protocol()


def test_content_binding_and_write_once(runner, tmp_path):
    path = tmp_path / "artifact.json"
    value = runner.write_identity(path, {"x": 1, "authorized": False})
    assert runner.read_identity(path) == value
    with pytest.raises(FileExistsError):
        runner.write_identity(path, {"x": 2})
    value["authorized"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="content binding"):
        runner.read_identity(path)


def test_prediction_has_no_truth_member_loader(runner, tmp_path, monkeypatch, geometry):
    protocol = copy.deepcopy(runner.load_protocol())
    protocol["output_root"] = str(tmp_path)
    protocol["inputs"]["source_truth"]["path"] = str(tmp_path / "must_not_open.npz")
    names = ["103.pkl", *[f"synthetic{i}.pkl" for i in range(13)]]
    values = np.repeat(geometry[None], 14, axis=0)
    monkeypatch.setattr(
        runner,
        "load_forecasts",
        lambda p: (
            names,
            {key: values for key in ("incumbent", "paired", "readout", "nominal")},
        ),
    )
    receipt = {"artifact_id": "synthetic-lock"}
    runner.predict(protocol, receipt)
    assert not (tmp_path / "must_not_open.npz").exists()
    seal = runner.read_identity(tmp_path / "prediction-seal.json")
    assert seal["ordinary_success"] == 14
    assert not seal["source_truth_opened"] and not seal["source_metrics_computed"]
    with pytest.raises(FileExistsError):
        runner.predict(protocol, receipt)


def test_scorer_rejects_missing_barrier_before_truth_access(runner, tmp_path):
    protocol = copy.deepcopy(runner.load_protocol())
    protocol["output_root"] = str(tmp_path)
    protocol["inputs"]["source_truth"]["path"] = str(tmp_path / "unavailable.npz")
    with pytest.raises(FileNotFoundError, match="prediction-seal"):
        runner.score(protocol, {"artifact_id": "synthetic"})


def test_synthetic_end_to_end_seal_then_score(runner, tmp_path, monkeypatch, geometry):
    protocol = copy.deepcopy(runner.load_protocol())
    protocol["output_root"] = str(tmp_path)
    names = ["103.pkl", *[f"synthetic{i}.pkl" for i in range(13)]]
    means = np.repeat(geometry[None], 14, axis=0)
    original = {
        key: means.copy() for key in ("incumbent", "paired", "readout", "nominal")
    }
    monkeypatch.setattr(runner, "load_forecasts", lambda p: (names, original))
    monkeypatch.setattr(runner, "registered_names", lambda p: names)
    receipt = {"artifact_id": "synthetic-lock"}
    runner.predict(protocol, receipt)
    target = np.zeros((14, 170, 12, 3))
    target[:, 50:] = means
    target[:, 50:, (3, 5, 7, 9)] += 0.002
    path = tmp_path / "synthetic-truth.npz"
    np.savez_compressed(path, names=names, targets=target)
    protocol["inputs"]["source_truth"].update(path=str(path), sha256=file_digest(path))
    runner.score(protocol, receipt)
    result = runner.read_identity(tmp_path / "result.json")
    assert result["metrics"][PRIMARY]["mean"]["all"][
        "coordinate_l1_mm"
    ] == pytest.approx(2)
    assert result["accounting"]["ordinary_success"] == 14
    assert not result["gate"]["passed"]
    assert not result["protected_data_access"]
    assert not result["new_transfer_or_target_execution_authorized"]
    with pytest.raises(ValueError, match="already exists"):
        runner.score(protocol, receipt)


def test_synthetic_seal_validation_and_comparator_tamper(
    runner, tmp_path, monkeypatch, geometry
):
    protocol = copy.deepcopy(runner.load_protocol())
    protocol["output_root"] = str(tmp_path)
    names = ["103.pkl", *[f"synthetic{i}.pkl" for i in range(13)]]
    values = np.repeat(geometry[None], 14, axis=0)
    original = {
        key: values.copy() for key in ("incumbent", "paired", "readout", "nominal")
    }
    monkeypatch.setattr(runner, "load_forecasts", lambda p: (names, original))
    monkeypatch.setattr(runner, "registered_names", lambda p: names)
    receipt = {"artifact_id": "synthetic-lock"}
    runner.predict(protocol, receipt)
    arrays, seal = runner.validated_predictions(protocol, receipt)
    assert arrays["names"].tolist() == names
    assert seal["ordinary_success"] == 14
    original["paired"][0, 0, 3, 0] += 1
    with pytest.raises(ValueError, match="comparator"):
        runner.validated_predictions(protocol, receipt)
    path = tmp_path / "predictions.npz"
    before = file_digest(path)
    with path.open("ab") as stream:
        stream.write(b"tamper")
    assert before != file_digest(path)
    with pytest.raises(ValueError, match="changed after seal"):
        runner.validated_predictions(protocol, receipt)


def test_svd_failure_exact_fallback(monkeypatch, geometry, config):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("synthetic failure")

    monkeypatch.setattr(np.linalg, "svd", fail)
    with pytest.raises(ProjectionUnavailable):
        tangent_project(np.zeros_like(geometry), geometry, config)
    arms, info = split_forecast(geometry, geometry, geometry, geometry, config)
    assert arms[PRIMARY] is geometry and info["exact_fallback"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_edge_length_m": 0},
        {"relative_svd_tolerance": np.nan},
        {"bootstrap_replicates": 0},
        {"clamped_nodes": (0, 0)},
        {"hidden_nodes": (0, 3)},
        {"node_count": 2},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        dataclasses.replace(SplitConfig(), **kwargs)
