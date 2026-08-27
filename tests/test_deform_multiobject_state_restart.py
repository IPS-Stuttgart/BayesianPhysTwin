"""Source-only contracts for fixed multi-object state/readout transfer."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_multiobject_restart import (
    config_for_object,
    load_protocol,
    summarize_predictions,
    transfer_assessment,
    validate_manifest,
)
from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    sparse_state_increments,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/sota/deform_multiobject_state_restart_v1.json"


@pytest.fixture
def protocol():
    return load_protocol(PROTOCOL)


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/remote"))
    spec = importlib.util.spec_from_file_location(
        "multi_runner", ROOT / "scripts/remote/run_deform_multiobject_state_restart.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_opened_rosters_and_fixed_sensing(protocol):
    assert sum(len(item["names"]) for item in protocol["objects"]) == 30
    for item in protocol["objects"]:
        config = config_for_object(protocol, item)
        assert len(config.observed_nodes) * len(config.observation_frames) == 8
        assert len(config.hidden_nodes) == 4
        assert config.forecast_end - config.prefix_length == 120
        assert not set(config.observed_nodes) & set(config.hidden_nodes)
        assert config.design_case == ("103.pkl" if item["object"] == "DLO2" else "")


@pytest.mark.parametrize(
    "field,value",
    [
        ("gains", {"primary": 0.25, "secondary": 0.25}),
        ("observation_frames", [49, 50]),
        ("protected_data_access", True),
        ("analysis_case_count", 28),
        ("future_free_node_truth_is_model_input", True),
        ("all_objects_and_noise_predictions_sealed_before_metrics", False),
    ],
)
def test_protocol_rejects_method_and_boundary_drift(protocol, tmp_path, field, value):
    protocol[field] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        load_protocol(path)


def test_only_dlo2_design_case_is_excluded(protocol, tmp_path):
    protocol["objects"][0]["excluded_design_case"] = "105.pkl"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="design case"):
        load_protocol(path)


@pytest.mark.parametrize("object_index", [0, 1, 2])
def test_sparse_inference_is_causal_and_zero_at_object_specific_clamps(
    protocol, runner, object_index
):
    config = config_for_object(protocol, protocol["objects"][object_index])
    raw = np.zeros((2, 500, config.node_count, 3), dtype=np.float32)
    initial, actions = runner.native.causal_model_inputs(raw, config)
    changed = raw.copy()
    free = [i for i in range(config.node_count) if i not in config.clamped_nodes]
    changed[:, config.prefix_length + 2 :, free] = 1e6
    initial2, actions2 = runner.native.causal_model_inputs(changed, config)
    np.testing.assert_array_equal(initial, initial2)
    np.testing.assert_array_equal(actions, actions2)
    reference = np.zeros((2, config.prefix_length, config.node_count, 3))
    observations = np.ones((2, 2, 4, 3)) * 0.001
    observations[:, 0] *= 0.5
    dx, dv = sparse_state_increments(reference, observations, config)
    assert np.count_nonzero(dx[:, config.clamped_nodes]) == 0
    assert np.count_nonzero(dv[:, config.clamped_nodes]) == 0
    np.testing.assert_allclose(dx[:, config.observed_nodes], 0.001)
    np.testing.assert_allclose(dv[:, config.observed_nodes], 0.0005 / 0.08)


def test_manifest_checks_exact_roster_hash_and_partition(protocol, tmp_path):
    item = copy.deepcopy(protocol["objects"][0])
    entries = {}
    for name in item["names"]:
        path = tmp_path / "DLO1" / "train" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        entries[name] = {"path": str(path), "sha256": file_digest(path)}
    manifest = {
        "dlo_type": "DLO1",
        "split": {"source_test": item["names"]},
        "trajectories": entries,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    item["manifest"] = {
        "path": str(path),
        "sha256": file_digest(path),
        "roster_key": "source_test",
    }
    assert validate_manifest(item) == manifest
    item["names"] = item["names"][::-1]
    with pytest.raises(ValueError, match="roster"):
        validate_manifest(item)
    item["names"] = item["names"][::-1]
    Path(entries[item["names"][0]]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        validate_manifest(item)


def test_noise_repetitions_do_not_become_independent_trajectories(protocol):
    config = dataclasses.replace(
        config_for_object(protocol, protocol["objects"][0]), bootstrap_replicates=80
    )
    truth = np.zeros((3, 120, config.node_count, 3))
    base = np.ones_like(truth) * 0.01
    points = base.copy()
    points[0] *= 0.5
    points[1] *= 0.75
    points[2] *= 1.1
    one = summarize_predictions(
        {"incumbent": base, "candidate": points}, truth, ["a", "b", "c"], config
    )
    many = summarize_predictions(
        {
            "incumbent": np.repeat(base[None], 16, 0),
            "candidate": np.repeat(points[None], 16, 0),
        },
        truth,
        ["a", "b", "c"],
        config,
    )
    for key in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm"):
        np.testing.assert_allclose(
            one["summaries"]["candidate"][key + "_delta_ci95"],
            many["summaries"]["candidate"][key + "_delta_ci95"],
        )
    assert many["summaries"]["candidate"]["case_count"] == 3
    assert many["summaries"]["candidate"]["noise_repetitions"] == 16


def test_metrics_cannot_hide_an_unsupported_case(protocol):
    config = config_for_object(protocol, protocol["objects"][2])
    truth = np.zeros((8, 120, 12, 3))
    base = np.ones_like(truth)
    candidate = base.copy()
    candidate[3, 5, 7, 1] = np.nan
    with pytest.raises(ValueError, match="denominator"):
        summarize_predictions(
            {"incumbent": base, "candidate": candidate},
            truth,
            protocol["objects"][2]["names"],
            config,
        )


def _mock_results(protocol):
    results = {}
    for item in protocol["objects"]:
        conditions = {}
        for condition in ("clean", *protocol["noise"]["conditions"]):
            arms = (
                protocol["clean_arms"]
                if condition == "clean"
                else protocol["noise_arms"]
            )
            rows = {}
            for arm in arms:
                factor = 0.8 if arm.startswith("incumbent_propagated") else 1.0
                row = {
                    "case_count": 13 if item["object"] == "DLO2" else 8,
                    "joint_wins": 6,
                }
                for metric in ("coordinate_l1_mm", "point_rmse_mm", "fde_mm"):
                    row[metric] = factor * 10
                    row[metric + "_change_percent"] = 100 * (factor - 1)
                row["late"] = {"point_rmse_mm": factor * 10}
                rows[arm] = row
            conditions[condition] = {"summaries": rows}
        results[item["object"]] = conditions
    return results


def test_primary_gate_cannot_be_rescued_by_discovery_or_secondary(protocol):
    results = _mock_results(protocol)
    assert transfer_assessment(protocol, results)["primary_transfer_gate_passed"]
    results["DLO3"]["clean"]["summaries"][protocol["primary_arm"]]["point_rmse_mm"] = 11
    decision = transfer_assessment(protocol, results)
    assert not decision["primary_transfer_gate_passed"]
    assert decision["checks"]["DLO1"]["point_rmse_improves"]
    assert decision["secondary_gain_cannot_rescue_primary_gate"]
    assert (
        decision["object_balanced"]["transfer_only"]["clean"]["incumbent"]["case_count"]
        == 16
    )


def test_equal_object_weighting_does_not_overweight_dlo2(protocol):
    results = _mock_results(protocol)
    results["DLO2"]["clean"]["summaries"]["incumbent"]["point_rmse_mm"] = 100
    assessment = transfer_assessment(protocol, results)
    assert (
        assessment["object_balanced"]["all_three_including_discovery"]["clean"][
            "incumbent"
        ]["point_rmse_mm"]
        == 40
    )
    assert (
        assessment["object_balanced"]["transfer_only"]["clean"]["incumbent"][
            "point_rmse_mm"
        ]
        == 10
    )
    del results["DLO3"]
    with pytest.raises(ValueError, match="all registered"):
        transfer_assessment(protocol, results)


def test_control_gate_rederives_checks_instead_of_trusting_passed(runner, protocol):
    controls = {
        "native_adapter_max_error_m": 0.0,
        "archived_gpu_replay_max_error_m": 0.0,
        "archived_gpu_replay_coordinate_rmse_m": 0.0,
        "zero_update_continuation_byte_identical": True,
        "incumbent_zero_update_returns_original_object": True,
        "synthetic_error_before_l2_m": 0.01,
        "synthetic_recovery_fraction": 1.0,
        "passed": True,
    }
    assert runner.control_gate(controls, protocol["controls"])
    controls["native_adapter_max_error_m"] = 1.0
    assert not runner.control_gate(controls, protocol["controls"])
    controls["native_adapter_max_error_m"] = float("nan")
    assert not runner.control_gate(controls, protocol["controls"])


def test_prediction_seal_is_write_once_and_binds_names(runner, tmp_path):
    path = tmp_path / "predictions.npz"
    points = np.zeros((2, 120, 12, 3))
    seal = runner.save_predictions(path, ["a", "b"], {"incumbent": points})
    assert seal["sha256"] == file_digest(path)
    assert seal["arrays"]["incumbent"] == array_digest(points)
    assert seal["arrays"]["names"] == array_digest(np.array(["a", "b"]))
    with pytest.raises(FileExistsError):
        runner.save_predictions(path, ["a", "b"], {"incumbent": points})
    points[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="ordinary success"):
        runner.save_predictions(
            tmp_path / "invalid.npz", ["a", "b"], {"incumbent": points}
        )


def test_barrier_requires_complete_roster_before_scoring(runner, protocol, tmp_path):
    barrier = {
        "schema": "deform-multiobject-prediction-barrier-v1",
        "source_receipt_sha256": "receipt",
        "protocol_sha256": file_digest(PROTOCOL),
        "ordinary_success": 30,
        "retained_technical_failure": 0,
        "unsealable": 0,
        "analysis_case_count": 29,
        "new_metrics_computed": False,
        "protected_data_access": False,
        "no_replacement": True,
        "objects": {},
    }
    (tmp_path / "preflight.json").write_text("{}")
    barrier["preflight_sha256"] = file_digest(tmp_path / "preflight.json")
    (tmp_path / "prediction_barrier.json").write_text(json.dumps(barrier))
    with pytest.raises(ValueError, match="denominator"):
        runner.validate_barrier(tmp_path, protocol, "receipt")
    (tmp_path / "failure.json").write_text("{}")
    with pytest.raises(ValueError, match="technical failure"):
        runner.validate_barrier(tmp_path, protocol, "receipt")


def test_prior_dlo2_native_transition_is_inherited_not_replaced(runner):
    assert runner.MultiObjectRod.advance is runner.native.NativeRod.advance
    assert runner.MultiObjectRod.initialize is runner.native.NativeRod.initialize
    assert runner.MultiObjectRod.rollout is runner.native.NativeRod.rollout


def test_no_data_access_when_freezing_a_dirty_tree(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **k: " M local-change"
    )
    with pytest.raises(ValueError, match="commit"):
        runner.freeze(tmp_path / "archive")
    assert not (tmp_path / "archive").exists()


def _synthetic_barrier(runner, protocol, tmp_path):
    protocol = copy.deepcopy(protocol)
    records = {}
    controls = {
        "native_adapter_max_error_m": 0.0,
        "archived_gpu_replay_max_error_m": 0.0,
        "archived_gpu_replay_coordinate_rmse_m": 0.0,
        "zero_update_continuation_byte_identical": True,
        "incumbent_zero_update_returns_original_object": True,
        "synthetic_error_before_l2_m": 0.01,
        "synthetic_error_after_l2_m": 0.0,
        "synthetic_recovery_fraction": 1.0,
        "passed": True,
    }
    for item in protocol["objects"]:
        directory = tmp_path / item["object"]
        directory.mkdir()
        shape = (len(item["names"]), 170, item["node_count"], 3)
        whole = np.ones(shape) * 0.01
        base = whole[:, 50:170]
        registered = directory / "registered.npz"
        with registered.open("xb") as stream:
            np.savez_compressed(stream, **{item["archive"]["incumbent_key"]: whole})
        item["archive"]["path"] = str(registered)
        item["archive"]["sha256"] = file_digest(registered)
        (directory / "controls.json").write_text(json.dumps(controls))
        files = {}
        for condition in ("clean", *protocol["noise"]["conditions"]):
            arms = (
                protocol["clean_arms"]
                if condition == "clean"
                else protocol["noise_arms"]
            )
            mean = (
                base
                if condition == "clean"
                else np.broadcast_to(base, (16, *base.shape))
            )
            files[condition] = runner.save_predictions(
                directory / f"{condition}.npz",
                item["names"],
                {arm: mean for arm in arms},
            )
        seal = {
            "schema": "deform-multiobject-object-prediction-seal-v1",
            "object": item["object"],
            "names": item["names"],
            "case_count": len(item["names"]),
            "files": files,
            "controls_sha256": file_digest(directory / "controls.json"),
            "incumbent_array_sha256": array_digest(base),
            "input_sha256s": {
                k: item[k]["sha256"] for k in ("archive", "checkpoint", "manifest")
            },
            "new_metrics_computed": False,
            "protected_data_access": False,
        }
        (directory / "prediction_seal.json").write_text(json.dumps(seal))
        records[item["object"]] = {
            "seal_sha256": file_digest(directory / "prediction_seal.json"),
            "case_count": len(item["names"]),
            "ordinary_success": len(item["names"]),
        }
    preflight = {
        "source_receipt_sha256": "receipt",
        "source_revision": "synthetic",
        "protocol_sha256": file_digest(PROTOCOL),
        "new_metrics_computed": False,
        "protected_data_access": False,
    }
    (tmp_path / "preflight.json").write_text(json.dumps(preflight))
    barrier = {
        "schema": "deform-multiobject-prediction-barrier-v1",
        "source_revision": "synthetic",
        "source_receipt_sha256": "receipt",
        "protocol_sha256": file_digest(PROTOCOL),
        "ordinary_success": 30,
        "retained_technical_failure": 0,
        "unsealable": 0,
        "analysis_case_count": 29,
        "new_metrics_computed": False,
        "protected_data_access": False,
        "no_replacement": True,
        "objects": records,
        "preflight_sha256": file_digest(tmp_path / "preflight.json"),
    }
    (tmp_path / "prediction_barrier.json").write_text(json.dumps(barrier))
    return protocol, barrier


def test_full_synthetic_barrier_and_forged_control(runner, protocol, tmp_path):
    protocol, barrier = _synthetic_barrier(runner, protocol, tmp_path)
    assert runner.validate_barrier(tmp_path, protocol, "receipt") == barrier
    directory = tmp_path / "DLO3"
    control_path = directory / "controls.json"
    controls = json.loads(control_path.read_text())
    controls["archived_gpu_replay_max_error_m"] = 0.5
    control_path.write_text(json.dumps(controls))
    seal_path = directory / "prediction_seal.json"
    seal = json.loads(seal_path.read_text())
    seal["controls_sha256"] = file_digest(control_path)
    seal_path.write_text(json.dumps(seal))
    barrier["objects"]["DLO3"]["seal_sha256"] = file_digest(seal_path)
    (tmp_path / "prediction_barrier.json").write_text(json.dumps(barrier))
    with pytest.raises(ValueError, match="does not rederive"):
        runner.validate_barrier(tmp_path, protocol, "receipt")


def test_independent_metric_formulas_have_correct_units_and_axes():
    spec = importlib.util.spec_from_file_location(
        "multi_verifier", ROOT / "scripts/verify_deform_multiobject_state_restart.py"
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    truth = np.zeros((2, 7, 4, 3))
    points = np.broadcast_to(np.array([0.003, 0.004, 0.0]), (3, 2, 7, 4, 3))
    scores = verifier.metrics(points, truth)
    assert scores["point_rmse_mm"].shape == (3, 2)
    np.testing.assert_allclose(scores["coordinate_l1_mm"], 7 / 3)
    np.testing.assert_allclose(scores["point_rmse_mm"], 5)
    np.testing.assert_allclose(scores["fde_mm"], 5)
