"""Complete synthetic custody/scoring artifacts; no public source payload access."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)
from bayesian_phystwin_experiments.deft_cross_branch_source import pack_branched_world
from bayesian_phystwin_experiments.deft_topology_observer import (
    ARMS,
    CASE_IDS,
    PRIMARY,
    permitted_inputs,
    score_case,
    score_study,
)

ROOT = Path(__file__).resolve().parents[1]


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/remote"))
    runner = _module(
        ROOT / "scripts/remote/run_deft_topology_observer.py", "_test_topology_runner"
    )
    verifier = _module(
        ROOT / "scripts/verify_deft_topology_observer.py", "_test_topology_verifier"
    )
    source_root, run_root, training = (
        tmp_path / name for name in ("source", "run", "training")
    )
    for path in (source_root, run_root, training):
        path.mkdir()
    monkeypatch.setattr(runner, "ROOT", source_root)
    monkeypatch.setattr(runner, "PROTOCOL", "protocol.json")
    protocol = json.loads(
        (ROOT / "configs/sota/deft_topology_observer_source_v1.json").read_text()
    )
    trajectories = {}
    for index, spec in enumerate(protocol["source_cases"]):
        raw = (
            np.arange(30000, dtype=np.float64).reshape(3, 500, 20) / 1000000
            + index * 0.1
        )
        payload = pickle.dumps(raw, protocol=4)
        (training / spec["filename"]).write_bytes(payload)
        spec["sha256"] = hashlib.sha256(payload).hexdigest()
        spec["git_blob"] = hashlib.sha1(
            b"blob " + str(len(payload)).encode() + b"\0" + payload
        ).hexdigest()
        trajectories[spec["id"]] = pack_branched_world(raw.transpose(1, 2, 0))
    protocol_path = source_root / "protocol.json"
    write_json_once(protocol_path, protocol)
    protocol_sha = file_digest(protocol_path)
    source_sha = "a" * 64
    staged, sealed, rows = {}, {}, {}
    for case in CASE_IDS:
        case_root = run_root / case
        case_root.mkdir()
        inputs = permitted_inputs(trajectories[case])
        np.savez_compressed(case_root / "permitted_inputs.npz", **inputs)
        write_json_once(
            case_root / "input_manifest.json",
            {
                "schema": runner.SCHEMA + "-input",
                "case_id": case,
                "source_receipt_sha256": source_sha,
                "protocol_sha256": protocol_sha,
                "future_free_node_values_published": False,
                "input_sha256": file_digest(case_root / "permitted_inputs.npz"),
                "array_sha256s": {
                    key: array_digest(value) for key, value in inputs.items()
                },
            },
        )
        staged[case] = {
            "status": "staged",
            "manifest_sha256": file_digest(case_root / "input_manifest.json"),
        }
        truth = trajectories[case][52:172]
        arrays = {arm: truth + (0.002 if arm == PRIMARY else 0.01) for arm in ARMS}
        np.savez_compressed(case_root / "predictions.npz", **arrays)
        write_json_once(
            case_root / "prediction_seal.json",
            {
                "schema": runner.SCHEMA + "-case-seal",
                "case_id": case,
                "source_receipt_sha256": source_sha,
                "protocol_sha256": protocol_sha,
                "source_future_scoring_opened": False,
                "controls": {"zero_update_byte_identical": True},
                "input_manifest_sha256": file_digest(case_root / "input_manifest.json"),
                "prediction_sha256": file_digest(case_root / "predictions.npz"),
                "array_sha256s": {
                    arm: array_digest(value) for arm, value in arrays.items()
                },
            },
        )
        sealed[case] = {
            "status": "ordinary-success",
            "seal_sha256": file_digest(case_root / "prediction_seal.json"),
        }
        rows[case] = score_case(arrays, truth)
    write_json_once(
        run_root / "input_barrier.json",
        {
            "schema": runner.SCHEMA + "-input-barrier",
            "source_receipt_sha256": source_sha,
            "protocol_sha256": protocol_sha,
            "cases": staged,
            "source_future_scoring_opened": False,
        },
    )
    write_json_once(
        run_root / "prediction_barrier.json",
        {
            "schema": runner.SCHEMA + "-prediction-barrier",
            "source_receipt_sha256": source_sha,
            "protocol_sha256": protocol_sha,
            "cases": sealed,
            "input_barrier_sha256": file_digest(run_root / "input_barrier.json"),
            "source_future_scoring_opened": False,
            "ordinary_successful_recordings": 3,
            "technical_failures": 0,
        },
    )
    write_json_once(
        run_root / "result.json",
        {
            "protocol_sha256": protocol_sha,
            "prediction_barrier_sha256": file_digest(
                run_root / "prediction_barrier.json"
            ),
            **score_study(rows),
            "ordinary_successful_recordings": 3,
            "technical_failures": 0,
            "unsealable": 0,
            "protected_data_read": False,
            "independent_confirmation": False,
        },
    )
    return SimpleNamespace(
        runner=runner,
        verifier=verifier,
        root=run_root,
        training=training,
        protocol=protocol_path,
        source_sha=source_sha,
    )


def test_synthetic_whole_study_and_independent_raw_identity_verification(bundle):
    barrier, predictions = bundle.runner.read_predictions(
        bundle.root,
        file_digest(bundle.root / "prediction_barrier.json"),
        bundle.source_sha,
    )
    assert set(predictions) == set(CASE_IDS)
    assert barrier["ordinary_successful_recordings"] == 3
    report = bundle.verifier.verify(bundle.root, bundle.training, bundle.protocol)
    assert report["verification_passed"]
    assert report["source_gate_passed"]
    assert report["metrics_verified"] == 512
    assert report["gate_checks_verified"] == 36
    assert report["prediction_arms_verified"] == 24


@pytest.mark.parametrize(
    "mutation", ["missing-case", "failure", "opened", "wrong-source"]
)
def test_incomplete_or_wrong_custody_never_reads_outcomes(
    bundle, monkeypatch, mutation
):
    path = bundle.root / "prediction_barrier.json"
    value = json.loads(path.read_text())
    if mutation == "missing-case":
        del value["cases"][CASE_IDS[-1]]
    elif mutation == "failure":
        value["ordinary_successful_recordings"] = 2
        value["technical_failures"] = 1
    elif mutation == "opened":
        value["source_future_scoring_opened"] = True
    else:
        value["source_receipt_sha256"] = "b" * 64
    path.write_text(json.dumps(value))
    monkeypatch.setattr(
        bundle.runner,
        "load_training_case",
        lambda *args: pytest.fail("outcome opened before complete barrier"),
    )
    args = SimpleNamespace(
        run_root=bundle.root,
        prediction_barrier_sha256=file_digest(path),
        source_receipt_sha256=bundle.source_sha,
    )
    with pytest.raises(ValueError):
        bundle.runner.score(args, {}, {})
    assert not (bundle.root / "score_attempt.json").exists()


def test_prediction_tamper_and_exact_fallback_tamper_are_rejected(bundle):
    root = bundle.root
    (root / CASE_IDS[0] / "predictions.npz").write_bytes(b"not the sealed prediction")
    with pytest.raises(ValueError, match="prediction or exact-fallback"):
        bundle.runner.read_predictions(
            root, file_digest(root / "prediction_barrier.json"), bundle.source_sha
        )


def test_independent_verifier_detects_wrong_hidden_identity_metric(bundle):
    path = bundle.root / "result.json"
    value = json.loads(path.read_text())
    value["per_recording"][CASE_IDS[0]][PRIMARY]["child1"]["point_rmse_mm"] += 0.1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="per-recording metric"):
        bundle.verifier.verify(bundle.root, bundle.training, bundle.protocol)


def test_independent_verifier_detects_changed_decision(bundle):
    path = bundle.root / "result.json"
    value = json.loads(path.read_text())
    value["source_gate_passed"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source decision"):
        bundle.verifier.verify(bundle.root, bundle.training, bundle.protocol)


def test_write_once_attempt_does_not_allow_retry(tmp_path):
    path = tmp_path / "prediction_attempt.json"
    write_json_once(path, {"attempt": 1})
    with pytest.raises(FileExistsError):
        write_json_once(path, {"attempt": 2})


def test_protocol_is_training_only_and_preserves_old_native_identity():
    protocol = json.loads(
        (ROOT / "configs/sota/deft_topology_observer_source_v1.json").read_text()
    )
    assert tuple(row["id"] for row in protocol["source_cases"]) == CASE_IDS
    assert tuple(protocol["arms"]) == ARMS
    assert protocol["inputs"]["point_observations_per_corrected_arm"] == 8
    assert (
        protocol["inputs"]["unique_point_observations_staged_for_both_policies"] == 12
    )
    assert protocol["inputs"]["state_update_gain"] == 1.0
    assert protocol["boundaries"]["split"] == "train"
    assert protocol["boundaries"]["parameter_checkpoint_or_gain_fitting"] is False
    assert (
        protocol["native_qualification"]["result_sha256"]
        == "87e1649ffdb34602a995ce0e7d4760925e5ad0a29bc4e722fbbc9ca6b34840b2"
    )
