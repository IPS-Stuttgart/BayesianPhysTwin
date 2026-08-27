"""Synthetic verification fixtures, independent of the selected recording."""

from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)
from bayesian_phystwin_experiments.deft_cross_branch_source import (
    ARMS,
    PRIMARY,
    score_cross_branch,
)

ROOT = Path(__file__).resolve().parents[1]


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "deft_independent_verifier", ROOT / "scripts/verify_deft_cross_branch_source.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path, monkeypatch):
    module = _verifier()
    source = tmp_path / "source.pkl"
    source.write_bytes(pickle.dumps(np.zeros((3, 500, 20)), protocol=4))
    monkeypatch.setattr(module, "SOURCE_FILE_SHA256", file_digest(source))
    truth = np.zeros((120, 3, 13, 3))
    arrays = {arm: np.ones_like(truth) * 0.02 for arm in ARMS}
    arrays[PRIMARY] *= 0.5
    with (tmp_path / "predictions.npz").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    write_json_once(
        tmp_path / "prediction_barrier.json",
        {
            "prediction_file_sha256": file_digest(tmp_path / "predictions.npz"),
            "array_sha256s": {
                arm: array_digest(value) for arm, value in arrays.items()
            },
        },
    )
    result = {
        **score_cross_branch(arrays, truth),
        "prediction_barrier_sha256": file_digest(tmp_path / "prediction_barrier.json"),
        "ordinary_successful_recordings": 1,
        "technical_failures": 0,
        "unsealable": 0,
        "protected_data_read": False,
        "public_evaluation_or_test_split_read": False,
    }
    write_json_once(tmp_path / "result.json", result)
    return module, source, result


def test_independent_recomputation_agrees_on_all_child_metrics_and_gate(
    tmp_path, monkeypatch
):
    module, source, _ = _fixture(tmp_path, monkeypatch)
    result = module.verify(tmp_path, source)
    assert result["verified_child_and_aggregate_metrics"] == 96
    assert result["verified_gate_checks"] == 18
    assert result["verified_prediction_arms"] == 8
    assert result["source_pilot_gate_passed"] is True


@pytest.mark.parametrize("mutation", ["metric", "gate", "count", "boundary"])
def test_verifier_rejects_result_tampering(tmp_path, monkeypatch, mutation):
    module, source, result = _fixture(tmp_path, monkeypatch)
    if mutation == "metric":
        result["per_arm"][PRIMARY]["child1"]["point_rmse_mm"] += 1
    elif mutation == "gate":
        result["source_pilot_gate_passed"] = False
    elif mutation == "count":
        result["ordinary_successful_recordings"] = 2
    else:
        result["protected_data_read"] = True
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        module.verify(tmp_path, source)


def test_verifier_rejects_prediction_or_source_byte_changes(tmp_path, monkeypatch):
    module, source, _ = _fixture(tmp_path, monkeypatch)
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source recording changed"):
        module.verify(tmp_path, source)
    source.write_bytes(source.read_bytes()[:-7])
    path = tmp_path / "predictions.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="prediction archive changed"):
        module.verify(tmp_path, source)
