"""Custody checks use temporary synthetic artifacts, never study inputs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin_experiments.dlolab_regret_artifacts as artifacts
from bayesian_phystwin_experiments.deform_state_restart import file_digest
from bayesian_phystwin_experiments.dlolab_native import DloLabConfig


def test_write_once_and_digest_reject_tampering(tmp_path):
    path = tmp_path / "record.json"
    artifacts.write_record(path, {"value": 1})
    with pytest.raises(FileExistsError):
        artifacts.write_record(path, {"value": 2})
    value = json.loads(path.read_text())
    value["value"] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity"):
        artifacts.read_record(path)


def test_array_bundle_binds_dtype_shape_bytes_and_members(tmp_path):
    expected = {"position": np.arange(6, dtype=np.float64).reshape(2, 3)}
    manifest = artifacts.write_bundle(tmp_path, expected)
    np.testing.assert_array_equal(
        artifacts.load_bundle(tmp_path, manifest)["position"], expected["position"]
    )
    with pytest.raises(FileExistsError):
        artifacts.write_bundle(tmp_path, expected)
    wrong = {**manifest, "file": "../arrays.npz"}
    with pytest.raises(ValueError, match="path"):
        artifacts.load_bundle(tmp_path, wrong)
    wrong = {**manifest, "arrays": {"position": "a" * 64}}
    with pytest.raises(ValueError, match="identity"):
        artifacts.load_bundle(tmp_path, wrong)
    path = tmp_path / "arrays.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes"):
        artifacts.load_bundle(tmp_path, manifest)


def _qualification(tmp_path):
    native = tmp_path / "src/bayesian_phystwin_experiments/dlolab_native.py"
    native.parent.mkdir(parents=True)
    native.write_text("fixture\n")
    trajectory = tmp_path / "trajectories.npz"
    trajectory.write_bytes(b"synthetic provenance fixture")
    value = {
        "schema": "dlolab-native-qualification-result-v1",
        "world_bank": True,
        "qualification_passed": True,
        "config_id": DloLabConfig().identity,
        "checks": {key: True for key in artifacts.QUALIFICATION_CHECKS},
        "protected_data_read": False,
        "method_outcomes_read": False,
        "source_sha256": {
            "src/bayesian_phystwin_experiments/dlolab_native.py": file_digest(native)
        },
        "source_revision": "a" * 40,
        "trajectories_sha256": file_digest(trajectory),
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(value))
    return path, value


@pytest.mark.parametrize("change", ["world_bank", "checks", "source", "trajectory"])
def test_qualification_is_not_just_a_pass_boolean(tmp_path, change):
    path, value = _qualification(tmp_path)
    artifacts.validate_qualification(path, tmp_path)
    if change == "world_bank":
        value["world_bank"] = False
    elif change == "checks":
        value["checks"][artifacts.QUALIFICATION_CHECKS[0]] = False
    elif change == "source":
        value["source_sha256"]["src/bayesian_phystwin_experiments/dlolab_native.py"] = (
            "b" * 64
        )
    else:
        value["trajectories_sha256"] = "b" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        artifacts.validate_qualification(path, tmp_path)


def test_lock_binds_output_source_runtime_and_protocol(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "code.py").write_text("fixture\n")
    monkeypatch.setattr(artifacts, "SOURCE_PATHS", ("code.py",))
    monkeypatch.setattr(artifacts, "clean_revision", lambda _: "a" * 40)
    monkeypatch.setattr(
        artifacts,
        "validate_qualification",
        lambda *_: {"path": str(tmp_path / "q"), "sha256": "b" * 64},
    )
    monkeypatch.setattr(artifacts, "runtime_identity", lambda: {"cpu": True})
    monkeypatch.setattr(artifacts, "verify_upstream", lambda _: {"revision": "c" * 40})
    output = tmp_path / "study"
    lock = artifacts.freeze(root, output, tmp_path, tmp_path / "q")
    assert artifacts.validate_lock(root, output) == lock
    with pytest.raises(FileExistsError):
        artifacts.freeze(root, output, tmp_path, tmp_path / "q")
    monkeypatch.setattr(artifacts, "runtime_identity", lambda: {"cpu": False})
    with pytest.raises(ValueError, match="runtime"):
        artifacts.validate_lock(root, output)
    monkeypatch.setattr(artifacts, "runtime_identity", lambda: {"cpu": True})
    (root / "code.py").write_text("changed\n")
    with pytest.raises(ValueError, match="source bytes"):
        artifacts.validate_lock(root, output)


def test_incomplete_prediction_barrier_cannot_open_outcomes(tmp_path, monkeypatch):
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "scripts/remote/run_dlolab_regret_source.py"
    )
    spec = importlib.util.spec_from_file_location("dlolab_runner_test", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    lock = {"artifact_id": "a" * 64, "upstream_root": str(tmp_path)}
    monkeypatch.setattr(runner, "validate_lock", lambda *_: lock)
    entered = []
    monkeypatch.setattr(runner, "start_prefix", lambda *_: entered.append(True))
    with pytest.raises(ValueError, match="cannot read"):
        runner.execute(tmp_path, "score")
    assert entered == []
    assert not (tmp_path / "score").exists()


@pytest.mark.parametrize("count,outcomes", [(63, False), (64, True)])
def test_stage_validates_complete_count_and_unopened_evaluation(
    tmp_path, count, outcomes
):
    directory = tmp_path / "predict"
    directory.mkdir()
    bundle = artifacts.write_bundle(
        directory, {"decisions": np.zeros((count, 7), dtype=int)}
    )
    lock = {"artifact_id": "a" * 64}
    artifacts.write_record(
        directory / "seal.json",
        {
            "schema": "dlolab-regret-stage-seal-v1",
            "stage": "predict",
            "lock_id": lock["artifact_id"],
            "status": "ordinary_success",
            "count": count,
            "protected_data_read": False,
            "evaluation_outcomes_generated": outcomes,
            "bundle": bundle,
        },
    )
    with pytest.raises(ValueError):
        artifacts.read_stage(tmp_path, "predict", lock)


def test_complete_staged_dry_run_never_generates_truth_in_predict(
    tmp_path, monkeypatch
):
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/remote/run_dlolab_regret_source.py"
    )
    spec = importlib.util.spec_from_file_location("dlolab_staged_dry_run", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    lock = {"artifact_id": "a" * 64, "upstream_root": str(tmp_path)}
    monkeypatch.setattr(runner, "validate_lock", lambda *_: lock)
    generated = []

    class Snapshot:
        field_digests = {"synthetic_memory": "b" * 64}

    class Runtime:
        model_id = "c" * 64

        def __init__(self, count):
            self.count = count
            self.initial_positions = np.zeros((count, 16, 3))

        def close(self):
            pass

    def prefix(_upstream, bending, _velocity):
        count = len(bending)
        return Runtime(count), np.zeros((count, 25, 16, 3)), Snapshot()

    def future(runtime, _snapshot):
        generated.append(runtime.count)
        return np.zeros((runtime.count, 9, 40, 16, 3))

    monkeypatch.setattr(runner, "start_prefix", prefix)
    monkeypatch.setattr(runner, "continue_all_actions", future)
    runner.execute(tmp_path, "bank")
    runner.execute(tmp_path, "calibrate")
    prediction = runner.execute(tmp_path, "predict")
    assert generated == [15, 39]
    assert prediction["evaluation_outcomes_generated"] is False
    assert prediction["count"] == 64
    assert len(prediction["command_sha256"]) == 64
    with pytest.raises(FileExistsError):
        runner.execute(tmp_path, "predict")
    assert generated == [15, 39]
    scored = runner.execute(tmp_path, "score")
    assert generated == [15, 39, 64]
    assert not scored["result"]["source_gate_passed"]
    assert scored["result"]["ordinary_evaluation_episodes"] == 64
    with pytest.raises(FileExistsError):
        runner.execute(tmp_path, "score")
    assert generated == [15, 39, 64]
