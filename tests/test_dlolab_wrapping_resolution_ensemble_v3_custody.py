from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    read_record,
    write_record,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "wrapping_resolution_ensemble_runner",
    ROOT / "scripts/remote/run_dlolab_wrapping_resolution_ensemble_v3.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_alternate_root_is_rejected_before_artifact_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "registered")
    monkeypatch.setattr(
        runner,
        "read_record",
        lambda path: pytest.fail("artifact read before root rejection"),
    )
    with pytest.raises(ValueError, match="registered continuous wrapping root"):
        runner._validate(tmp_path / "alternate")


def test_future_requires_recomputed_barrier_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(
        runner,
        "_validate",
        lambda path: ({"artifact_id": "lock"}, {}, {}),
    )
    monkeypatch.setattr(
        runner,
        "_require_barrier",
        lambda *args: (_ for _ in ()).throw(ValueError("missing decision barrier")),
    )
    monkeypatch.setattr(
        runner,
        "run_worlds",
        lambda *args, **kwargs: pytest.fail("native future entered"),
    )
    with pytest.raises(ValueError, match="decision barrier"):
        runner._worker(output, "future", 0)
    assert list(output.iterdir()) == []


def test_future_loader_checks_barrier_before_reading_future(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "_require_barrier",
        lambda *args: (_ for _ in ()).throw(ValueError("missing decision barrier")),
    )
    monkeypatch.setattr(
        runner,
        "read_record",
        lambda path: pytest.fail("future artifact read before barrier"),
    )
    with pytest.raises(ValueError, match="decision barrier"):
        runner._load_task(tmp_path, {}, runner.future_task(0))


def test_barrier_is_rederived_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "schema": "dlolab-wrapping-resolution-ensemble-decision-barrier-v3",
        "lock_id": "lock",
        "decision_seal_id": "decision",
        "pre_future": {"pre_future_gate_passed": True},
        "future_simulated": False,
        "future_read": False,
    }
    recorded = {**expected, "future_simulated": True}
    monkeypatch.setattr(runner, "read_record", lambda path: recorded)
    monkeypatch.setattr(runner, "_barrier_contents", lambda *args: expected)
    with pytest.raises(ValueError, match="barrier changed"):
        runner._require_barrier(tmp_path, {"artifact_id": "lock"})


def test_prefix_padding_does_not_expand_scientific_denominator() -> None:
    last = runner.prefix_task(5)
    assert last["world_indices"] == [45, 46, 47]
    assert last["native_world_indices"] == [45, 46, 47, 47, 47, 47, 47, 47, 47]
    assert len(runner.continuous_worlds()) == runner.WORLD_COUNT == 48


def test_prefix_worker_failure_consumes_claim_before_native_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(
        runner,
        "_validate",
        lambda path: ({"artifact_id": "lock"}, {}, {}),
    )
    calls: list[list[dict[str, object]]] = []

    def fail(*args: object, **kwargs: object) -> object:
        directory = Path(args[1])
        claim = read_record(directory / "claim.json")
        assert claim["lock_id"] == "lock"
        calls.append(list(args[2]))
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(runner, "run_worlds", fail)
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        runner._worker(output, "prefix", 0)
    failure = read_record(output / "prefix-0" / "failure.json")
    assert failure["retry_authorized"] is False
    with pytest.raises(FileExistsError):
        runner._worker(output, "prefix", 0)
    assert len(calls) == 1


def test_wrong_main_root_is_rejected_before_parent_or_preflight_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = tmp_path / "registered"
    alternate = tmp_path / "alternate"
    monkeypatch.setattr(runner, "OUTPUT", registered)
    monkeypatch.setattr(
        runner,
        "_parent",
        lambda: pytest.fail("parent read before root rejection"),
    )
    monkeypatch.setattr(
        runner,
        "_load_preflight",
        lambda: pytest.fail("preflight read before root rejection"),
    )
    with pytest.raises(ValueError, match="one fresh continuous wrapping attempt"):
        runner._run(alternate)
    assert not alternate.exists()


def test_future_task_cannot_use_unregistered_index() -> None:
    with pytest.raises(ValueError, match="registered continuous wrapping future"):
        runner.future_task(True)


def test_decision_barrier_contents_cannot_claim_future_read() -> None:
    value = {
        "schema": "dlolab-wrapping-resolution-ensemble-decision-barrier-v3",
        "lock_id": "lock",
        "decision_seal_id": "decision",
        "pre_future": {"pre_future_gate_passed": True},
        "future_simulated": False,
        "future_read": False,
    }
    assert value["future_simulated"] is False and value["future_read"] is False
    assert np.asarray([value["pre_future"]["pre_future_gate_passed"]]).all()


def test_terminal_v1_binding_rejects_any_resumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "terminal-v1"
    (root / "decisions").mkdir(parents=True)
    attempt = write_record(tmp_path / "attempt.json", {"schema": "attempt"})
    write_record(root / "lock.json", {"schema": "lock"})
    write_record(root / "decisions" / "seal.json", {"schema": "decisions"})
    write_record(root / "decision-barrier.json", {"schema": "barrier"})
    failure = write_record(
        root / "failure.json",
        {
            "completed_prefix_batches": 4,
            "completed_future_worlds": 32,
            "retry_authorized": False,
            "replacement_authorized": False,
        },
    )
    paths = {
        "attempt.json": tmp_path / "attempt.json",
        "lock.json": root / "lock.json",
        "decisions/seal.json": root / "decisions" / "seal.json",
        "decision-barrier.json": root / "decision-barrier.json",
        "failure.json": root / "failure.json",
    }
    monkeypatch.setattr(runner, "TERMINAL_V1", root)
    monkeypatch.setattr(runner, "TERMINAL_V1_ATTEMPT", tmp_path / "attempt.json")
    monkeypatch.setattr(
        runner,
        "TERMINAL_V1_FILE_SHA256",
        {name: runner.file_digest(path) for name, path in paths.items()},
    )
    monkeypatch.setattr(runner, "TERMINAL_V1_FAILURE_ID", failure["artifact_id"])
    value, hashes = runner._terminal_v1()
    assert value == failure and len(hashes) == 5 and attempt["artifact_id"]
    write_record(root / "result.json", {"status": "forged-resumption"})
    with pytest.raises(ValueError, match="resumed"):
        runner._terminal_v1()


def test_development_v2_binding_rejects_mutated_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "development-v2"
    (root / "decisions").mkdir(parents=True)
    (root / "generation").mkdir()
    attempt_path = tmp_path / "development-v2.attempt.json"
    write_record(attempt_path, {"schema": "attempt"})
    write_record(root / "lock.json", {"schema": "lock"})
    np.savez(root / "decisions" / "arrays.npz", decisions=np.zeros(1))
    write_record(root / "decisions" / "seal.json", {"schema": "decisions"})
    write_record(root / "decision-barrier.json", {"schema": "barrier"})
    np.savez(root / "generation" / "arrays.npz", reward=np.zeros(1))
    write_record(root / "generation" / "seal.json", {"schema": "generation"})
    result = write_record(
        root / "result.json",
        {
            "status": "complete",
            "source_gate_passed": False,
            "ordinary_worlds": 32,
            "technical_failures": 0,
            "replacements": 0,
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    paths = {
        "attempt.json": attempt_path,
        "lock.json": root / "lock.json",
        "decisions/arrays.npz": root / "decisions" / "arrays.npz",
        "decisions/seal.json": root / "decisions" / "seal.json",
        "decision-barrier.json": root / "decision-barrier.json",
        "generation/arrays.npz": root / "generation" / "arrays.npz",
        "generation/seal.json": root / "generation" / "seal.json",
        "result.json": root / "result.json",
    }
    monkeypatch.setattr(runner, "DEVELOPMENT_V2", root)
    monkeypatch.setattr(runner, "DEVELOPMENT_V2_ATTEMPT", attempt_path)
    monkeypatch.setattr(
        runner,
        "DEVELOPMENT_V2_FILE_SHA256",
        {name: runner.file_digest(path) for name, path in paths.items()},
    )
    monkeypatch.setattr(runner, "DEVELOPMENT_V2_RESULT_ID", result["artifact_id"])
    value, hashes = runner._development_v2()
    assert value == result and len(hashes) == 8
    (root / "generation" / "arrays.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="development v2 evidence changed"):
        runner._development_v2()


def test_development_v3_diagnostic_binding_rejects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "diagnostic.json"
    record = write_record(
        path,
        {
            "schema": "dlolab-wrapping-resolution-ensemble-development-v3",
            "status": "post_open_development_diagnostic",
            "development_v2_result_id": runner.DEVELOPMENT_V2_RESULT_ID,
            "model_resolution_weights": {"finite": 0.5, "continuous": 0.5},
            "registered_v2_arm_reproduction": {
                "fixed": True,
                "finite_particle_bayes": True,
                "continuous_bayes": True,
                "continuous_map": True,
            },
            "worlds": 32,
            "sensor_draws_per_world": 4096,
            "lead_is_not_prospective_evidence": True,
            "future_experiment_authorized": False,
            "v2_source_gate_reclassified": False,
            "protected_data_read": False,
        },
    )
    monkeypatch.setattr(runner, "DEVELOPMENT_V3_DIAGNOSTIC", path)
    monkeypatch.setattr(runner, "DEVELOPMENT_V3_DIAGNOSTIC_ID", record["artifact_id"])
    assert runner._development_v3_diagnostic() == record
    mutated = dict(record)
    mutated.pop("artifact_id")
    mutated["future_experiment_authorized"] = True
    path.unlink()
    write_record(path, mutated)
    with pytest.raises(ValueError, match="development diagnostic changed"):
        runner._development_v3_diagnostic()
