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
    "wrapping_risk_guard_runner",
    ROOT / "scripts/remote/run_dlolab_wrapping_risk_guard_v4.py",
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
        runner, "_validate", lambda path: ({"artifact_id": "lock"}, {}, {})
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


def test_barrier_is_rederived_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "schema": "dlolab-wrapping-risk-guard-decision-barrier-v4",
        "lock_id": "lock",
        "decision_seal_id": "decision",
        "pre_future": {"pre_future_gate_passed": True},
        "future_simulated": False,
        "future_read": False,
    }
    monkeypatch.setattr(
        runner, "read_record", lambda path: {**expected, "future_read": True}
    )
    monkeypatch.setattr(runner, "_barrier_contents", lambda *args: expected)
    with pytest.raises(ValueError, match="barrier changed"):
        runner._require_barrier(tmp_path, {"artifact_id": "lock"})


def test_complete_roster_has_no_padding_or_replacement_slots() -> None:
    assert runner.PREFIX_BATCH_COUNT == 8
    assert runner.WORLD_COUNT == 72
    assert runner.prefix_task(7)["world_indices"] == list(range(63, 72))
    assert runner.prefix_task(7)["native_world_indices"] == list(range(63, 72))


def test_prefix_failure_consumes_claim_before_native_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(
        runner, "_validate", lambda path: ({"artifact_id": "lock"}, {}, {})
    )
    calls: list[list[dict[str, object]]] = []

    def fail(*args: object, **kwargs: object) -> object:
        directory = Path(args[1])
        assert read_record(directory / "claim.json")["lock_id"] == "lock"
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


def test_development_v3_binding_rejects_mutated_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "development-v3"
    (root / "decisions").mkdir(parents=True)
    (root / "generation").mkdir()
    attempt_path = tmp_path / "development-v3.attempt.json"
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
            "ordinary_worlds": 48,
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
    monkeypatch.setattr(runner, "DEVELOPMENT_V3", root)
    monkeypatch.setattr(runner, "DEVELOPMENT_V3_ATTEMPT", attempt_path)
    monkeypatch.setattr(
        runner,
        "DEVELOPMENT_V3_FILE_SHA256",
        {name: runner.file_digest(path) for name, path in paths.items()},
    )
    monkeypatch.setattr(runner, "DEVELOPMENT_V3_RESULT_ID", result["artifact_id"])
    value, hashes = runner._development_v3()
    assert value == result and len(hashes) == 8
    (root / "generation" / "arrays.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="development v3 evidence changed"):
        runner._development_v3()


def test_development_v4_diagnostic_rejects_reclassification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "diagnostic.json"
    record = write_record(
        path,
        {
            "status": "post_open_development_diagnostic",
            "development_result_ids": {
                "v2": runner.DEVELOPMENT_V2_RESULT_ID,
                "v3": runner.DEVELOPMENT_V3_RESULT_ID,
            },
            "selected_probability": 0.975,
            "selected_development_metrics": {
                "worlds": 80,
                "worlds_harmed_beyond_numeric_margin": 0,
            },
            "lead_is_not_prospective_evidence": True,
            "future_experiment_automatically_authorized": False,
            "development_v2_v3_results_reclassified": False,
            "protected_data_read": False,
        },
    )
    monkeypatch.setattr(runner, "DEVELOPMENT_V4_DIAGNOSTIC", path)
    monkeypatch.setattr(
        runner, "DEVELOPMENT_V4_DIAGNOSTIC_SHA256", runner.file_digest(path)
    )
    monkeypatch.setattr(runner, "DEVELOPMENT_V4_DIAGNOSTIC_ID", record["artifact_id"])
    assert runner._development_v4_diagnostic() == record
    mutated = dict(record)
    mutated.pop("artifact_id")
    mutated["future_experiment_automatically_authorized"] = True
    path.unlink()
    write_record(path, mutated)
    monkeypatch.setattr(
        runner, "DEVELOPMENT_V4_DIAGNOSTIC_SHA256", runner.file_digest(path)
    )
    with pytest.raises(ValueError, match="development diagnostic changed"):
        runner._development_v4_diagnostic()
