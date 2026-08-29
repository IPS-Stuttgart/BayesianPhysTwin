from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    read_record,
    write_record,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "wrapping_risk_certified_guard_v9_runner",
    ROOT / "scripts/remote/run_dlolab_wrapping_certified_guard_v9.py",
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
        "schema": "dlolab-wrapping-risk-certified-guard-decision-barrier-v9",
        "lock_id": "lock",
        "decision_seal_id": "decision",
        "pre_future": {"pre_future_gate_passed": True},
        "future_simulated": False,
        "future_read": False,
        "protected_data_read": False,
    }
    monkeypatch.setattr(
        runner, "read_record", lambda path: {**expected, "future_read": True}
    )
    monkeypatch.setattr(runner, "_barrier_contents", lambda *args: expected)
    with pytest.raises(ValueError, match="barrier changed"):
        runner._require_barrier(tmp_path, {"artifact_id": "lock"})


def test_complete_roster_has_no_padding_or_replacement_slots() -> None:
    assert runner.PREFIX_BATCH_COUNT == 32
    assert runner.WORLD_COUNT == 288
    assert runner.prefix_task(31)["world_indices"] == list(range(279, 288))
    assert runner.prefix_task(31)["native_world_indices"] == list(range(279, 288))


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
        directory = Path(str(args[1]))
        claim = read_record(directory / "claim.json")
        assert claim["lock_id"] == "lock"
        assert claim["retry_authorized"] is False
        assert claim["replacement_authorized"] is False
        assert claim["protected_data_read"] is False
        calls.append(cast(list[dict[str, object]], args[2]))
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(runner, "run_worlds", fail)
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        runner._worker(output, "prefix", 0)
    failure = read_record(output / "prefix-0" / "failure.json")
    assert failure["retry_authorized"] is False
    assert failure["replacement_authorized"] is False
    assert failure["protected_data_read"] is False
    with pytest.raises(FileExistsError):
        runner._worker(output, "prefix", 0)
    assert len(calls) == 1


def test_terminal_v4_summary_is_semantically_revalidated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = {
        "artifact_id": (
            "ef75f43b46654530ed8a788303feee13c36a3d448566041b42707fe898e07873"
        ),
        "failure_id": (
            "003be585e995ad8e38818cbb341fe9d39c8344d2dd8bc59d4bd6ace61945443f"
        ),
        "status": "terminal_technical_failure",
        "ordinary_future_worlds": 69,
        "registered_future_worlds": 72,
        "task_value_scored": False,
        "scientific_result_available": False,
        "retry_authorized": False,
        "replacement_authorized": False,
        "protected_data_read": False,
    }
    monkeypatch.setattr(runner, "_summary", lambda *args, **kwargs: dict(terminal))
    assert runner._terminal_v4() == terminal
    terminal["retry_authorized"] = True
    with pytest.raises(ValueError, match="terminal v4 summary lineage"):
        runner._terminal_v4()


def test_runtime_v7_summary_is_semantically_revalidated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = {
        "artifact_id": (
            "24bc06374ff8e5c392304b1b3091e346172b41e1ac8a22081d1efdaa52ff611e"
        ),
        "status": "complete",
        "constructor_successes": 24,
        "full_rollout_successes": 4,
        "qualification_passed": True,
        "scientific_outcome_scored": False,
        "retry_authorized": False,
        "protected_data_read": False,
    }
    monkeypatch.setattr(
        runner, "_summary", lambda *args, **kwargs: dict(qualification)
    )
    assert runner._runtime_v7() == qualification
    qualification["scientific_outcome_scored"] = True
    with pytest.raises(ValueError, match="runtime v7 qualification summary"):
        runner._runtime_v7()


def test_native_process_signal_is_persisted_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    task = runner.prefix_task(0)

    def fail_process(*args: Any, **kwargs: Any) -> SimpleNamespace:
        directory = output / task["name"]
        directory.mkdir()
        write_record(
            directory / "claim.json",
            {
                "schema": "dlolab-wrapping-risk-certified-guard-claim-v9",
                "lock_id": "lock",
                "task": task,
                "authorization": {"gate": "prefix_only_before_futures"},
                "retry_authorized": False,
                "replacement_authorized": False,
                "protected_data_read": False,
            },
        )
        return SimpleNamespace(returncode=-11)

    monkeypatch.setattr(runner.subprocess, "run", fail_process)
    with pytest.raises(RuntimeError, match="exited -11; no retry"):
        runner._execute(output, "prefix", 0)
    failure = read_record(output / task["name"] / "process-failure.json")
    assert failure["returncode"] == -11
    assert failure["retry_authorized"] is False
    assert failure["replacement_authorized"] is False
    assert failure["protected_data_read"] is False


def test_source_manifest_reads_only_compact_prior_summaries() -> None:
    terminal_v4 = [name for name in runner.NEW_SOURCES if "risk_guard_source_v4" in name]
    runtime_v7 = [
        name for name in runner.NEW_SOURCES if "runtime_qualification_v7" in name
    ]
    calibration_v8 = [
        name for name in runner.NEW_SOURCES if "risk_guard_source_v8/summary" in name
    ]
    assert terminal_v4 == [
        "results/sota/dlolab_wrapping_risk_guard_source_v4/summary.json"
    ]
    assert runtime_v7 == [
        "results/sota/dlolab_wrapping_risk_guard_runtime_qualification_v7/summary.json"
    ]
    assert calibration_v8 == [
        "results/sota/dlolab_wrapping_risk_guard_source_v8/summary.json"
    ]
    assert all(
        "generation" not in name and "future-" not in name
        for name in runner.NEW_SOURCES
    )


def test_attempt_boundary_fields_are_required_by_validator_source() -> None:
    source = (ROOT / "scripts/remote/run_dlolab_wrapping_certified_guard_v9.py").read_text()
    for expression in (
        'attempt.get("terminal_v4_partial_future_payload_read") is not False',
        'attempt.get("runtime_v7_arrays_read") is not False',
        'attempt.get("calibration_v8_raw_arrays_read") is not False',
        'record.get("replacement_authorized") is not False',
        'record.get("protected_data_read") is not False',
    ):
        assert expression in source
