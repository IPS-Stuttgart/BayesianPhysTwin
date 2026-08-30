from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    read_record,
    write_record,
)

SPEC = importlib.util.spec_from_file_location(
    "unknotting_headroom_runner_v1",
    Path(__file__).resolve().parents[1]
    / "scripts/remote/run_dlolab_unknotting_headroom_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _frozen_root(tmp_path, monkeypatch, mutation=None):
    root, output, attempt_path = (
        tmp_path / "code",
        tmp_path / "run",
        tmp_path / "attempt.json",
    )
    root.mkdir()
    output.mkdir()
    (root / "frozen.py").write_text("pass\n")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "ATTEMPT", attempt_path)
    monkeypatch.setattr(runner, "SOURCES", ("frozen.py",))
    monkeypatch.setattr(runner, "clean_revision", lambda _: "a" * 40)
    monkeypatch.setattr(runner, "source", lambda: {"native": "frozen"})
    monkeypatch.setattr(runner, "runtime", lambda: {"device": "cpu"})
    launch = {
        "revision": "a" * 40,
        "protocol": runner.protocol(),
        "output_root": str(output),
        "source_sha256": {"frozen.py": runner.file_digest(root / "frozen.py")},
        "runtime": runner.runtime(),
        "native_source": runner.source(),
    }
    attempt = write_record(
        attempt_path,
        {
            "schema": "dlolab-unknotting-headroom-attempt-v1",
            **launch,
            "retry_authorized": False,
            "protected_data_read": False,
        },
    )
    value = {
        "schema": "dlolab-unknotting-headroom-lock-v1",
        "attempt_id": attempt["artifact_id"],
        **launch,
    }
    if mutation == "revision":
        value["revision"] = "b" * 40
    elif mutation == "source":
        value["source_sha256"]["frozen.py"] = "0" * 64
    elif mutation == "runtime":
        value["runtime"] = {"device": "gpu"}
    elif mutation == "protocol":
        value["protocol"]["world_count"] = 8
    elif mutation == "native":
        value["native_source"] = {"native": "other"}
    return output, write_record(output / "lock.json", value)


def test_exact_lock_roundtrip(tmp_path, monkeypatch) -> None:
    output, lock = _frozen_root(tmp_path, monkeypatch)
    assert runner.validate_lock(output) == lock


@pytest.mark.parametrize(
    "mutation", ["revision", "source", "runtime", "protocol", "native"]
)
def test_source_runtime_and_protocol_custody_fail_closed(
    tmp_path, monkeypatch, mutation
) -> None:
    output, _ = _frozen_root(tmp_path, monkeypatch, mutation)
    with pytest.raises(ValueError, match="frozen"):
        runner.validate_lock(output)


@pytest.mark.parametrize("terminal", ["failure.json", "result.json"])
def test_terminal_result_disallows_worker(tmp_path, monkeypatch, terminal) -> None:
    output, _ = _frozen_root(tmp_path, monkeypatch)
    write_record(output / terminal, {"status": "terminal"})
    with pytest.raises(ValueError, match="no retry"):
        runner.validate_lock(output)


def test_wrong_root_rejected_before_source_access(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "registered")
    monkeypatch.setattr(runner, "ATTEMPT", tmp_path / "attempt.json")
    with pytest.raises(ValueError, match="fresh"):
        runner.run(tmp_path / "unregistered")
    assert not (tmp_path / "unregistered").exists()
    assert not runner.ATTEMPT.exists()


def test_runtime_preflight_fails_before_attempt_or_root(tmp_path, monkeypatch) -> None:
    root, output, attempt = (
        tmp_path / "code",
        tmp_path / "run",
        tmp_path / "attempt.json",
    )
    root.mkdir()
    (root / "frozen.py").write_text("pass\n")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "ATTEMPT", attempt)
    monkeypatch.setattr(runner, "SOURCES", ("frozen.py",))
    monkeypatch.setattr(runner, "clean_revision", lambda _: "a" * 40)
    monkeypatch.setattr(runner, "source", lambda: {"native": "frozen"})

    def fail_runtime():
        raise ValueError("synthetic runtime mismatch")

    monkeypatch.setattr(runner, "runtime", fail_runtime)
    with pytest.raises(ValueError, match="runtime mismatch"):
        runner.run(output)
    assert not attempt.exists()
    assert not output.exists()


def test_existing_attempt_rejects_successor_before_root(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"
    attempt = tmp_path / "attempt.json"
    attempt.write_text("consumed\n")
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "ATTEMPT", attempt)
    with pytest.raises(ValueError, match="fresh"):
        runner.run(output)
    assert not output.exists()


def test_worker_claim_precedes_native_failure_and_consumes_attempt(
    tmp_path, monkeypatch
) -> None:
    output, lock = _frozen_root(tmp_path, monkeypatch)
    called = []

    def fail(upstream, directory, world):
        claim = read_record(directory / "claim.json")
        assert claim["lock_id"] == lock["artifact_id"]
        assert claim["task"]["world"] == world
        called.append(world)
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(runner, "run_world", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        runner.worker(output, 0)
    failure = read_record(output / runner.task(0)["name"] / "failure.json")
    assert not failure["retry_authorized"]
    with pytest.raises(FileExistsError):
        runner.worker(output, 0)
    assert len(called) == 1


def test_previous_world_qa_is_rederived_before_next_world(
    tmp_path, monkeypatch
) -> None:
    output, lock = _frozen_root(tmp_path, monkeypatch)
    directory = output / runner.task(0)["name"]
    directory.mkdir()
    write_record(
        directory / "qa.json",
        {"lock_id": lock["artifact_id"], "seal_id": "seal", "qa": {"passed": True}},
    )
    monkeypatch.setattr(
        runner,
        "load_task",
        lambda *args: ({"artifact_id": "seal"}, {}, {"passed": False}),
    )
    with pytest.raises(ValueError, match="requalify"):
        runner.worker(output, 1)
