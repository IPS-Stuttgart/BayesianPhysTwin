import importlib.util
from pathlib import Path

import pytest

from bayesian_phystwin_experiments.dlolab_regret_artifacts import (
    read_record,
    write_record,
)

SPEC = importlib.util.spec_from_file_location(
    "wrapping_runner",
    Path(__file__).resolve().parents[1]
    / "scripts/remote/run_dlolab_wrapping_source.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def prepare_code(tmp_path, monkeypatch):
    root, output = tmp_path / "code", tmp_path / "run"
    root.mkdir()
    (root / "frozen.py").write_text("pass\n")
    verifier = "scripts/verify_dlolab_wrapping_source.py"
    (root / "scripts").mkdir()
    (root / verifier).write_bytes(
        (Path(__file__).resolve().parents[1] / verifier).read_bytes()
    )
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "SOURCES", ("frozen.py", verifier))
    monkeypatch.setattr(runner, "clean_revision", lambda _: "a" * 40)
    monkeypatch.setattr(
        runner,
        "source",
        lambda: {
            "asset_archive_sha256": "acd483e232f1bb1fbf34078b154825fab3d2ee63b0aa4efc253c4411b368e421"
        },
    )
    monkeypatch.setattr(runner, "runtime", lambda: {"device": "cpu"})
    return root, output


def frozen_root(tmp_path, monkeypatch, mutation=None):
    root, output = prepare_code(tmp_path, monkeypatch)
    output.mkdir()
    data = {
        "schema": "dlolab-wrapping-source-lock-v1",
        "revision": "a" * 40,
        "protocol": runner.protocol(),
        "output_root": str(output),
        "source_sha256": {p: runner.file_digest(root / p) for p in runner.SOURCES},
        "runtime": runner.runtime(),
        "native_source": runner.source(),
    }
    if mutation == "revision":
        data["revision"] = "b" * 40
    elif mutation == "source":
        data["source_sha256"]["frozen.py"] = "0" * 64
    elif mutation == "runtime":
        data["runtime"] = {"device": "gpu"}
    elif mutation == "protocol":
        data["protocol"]["native_steps"] -= 1
    elif mutation == "native":
        data["native_source"] = {"native": "other"}
    lock = write_record(output / "lock.json", data)
    return output, lock


def test_exact_lock_roundtrip(tmp_path, monkeypatch):
    output, lock = frozen_root(tmp_path, monkeypatch)
    assert runner.validate_lock(output) == lock


@pytest.mark.parametrize(
    "mutation", ["revision", "source", "runtime", "protocol", "native"]
)
def test_source_runtime_method_custody_fail_closed(tmp_path, monkeypatch, mutation):
    output, _ = frozen_root(tmp_path, monkeypatch, mutation)
    with pytest.raises(ValueError, match="frozen"):
        runner.validate_lock(output)


@pytest.mark.parametrize("terminal", ["failure.json", "result.json"])
def test_terminal_outcome_disallows_worker(tmp_path, monkeypatch, terminal):
    output, _ = frozen_root(tmp_path, monkeypatch)
    write_record(output / terminal, {"status": "terminal"})
    with pytest.raises(ValueError, match="no retry"):
        runner.worker(output, 0)


def test_wrong_root_rejected_before_any_source_access(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "registered")
    with pytest.raises(ValueError, match="fresh"):
        runner.run(tmp_path / "unregistered")
    assert not (tmp_path / "unregistered").exists()


def test_worker_failure_consumes_attempt_before_native_call(tmp_path, monkeypatch):
    output, lock = frozen_root(tmp_path, monkeypatch)
    called = []

    def fail(upstream, directory, world):
        claim = read_record(directory / "claim.json")
        assert (
            claim["lock_id"] == lock["artifact_id"] and claim["task"]["world"] == world
        )
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


def test_failed_qa_not_overridden_by_forged_stored_pass(tmp_path, monkeypatch):
    output, lock = frozen_root(tmp_path, monkeypatch)
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
    with pytest.raises(ValueError, match="rederived"):
        runner.prerequisites(output, lock, 1)


def test_later_world_cannot_bypass_missing_nominal_repeats(tmp_path, monkeypatch):
    output, _ = frozen_root(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="cannot read"):
        runner.worker(output, 3)
    assert not (output / runner.task(3)["name"]).exists()


def test_unregistered_worker_has_no_execution_path(tmp_path, monkeypatch):
    output, lock = frozen_root(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unregistered"):
        runner.prerequisites(output, lock, 11)


@pytest.mark.parametrize("technical", [False, True])
def test_terminal_accounting_and_alternate_verifier(tmp_path, monkeypatch, technical):
    from test_dlolab_wrapping_source import checker, fixture

    root, output = prepare_code(tmp_path, monkeypatch)

    def native(upstream, directory, world):
        if technical:
            raise RuntimeError("synthetic runtime failure")
        data, report = fixture()
        data["post_pos_m"][:, :, 0, 0] += 0.001
        return data, report

    monkeypatch.setattr(runner, "run_world", native)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: runner.worker(output, int(command[-1])),
    )
    if technical:
        with pytest.raises(RuntimeError, match="synthetic"):
            runner.run(output)
    else:
        runner.run(output)
    result = read_record(output / "result.json")
    assert result["attempted_batches"] == 1 and result["unrun_batches"] == 10
    assert result["completed_native_trajectories"] == (0 if technical else 9)
    assert result["ordinary_trajectories"] == result["qualified_trajectories"] == 0
    assert not result["source_gate_passed"]
    monkeypatch.setattr(
        checker.subprocess,
        "check_output",
        lambda command, **kwargs: (root / command[-1].split(":", 1)[1]).read_bytes(),
    )
    checked = checker.verify(output, root)
    assert checked["passed"] and not checked["source_value_analyzed"]
    assert (
        not checked["new_empirical_execution"]
        and not checked["independent_human_review"]
    )
    with pytest.raises(ValueError, match="fresh"):
        runner.run(output)
