from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "wrapping_continuous_bayes_runner",
    ROOT / "scripts/remote/run_dlolab_wrapping_continuous_bayes_v1.py",
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
        "schema": "dlolab-wrapping-continuous-decision-barrier-v1",
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
    last = runner.prefix_task(3)
    assert last["world_indices"] == [27, 28, 29, 30, 31]
    assert last["native_world_indices"] == [27, 28, 29, 30, 31, 31, 31, 31, 31]
    assert len(runner.continuous_worlds()) == runner.WORLD_COUNT == 32


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
        "schema": "dlolab-wrapping-continuous-decision-barrier-v1",
        "lock_id": "lock",
        "decision_seal_id": "decision",
        "pre_future": {"pre_future_gate_passed": True},
        "future_simulated": False,
        "future_read": False,
    }
    assert value["future_simulated"] is False and value["future_read"] is False
    assert np.asarray([value["pre_future"]["pre_future_gate_passed"]]).all()


def test_compact_terminal_summary_is_content_bound_and_unscored() -> None:
    path = (
        ROOT
        / "results/sota/dlolab_wrapping_continuous_bayes_source_v1/summary.json"
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    artifact_id = summary.pop("artifact_id")
    assert content_id(summary) == artifact_id
    assert summary["status"] == "terminal_technical_failure"
    assert summary["completed_future_worlds"] == 32
    assert summary["task_value_scored"] is False
    assert summary["retry_authorized"] is False
