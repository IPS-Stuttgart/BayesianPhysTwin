"""Fail-closed custody tests for the independent-action v3 runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_native import file_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slingshot_policy_certificate_source_v3_runner",
    ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v3.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _lock() -> dict[str, Any]:
    return {
        "artifact_id": "lock",
        "controls": np.zeros((8, 3, 6), dtype=np.float64).tolist(),
        "assets_root": "/unused",
        "runtime": {},
    }


def test_qualification_result_is_exactly_bound() -> None:
    assert file_digest(runner.QUALIFICATION_SUMMARY) == runner.QUALIFICATION_RESULT_SHA256
    value = runner.load_qualification()
    assert value["artifact_id"] == runner.QUALIFICATION_RESULT_ID
    assert value["qualification_passed"] is True


def test_alternate_root_is_rejected_before_parent_or_qualification_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "registered")
    monkeypatch.setattr(
        runner, "load_parent", lambda: pytest.fail("parent read before root rejection")
    )
    monkeypatch.setattr(
        runner,
        "load_qualification",
        lambda: pytest.fail("qualification read before root rejection"),
    )
    with pytest.raises(ValueError, match="fresh registered"):
        runner.freeze(tmp_path / "alternate")


def test_calibration_future_reproduces_candidates_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(runner, "validate_lock", lambda path: _lock())
    monkeypatch.setattr(
        runner,
        "load_candidates",
        lambda *args: (_ for _ in ()).throw(ValueError("candidate reproduction failed")),
    )
    monkeypatch.setattr(
        runner,
        "run_registered_world",
        lambda *args, **kwargs: pytest.fail("native future entered"),
    )
    with pytest.raises(ValueError, match="candidate reproduction failed"):
        runner.worker(output, "calibration", "future", 0, 0)
    assert list(output.iterdir()) == []


def test_evaluation_future_reproduces_decision_barrier_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(runner, "validate_lock", lambda path: _lock())
    monkeypatch.setattr(
        runner,
        "load_evaluation_decisions",
        lambda *args: (_ for _ in ()).throw(ValueError("decision reproduction failed")),
    )
    monkeypatch.setattr(
        runner,
        "run_registered_world",
        lambda *args, **kwargs: pytest.fail("native future entered"),
    )
    with pytest.raises(ValueError, match="decision reproduction failed"):
        runner.worker(output, "evaluation", "future", 0, 0)
    assert list(output.iterdir()) == []


def test_future_failure_consumes_claim_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(runner, "validate_lock", lambda path: _lock())
    monkeypatch.setattr(
        runner,
        "load_candidates",
        lambda *args: ({"artifact_id": "candidate"}, {}),
    )

    def fail(*args: object, **kwargs: object) -> object:
        directory = Path(str(args[1]))
        claim = read_record(directory / "claim.json")
        assert claim["retry_authorized"] is False
        assert claim["replacement_authorized"] is False
        raise RuntimeError("synthetic independent failure")

    monkeypatch.setattr(runner, "run_registered_world", fail)
    with pytest.raises(RuntimeError, match="synthetic independent failure"):
        runner.worker(output, "calibration", "future", 0, 0)
    failure = read_record(output / "calibration-future-000-action-00/failure.json")
    assert failure["retry_authorized"] is False
    assert failure["replacement_authorized"] is False
    with pytest.raises(FileExistsError):
        runner.worker(output, "calibration", "future", 0, 0)


def test_world_qualification_requires_all_eight_action_seals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[int] = []

    def load(*args: object, **kwargs: object) -> object:
        action = int(args[4])
        called.append(action)
        if action == 7:
            raise ValueError("missing final action")
        return {"artifact_id": f"seal-{action}", "native": {}}, {}

    monkeypatch.setattr(runner, "load_future_action", load)
    with pytest.raises(ValueError, match="missing final action"):
        runner._world_qualification(
            tmp_path,
            _lock(),
            "calibration",
            0,
            write=True,
            authorization={"gate": "test"},
        )
    assert called == list(range(8))
    assert not (tmp_path / "calibration-future-000-qualification.json").exists()


def test_runner_binds_one_attempt_and_complete_independent_denominator() -> None:
    source = (
        ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v3.py"
    ).read_text()
    for expression in (
        '"attempt_number": 1',
        '"retry_authorized": False',
        '"replacement_authorized": False',
        "for action_index in range(ACTION_COUNT)",
        "independent_world_qa(",
        "load_evaluation_decisions(output, lock)",
        '"partial_score_authorized": False',
    ):
        assert expression in source
    assert runner.FUTURE_WORKERS == 8
    assert runner.COUNTS == {"calibration": 128, "evaluation": 288}
