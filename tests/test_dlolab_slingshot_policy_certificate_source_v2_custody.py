from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slingshot_policy_certificate_source_v2_runner",
    ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v2.py",
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


def test_alternate_root_is_rejected_before_parent_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "registered")
    monkeypatch.setattr(
        runner,
        "load_parent",
        lambda: pytest.fail("parent read before output-root rejection"),
    )
    with pytest.raises(ValueError, match="registered one-attempt root"):
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
        "run_registered_worlds",
        lambda *args, **kwargs: pytest.fail("native future entered"),
    )
    with pytest.raises(ValueError, match="candidate reproduction failed"):
        runner.worker(output, "calibration", "future", 0)
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
        "run_registered_worlds",
        lambda *args, **kwargs: pytest.fail("native future entered"),
    )
    with pytest.raises(ValueError, match="decision reproduction failed"):
        runner.worker(output, "evaluation", "future", 0)
    assert list(output.iterdir()) == []


def test_prefix_failure_consumes_claim_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(runner, "validate_lock", lambda path: _lock())

    def fail(*args: object, **kwargs: object) -> object:
        directory = Path(str(args[1]))
        claim = read_record(directory / "claim.json")
        assert claim["retry_authorized"] is False
        assert claim["replacement_authorized"] is False
        assert claim["protected_data_read"] is False
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(runner, "run_registered_worlds", fail)
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        runner.worker(output, "calibration", "prefix", 0)
    failure = read_record(output / "calibration-prefix-00/failure.json")
    assert failure["retry_authorized"] is False
    assert failure["replacement_authorized"] is False
    assert failure["protected_data_read"] is False
    with pytest.raises(FileExistsError):
        runner.worker(output, "calibration", "prefix", 0)


def test_compact_calibration_rejects_changed_realized_gain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rewards = np.zeros((128, 7), dtype=np.float64)
    candidate = {
        "candidate_actions": np.zeros(128, dtype=np.int64),
        "predicted_gain": np.zeros(128, dtype=np.float64),
        "mean_raw_upper": np.zeros((128, 7), dtype=np.float64),
    }
    calibration, realized = runner.calibrate(candidate, rewards)
    simultaneous = runner.calibrate_simultaneous_guard(candidate, rewards)
    future_ids = [f"future-{index:03d}" for index in range(128)]
    seal = {
        "schema": "dlolab-slingshot-policy-calibration-v2",
        "lock_id": "lock",
        "candidate_seal_id": "candidate",
        "future_seal_ids": future_ids,
        "all_native_qa": True,
        "policy_calibration": runner.dataclasses.asdict(calibration),
        "simultaneous_calibration": runner.dataclasses.asdict(simultaneous),
        "bundle": {},
        "evaluation_prefix_read": False,
        "evaluation_future_simulated": False,
        "evaluation_future_read": False,
        "protected_data_read": False,
        "artifact_id": "calibration",
    }
    changed = np.array(realized, copy=True)
    changed[0] = 1.0
    monkeypatch.setattr(runner, "_calibration_seal", lambda *args: seal)
    monkeypatch.setattr(
        runner,
        "load_native_bundle",
        lambda *args: {"rewards": rewards, "realized_candidate_gain": changed},
    )
    monkeypatch.setattr(
        runner,
        "load_candidates",
        lambda *args: ({"artifact_id": "candidate"}, candidate),
    )
    monkeypatch.setattr(
        runner,
        "_task_records",
        lambda output, lock, role, kind, index: {
            "artifact_id": f"future-{index:03d}"
        },
    )
    with pytest.raises(ValueError, match="calibration does not reproduce"):
        runner.load_calibration(tmp_path, _lock())


def test_complete_rosters_have_no_replacement_slots() -> None:
    assert runner.COUNTS == {"calibration": 128, "evaluation": 288}
    assert runner.prefix_batch_count("calibration") == 16
    assert runner.prefix_batch_count("evaluation") == 36
    assert runner.future_task("evaluation", 287)["world_index"] == 287


def test_runner_binds_reproduced_authorization_and_frozen_sources() -> None:
    source = (
        ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v2.py"
    ).read_text()
    for expression in (
        'candidate, _ = load_candidates(output, lock, role)',
        'decision, _, barrier = load_evaluation_decisions(output, lock)',
        'array_digest(arrays["realized_candidate_gain"]) != array_digest(realized)',
        'observed != expected',
        'directory.mkdir(exist_ok=False)',
        '"retry_authorized": False',
        '"replacement_authorized": False',
    ):
        assert expression in source
    assert "tests/test_dlolab_slingshot_policy_certificate_source_v2_custody.py" in (
        runner.SOURCES
    )
    assert "scripts/remote/verify_dlolab_slingshot_policy_certificate_source_v2.py" in (
        runner.SOURCES
    )
