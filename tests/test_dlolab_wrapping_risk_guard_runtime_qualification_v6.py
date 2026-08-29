from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_native import array_digest
from bayesian_phystwin_experiments.dlolab_regret_artifacts import read_record

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "wrapping_runtime_qualification_v6",
    ROOT / "scripts/remote/run_dlolab_wrapping_risk_guard_runtime_qualification_v6.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _constructor_data() -> dict[str, np.ndarray]:
    return {
        "rod_pos_m": np.zeros((9, 50, 3), dtype=np.float64),
        "rod_vel_m_s": np.zeros((9, 50, 3), dtype=np.float64),
        "post_pos_m": np.zeros((9, 3, 3), dtype=np.float64),
        "gripper_pos_m": np.zeros((9, 2, 3), dtype=np.float64),
        "robot_qpos": np.zeros((9, 14), dtype=np.float64),
        "memory_state": np.zeros((1,), dtype=np.float64),
    }


def test_plan_freezes_distinct_source_only_qualification() -> None:
    value = runner.plan()
    assert value["native_workload"]["constructor_processes"] == 24
    assert value["native_workload"]["full_rollout_processes"] == 4
    assert value["gate"]["retry_authorized"] is False
    assert value["gate"]["study_automatically_authorized"] is False
    assert value["boundaries"] == {
        "fresh_scientific_worlds_defined": False,
        "physical_execution": False,
        "protected_data_read": False,
        "scientific_outcome_scored": False,
        "v4_partial_future_artifacts_read": False,
        "v4_retry": False,
        "v5_runtime_artifacts_read": False,
        "v5_retry": False,
    }


def test_registered_tasks_are_complete_and_unpadded() -> None:
    assert runner._task("constructor", 23)["name"] == "constructor-23"
    assert runner._task("full", 3)["name"] == "full-rollout-03"
    assert runner._task("full", 3)["worlds"] == [runner.preflight_world()] * 9
    with pytest.raises(ValueError, match="registered runtime qualification task"):
        runner._task("constructor", 24)
    with pytest.raises(ValueError, match="registered runtime qualification task"):
        runner._task("prefix", 0)


def test_constructor_qa_binds_initial_state_and_no_reward() -> None:
    data = _constructor_data()
    native = {
        "native_steps": 0,
        "worlds": [runner.preflight_world()] * 9,
        "constructor_completed": True,
        "init_cmaes_env_completed": True,
        "future_simulated": False,
        "reward_exposed": False,
        "parameter_randomization_deferred": True,
        "world_realization": {},
        "state_sha256": {
            name: array_digest(value) for name, value in sorted(data.items())
        },
    }
    assert runner._constructor_qa(data, native)["qa_passed"] is True
    native["world_realization"] = {"bending": [10000.0] * 9}
    assert runner._constructor_qa(data, native)["qa_passed"] is False
    native["world_realization"] = {}
    native["reward_exposed"] = True
    assert runner._constructor_qa(data, native)["qa_passed"] is False


def test_alternate_root_is_rejected_before_artifact_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "registered")
    monkeypatch.setattr(
        runner,
        "read_record",
        lambda path: pytest.fail("artifact read before root rejection"),
    )
    with pytest.raises(ValueError, match="registered runtime qualification root"):
        runner._validate_lock(tmp_path / "alternate")


def test_existing_attempt_rejects_campaign_before_runtime_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    attempt = tmp_path / "attempt.json"
    attempt.write_text("consumed", encoding="utf-8")
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "ATTEMPT", attempt)
    monkeypatch.setattr(
        runner,
        "runtime_identity_v6",
        lambda: pytest.fail("runtime read after consumed attempt"),
    )
    with pytest.raises(ValueError, match="one fresh runtime qualification attempt"):
        runner._run(output)


def test_signal_exit_is_sealed_as_terminal_process_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=-11),
    )
    lock = {"artifact_id": "lock"}
    with pytest.raises(RuntimeError, match="exited -11; no retry"):
        runner._execute(output, lock, "constructor", 0)
    failure = read_record(output / "constructor-00" / "process-failure.json")
    assert failure["returncode"] == -11
    assert failure["retry_authorized"] is False
    assert failure["replacement_authorized"] is False
    assert not (output / "constructor-00" / "seal.json").exists()


def test_worker_requires_parent_claim_before_native_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    (output / "constructor-00").mkdir(parents=True)
    monkeypatch.setattr(runner, "_validate_lock", lambda path: {"artifact_id": "lock"})
    monkeypatch.setattr(
        runner,
        "run_constructor_probe",
        lambda *args, **kwargs: pytest.fail("native constructor entered"),
    )
    with pytest.raises(ValueError, match="cannot read DLO-Lab study record"):
        runner._worker(output, "constructor", 0)


def test_source_lock_includes_runtime_plan_and_v4_terminal_summary() -> None:
    assert (
        "configs/sota/dlolab_wrapping_risk_guard_runtime_qualification_v6.json"
        in runner.SOURCE_PATHS
    )
    assert (
        "results/sota/dlolab_wrapping_risk_guard_source_v4/summary.json"
        in runner.SOURCE_PATHS
    )
    assert (
        "results/sota/dlolab_wrapping_risk_guard_runtime_qualification_v5/summary.json"
        in runner.SOURCE_PATHS
    )
