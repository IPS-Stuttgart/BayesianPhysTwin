from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_slingshot_belief_native import (
    PrefixComplete,
    PrefixTrace,
    run_registered_worlds,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slingshot_belief_runner", ROOT / "scripts/remote/run_dlolab_slingshot_belief.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_native_prefix_stop_occurs_exactly_before_frame300():
    trace = PrefixTrace()
    for i in range(299):
        trace.append({"position": np.array([i], dtype=float)})
    with pytest.raises(PrefixComplete):
        trace.append({"position": np.array([299.0])})
    assert trace.arrays()["position"].shape == (300, 1)
    with pytest.raises(ValueError, match="crossed"):
        trace.append({"position": np.array([300.0])})
    with pytest.raises(ValueError, match="incomplete"):
        PrefixTrace().arrays()


def test_source_particles_are_reused_and_padding_does_not_create_cases():
    new = []
    for i in range(27):
        try:
            new.append(runner.task("particle", i))
        except ValueError as error:
            assert "reused" in str(error)
    assert len(new) == 18
    last = runner.task("calibration-prefix", 2)
    assert last["case_indices"] == [16, 17, 18]
    assert len(last["worlds"]) == 8
    assert last["worlds"][2:] == [last["worlds"][2]] * 6
    assert len(runner.task("qualification", 0)["worlds"]) == 8
    for kind, index in (
        ("evaluation-prefix", 4),
        ("evaluation-future", 32),
        ("particle", 27),
        ("qualification", 1),
        ("target", 0),
        ("particle", True),
    ):
        with pytest.raises(ValueError):
            runner.task(kind, index)


def test_alternate_roots_are_rejected_before_any_artifact_read(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "read_record",
        lambda path: pytest.fail("artifact read before root gate"),
    )
    with pytest.raises(ValueError, match="write-once root"):
        runner.validate_lock(tmp_path)


def test_full_future_requires_all_prediction_seals_before_native_claim(
    tmp_path, monkeypatch
):
    lock = {"artifact_id": "lock"}
    monkeypatch.setattr(runner, "validate_lock", lambda output: (lock, {}, []))
    monkeypatch.setattr(runner, "require_qualification", lambda *args: None)
    monkeypatch.setattr(
        runner, "load_bank", lambda *args: ({"artifact_id": "bank"}, {})
    )
    monkeypatch.setattr(
        runner, "load_calibrator", lambda *args: ({"artifact_id": "cal"}, {})
    )

    def refuse(*args):
        raise ValueError("missing prediction barrier")

    monkeypatch.setattr(runner, "require_barrier", refuse)
    monkeypatch.setattr(
        runner,
        "run_registered_worlds",
        lambda *args, **kwargs: pytest.fail("native future entered"),
    )
    with pytest.raises(ValueError, match="barrier"):
        runner.worker(tmp_path, "evaluation-future", 0)
    assert list(tmp_path.iterdir()) == []


def test_barrier_is_rederived_and_cannot_drop_a_case(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "load_prediction",
        lambda output, lock, role, i: ({"artifact_id": f"prediction-{i}"}, {}),
    )
    content = runner.barrier_contents(tmp_path, {"artifact_id": "lock"}, "evaluation")
    assert len(content["prediction_seals"]) == 32
    content["count"] = 31
    monkeypatch.setattr(runner, "read_record", lambda path: content)
    with pytest.raises(ValueError, match="complete prediction barrier"):
        runner.require_barrier(tmp_path, {"artifact_id": "lock"}, "evaluation")


def test_invalid_native_worlds_are_rejected_before_import(tmp_path):
    world = {"x_offset_m": 0.0, "bending_E": 1e5, "stretching_K": 8e5}
    with pytest.raises(ValueError, match="eight"):
        run_registered_worlds(
            tmp_path, tmp_path, np.zeros((8, 3, 6)), [world] * 7, prefix_only=True
        )
    bad = {**world, "x_offset_m": 10.0}
    with pytest.raises(ValueError, match="parameters"):
        run_registered_worlds(
            tmp_path, tmp_path, np.zeros((8, 3, 6)), [bad] * 8, prefix_only=True
        )


def test_calibrator_values_are_recomputed_not_trusted(tmp_path, monkeypatch):
    from bayesian_phystwin_experiments.coupled_action_regret import RegretCalibration

    monkeypatch.setattr(
        runner,
        "read_record",
        lambda path: {
            "lock_id": "lock",
            "count": 19,
            "evaluation_futures_read": False,
            "calibrations": {
                m: {"coverage": 0.9, "count": 19, "rank": 18, "offset": 0.0}
                for m in runner.MODES
            },
            "future_seals": ["x"] * 19,
            "native_qa": [{"qa_passed": True}] * 19,
        },
    )
    monkeypatch.setattr(
        runner,
        "future_table",
        lambda *args: (np.zeros((19, 7)), ["x"] * 19, [{"qa_passed": True}] * 19),
    )
    monkeypatch.setattr(runner, "load_prediction", lambda *args: ({}, {}))
    monkeypatch.setattr(
        runner,
        "calibrate",
        lambda *args: {m: RegretCalibration(0.9, 19, 18, 0.1) for m in runner.MODES},
    )
    with pytest.raises(ValueError, match="calibration arithmetic"):
        runner.load_calibrator(tmp_path, {"artifact_id": "lock"})


@pytest.mark.parametrize("mutation", [None, "control", "future", "parameter"])
def test_prefix_seal_binds_controls_parameters_and_exact_member_budget(
    tmp_path, mutation
):
    spec = runner.task("calibration-prefix", 0)
    directory = tmp_path / spec["name"]
    directory.mkdir()
    bank = np.zeros((8, 3, 6), dtype=np.float64)
    lock = {"artifact_id": "lock", "controls": bank.tolist()}
    arrays = {
        name: np.zeros((300, 8, 12, 3) if name.startswith("rod_") else (300, 8, 3))
        for name in runner.TRACE_NAMES
    }
    arrays["controls"] = bank.copy()
    if mutation == "control":
        arrays["controls"][0, 0, 0] = 0.1
    if mutation == "future":
        arrays["future_reward"] = np.zeros(8)
    realization = {
        "bending": [[w["bending_E"] for w in spec["worlds"]]],
        "stretching": [[w["stretching_K"] for w in spec["worlds"]]],
        "sphere_initial_position_m": [
            [0.12 + w["x_offset_m"], 0.06, 0.2] for w in spec["worlds"]
        ],
        "cube_initial_position_m": [
            [0.12 + w["x_offset_m"], 0.23, 0.22] for w in spec["worlds"]
        ],
    }
    if mutation == "parameter":
        realization["bending"][0][0] += 1
    claim = runner.write_record(
        directory / "claim.json", {"lock_id": "lock", "task": spec}
    )
    bundle = runner.write_native_bundle(directory, arrays)
    runner.write_record(
        directory / "seal.json",
        {
            "lock_id": "lock",
            "task": spec,
            "claim_id": claim["artifact_id"],
            "bundle": bundle,
            "native": {
                "native_steps": 300,
                "future_simulated": False,
                "reward_scored": False,
                "world_realization": realization,
            },
        },
    )
    if mutation is None:
        _, loaded = runner.load_task(tmp_path, lock, spec)
        assert loaded["rod_pos_m"].shape == (300, 8, 12, 3)
    else:
        with pytest.raises(ValueError):
            runner.load_task(tmp_path, lock, spec)


def test_qualification_success_boolean_cannot_override_failed_arithmetic(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "read_record", lambda path: {"passed": True})
    monkeypatch.setattr(runner, "qualification", lambda *args: {"passed": False})
    with pytest.raises(ValueError, match="qualification gate failed"):
        runner.require_qualification(tmp_path, {}, [])
