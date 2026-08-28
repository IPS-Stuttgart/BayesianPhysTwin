from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.dlolab_slingshot_probe import (
    full_task,
    material_information,
    prefix_task,
    probe_controls,
    protocol,
    select_probe,
    source_information_value,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slingshot_probe_runner", ROOT / "scripts/remote/run_dlolab_slingshot_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def commands():
    value = np.zeros((8, 3, 6), dtype=np.float64)
    value[:, 0, :3] = [0, -0.03, 0.02]
    value[:, 1, :3] = [0.06, -0.07, 0]
    value[:, 2, :3] = [0, -0.08, 0]
    value[:, :, 3:] = 0.1
    value[1, 1, 0] = 0.05
    return value


@pytest.mark.parametrize("index,fraction", [(0, 0.25), (1, 0.5)])
def test_frontload_preserves_command_sum_rotation_future_choices_and_source(
    index, fraction
):
    old = commands()
    before = old.tobytes()
    new = probe_controls(old, index)
    assert old.tobytes() == before
    np.testing.assert_allclose(
        new[:, :2, :3].sum(axis=1), old[:, :2, :3].sum(axis=1), atol=1e-15
    )
    np.testing.assert_array_equal(new[:, :, 3:], old[:, :, 3:])
    np.testing.assert_array_equal(new[:, 2], old[:, 2])
    np.testing.assert_array_equal(new[5], new[7])
    np.testing.assert_allclose(
        new[:, 0, :3] - old[:, 0, :3], np.broadcast_to(fraction * old[5, 1, :3], (8, 3))
    )
    assert np.max(np.linalg.norm(new[:, :, :3], axis=-1)) <= 0.1


@pytest.mark.parametrize("index", [-1, 2, True, 0.0])
def test_unregistered_probe_refused(index):
    with pytest.raises(ValueError, match="unregistered"):
        probe_controls(commands(), index)


def test_invalid_shared_prefix_limits_and_duplicate_fail():
    for mutation in ("prefix", "duplicate", "limit", "nan", "dtype"):
        value = commands()
        if mutation == "prefix":
            value[0, 0, 0] += 0.01
        elif mutation == "duplicate":
            value[7, 2, 0] += 0.01
        elif mutation == "limit":
            value[:, 0, 0] = 0.2
        elif mutation == "nan":
            value[:, 0, 0] = np.nan
        else:
            value = value.astype(np.float32)
        with pytest.raises(ValueError):
            probe_controls(value, 0)


def test_complete_nine_world_denominator_padding_not_extra_cases():
    first, last = prefix_task(0, 0), prefix_task(0, 1)
    assert first["world_indices"] + last["world_indices"] == list(range(9, 18))
    assert last["worlds"] == [last["worlds"][0]] * 8
    assert all(w["x_offset_m"] == 0 for w in first["worlds"] + last["worlds"])
    assert full_task(1, 26)["world_indices"] == [26]
    for probe, index in ((False, 0), (2, 0), (0, 2)):
        with pytest.raises(ValueError):
            prefix_task(probe, index)
    for probe, index in ((False, 0), (2, 0), (0, 27)):
        with pytest.raises(ValueError):
            full_task(probe, index)


def strong_history():
    value = np.zeros((9, 3, 4, 3))
    value[5, 1, :, 0] = 0.02
    value[3, 1, :, 0] = -0.02
    return value


def test_material_sensitivity_and_gate_are_not_posterior_performance():
    strong, weak = strong_history(), np.zeros((9, 3, 4, 3))
    info = material_information(strong)
    assert info["whitened_stretching_secant_norm"] > 1
    assert info["whitened_bending_secant_norm"] == 0
    result = select_probe([weak, strong], [True, True])
    assert result["selected_probe"] == 1 and result["source_bank_authorized"]
    assert result["method_evaluation_authorized"] is False
    assert select_probe([strong, strong], [True, True])["selected_probe"] == 0
    assert select_probe([weak, weak], [True, True])["source_bank_authorized"] is False
    assert (
        select_probe([strong, strong], [True, False])["source_bank_authorized"] is False
    )
    with pytest.raises(ValueError):
        select_probe([strong], [True])
    with pytest.raises(ValueError):
        material_information(strong[:8])


def test_uninformative_source_cannot_invent_information_value():
    prefix = np.zeros((27, 3, 4, 3))
    reward = np.zeros((27, 7))
    reward[:13, 0] = 1
    reward[13:, 1] = 1
    result = source_information_value(prefix, reward, np.full(27, 1 / 27))
    assert result["best_blind_action"] == 1
    assert abs(result["information_gain"]) < 1e-12
    assert result["perfect_information_reward"] > result["posterior_mean_reward"]
    assert result["integration_only_not_out_of_sample"]


def test_root_gate_precedes_artifact_access(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "read_record", lambda *args: pytest.fail("artifact accessed")
    )
    with pytest.raises(ValueError, match="write-once"):
        runner.validate(tmp_path)


def test_future_worker_cannot_bypass_prefix_arithmetic(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "validate", lambda *args: ({}, {}, {}))
    monkeypatch.setattr(
        runner,
        "require_prefix",
        lambda *args: (_ for _ in ()).throw(ValueError("prefix gate")),
    )
    monkeypatch.setattr(
        runner,
        "run_registered_worlds",
        lambda *args, **kwargs: pytest.fail("native future entered"),
    )
    with pytest.raises(ValueError, match="prefix gate"):
        runner.worker(tmp_path, "source", 0, 0)
    assert list(tmp_path.iterdir()) == []


def test_stored_success_boolean_does_not_authorize_failed_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "read_record", lambda *args: {"source_bank_authorized": True}
    )
    monkeypatch.setattr(
        runner, "prefix_result", lambda *args: {"source_bank_authorized": False}
    )
    with pytest.raises(ValueError, match="rederived"):
        runner.require_prefix(tmp_path, {}, {})


def test_protocol_is_source_only_without_extra_time_or_observations():
    value = protocol()
    assert value["macro_count"] == 3 and value["prefix_steps"] == 300
    assert value["observation_frames"] == [139, 219, 299]
    assert value["full_source_world_count_if_prefix_passes"] == 27
    assert not any(
        value[k]
        for k in (
            "gpu_work",
            "new_recordings",
            "retry_authorized",
            "method_evaluation_authorized",
            "protected_data_read",
            "calibration_or_evaluation_worlds_read",
        )
    )


def test_exact_committed_source_controls_admit_both_registered_probes():
    source_lock = json.loads(
        (
            ROOT / "results/source/dlolab_slingshot_belief_control_v1/lock.json"
        ).read_text()
    )
    original = np.asarray(source_lock["controls"], dtype=np.float64)
    for i in range(2):
        changed = probe_controls(original, i)
        assert np.max(np.linalg.norm(changed[:, :, :3], axis=-1)) <= 0.1 + 1e-12


def test_source_driver_uses_exact_controls_for_native_qa_and_retains_denominator(
    tmp_path, monkeypatch
):
    output = tmp_path / "new-screen"
    original = commands()
    old = {
        "controls": original.tolist(),
        "source_sha256": {},
        "protocol": {"prior_weights": [1 / 27] * 27},
    }
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "clean_revision", lambda *args: "test-commit")
    monkeypatch.setattr(runner, "file_digest", lambda *args: "test-digest")
    monkeypatch.setattr(runner, "source", lambda: (old, {"reward": np.zeros((27, 7))}))
    monkeypatch.setattr(
        runner,
        "prefix_result",
        lambda *args: {"source_bank_authorized": True, "selected_probe": 0},
    )
    launched = []
    monkeypatch.setattr(runner, "launch", lambda *args: launched.append(args[2:]))
    data = {
        "rod_pos_m": np.zeros((900, 8, 12, 3)),
        "sphere_pos_m": np.zeros((900, 8, 3)),
        "cube_pos_m": np.zeros((900, 8, 3)),
        "gripper_pos_m": np.zeros((900, 8, 3)),
    }

    def load(output, lock, old, spec):
        value = {k: v[:300] if spec["prefix_only"] else v for k, v in data.items()}
        return {"artifact_id": spec["name"], "native": {}}, value

    checked = []

    def qa(value, native, expected_controls, prefix):
        np.testing.assert_array_equal(expected_controls, probe_controls(original, 0))
        checked.append(prefix is not None)
        return {"qa_passed": True, "metrics": [{"native_reward": 0.0}] * 8}

    monkeypatch.setattr(runner, "load_task", load)
    monkeypatch.setattr(runner, "native_qa", qa)
    monkeypatch.setattr(
        runner,
        "source_information_value",
        lambda *args: {
            "information_gain": 0.0,
            "posterior_gain_over_map": 0.0,
            "best_blind_reward": 0.0,
            "posterior_mean_reward": 0.0,
        },
    )
    runner.run(output)
    result = runner.read_record(output / "result.json")
    assert len(launched) == 31 and len(checked) == 27 and sum(checked) == 9
    assert result["source_worlds_generated"] == 27
    assert result["source_gate_passed"] is False
    with pytest.raises(FileExistsError):
        runner.run(output)
    assert len(launched) == 31
