import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.phystwin_confirmation_lock import confirmation_output_lock
from bayesian_phystwin.phystwin_confirmatory import (
    PhysTwinConfirmatoryProtocol,
    _fit_case,
    _cohort_readout,
    _lock_protocol,
    _seal_case_cache,
    _split_for_case,
    _validate_cached_case,
    run_phystwin_confirmatory_benchmark,
)


def test_split_uses_first_three_quarters_of_released_training(tmp_path: Path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "split.json").write_text(
        json.dumps({"frame_len": 85, "train": [0, 59], "test": [59, 85]})
    )

    assert _split_for_case(case, 0.75) == (44, 59, 85)


def test_protocol_lock_is_idempotent_and_rejects_changes(tmp_path: Path):
    specification = {"protocol": {"maximum_residual_m": 0.01}}

    first = _lock_protocol(tmp_path, specification)
    second = _lock_protocol(tmp_path, specification)

    assert first == second
    with pytest.raises(RuntimeError, match="different locked protocol"):
        _lock_protocol(tmp_path, {"protocol": {"maximum_residual_m": 0.03}})


def test_protocol_lock_normalizes_json_tuple_round_trip(tmp_path: Path):
    specification = {"protocol": {"rank_candidates": (1, 2, 4)}}

    first = _lock_protocol(tmp_path, specification)
    second = _lock_protocol(tmp_path, specification)

    assert first == second
    assert second["specification"]["protocol"]["rank_candidates"] == [1, 2, 4]


def test_cohort_readout_aggregates_case_changes() -> None:
    case_results = {
        "a": {
            "accepted_on_validation": True,
            "future_percent_change": {
                "chamfer_distance_m": -10.0,
                "track_error_m": -5.0,
            },
        },
        "b": {
            "accepted_on_validation": False,
            "future_percent_change": {
                "chamfer_distance_m": 2.0,
                "track_error_m": -1.0,
            },
        },
    }

    result = _cohort_readout(("a", "b"), case_results, {"bootstrap": {}})

    assert result["validation_acceptance_count"] == 1
    assert result["improved_case_count"] == {
        "chamfer_distance_m": 1,
        "track_error_m": 2,
    }
    assert result["improved_on_both_count"] == 1


def test_confirmatory_defaults_keep_development_cases_explicit():
    protocol = PhysTwinConfirmatoryProtocol()

    assert protocol.maximum_residual_m == 0.01
    assert protocol.development_cases == (
        "single_lift_sloth",
        "double_lift_sloth",
        "double_stretch_sloth",
    )


def test_confirmatory_rejects_nonpositive_workers(tmp_path: Path):
    with pytest.raises(ValueError, match="workers must be positive"):
        run_phystwin_confirmatory_benchmark(tmp_path, tmp_path / "out", workers=0)


def test_confirmatory_rejects_a_second_output_owner(tmp_path: Path):
    output = tmp_path / "out"

    with confirmation_output_lock(output):
        with pytest.raises(RuntimeError, match="another PhysTwin confirmation"):
            run_phystwin_confirmatory_benchmark(tmp_path, output)


def test_confirmatory_rejects_duplicate_manifest_cases(tmp_path: Path):
    (tmp_path / "evaluation_subset_manifest.json").write_text(
        json.dumps({"selected_cases": ["case_a", "case_a"]})
    )
    with pytest.raises(ValueError, match="duplicate cases"):
        run_phystwin_confirmatory_benchmark(tmp_path, tmp_path / "out")


def test_cached_case_rejects_changed_inputs(tmp_path: Path):
    case = tmp_path / "case"
    case.mkdir()
    inputs = {}
    for name, filename in (
        ("final_data", "final_data.pkl"),
        ("baseline_trajectory", "inference.pkl"),
        ("gt_track_3d", "gt_track_3d.pkl"),
    ):
        path = case / filename
        path.write_bytes(name.encode())
        inputs[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    output = tmp_path / "output"
    output.mkdir()
    (output / "trajectory.pkl").write_bytes(b"trajectory")
    summary = {"schema_version": 1, "config": {"rank": 2}, "inputs": inputs}
    (case / "split.json").write_text(
        json.dumps({"frame_len": 3, "train": [0, 2], "test": [2, 3]})
    )
    _seal_case_cache(summary, case, output)

    _validate_cached_case(summary, {"rank": 2}, case, output, "case")
    (case / "inference.pkl").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="baseline_trajectory"):
        _validate_cached_case(summary, {"rank": 2}, case, output, "case")


def test_cached_case_rejects_changed_outputs(tmp_path: Path):
    case = tmp_path / "case"
    case.mkdir()
    inputs = {}
    for name, filename in (
        ("final_data", "final_data.pkl"),
        ("baseline_trajectory", "inference.pkl"),
        ("gt_track_3d", "gt_track_3d.pkl"),
    ):
        path = case / filename
        path.write_bytes(name.encode())
        inputs[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (case / "split.json").write_text(
        json.dumps({"frame_len": 3, "train": [0, 2], "test": [2, 3]})
    )
    output = tmp_path / "output"
    output.mkdir()
    trajectory = output / "trajectory.pkl"
    trajectory.write_bytes(b"trajectory")
    summary = {"schema_version": 1, "config": {"rank": 2}, "inputs": inputs}
    _seal_case_cache(summary, case, output)
    trajectory.write_bytes(b"modified")

    with pytest.raises(RuntimeError, match="summary, implementation, source, or output"):
        _validate_cached_case(summary, {"rank": 2}, case, output, "case")


def test_cached_case_rejects_changed_summary_body(tmp_path: Path):
    case = tmp_path / "case"
    case.mkdir()
    inputs = {}
    for name, filename in (
        ("final_data", "final_data.pkl"),
        ("baseline_trajectory", "inference.pkl"),
        ("gt_track_3d", "gt_track_3d.pkl"),
    ):
        path = case / filename
        path.write_bytes(name.encode())
        inputs[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (case / "split.json").write_text(
        json.dumps({"frame_len": 3, "train": [0, 2], "test": [2, 3]})
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "trajectory.pkl").write_bytes(b"trajectory")
    summary = {
        "schema_version": 1,
        "config": {"rank": 2},
        "inputs": inputs,
        "selection": {"accepted": True},
    }
    _seal_case_cache(summary, case, output)
    summary["selection"]["accepted"] = False

    with pytest.raises(RuntimeError, match="summary, implementation, source, or output"):
        _validate_cached_case(summary, {"rank": 2}, case, output, "case")


def test_action_fit_rejects_input_changed_while_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root = tmp_path / "data"
    case = data_root / "case_a"
    case.mkdir(parents=True)
    (case / "split.json").write_text(
        json.dumps({"frame_len": 10, "train": [0, 8], "test": [8, 10]})
    )
    for filename in ("final_data.pkl", "inference.pkl", "gt_track_3d.pkl"):
        (case / filename).write_bytes(filename.encode())

    def fake_fit(*args, **kwargs):
        output = Path(args[3])
        output.mkdir(parents=True)
        (output / "trajectory.pkl").write_bytes(b"trajectory")
        (case / "inference.pkl").write_bytes(b"changed during fit")
        return {"inputs": {}}

    monkeypatch.setattr(
        "bayesian_phystwin.phystwin_confirmatory.fit_action_conditioned_residual_dynamics",
        fake_fit,
    )

    with pytest.raises(RuntimeError, match="sources changed while the fit was running"):
        _fit_case(
            (
                data_root,
                tmp_path / "output",
                "case_a",
                PhysTwinConfirmatoryProtocol(),
                False,
            )
        )
