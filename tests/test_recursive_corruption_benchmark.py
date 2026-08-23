from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from bayesian_phystwin.recursive_corruption_benchmark import (
    CONDITIONS,
    RecursiveCorruptionBenchmarkConfig,
    generate_corrupted_sequence,
    run_recursive_corruption_benchmark,
    write_recursive_corruption_csv,
    write_recursive_corruption_json,
)

ROOT = Path(__file__).resolve().parents[1]
SELECTIVITY_SCRIPT = (
    ROOT / "scripts" / "science" / "analyze_recursive_corruption_selectivity_v1.py"
)


def _load_selectivity_script() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "bpt_recursive_corruption_selectivity_test_module",
        SELECTIVITY_SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


selectivity = _load_selectivity_script()


def _small_config() -> RecursiveCorruptionBenchmarkConfig:
    return RecursiveCorruptionBenchmarkConfig(
        step_count=96,
        corruption_start=30,
        corruption_length=16,
        recovery_window=24,
    )


def test_generated_sequences_are_deterministic_and_immutable() -> None:
    config = _small_config()
    first = generate_corrupted_sequence("identity_switch", seed=7, config=config)
    second = generate_corrupted_sequence("identity_switch", seed=7, config=config)
    assert np.array_equal(first.true_position_m, second.true_position_m)
    assert np.array_equal(first.observation_m, second.observation_m)
    assert not first.true_position_m.flags.writeable
    with pytest.raises(ValueError):
        first.true_position_m[0] = 1.0


def test_guard_rejects_stale_observations_and_preserves_exact_fallback() -> None:
    result = run_recursive_corruption_benchmark(
        seeds=[3],
        conditions=["delayed_observation"],
        config=_small_config(),
    )
    guarded = next(
        record
        for record in result["records"]
        if record["method"] == "guarded_recursive"
    )
    unguarded = next(
        record
        for record in result["records"]
        if record["method"] == "recursive_gaussian"
    )
    assert guarded["fallback_reasons"]["stale-observation"] == 16
    assert guarded["exact_fallback_violation_count"] == 0
    assert guarded["corruption_rmse_m"] < unguarded["corruption_rmse_m"]


def test_guard_reduces_materially_harmful_updates_under_corruption() -> None:
    result = run_recursive_corruption_benchmark(
        seeds=range(5),
        conditions=[
            "outlier_burst",
            "coherent_drift",
            "identity_switch",
            "delayed_observation",
        ],
        config=_small_config(),
    )
    aggregate = result["aggregate"]["all_corruptions"]
    guarded = aggregate["guarded_recursive"]
    unguarded = aggregate["recursive_gaussian"]
    last = aggregate["last_residual"]
    assert (
        guarded["materially_harmful_accepted_update_count_mean"]
        < unguarded["materially_harmful_accepted_update_count_mean"]
    )
    assert guarded["rmse_m_mean"] < unguarded["rmse_m_mean"]
    assert guarded["rmse_m_mean"] < last["rmse_m_mean"]
    assert result["summary"]["guarded_exact_fallback_violation_count"] == 0


def test_clean_only_result_has_bounded_empty_corruption_summary() -> None:
    result = run_recursive_corruption_benchmark(
        seeds=[0],
        conditions=["clean"],
        config=_small_config(),
    )
    summary = result["summary"]
    assert summary["corrupted_sequence_count_per_method"] == 0
    assert summary["guarded_vs_last_residual_rmse_change_percent"] is None
    json.dumps(result, allow_nan=False)


def test_result_is_finite_json_and_writers_are_deterministic(
    tmp_path: Path,
) -> None:
    result = run_recursive_corruption_benchmark(
        seeds=[0, 1],
        conditions=["clean", "missing_burst"],
        config=_small_config(),
    )
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    assert "NaN" not in encoded
    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"
    write_recursive_corruption_json(result, json_path)
    write_recursive_corruption_csv(result, csv_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == result
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == (
        1 + len(result["records"])
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_count": 10},
        {"residual_persistence": 1.0},
        {"minimum_reliability": 0.0},
        {"corruption_start": 130, "corruption_length": 20},
        {"delay_steps": 60},
    ],
)
def test_invalid_configurations_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RecursiveCorruptionBenchmarkConfig(**kwargs)


def test_all_declared_conditions_execute() -> None:
    result = run_recursive_corruption_benchmark(
        seeds=[0],
        conditions=CONDITIONS,
        config=_small_config(),
    )
    assert result["conditions"] == list(CONDITIONS)
    assert len(result["records"]) == len(CONDITIONS) * 5


def test_selectivity_curve_is_deterministic_and_never_selects_a_threshold() -> None:
    arguments = {
        "seeds": [0, 1],
        "conditions": ["outlier_burst", "missing_burst"],
        "maximum_nis_grid": [9.0, 1.0, 4.0],
        "config": _small_config(),
    }
    first = selectivity.analyze_recursive_corruption_selectivity(**arguments)
    second = selectivity.analyze_recursive_corruption_selectivity(**arguments)

    assert first == second
    assert first["maximum_nis_grid"] == [1.0, 4.0, 9.0]
    assert first["selection_authorized"] is False
    assert len(first["report_id"]) == 64
    assert json.loads(json.dumps(first, allow_nan=False)) == first

    curve = first["curve"]
    assert isinstance(curve, list)
    assert len(curve) == 3
    for point in curve:
        assert point["exact_fallback_violation_count"] == 0
        total_fraction = point["acceptance_fraction"] + point["fallback_fraction"]
        assert total_fraction == pytest.approx(1.0)


def test_selectivity_curve_rejects_target_like_or_ambiguous_sweeps() -> None:
    with pytest.raises(ValueError, match="corrupted conditions only"):
        selectivity.analyze_recursive_corruption_selectivity(
            seeds=[0],
            conditions=["clean"],
            maximum_nis_grid=[1.0],
            config=_small_config(),
        )
    with pytest.raises(ValueError, match="unique"):
        selectivity.analyze_recursive_corruption_selectivity(
            seeds=[0],
            conditions=["outlier_burst"],
            maximum_nis_grid=[1.0, 1.0],
            config=_small_config(),
        )


def test_selectivity_report_writer_is_atomic_and_non_overwriting(
    tmp_path: Path,
) -> None:
    report = selectivity.analyze_recursive_corruption_selectivity(
        seeds=[0],
        conditions=["identity_switch"],
        maximum_nis_grid=[1.0, 9.0],
        config=_small_config(),
    )
    destination = tmp_path / "selectivity.json"
    selectivity.write_selectivity_report(report, destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        selectivity.write_selectivity_report(report, destination)
    selectivity.write_selectivity_report(report, destination, force=True)
    assert json.loads(destination.read_text(encoding="utf-8")) == report
