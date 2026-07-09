import csv
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.synthetic_benchmark import (
    METHODS,
    SyntheticBenchmarkConfig,
    _temporally_smoothed_bias_probability,
    make_action,
    parameter_grid,
    run_synthetic_benchmark,
    simulate_parameter_particles,
    write_benchmark_csv,
    write_benchmark_json,
    write_reliability_csv,
)


def _small_config() -> SyntheticBenchmarkConfig:
    return SyntheticBenchmarkConfig(
        node_count=4,
        step_count=30,
        train_step_count=20,
        stiffness_count=5,
        damping_count=5,
        control_scale_count=5,
    )


def test_fixed_graph_simulation_is_finite_and_parameter_dependent() -> None:
    config = _small_config()
    particles = np.array([[6.0, 0.45, 1.0], [8.0, 0.45, 1.0]])

    trajectories = simulate_parameter_particles(
        particles,
        make_action(config, "dynamic"),
        config,
    )

    assert trajectories.shape == (2, config.step_count, config.node_count)
    assert np.all(np.isfinite(trajectories))
    assert not np.allclose(trajectories[0], trajectories[1])


def test_bias_gate_rejects_isolated_cue_and_accepts_persistent_cue() -> None:
    raw = np.zeros((30, 3), dtype=float)
    raw[10, 0] = 1.0
    raw[8:22, 1] = 1.0

    probability = _temporally_smoothed_bias_probability(
        raw.reshape(-1),
        step_count=30,
        node_count=3,
    ).reshape(30, 3)

    assert probability[:, 0].max() < 0.05
    assert probability[:, 1].max() > 0.5


def test_correlated_benchmark_runs_all_required_baselines() -> None:
    config = _small_config()

    result = run_synthetic_benchmark(
        seeds=[0],
        conditions=["correlated"],
        action_modes=["dynamic"],
        config=config,
    )

    assert result["parameter_grid_size"] == parameter_grid(config).shape[0]
    assert set(result["runs"][0]["methods"]) == set(METHODS)
    assert len(result["aggregate"]) == len(METHODS)
    assert len(result["reliability_aggregate"]) == 3
    assert result["runs"][0]["corruption_counts"]["drift"] > 0
    assert np.isfinite(
        result["runs"][0]["reliability"]["markov_posterior"]["brier_score"]
    )
    for method in METHODS:
        assert np.isfinite(result["runs"][0]["methods"][method]["state"]["future_rmse"])


def test_benchmark_writes_json_and_aggregate_csv(tmp_path: Path) -> None:
    result = run_synthetic_benchmark(
        seeds=[1],
        conditions=["clean"],
        action_modes=["quasi_static"],
        config=_small_config(),
    )
    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "aggregate.csv"
    reliability_path = tmp_path / "reliability.csv"

    write_benchmark_json(result, json_path)
    write_benchmark_csv(result, csv_path)
    write_reliability_csv(result, reliability_path)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with reliability_path.open("r", encoding="utf-8", newline="") as handle:
        reliability_rows = list(csv.DictReader(handle))
    assert loaded["schema_version"] == 2
    assert len(rows) == len(METHODS)
    assert "state_future_rmse_mean" in rows[0]
    assert len(reliability_rows) == 3
    assert "brier_score_mean" in reliability_rows[0]
