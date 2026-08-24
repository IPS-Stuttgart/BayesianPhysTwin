from __future__ import annotations

from pathlib import Path

import pytest

from bayesian_phystwin.synthetic_benchmark import SyntheticBenchmarkConfig
from bayesian_phystwin_experiments.synthetic_benchmark_sbc_v1 import (
    SyntheticBenchmarkSBCConfigV1,
    run_synthetic_benchmark_sbc_v1,
    write_synthetic_benchmark_sbc_v1,
)


def _small_model() -> SyntheticBenchmarkConfig:
    return SyntheticBenchmarkConfig(
        node_count=4,
        step_count=30,
        train_step_count=20,
        stiffness_count=5,
        damping_count=5,
        control_scale_count=5,
    )


def test_matched_posterior_beats_deliberate_dispersion_controls() -> None:
    result = run_synthetic_benchmark_sbc_v1(
        config=SyntheticBenchmarkSBCConfigV1(
            replicate_count=600,
            seed=20260824,
            bin_count=10,
        ),
        benchmark_config=_small_model(),
    )

    assert result["parameter_grid_size"] == 125
    assert result["action_mode_counts"] == {"dynamic": 300, "quasi_static": 300}
    separation = result["normative_control_separation"]
    assert separation["matched_has_smallest_mean_ks"] is True
    assert separation["matched_has_smallest_90_coverage_error"] is True
    matched = result["diagnostics"]["matched_likelihood"]
    under = result["diagnostics"]["underdispersed_0.5x"]
    assert matched["mean_ks_distance"] < under["mean_ks_distance"]
    assert (
        matched["mean_absolute_90_coverage_error"]
        < under["mean_absolute_90_coverage_error"]
    )


def test_result_is_deterministic_and_content_addressed() -> None:
    config = SyntheticBenchmarkSBCConfigV1(replicate_count=80, seed=7)
    first = run_synthetic_benchmark_sbc_v1(
        config=config,
        benchmark_config=_small_model(),
    )
    second = run_synthetic_benchmark_sbc_v1(
        config=config,
        benchmark_config=_small_model(),
    )
    assert first["result_id"] == second["result_id"]
    assert first == second


def test_atomic_writer_refuses_silent_replacement(tmp_path: Path) -> None:
    result = run_synthetic_benchmark_sbc_v1(
        config=SyntheticBenchmarkSBCConfigV1(replicate_count=40, seed=5),
        benchmark_config=_small_model(),
    )
    output = tmp_path / "sbc.json"
    write_synthetic_benchmark_sbc_v1(result, output)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_synthetic_benchmark_sbc_v1(result, output)
    write_synthetic_benchmark_sbc_v1(result, output, overwrite=True)
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_config_requires_matched_arm_and_independent_replicates() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        SyntheticBenchmarkSBCConfigV1(replicate_count=10)
    with pytest.raises(ValueError, match="include the matched"):
        SyntheticBenchmarkSBCConfigV1(likelihood_scale_multipliers=(0.5, 2.0))
