import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.cli.counterfactual_benchmark import _parse_seeds
from causal4d.evaluation import (
    run_counterfactual_benchmark,
    write_benchmark_artifacts,
)


def _config() -> CounterfactualBenchmarkConfig:
    return CounterfactualBenchmarkConfig(
        frame_count=18,
        training_repeats=1,
        parameter_grid_count=3,
        fit_frame_stride=3,
    )


def test_evaluation_covers_all_objects_worlds_methods_and_metric_families() -> None:
    result = run_counterfactual_benchmark(seeds=[0, 1], config=_config())

    assert len(result["interventions"]) == 2 * 3 * 2 * 3
    assert len(result["parameter_recovery"]) == 2 * 3 * 3
    assert len(result["fit_diagnostics"]) == 2 * 3
    assert len(result["aggregate"]["interventions"]) == 2 * 3
    assert {row["method"] for row in result["interventions"]} == {
        "generative_only",
        "physics_only",
        "hybrid",
    }
    assert {row["world_condition"] for row in result["interventions"]} == {
        "matched_contact",
        "shifted_contact",
    }
    for row in result["interventions"]:
        for key in (
            "trajectory_rmse_m",
            "relative_intervention_rmse",
            "fde_m",
            "coverage",
            "mean_interval_width_m",
            "nees",
        ):
            assert np.isfinite(row[key])
    for row in result["parameter_recovery"]:
        assert np.isfinite(row["absolute_error"])
        assert np.isfinite(row["crps"])
        assert 0.0 <= row["posterior_normalized_entropy"] <= 1.0


def test_artifact_bundle_is_deterministic_and_checksummed(tmp_path: Path) -> None:
    result = run_counterfactual_benchmark(seeds=[4], config=_config())
    first_paths = write_benchmark_artifacts(result, tmp_path / "first")
    second_paths = write_benchmark_artifacts(result, tmp_path / "second")

    assert set(first_paths) == {
        "summary",
        "protocol",
        "interventions",
        "parameter_recovery",
        "fit_diagnostics",
        "manifest",
    }
    for name in first_paths:
        first = Path(first_paths[name])
        second = Path(second_paths[name])
        assert first.read_bytes() == second.read_bytes()

    manifest = json.loads(Path(first_paths["manifest"]).read_text(encoding="utf-8"))
    for filename, metadata in manifest["artifacts"].items():
        payload = (tmp_path / "first" / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
        assert len(payload) == metadata["bytes"]


def test_seed_parser_and_runner_reject_invalid_seed_sets() -> None:
    assert _parse_seeds("1:5:2") == [1, 3]
    assert _parse_seeds("2,4") == [2, 4]
    with pytest.raises(ValueError, match="at least one seed"):
        run_counterfactual_benchmark(seeds=[], config=_config())
    with pytest.raises(ValueError, match="unique"):
        run_counterfactual_benchmark(seeds=[1, 1], config=_config())
