import hashlib
import json
from pathlib import Path

from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.cli.latent_contact_benchmark import _parse_seeds
from causal4d.contact_evaluation import (
    run_latent_contact_benchmark,
    write_latent_contact_artifacts,
)
from causal4d.contact_inference import LatentContactConfig


def _configs() -> tuple[CounterfactualBenchmarkConfig, LatentContactConfig]:
    benchmark = CounterfactualBenchmarkConfig(
        frame_count=18,
        training_repeats=1,
        parameter_grid_count=3,
        fit_frame_stride=3,
    )
    contact = LatentContactConfig(
        parameter_particle_count=3,
        observation_fraction=0.20,
        likelihood_scales_m=(0.002,),
        likelihood_powers=(1.0,),
        dynamic_likelihood_weights=(0.0,),
        posterior_temperatures=(1.0,),
    )
    return benchmark, contact


def test_latent_contact_evaluation_covers_settings_controls_and_topology_folds() -> (
    None
):
    benchmark, contact = _configs()
    result = run_latent_contact_benchmark(
        seeds=[0],
        benchmark_config=benchmark,
        contact_config=contact,
    )

    assert len(result["interventions"]) == 3 * 2 * 2 * 4
    assert len(result["contact_recovery"]) == 3 * 2 * 2
    assert len(result["fold_calibration"]) == 3
    assert {row["setting"] for row in result["interventions"]} == {
        "pre_intervention",
        "online_adaptation",
    }
    assert {row["method"] for row in result["interventions"]} == {
        "nominal_physics",
        "latent_contact",
        "oracle_contact",
        "oracle_contact_theta",
    }
    for row in result["fold_calibration"]:
        sources = set(row["source_objects"].split(";"))
        assert row["held_out_object"] not in sources
        assert row["source_excludes_target"]
        assert row["contact_hypothesis_count"] > 1
    assert len(result["aggregate"]["held_out_topology"]) == 3
    assert {gate["name"] for gate in result["success_gates"]["gates"]} >= {
        "shifted_oracle_gap_closure",
        "maximum_online_coverage_error",
        "shifted_node_accuracy",
        "shifted_gain_coverage",
        "shifted_delay_coverage",
        "held_out_topology_count",
    }


def test_latent_contact_artifacts_are_deterministic_and_checksummed(
    tmp_path: Path,
) -> None:
    benchmark, contact = _configs()
    result = run_latent_contact_benchmark(
        seeds=[1],
        benchmark_config=benchmark,
        contact_config=contact,
    )
    first = write_latent_contact_artifacts(result, tmp_path / "first")
    second = write_latent_contact_artifacts(result, tmp_path / "second")

    assert set(first) == {
        "summary",
        "protocol",
        "interventions",
        "contact_recovery",
        "fold_calibration",
        "success_gates",
        "manifest",
    }
    for name in first:
        assert Path(first[name]).read_bytes() == Path(second[name]).read_bytes()
    manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
    for filename, metadata in manifest["artifacts"].items():
        payload = (tmp_path / "first" / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
        assert len(payload) == metadata["bytes"]


def test_latent_contact_seed_parser() -> None:
    assert _parse_seeds("0:5:2") == [0, 2, 4]
    assert _parse_seeds("3,7") == [3, 7]
