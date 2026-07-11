"""End-to-end runner and artifact writers for the counterfactual benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from causal4d.baselines import fit_baselines
from causal4d.benchmark import (
    CounterfactualBenchmarkConfig,
    build_protocol,
    generate_episodes,
    make_parameter_grid,
    protocol_manifest,
)
from causal4d.metrics import (
    aggregate_interventions,
    aggregate_parameters,
    intervention_metrics,
    parameter_recovery_rows,
    posterior_ambiguity,
)


def run_counterfactual_benchmark(
    *,
    seeds: Sequence[int],
    config: CounterfactualBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Fit on repeated actions and evaluate one untouched intervention per object."""

    cfg = config or CounterfactualBenchmarkConfig()
    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_seeds:
        raise ValueError("at least one seed is required")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be unique")

    protocols = build_protocol(cfg)
    intervention_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []

    for seed in normalized_seeds:
        for object_index, protocol in enumerate(protocols):
            object_seed = seed * 10_000 + object_index * 101
            training, validation, held_out = generate_episodes(
                protocol,
                cfg,
                seed=object_seed,
            )
            particles = make_parameter_grid(protocol.graph_object, cfg)
            baselines = fit_baselines(training, validation, particles, cfg)
            ambiguity = posterior_ambiguity(baselines.physics.posterior)
            fit_rows.append(
                {
                    "seed": seed,
                    "object": protocol.graph_object.name,
                    "training_episode_count": len(training),
                    "parameter_particle_count": int(particles.shape[0]),
                    "hybrid_residual_scale": baselines.hybrid.residual_scale,
                    "hybrid_validation_rmse_m": baselines.hybrid.validation_rmse_m,
                    **ambiguity,
                }
            )
            for row in parameter_recovery_rows(
                baselines.physics.posterior,
                protocol.graph_object.true_parameters,
                confidence_level=cfg.confidence_level,
            ):
                parameter_rows.append(
                    {
                        "seed": seed,
                        "object": protocol.graph_object.name,
                        **row,
                    }
                )

            for episode in held_out:
                for prediction in baselines.predict_all(episode):
                    intervention_rows.append(
                        {
                            "seed": seed,
                            "object": protocol.graph_object.name,
                            "action": episode.action.action_id,
                            "world_condition": episode.condition.name,
                            "method": prediction.method,
                            **intervention_metrics(
                                prediction,
                                episode.truth,
                                confidence_level=cfg.confidence_level,
                                gross_failure_threshold_m=cfg.gross_failure_threshold_m,
                            ),
                        }
                    )

    return {
        "schema_version": 1,
        "benchmark": "causal4d-controlled-counterfactual-v1",
        "config": cfg.as_dict(),
        "seeds": normalized_seeds,
        "protocol": protocol_manifest(protocols, cfg),
        "interventions": intervention_rows,
        "parameter_recovery": parameter_rows,
        "fit_diagnostics": fit_rows,
        "aggregate": {
            "interventions": aggregate_interventions(intervention_rows),
            "parameter_recovery": aggregate_parameters(parameter_rows),
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty artifact: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_benchmark_artifacts(
    result: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Write a deterministic, checksummed benchmark result bundle."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    protocol_path = output / "protocol.json"
    intervention_path = output / "interventions.csv"
    parameter_path = output / "parameter_recovery.csv"
    fit_path = output / "fit_diagnostics.csv"
    manifest_path = output / "manifest.json"

    _write_json(
        summary_path,
        {
            "schema_version": result["schema_version"],
            "benchmark": result["benchmark"],
            "config": result["config"],
            "seeds": result["seeds"],
            "aggregate": result["aggregate"],
        },
    )
    _write_json(protocol_path, result["protocol"])
    _write_csv(intervention_path, result["interventions"])
    _write_csv(parameter_path, result["parameter_recovery"])
    _write_csv(fit_path, result["fit_diagnostics"])

    artifacts = [
        summary_path,
        protocol_path,
        intervention_path,
        parameter_path,
        fit_path,
    ]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "benchmark": result["benchmark"],
            "artifacts": {
                path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in artifacts
            },
        },
    )
    return {
        "summary": str(summary_path),
        "protocol": str(protocol_path),
        "interventions": str(intervention_path),
        "parameter_recovery": str(parameter_path),
        "fit_diagnostics": str(fit_path),
        "manifest": str(manifest_path),
    }
