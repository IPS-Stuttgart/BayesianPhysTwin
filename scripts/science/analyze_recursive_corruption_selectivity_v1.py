#!/usr/bin/env python3
"""Build a diagnostic innovation-threshold curve for recursive corruption runs.

The curve deliberately evaluates generated truth and is therefore controlled,
retrospective mechanism analysis. It does not select or authorize a threshold
for a real provider, physical object/session cohort, or deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Final, cast

from bayesian_phystwin.recursive_corruption_benchmark import (
    CONDITIONS,
    RECURSIVE_CORRUPTION_SCHEMA,
    RECURSIVE_CORRUPTION_SCHEMA_VERSION,
    RecursiveCorruptionBenchmarkConfig,
    run_recursive_corruption_benchmark,
)

SCHEMA: Final = "bayesian-phystwin.recursive-corruption-selectivity"
SCHEMA_VERSION: Final = 1
DEFAULT_MAXIMUM_NIS_GRID: Final[tuple[float, ...]] = (
    1.0,
    2.0,
    4.0,
    9.0,
    16.0,
    36.0,
    1_000_000.0,
)
DEFAULT_CONDITIONS: Final[tuple[str, ...]] = tuple(
    condition for condition in CONDITIONS if condition != "clean"
)


def _canonical_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    thresholds: list[float] = []
    for raw in values:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("maximum-NIS thresholds must be finite positive numbers")
        threshold = float(raw)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("maximum-NIS thresholds must be finite positive numbers")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("at least one maximum-NIS threshold is required")
    if len(thresholds) != len(set(thresholds)):
        raise ValueError("maximum-NIS thresholds must be unique")
    return tuple(sorted(thresholds))


def _canonical_conditions(values: Sequence[str]) -> tuple[str, ...]:
    conditions = tuple(values)
    if not conditions:
        raise ValueError("at least one corrupted condition is required")
    if any(type(condition) is not str for condition in conditions):
        raise ValueError("conditions must be literal strings")
    if "clean" in conditions:
        raise ValueError("the selectivity curve requires corrupted conditions only")
    unknown = sorted(set(conditions) - set(DEFAULT_CONDITIONS))
    if unknown:
        raise ValueError(f"unknown corrupted conditions: {unknown}")
    if len(conditions) != len(set(conditions)):
        raise ValueError("conditions must be unique")
    return conditions


def _canonical_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("seeds must be nonnegative integers")
        seeds.append(value)
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    return tuple(seeds)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _finite_metric(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric metric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite numeric metric")
    return result


def _optional_finite_metric(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_metric(value, name=name)


def _relative_change_percent(candidate: float, reference: float) -> float | None:
    if reference == 0.0:
        return None
    return 100.0 * (candidate / reference - 1.0)


def _method_aggregate(
    result: Mapping[str, Any],
    method: str,
) -> Mapping[str, Any]:
    aggregate = _mapping(result.get("aggregate"), name="benchmark aggregate")
    all_corruptions = _mapping(
        aggregate.get("all_corruptions"),
        name="all-corruptions aggregate",
    )
    return _mapping(all_corruptions.get(method), name=f"{method} aggregate")


def _reference_metrics(result: Mapping[str, Any]) -> dict[str, object]:
    references: dict[str, object] = {}
    for method in ("last_residual", "recursive_gaussian"):
        aggregate = _method_aggregate(result, method)
        references[method] = {
            "rmse_m": _finite_metric(
                aggregate.get("rmse_m_mean"),
                name=f"{method} RMSE",
            ),
            "gaussian_nll": _optional_finite_metric(
                aggregate.get("gaussian_nll_mean"),
                name=f"{method} Gaussian NLL",
            ),
            "coverage_90": _optional_finite_metric(
                aggregate.get("coverage_90_mean"),
                name=f"{method} coverage",
            ),
            "mean_full_interval_width_90_m": _optional_finite_metric(
                aggregate.get("mean_full_interval_width_90_m_mean"),
                name=f"{method} interval width",
            ),
        }
    return references


def _canonical_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def analyze_recursive_corruption_selectivity(
    *,
    seeds: Sequence[int] = tuple(range(50)),
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    maximum_nis_grid: Sequence[float] = DEFAULT_MAXIMUM_NIS_GRID,
    config: RecursiveCorruptionBenchmarkConfig | None = None,
) -> dict[str, object]:
    """Evaluate a predeclared innovation-threshold grid without selecting a winner."""

    canonical_seeds = _canonical_seeds(seeds)
    canonical_conditions = _canonical_conditions(conditions)
    thresholds = _canonical_thresholds(maximum_nis_grid)
    base_config = RecursiveCorruptionBenchmarkConfig() if config is None else config
    forecast_count = base_config.step_count - 1

    curve: list[dict[str, object]] = []
    references: dict[str, object] | None = None
    for threshold in thresholds:
        threshold_config = replace(base_config, maximum_nis=threshold)
        benchmark = run_recursive_corruption_benchmark(
            seeds=canonical_seeds,
            conditions=canonical_conditions,
            config=threshold_config,
        )
        current_references = _reference_metrics(benchmark)
        if references is None:
            references = current_references
        elif current_references != references:
            raise ValueError(
                "non-guard reference metrics changed across the maximum-NIS grid"
            )

        guarded = _method_aggregate(benchmark, "guarded_recursive")
        unguarded = _method_aggregate(benchmark, "recursive_gaussian")
        last = _method_aggregate(benchmark, "last_residual")
        sequence_count = int(
            _finite_metric(guarded.get("sequence_count"), name="sequence count")
        )
        accepted_mean = _finite_metric(
            guarded.get("accepted_update_count_mean"),
            name="accepted-update mean",
        )
        fallback_mean = _finite_metric(
            guarded.get("fallback_count_mean"),
            name="fallback mean",
        )
        if not math.isclose(
            accepted_mean + fallback_mean,
            float(forecast_count),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("accepted and fallback counts do not cover every forecast")
        harmful_mean = _finite_metric(
            guarded.get("materially_harmful_accepted_update_count_mean"),
            name="harmful accepted-update mean",
        )
        guarded_rmse = _finite_metric(guarded.get("rmse_m_mean"), name="guarded RMSE")
        unguarded_rmse = _finite_metric(
            unguarded.get("rmse_m_mean"),
            name="unguarded RMSE",
        )
        last_rmse = _finite_metric(last.get("rmse_m_mean"), name="last-residual RMSE")
        guarded_nll = _finite_metric(
            guarded.get("gaussian_nll_mean"),
            name="guarded Gaussian NLL",
        )
        unguarded_nll = _finite_metric(
            unguarded.get("gaussian_nll_mean"),
            name="unguarded Gaussian NLL",
        )
        records = benchmark.get("records")
        if not isinstance(records, list):
            raise ValueError("benchmark records must be a list")
        guarded_records = [
            record
            for record in records
            if isinstance(record, Mapping)
            and record.get("method") == "guarded_recursive"
        ]
        if len(guarded_records) != sequence_count:
            raise ValueError("guarded record count does not match the aggregate")
        exact_fallback_violations = sum(
            int(record["exact_fallback_violation_count"])
            for record in guarded_records
        )

        curve.append(
            {
                "maximum_nis": threshold,
                "sequence_count": sequence_count,
                "forecast_count_per_sequence": forecast_count,
                "acceptance_fraction": accepted_mean / forecast_count,
                "fallback_fraction": fallback_mean / forecast_count,
                "materially_harmful_fraction_of_accepted_updates": (
                    harmful_mean / accepted_mean if accepted_mean > 0.0 else None
                ),
                "deployed_rmse_m": guarded_rmse,
                "rmse_change_vs_last_residual_percent": _relative_change_percent(
                    guarded_rmse,
                    last_rmse,
                ),
                "rmse_change_vs_recursive_gaussian_percent": _relative_change_percent(
                    guarded_rmse,
                    unguarded_rmse,
                ),
                "gaussian_nll": guarded_nll,
                "gaussian_nll_regret_vs_recursive_per_sequence": (
                    guarded_nll - unguarded_nll
                )
                * forecast_count,
                "coverage_90": _finite_metric(
                    guarded.get("coverage_90_mean"),
                    name="guarded coverage",
                ),
                "mean_full_interval_width_90_m": _finite_metric(
                    guarded.get("mean_full_interval_width_90_m_mean"),
                    name="guarded interval width",
                ),
                "exact_fallback_violation_count": exact_fallback_violations,
                "fallback_reason_totals": dict(
                    _mapping(
                        guarded.get("fallback_reason_totals"),
                        name="fallback reasons",
                    )
                ),
            }
        )

    assert references is not None
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "benchmark_schema": RECURSIVE_CORRUPTION_SCHEMA,
        "benchmark_schema_version": RECURSIVE_CORRUPTION_SCHEMA_VERSION,
        "analysis_status": "controlled-retrospective-diagnostic",
        "selection_authorized": False,
        "seeds": list(canonical_seeds),
        "conditions": list(canonical_conditions),
        "maximum_nis_grid": list(thresholds),
        "base_config": asdict(base_config),
        "reference_methods": references,
        "curve": curve,
        "interpretation": (
            "Acceptance and deployed risk are evaluated on generated truth to diagnose "
            "the innovation guard. No threshold is selected or promoted."
        ),
        "scientific_boundary": (
            "Controlled mechanism analysis only; no real-provider competence, "
            "physical-object transfer, calibration, intervention benefit, deployment "
            "safety, or state-of-the-art claim is authorized."
        ),
    }
    payload["report_id"] = _canonical_id(payload)
    json.dumps(payload, allow_nan=False, sort_keys=True)
    return payload


def write_selectivity_report(
    report: Mapping[str, object],
    path: str | Path,
    *,
    force: bool = False,
) -> None:
    """Publish a finite report atomically and refuse overwrite by default."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, destination)
            temporary = None
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise FileExistsError(f"refusing to overwrite {destination}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_seeds(value: str) -> tuple[int, ...]:
    if ":" in value:
        parts = value.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("seed ranges must use START:STOP")
        try:
            start, stop = (int(part) for part in parts)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "seed ranges must contain integers"
            ) from error
        if start < 0 or stop <= start:
            raise argparse.ArgumentTypeError("seed ranges require 0 <= START < STOP")
        return tuple(range(start, stop))
    try:
        seeds = tuple(int(part) for part in value.split(",") if part)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    try:
        return _canonical_seeds(seeds)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parse_conditions(value: str) -> tuple[str, ...]:
    try:
        return _canonical_conditions(tuple(part for part in value.split(",") if part))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        raw = tuple(float(part) for part in value.split(",") if part)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "maximum-NIS grid must contain comma-separated numbers"
        ) from error
    try:
        return _canonical_thresholds(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("0:50"))
    parser.add_argument(
        "--conditions",
        type=_parse_conditions,
        default=DEFAULT_CONDITIONS,
    )
    parser.add_argument(
        "--maximum-nis-grid",
        type=_parse_thresholds,
        default=DEFAULT_MAXIMUM_NIS_GRID,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = analyze_recursive_corruption_selectivity(
        seeds=arguments.seeds,
        conditions=arguments.conditions,
        maximum_nis_grid=arguments.maximum_nis_grid,
    )
    write_selectivity_report(report, arguments.output, force=arguments.force)
    print(
        json.dumps(
            {
                "report_id": report["report_id"],
                "threshold_count": len(report["curve"]),
                "selection_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
