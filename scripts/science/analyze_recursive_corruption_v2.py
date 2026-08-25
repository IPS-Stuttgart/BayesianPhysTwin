#!/usr/bin/env python3
"""Reproduce every recursive-corruption v2 analysis product from retained inputs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tempfile
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist, fmean, stdev
from typing import Any, Final, cast

import numpy as np

from bayesian_phystwin_experiments.recursive_corruption_benchmark_v2 import (
    CONDITIONS,
    FALLBACK_REASONS,
    IMPERFECT_CUE_CONDITIONS,
    METHODS,
    PRIMARY_ENDPOINTS,
    SCHEMA,
    SCHEMA_VERSION,
    STRESS_CONDITIONS,
    TRACE_SCHEMA,
    TRACE_SCHEMA_VERSION,
    RecursiveCorruptionV2Config,
    sha256_file,
)

ANALYSIS_SCHEMA: Final = "bayesian-phystwin.recursive-corruption-analysis-v2"
ANALYSIS_SCHEMA_VERSION: Final = 2
PRIMARY_CLASSIFICATION: Final = "primary-preregistered"
SECONDARY_CLASSIFICATION: Final = "secondary-registered"

METRICS: Final[tuple[str, ...]] = (
    "rmse_m",
    "corruption_rmse_m",
    "recovery_rmse_m",
    "maximum_absolute_error_m",
    "recovery_half_life_steps",
    "accepted_update_count",
    "fallback_count",
    "materially_harmful_accepted_update_count",
    "exact_fallback_violation_count",
    "gaussian_nll",
    "coverage_90",
    "mean_full_interval_width_90_m",
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _records(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("records must be a nonempty list")
    if any(not isinstance(record, Mapping) for record in value):
        raise ValueError("every record must be a mapping")
    return cast(list[Mapping[str, Any]], value)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric")
    return result


def _optional_finite(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, name=name)


def _mean_sem(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("mean/SEM requires at least one value")
    mean = fmean(values)
    sem = 0.0 if len(values) == 1 else stdev(values) / math.sqrt(len(values))
    return mean, sem


def _bootstrap_interval(
    values: Sequence[float],
    *,
    seed: int,
    replicates: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap requires at least two finite seed values")
    rng = np.random.default_rng(np.random.SeedSequence([seed, array.size]))
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    means = np.mean(array[indices], axis=1)
    alpha = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha], method="linear")
    return float(lower), float(upper)


def _wilson_interval(
    events: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if total <= 0 or events < 0 or events > total:
        raise ValueError("Wilson interval requires 0 <= events <= total")
    z = NormalDist().inv_cdf(0.5 + 0.5 * confidence)
    fraction = events / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (fraction + z2 / (2.0 * total)) / denominator
    radius = (
        z
        / denominator
        * math.sqrt(fraction * (1.0 - fraction) / total + z2 / (4.0 * total * total))
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _load_trace_archive(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path, "r") as archive:
        metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
        for name in sorted(archive.namelist()):
            if not name.endswith(".npy"):
                continue
            arrays[name[:-4]] = np.lib.format.read_array(
                io.BytesIO(archive.read(name)),
                allow_pickle=False,
            )
    if metadata.get("schema") != TRACE_SCHEMA:
        raise ValueError("unexpected trace schema")
    if metadata.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError("unexpected trace schema version")
    return metadata, arrays


def _paired_seed_vector(
    by_key: Mapping[tuple[int, str, str], Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    conditions: Sequence[str],
    candidate: str,
    reference: str,
    metric: str,
    reduction: str,
) -> list[float]:
    vector: list[float] = []
    for seed in seeds:
        differences = [
            _finite(by_key[(seed, condition, candidate)].get(metric), name=metric)
            - _finite(by_key[(seed, condition, reference)].get(metric), name=metric)
            for condition in conditions
        ]
        if reduction == "mean":
            vector.append(fmean(differences))
        elif reduction == "sum":
            vector.append(float(sum(differences)))
        else:
            raise ValueError(f"unknown reduction {reduction!r}")
    return vector


def _contrast(
    *,
    name: str,
    classification: str,
    candidate: str,
    reference: str,
    metric: str,
    conditions: Sequence[str],
    reduction: str,
    by_key: Mapping[tuple[int, str, str], Mapping[str, Any]],
    seeds: Sequence[int],
    config: RecursiveCorruptionV2Config,
    stream: int,
) -> dict[str, object]:
    vector = _paired_seed_vector(
        by_key,
        seeds=seeds,
        conditions=conditions,
        candidate=candidate,
        reference=reference,
        metric=metric,
        reduction=reduction,
    )
    estimate = fmean(vector)
    interval = _bootstrap_interval(
        vector,
        seed=config.bootstrap_seed + stream,
        replicates=config.bootstrap_replicates,
    )
    return {
        "name": name,
        "classification": classification,
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "conditions": list(conditions),
        "condition_reduction": reduction,
        "independent_unit": "seed-domain",
        "independent_seed_count": len(vector),
        "estimate": estimate,
        "paired_95_interval": list(interval),
        "lower_is_better": True,
        "favorable_interval": interval[1] < 0.0,
        "seed_values": vector,
    }


def _condition_summary(
    *,
    records: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
    methods: Sequence[str],
) -> dict[str, dict[str, dict[str, object]]]:
    output: dict[str, dict[str, dict[str, object]]] = {}
    for condition in conditions:
        output[condition] = {}
        for method in methods:
            selected = [
                record
                for record in records
                if record.get("condition") == condition
                and record.get("method") == method
            ]
            summary: dict[str, object] = {"seed_count": len(selected)}
            for metric in METRICS:
                values = [
                    _optional_finite(
                        record.get(metric),
                        name=f"{condition}.{method}.{metric}",
                    )
                    for record in selected
                ]
                finite_values = [value for value in values if value is not None]
                if not finite_values:
                    summary[f"{metric}_mean"] = None
                    summary[f"{metric}_sem"] = None
                else:
                    mean, sem = _mean_sem(finite_values)
                    summary[f"{metric}_mean"] = mean
                    summary[f"{metric}_sem"] = sem
            reason_totals: Counter[str] = Counter()
            for record in selected:
                reasons = _mapping(
                    record.get("fallback_reasons"),
                    name="fallback reasons",
                )
                reason_totals.update(
                    {str(key): int(value) for key, value in reasons.items()}
                )
            summary["fallback_reason_totals"] = {
                reason: int(reason_totals.get(reason, 0))
                for reason in FALLBACK_REASONS
                if reason != "none"
            }
            output[condition][method] = summary
    return output


def _stress_summary(
    *,
    by_key: Mapping[tuple[int, str, str], Mapping[str, Any]],
    seeds: Sequence[int],
    conditions: Sequence[str],
    methods: Sequence[str],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for method in methods:
        summary: dict[str, object] = {"seed_count": len(seeds)}
        for metric in METRICS:
            seed_values: list[float] = []
            for seed in seeds:
                values = [
                    _optional_finite(
                        by_key[(seed, condition, method)].get(metric),
                        name=f"{seed}.{condition}.{method}.{metric}",
                    )
                    for condition in conditions
                ]
                finite_values = [value for value in values if value is not None]
                if finite_values:
                    seed_values.append(fmean(finite_values))
            if not seed_values:
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_sem"] = None
                summary[f"{metric}_seed_values"] = None
            else:
                mean, sem = _mean_sem(seed_values)
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_sem"] = sem
                summary[f"{metric}_seed_values"] = seed_values
        output[method] = summary
    return output


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _render_note(analysis: Mapping[str, Any]) -> str:
    primary = _mapping(analysis["primary_endpoints"], name="primary endpoints")
    secondary = _mapping(analysis["secondary_endpoints"], name="secondary endpoints")
    stress = _mapping(analysis["stress_summary"], name="stress summary")
    coequal = _mapping(analysis["coequal_review"], name="coequal review")

    def mm(value: object) -> str:
        return f"{1000.0 * float(value):.3f}"

    p1 = _mapping(primary[PRIMARY_ENDPOINTS[0]], name="primary endpoint 1")
    p2 = _mapping(primary[PRIMARY_ENDPOINTS[1]], name="primary endpoint 2")
    clean = _mapping(
        secondary["clean_guarded_recursive_minus_recursive_gaussian_rmse"],
        name="clean",
    )
    corruption = _mapping(
        secondary[
            "stress_corruption_rmse_guarded_recursive_minus_guarded_last_residual"
        ],
        name="corruption",
    )
    recovery = _mapping(
        secondary["stress_recovery_rmse_guarded_recursive_minus_guarded_last_residual"],
        name="recovery",
    )
    worst = _mapping(analysis["worst_seed_harm"], name="worst seed")

    lines = [
        "# Recursive-corruption benchmark v2 result",
        "",
        "## Evidence question",
        "",
        (
            "Version 2 gives `guarded_last_residual` and `guarded_recursive` "
            "the same reliability, lineage, innovation, trust-region, and "
            "exact-fallback information. It therefore separates the value of "
            "Gaussian recursive belief propagation from the value of the guard "
            "itself."
        ),
        "",
        "## Primary preregistered endpoints",
        "",
        (
            "1. Stress full-sequence RMSE, guarded recursion minus guarded "
            "last residual: "
            f"{mm(p1['estimate'])} mm [{mm(p1['paired_95_interval'][0])}, "
            f"{mm(p1['paired_95_interval'][1])}] mm."
        ),
        (
            f"2. Materially harmful accepted updates per seed, guarded recursion minus "
            f"unguarded Gaussian recursion: {float(p2['estimate']):.2f} "
            f"[{float(p2['paired_95_interval'][0]):.2f}, "
            f"{float(p2['paired_95_interval'][1]):.2f}]."
        ),
        "",
        (
            "Only these two endpoints are primary. Every other contrast below "
            "is secondary registered companion evidence."
        ),
        "",
        "## Co-equal companion evidence",
        "",
        (
            f"- Corruption-window RMSE contrast versus guarded last residual: "
            f"{mm(corruption['estimate'])} mm "
            f"[{mm(corruption['paired_95_interval'][0])}, "
            f"{mm(corruption['paired_95_interval'][1])}] mm."
        ),
        (
            f"- Recovery-window RMSE contrast versus guarded last residual: "
            f"{mm(recovery['estimate'])} mm [{mm(recovery['paired_95_interval'][0])}, "
            f"{mm(recovery['paired_95_interval'][1])}] mm."
        ),
        (
            f"- Clean-control cost versus unguarded Gaussian recursion: "
            f"{mm(clean['estimate'])} mm [{mm(clean['paired_95_interval'][0])}, "
            f"{mm(clean['paired_95_interval'][1])}] mm; preregistered margin "
            f"{mm(analysis['config']['clean_noninferiority_margin_m'])} mm."
        ),
        (
            f"- Harmful-seed fraction versus guarded last residual: "
            f"{100.0 * float(worst['harmful_seed_fraction']):.1f}% "
            f"[{100.0 * float(worst['harmful_seed_fraction_interval'][0]):.1f}, "
            f"{100.0 * float(worst['harmful_seed_fraction_interval'][1]):.1f}]%."
        ),
        (
            f"- Worst observed seed-domain regret versus guarded last residual: "
            f"{mm(worst['maximum_seed_regret_m'])} mm."
        ),
        "",
        "## Stress-set descriptive means",
        "",
    ]
    for method in METHODS:
        method_summary = _mapping(stress[method], name=method)
        lines.append(
            f"- `{method}`: {mm(method_summary['rmse_m_mean'])} mm full RMSE; "
            f"{mm(method_summary['corruption_rmse_m_mean'])} mm "
            "corruption-window RMSE; "
            f"{mm(method_summary['recovery_rmse_m_mean'])} mm recovery RMSE."
        )
    lines.extend(
        [
            "",
            "## Registered decision",
            "",
            "All co-equal criteria passed: "
            f"**{str(coequal['all_criteria_passed']).lower()}**.",
            "",
        ]
    )
    for criterion, passed in _mapping(coequal["criteria"], name="criteria").items():
        lines.append(f"- `{criterion}`: {str(bool(passed)).lower()}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            str(analysis["scientific_boundary"]),
            "",
            (
                "The independent units are freshly registered seed-domains. "
                "Physical dynamics, action trajectories, discrepancy dynamics, "
                "observation noise, corruption timing, and corruption severity "
                "vary across seeds. The result remains a controlled synthetic "
                "mechanism study and does not establish real-provider or "
                "physical-object transfer."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    result_path: Path,
    trace_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != SCHEMA or result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected v2 result schema")
    seeds = tuple(int(seed) for seed in result["seeds"])
    conditions = tuple(str(condition) for condition in result["conditions"])
    methods = tuple(str(method) for method in result["methods"])
    if conditions != CONDITIONS:
        raise ValueError("analysis requires the complete registered condition roster")
    if methods != METHODS:
        raise ValueError("analysis requires the complete registered method roster")
    if tuple(result.get("primary_endpoints", ())) != PRIMARY_ENDPOINTS:
        raise ValueError("primary endpoint roster changed")
    config = RecursiveCorruptionV2Config(**_mapping(result["config"], name="config"))
    records = _records(result.get("records"))
    expected_count = len(seeds) * len(conditions) * len(methods)
    if len(records) != expected_count:
        raise ValueError("record matrix is incomplete")
    by_key = {
        (int(record["seed"]), str(record["condition"]), str(record["method"])): record
        for record in records
    }
    if len(by_key) != expected_count:
        raise ValueError("record matrix contains duplicate keys")

    trace_metadata, arrays = _load_trace_archive(trace_path)
    if trace_metadata.get("result_id") != result.get("result_id"):
        raise ValueError("trace archive does not bind the exact result")
    if tuple(trace_metadata.get("seeds", ())) != seeds:
        raise ValueError("trace seed roster mismatch")
    if tuple(trace_metadata.get("conditions", ())) != conditions:
        raise ValueError("trace condition roster mismatch")
    if tuple(trace_metadata.get("methods", ())) != methods:
        raise ValueError("trace method roster mismatch")

    condition_summary = _condition_summary(
        records=records,
        conditions=conditions,
        methods=methods,
    )
    stress_summary = _stress_summary(
        by_key=by_key,
        seeds=seeds,
        conditions=STRESS_CONDITIONS,
        methods=methods,
    )

    primary: dict[str, dict[str, object]] = {}
    primary[PRIMARY_ENDPOINTS[0]] = _contrast(
        name=PRIMARY_ENDPOINTS[0],
        classification=PRIMARY_CLASSIFICATION,
        candidate="guarded_recursive",
        reference="guarded_last_residual",
        metric="rmse_m",
        conditions=STRESS_CONDITIONS,
        reduction="mean",
        by_key=by_key,
        seeds=seeds,
        config=config,
        stream=1,
    )
    primary[PRIMARY_ENDPOINTS[1]] = _contrast(
        name=PRIMARY_ENDPOINTS[1],
        classification=PRIMARY_CLASSIFICATION,
        candidate="guarded_recursive",
        reference="recursive_gaussian",
        metric="materially_harmful_accepted_update_count",
        conditions=STRESS_CONDITIONS,
        reduction="sum",
        by_key=by_key,
        seeds=seeds,
        config=config,
        stream=2,
    )

    secondary_specs = (
        (
            "stress_corruption_rmse_guarded_recursive_minus_guarded_last_residual",
            "guarded_recursive",
            "guarded_last_residual",
            "corruption_rmse_m",
            STRESS_CONDITIONS,
            "mean",
        ),
        (
            "stress_recovery_rmse_guarded_recursive_minus_guarded_last_residual",
            "guarded_recursive",
            "guarded_last_residual",
            "recovery_rmse_m",
            STRESS_CONDITIONS,
            "mean",
        ),
        (
            "clean_guarded_recursive_minus_recursive_gaussian_rmse",
            "guarded_recursive",
            "recursive_gaussian",
            "rmse_m",
            ("clean",),
            "mean",
        ),
        (
            "clean_guarded_recursive_minus_guarded_last_residual_rmse",
            "guarded_recursive",
            "guarded_last_residual",
            "rmse_m",
            ("clean",),
            "mean",
        ),
        (
            "imperfect_cue_rmse_guarded_recursive_minus_guarded_last_residual",
            "guarded_recursive",
            "guarded_last_residual",
            "rmse_m",
            IMPERFECT_CUE_CONDITIONS,
            "mean",
        ),
        (
            "stress_full_rmse_guarded_recursive_minus_physical_baseline",
            "guarded_recursive",
            "physical_baseline",
            "rmse_m",
            STRESS_CONDITIONS,
            "mean",
        ),
        (
            "stress_full_rmse_guarded_last_residual_minus_last_residual",
            "guarded_last_residual",
            "last_residual",
            "rmse_m",
            STRESS_CONDITIONS,
            "mean",
        ),
    )
    secondary: dict[str, dict[str, object]] = {}
    for stream, (
        name,
        candidate,
        reference,
        metric,
        selected_conditions,
        reduction,
    ) in enumerate(secondary_specs, start=10):
        secondary[name] = _contrast(
            name=name,
            classification=SECONDARY_CLASSIFICATION,
            candidate=candidate,
            reference=reference,
            metric=metric,
            conditions=selected_conditions,
            reduction=reduction,
            by_key=by_key,
            seeds=seeds,
            config=config,
            stream=stream,
        )

    seed_regret = cast(
        list[float],
        primary[PRIMARY_ENDPOINTS[0]]["seed_values"],
    )
    harmful_events = sum(value > config.harmful_seed_margin_m for value in seed_regret)
    harmful_fraction = harmful_events / len(seed_regret)
    harmful_interval = _wilson_interval(harmful_events, len(seed_regret))
    worst_seed = {
        "reference": "guarded_last_residual",
        "metric": "mean stress full-sequence RMSE regret",
        "harmful_seed_margin_m": config.harmful_seed_margin_m,
        "harmful_seed_count": harmful_events,
        "seed_count": len(seed_regret),
        "harmful_seed_fraction": harmful_fraction,
        "harmful_seed_fraction_interval": list(harmful_interval),
        "maximum_seed_regret_m": max(seed_regret),
        "minimum_seed_regret_m": min(seed_regret),
        "p95_seed_regret_m": float(np.quantile(seed_regret, 0.95, method="linear")),
    }

    p1 = primary[PRIMARY_ENDPOINTS[0]]
    p2 = primary[PRIMARY_ENDPOINTS[1]]
    corruption = secondary[
        "stress_corruption_rmse_guarded_recursive_minus_guarded_last_residual"
    ]
    recovery = secondary[
        "stress_recovery_rmse_guarded_recursive_minus_guarded_last_residual"
    ]
    clean = secondary["clean_guarded_recursive_minus_recursive_gaussian_rmse"]
    guarded_fallback_violations = sum(
        int(record["exact_fallback_violation_count"])
        for record in records
        if record["method"] in {"guarded_last_residual", "guarded_recursive"}
    )
    criteria = {
        "primary_matched_rmse_interval_below_zero": (
            float(p1["paired_95_interval"][1]) < 0.0
        ),
        "primary_harmful_update_interval_below_zero": (
            float(p2["paired_95_interval"][1]) < 0.0
        ),
        "corruption_window_interval_below_zero": (
            float(corruption["paired_95_interval"][1]) < 0.0
        ),
        "recovery_window_interval_below_zero": (
            float(recovery["paired_95_interval"][1]) < 0.0
        ),
        "clean_cost_within_preregistered_margin": float(clean["paired_95_interval"][1])
        <= config.clean_noninferiority_margin_m,
        "harmful_seed_fraction_within_limit": harmful_fraction
        <= config.maximum_harmful_seed_fraction,
        "worst_seed_regret_within_limit": max(seed_regret)
        <= config.maximum_worst_seed_regret_m,
        "exact_fallback_identity_preserved": guarded_fallback_violations == 0,
    }

    analysis: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "result_id": result["result_id"],
        "source_result_sha256": sha256_file(result_path),
        "source_trace_sha256": sha256_file(trace_path),
        "config": asdict(config),
        "independent_unit": "fresh seed-domain",
        "seed_count": len(seeds),
        "fresh_seed_roster": list(seeds),
        "dynamics_vary_across_seeds": True,
        "conditions_nested_within_seed": True,
        "primary_endpoint_names": list(PRIMARY_ENDPOINTS),
        "primary_endpoints": primary,
        "secondary_endpoints": secondary,
        "condition_summary": condition_summary,
        "stress_summary": stress_summary,
        "worst_seed_harm": worst_seed,
        "guarded_exact_fallback_violation_count": guarded_fallback_violations,
        "coequal_review": {
            "criteria": criteria,
            "all_criteria_passed": all(criteria.values()),
            "retuning_on_this_roster_authorized": False,
        },
        "metric_support": {
            "materially_harmful_accepted_update_count": {
                "defined_for_methods": [
                    "last_residual",
                    "guarded_last_residual",
                    "recursive_gaussian",
                    "guarded_recursive",
                ],
                "undefined_for_methods": ["physical_baseline"],
            },
            "gaussian_nll": {
                "defined_for_methods": ["recursive_gaussian", "guarded_recursive"],
                "undefined_for_methods": [
                    "physical_baseline",
                    "last_residual",
                    "guarded_last_residual",
                ],
            },
        },
        "scientific_boundary": result["scientific_boundary"],
        "access_boundary": result["access_boundary"],
    }

    absolute_error = arrays.get("absolute_error_m")
    if absolute_error is None:
        raise ValueError("trace archive lacks absolute_error_m")
    expected_shape = (len(seeds), len(conditions), len(methods), config.step_count - 1)
    if absolute_error.shape != expected_shape:
        raise ValueError("absolute-error trace shape mismatch")
    seed_domains = result["seed_domains"]
    if not isinstance(seed_domains, list) or len(seed_domains) != len(seeds):
        raise ValueError("seed-domain roster mismatch")
    time_rows: list[dict[str, object]] = []
    relative_min = -20
    relative_max = config.recovery_window
    for condition_index, condition in enumerate(conditions):
        for method_index, method in enumerate(methods):
            for relative_step in range(relative_min, relative_max + 1):
                values: list[float] = []
                times: list[float] = []
                for seed_index, raw_domain in enumerate(seed_domains):
                    domain = _mapping(raw_domain, name="seed domain")
                    absolute_step = int(domain["corruption_start"]) + relative_step
                    if 0 <= absolute_step < config.step_count - 1:
                        values.append(
                            float(
                                absolute_error[
                                    seed_index,
                                    condition_index,
                                    method_index,
                                    absolute_step,
                                ]
                            )
                        )
                        times.append(relative_step * float(domain["time_step"]))
                mean, sem = _mean_sem(values)
                time_rows.append(
                    {
                        "condition": condition,
                        "method": method,
                        "relative_step": relative_step,
                        "mean_relative_time_s": fmean(times),
                        "seed_count": len(values),
                        "mean_absolute_error_m": mean,
                        "sem_absolute_error_m": sem,
                    }
                )

    condition_rows: list[dict[str, object]] = []
    for condition in conditions:
        for method in methods:
            summary = condition_summary[condition][method]
            row: dict[str, object] = {"condition": condition, "method": method}
            for metric in METRICS:
                row[f"{metric}_mean"] = summary[f"{metric}_mean"]
                row[f"{metric}_sem"] = summary[f"{metric}_sem"]
            for reason in FALLBACK_REASONS:
                if reason != "none":
                    row[f"fallback_reason__{reason}"] = summary[
                        "fallback_reason_totals"
                    ][reason]
            condition_rows.append(row)

    endpoint_rows: list[dict[str, object]] = []
    for collection_name, collection in (
        ("primary", primary),
        ("secondary", secondary),
    ):
        for endpoint_name, endpoint in collection.items():
            endpoint_rows.append(
                {
                    "endpoint_group": collection_name,
                    "endpoint_name": endpoint_name,
                    "classification": endpoint["classification"],
                    "candidate": endpoint["candidate"],
                    "reference": endpoint["reference"],
                    "metric": endpoint["metric"],
                    "conditions": ",".join(endpoint["conditions"]),
                    "estimate": endpoint["estimate"],
                    "ci95_lower": endpoint["paired_95_interval"][0],
                    "ci95_upper": endpoint["paired_95_interval"][1],
                    "favorable_interval": endpoint["favorable_interval"],
                    "independent_seed_count": endpoint["independent_seed_count"],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_payloads: dict[str, bytes] = {
        "analysis.json": _canonical_json_bytes(analysis),
        "metric-support.json": _canonical_json_bytes(analysis["metric_support"]),
        "result-note.md": (_render_note(analysis).rstrip() + "\n").encode("utf-8"),
    }
    for name, content in output_payloads.items():
        (output_dir / name).write_bytes(content)

    condition_fieldnames = list(condition_rows[0].keys())
    _write_csv(
        output_dir / "condition-summary.csv",
        condition_fieldnames,
        condition_rows,
    )
    _write_csv(
        output_dir / "endpoint-summary.csv",
        list(endpoint_rows[0].keys()),
        endpoint_rows,
    )
    _write_csv(
        output_dir / "time-summary.csv",
        list(time_rows[0].keys()),
        time_rows,
    )

    figure_data = {
        "schema": "bayesian-phystwin.recursive-corruption-figure-data-v2",
        "schema_version": 2,
        "result_id": result["result_id"],
        "condition_summary_csv_sha256": sha256_file(
            output_dir / "condition-summary.csv"
        ),
        "endpoint_summary_csv_sha256": sha256_file(output_dir / "endpoint-summary.csv"),
        "time_summary_csv_sha256": sha256_file(output_dir / "time-summary.csv"),
        "primary_endpoint_names": list(PRIMARY_ENDPOINTS),
        "recommended_panels": [
            "full-corruption-recovery RMSE by condition for matched guarded arms",
            "harmful accepted updates for last-residual and Gaussian arms",
            "corruption-aligned per-step absolute-error traces",
            "clean-control and worst-seed harm annotations",
        ],
    }
    (output_dir / "figure-data.json").write_bytes(_canonical_json_bytes(figure_data))

    output_files = (
        "analysis.json",
        "metric-support.json",
        "result-note.md",
        "condition-summary.csv",
        "endpoint-summary.csv",
        "time-summary.csv",
        "figure-data.json",
    )
    manifest = {
        "schema": "bayesian-phystwin.recursive-corruption-analysis-manifest-v2",
        "schema_version": 2,
        "analysis_generator_sha256": sha256_file(Path(__file__)),
        "inputs": {
            result_path.name: sha256_file(result_path),
            trace_path.name: sha256_file(trace_path),
        },
        "outputs": {name: sha256_file(output_dir / name) for name in output_files},
        "all_coequal_criteria_passed": analysis["coequal_review"][
            "all_criteria_passed"
        ],
        "scientific_boundary": analysis["scientific_boundary"],
    }
    (output_dir / "analysis-manifest.json").write_bytes(_canonical_json_bytes(manifest))
    return analysis


def check_reproduction(
    *,
    result_path: Path,
    trace_path: Path,
    output_dir: Path,
) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        regenerated = Path(temporary)
        analyze(result_path=result_path, trace_path=trace_path, output_dir=regenerated)
        expected_names = {
            "analysis.json",
            "metric-support.json",
            "result-note.md",
            "condition-summary.csv",
            "endpoint-summary.csv",
            "time-summary.csv",
            "figure-data.json",
            "analysis-manifest.json",
        }
        observed_names = {path.name for path in output_dir.iterdir() if path.is_file()}
        missing = expected_names - observed_names
        if missing:
            raise ValueError(
                f"retained analysis outputs are missing: {sorted(missing)}"
            )
        for name in sorted(expected_names):
            if (regenerated / name).read_bytes() != (output_dir / name).read_bytes():
                raise ValueError(
                    f"retained analysis output is not reproducible: {name}"
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        check_reproduction(
            result_path=args.result,
            trace_path=args.traces,
            output_dir=args.output_dir,
        )
    else:
        analysis = analyze(
            result_path=args.result,
            trace_path=args.traces,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "result_id": analysis["result_id"],
                    "all_coequal_criteria_passed": analysis["coequal_review"][
                        "all_criteria_passed"
                    ],
                    "primary_endpoint_names": analysis["primary_endpoint_names"],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
