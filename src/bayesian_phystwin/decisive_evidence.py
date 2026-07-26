"""Matched selective-risk diagnostics for prospective PhysTwin evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DECISIVE_EVIDENCE_INPUT_CONTRACT = "bayesian-phystwin-decisive-evidence-v1"
DECISIVE_EVIDENCE_SUMMARY_CONTRACT = (
    "bayesian-phystwin-decisive-evidence-summary-v1"
)
DEFAULT_TARGET_COVERAGES = tuple(float(value) / 10.0 for value in range(11))
DEFAULT_REGRESSION_QUANTILES = (0.90, 0.95)
DEFAULT_RELIABILITY_EDGES = (0.0, 0.25, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class IntervalObservation:
    """One deployed predictive interval observation."""

    nominal_coverage: float
    covered: bool
    width: float


@dataclass(frozen=True)
class EvidenceRecord:
    """One method outcome for one metric and one statistical unit."""

    unit_id: str
    group_id: str
    metric: str
    method: str
    loss: float
    fallback_loss: float
    risk_score: float
    accepted: bool
    deployed_loss: float
    horizon: str
    reliability: float | None
    identifiable_rank: int | None
    intervals: tuple[IntervalObservation, ...]


@dataclass(frozen=True)
class EvidenceBundle:
    """Validated prospective-evidence bundle."""

    protocol_id: str
    statistical_unit: str
    claim_boundary: str
    reference_method: str | None
    records: tuple[EvidenceRecord, ...]


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _optional_reliability(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _number(value, name=name, minimum=0.0, maximum=1.0)


def _optional_rank(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _horizon(value: object, *, name: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonempty label or nonnegative number")
    if isinstance(value, str):
        return _text(value, name=name)
    if isinstance(value, (int, float)):
        number = _number(value, name=name, minimum=0.0)
        return f"{number:.12g}"
    raise ValueError(f"{name} must be a nonempty label or nonnegative number")


def _parse_intervals(value: object, *, record_name: str) -> tuple[IntervalObservation, ...]:
    if value is None:
        return ()
    raw_intervals = _sequence(value, name=f"{record_name}.intervals")
    intervals: list[IntervalObservation] = []
    seen_coverages: set[float] = set()
    for index, raw_interval in enumerate(raw_intervals):
        name = f"{record_name}.intervals[{index}]"
        interval = _mapping(raw_interval, name=name)
        nominal = _number(
            interval.get("nominal_coverage"),
            name=f"{name}.nominal_coverage",
            minimum=0.0,
            maximum=1.0,
        )
        if nominal in {0.0, 1.0}:
            raise ValueError(f"{name}.nominal_coverage must lie strictly inside (0, 1)")
        if nominal in seen_coverages:
            raise ValueError(f"{record_name} repeats nominal coverage {nominal}")
        seen_coverages.add(nominal)
        intervals.append(
            IntervalObservation(
                nominal_coverage=nominal,
                covered=_boolean(interval.get("covered"), name=f"{name}.covered"),
                width=_number(
                    interval.get("width"), name=f"{name}.width", minimum=0.0
                ),
            )
        )
    return tuple(sorted(intervals, key=lambda item: item.nominal_coverage))


def _parse_record(value: object, *, index: int) -> EvidenceRecord:
    name = f"records[{index}]"
    record = _mapping(value, name=name)
    accepted = _boolean(record.get("accepted"), name=f"{name}.accepted")
    loss = _number(record.get("loss"), name=f"{name}.loss", minimum=0.0)
    fallback_loss = _number(
        record.get("fallback_loss"), name=f"{name}.fallback_loss", minimum=0.0
    )
    deployed_loss = _number(
        record.get("deployed_loss"), name=f"{name}.deployed_loss", minimum=0.0
    )
    expected_deployed = loss if accepted else fallback_loss
    if deployed_loss != expected_deployed:
        behavior = "raw method loss" if accepted else "exact fallback loss"
        raise ValueError(
            f"{name}.deployed_loss must equal the {behavior}; "
            f"expected {expected_deployed!r}, got {deployed_loss!r}"
        )
    unit_id = _text(record.get("unit_id"), name=f"{name}.unit_id")
    raw_group = record.get("group_id", unit_id)
    return EvidenceRecord(
        unit_id=unit_id,
        group_id=_text(raw_group, name=f"{name}.group_id"),
        metric=_text(record.get("metric"), name=f"{name}.metric"),
        method=_text(record.get("method"), name=f"{name}.method"),
        loss=loss,
        fallback_loss=fallback_loss,
        risk_score=_number(record.get("risk_score"), name=f"{name}.risk_score"),
        accepted=accepted,
        deployed_loss=deployed_loss,
        horizon=_horizon(record.get("horizon"), name=f"{name}.horizon"),
        reliability=_optional_reliability(
            record.get("reliability"), name=f"{name}.reliability"
        ),
        identifiable_rank=_optional_rank(
            record.get("identifiable_rank"), name=f"{name}.identifiable_rank"
        ),
        intervals=_parse_intervals(record.get("intervals"), record_name=name),
    )


def parse_decisive_evidence(payload: Mapping[str, object]) -> EvidenceBundle:
    """Validate a decisive-evidence bundle and its matched fallback contract."""

    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("schema_version must be the integer 1")
    contract = payload.get("contract", DECISIVE_EVIDENCE_INPUT_CONTRACT)
    if contract != DECISIVE_EVIDENCE_INPUT_CONTRACT:
        raise ValueError(
            f"contract must be {DECISIVE_EVIDENCE_INPUT_CONTRACT!r}"
        )
    raw_reference = payload.get("reference_method")
    reference_method = (
        None
        if raw_reference is None
        else _text(raw_reference, name="reference_method")
    )
    records = tuple(
        _parse_record(value, index=index)
        for index, value in enumerate(
            _sequence(payload.get("records"), name="records")
        )
    )
    if not records:
        raise ValueError("records must not be empty")

    by_unit: dict[tuple[str, str], list[EvidenceRecord]] = {}
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        unique_key = (record.metric, record.unit_id, record.method)
        if unique_key in seen:
            raise ValueError(
                "duplicate metric/unit/method record: " + "/".join(unique_key)
            )
        seen.add(unique_key)
        by_unit.setdefault((record.metric, record.unit_id), []).append(record)

    methods_by_metric: dict[str, frozenset[str]] = {}
    for (metric, unit_id), unit_records in by_unit.items():
        methods = frozenset(record.method for record in unit_records)
        expected_methods = methods_by_metric.setdefault(metric, methods)
        if methods != expected_methods:
            raise ValueError(
                f"{metric}/{unit_id} has methods {sorted(methods)}, expected "
                f"{sorted(expected_methods)}; matched comparisons require every "
                "method on every unit"
            )
        first = unit_records[0]
        for record in unit_records[1:]:
            if record.fallback_loss != first.fallback_loss:
                raise ValueError(
                    f"{metric}/{unit_id} has method-dependent fallback losses"
                )
            if record.group_id != first.group_id:
                raise ValueError(f"{metric}/{unit_id} has inconsistent group_id values")
            if record.horizon != first.horizon:
                raise ValueError(f"{metric}/{unit_id} has inconsistent horizon values")

    if reference_method is not None:
        missing = sorted(
            metric
            for metric, methods in methods_by_metric.items()
            if reference_method not in methods
        )
        if missing:
            raise ValueError(
                f"reference method {reference_method!r} is absent for metrics {missing}"
            )

    return EvidenceBundle(
        protocol_id=_text(payload.get("protocol_id"), name="protocol_id"),
        statistical_unit=_text(
            payload.get("statistical_unit"), name="statistical_unit"
        ),
        claim_boundary=_text(
            payload.get("claim_boundary"), name="claim_boundary"
        ),
        reference_method=reference_method,
        records=records,
    )


def _validated_probabilities(
    values: Sequence[float], *, name: str, include_endpoints: bool
) -> tuple[float, ...]:
    parsed = tuple(
        _number(value, name=f"{name}[{index}]", minimum=0.0, maximum=1.0)
        for index, value in enumerate(values)
    )
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    if not include_endpoints and any(value in {0.0, 1.0} for value in parsed):
        raise ValueError(f"{name} values must lie strictly inside (0, 1)")
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError(f"{name} must be strictly increasing")
    return parsed


def _validated_reliability_edges(values: Sequence[float]) -> tuple[float, ...]:
    edges = _validated_probabilities(
        values, name="reliability_edges", include_endpoints=True
    )
    if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("reliability_edges must start at 0 and end at 1")
    return edges


def _mean(values: np.ndarray) -> float | None:
    return None if not len(values) else float(np.mean(values))


def _regression_summary(
    losses: np.ndarray,
    baselines: np.ndarray,
    *,
    quantiles: tuple[float, ...],
) -> dict[str, object]:
    eligible = baselines > 0.0
    changes = losses[eligible] / baselines[eligible] - 1.0
    if not len(changes):
        return {
            "eligible_count": 0,
            "improved_count": 0,
            "unchanged_count": 0,
            "regressed_count": 0,
            "mean_relative_change": None,
            "worst_relative_change": None,
            "worst_relative_regression": None,
            "high_quantiles": [
                {"quantile": quantile, "relative_change": None}
                for quantile in quantiles
            ],
        }
    worst = float(np.max(changes))
    return {
        "eligible_count": int(len(changes)),
        "improved_count": int(np.sum(changes < 0.0)),
        "unchanged_count": int(np.sum(changes == 0.0)),
        "regressed_count": int(np.sum(changes > 0.0)),
        "mean_relative_change": float(np.mean(changes)),
        "worst_relative_change": worst,
        "worst_relative_regression": max(0.0, worst),
        "high_quantiles": [
            {
                "quantile": quantile,
                "relative_change": float(np.quantile(changes, quantile)),
            }
            for quantile in quantiles
        ],
    }


def _loss_summary(
    losses: np.ndarray,
    baselines: np.ndarray,
    *,
    quantiles: tuple[float, ...],
) -> dict[str, object]:
    mean_loss = _mean(losses)
    mean_baseline = _mean(baselines)
    relative_change = (
        None
        if mean_loss is None or mean_baseline is None or mean_baseline <= 0.0
        else mean_loss / mean_baseline - 1.0
    )
    return {
        "count": int(len(losses)),
        "mean_loss": mean_loss,
        "mean_fallback_loss": mean_baseline,
        "mean_absolute_change_vs_fallback": _mean(losses - baselines),
        "relative_change_of_means_vs_fallback": relative_change,
        "per_unit_regression_vs_fallback": _regression_summary(
            losses, baselines, quantiles=quantiles
        ),
    }


def _operational_summary(
    records: Sequence[EvidenceRecord], *, quantiles: tuple[float, ...]
) -> dict[str, object]:
    losses = np.asarray([record.loss for record in records], dtype=float)
    fallbacks = np.asarray([record.fallback_loss for record in records], dtype=float)
    deployed = np.asarray([record.deployed_loss for record in records], dtype=float)
    accepted = np.asarray([record.accepted for record in records], dtype=bool)
    harmful = accepted & (losses > fallbacks)
    accepted_count = int(np.sum(accepted))
    return {
        "unit_count": int(len(records)),
        "accepted_count": accepted_count,
        "coverage": None if not records else accepted_count / len(records),
        "fallback_count": int(len(records) - accepted_count),
        "fallback_frequency": (
            None if not records else 1.0 - accepted_count / len(records)
        ),
        "exact_fallback_verified": True,
        "harmful_accepted_count": int(np.sum(harmful)),
        "harmful_update_frequency_all_units": (
            None if not records else float(np.mean(harmful))
        ),
        "harmful_update_frequency_accepted": (
            None if not accepted_count else float(np.sum(harmful) / accepted_count)
        ),
        "raw_method": _loss_summary(losses, fallbacks, quantiles=quantiles),
        "deployed": _loss_summary(deployed, fallbacks, quantiles=quantiles),
    }


def _pairwise_summary(
    candidate: Sequence[EvidenceRecord],
    reference: Sequence[EvidenceRecord],
    *,
    deployed: bool,
    quantiles: tuple[float, ...],
) -> dict[str, object]:
    candidate_by_unit = {record.unit_id: record for record in candidate}
    reference_by_unit = {record.unit_id: record for record in reference}
    units = tuple(sorted(candidate_by_unit))
    if set(units) != set(reference_by_unit):
        raise AssertionError("validated method unit sets diverged")
    attribute = "deployed_loss" if deployed else "loss"
    candidate_losses = np.asarray(
        [getattr(candidate_by_unit[unit], attribute) for unit in units], dtype=float
    )
    reference_losses = np.asarray(
        [getattr(reference_by_unit[unit], attribute) for unit in units], dtype=float
    )
    mean_reference = float(np.mean(reference_losses))
    return {
        "reference_method_mean_loss": mean_reference,
        "candidate_method_mean_loss": float(np.mean(candidate_losses)),
        "mean_loss_difference": float(np.mean(candidate_losses - reference_losses)),
        "relative_change_of_means": (
            None
            if mean_reference <= 0.0
            else float(np.mean(candidate_losses)) / mean_reference - 1.0
        ),
        "unit_wins": int(np.sum(candidate_losses < reference_losses)),
        "unit_ties": int(np.sum(candidate_losses == reference_losses)),
        "unit_losses": int(np.sum(candidate_losses > reference_losses)),
        "per_unit_regression_vs_reference": _regression_summary(
            candidate_losses, reference_losses, quantiles=quantiles
        ),
    }


def _coverage_count(target: float, count: int) -> int:
    return min(count, max(0, int(np.floor(target * count + 0.5))))


def _risk_coverage_curves(
    records_by_method: Mapping[str, Sequence[EvidenceRecord]],
    *,
    coverages: tuple[float, ...],
    quantiles: tuple[float, ...],
    reference_method: str | None,
) -> dict[str, object]:
    methods = tuple(sorted(records_by_method))
    unit_count = len(records_by_method[methods[0]])
    by_method_unit = {
        method: {record.unit_id: record for record in records}
        for method, records in records_by_method.items()
    }
    unit_order = tuple(sorted(by_method_unit[methods[0]]))
    curves: dict[str, list[dict[str, object]]] = {method: [] for method in methods}
    deployed_cache: dict[tuple[str, int], np.ndarray] = {}

    for coverage_index, target in enumerate(coverages):
        accepted_count = _coverage_count(target, unit_count)
        for method in methods:
            records = records_by_method[method]
            ranked = sorted(records, key=lambda record: (record.risk_score, record.unit_id))
            accepted_ids = {record.unit_id for record in ranked[:accepted_count]}
            losses = np.asarray(
                [by_method_unit[method][unit].loss for unit in unit_order], dtype=float
            )
            fallbacks = np.asarray(
                [by_method_unit[method][unit].fallback_loss for unit in unit_order],
                dtype=float,
            )
            accepted = np.asarray(
                [unit in accepted_ids for unit in unit_order], dtype=bool
            )
            deployed_losses = np.where(accepted, losses, fallbacks)
            deployed_cache[(method, coverage_index)] = deployed_losses
            harmful = accepted & (losses > fallbacks)
            boundary_tie_split = (
                0 < accepted_count < unit_count
                and ranked[accepted_count - 1].risk_score
                == ranked[accepted_count].risk_score
            )
            curves[method].append(
                {
                    "target_coverage": target,
                    "coverage": (
                        None if not unit_count else accepted_count / unit_count
                    ),
                    "accepted_count": accepted_count,
                    "fallback_count": unit_count - accepted_count,
                    "fallback_frequency": (
                        None if not unit_count else 1.0 - accepted_count / unit_count
                    ),
                    "maximum_accepted_risk_score": (
                        None
                        if not accepted_count
                        else float(ranked[accepted_count - 1].risk_score)
                    ),
                    "boundary_tie_split_by_unit_id": boundary_tie_split,
                    "selective_mean_loss": (
                        None if not accepted_count else float(np.mean(losses[accepted]))
                    ),
                    "harmful_accepted_count": int(np.sum(harmful)),
                    "harmful_update_frequency_accepted": (
                        None
                        if not accepted_count
                        else float(np.sum(harmful) / accepted_count)
                    ),
                    "deployed": _loss_summary(
                        deployed_losses, fallbacks, quantiles=quantiles
                    ),
                }
            )

    if reference_method is not None:
        reference_records = records_by_method[reference_method]
        reference_by_unit = {record.unit_id: record for record in reference_records}
        for method in methods:
            for coverage_index, _target in enumerate(coverages):
                if method == reference_method:
                    curves[method][coverage_index]["vs_reference_method"] = None
                    continue
                candidate_losses = deployed_cache[(method, coverage_index)]
                reference_losses = deployed_cache[(reference_method, coverage_index)]
                mean_reference = float(np.mean(reference_losses))
                curves[method][coverage_index]["vs_reference_method"] = {
                    "reference_method": reference_method,
                    "reference_mean_loss": mean_reference,
                    "mean_loss_difference": float(
                        np.mean(candidate_losses - reference_losses)
                    ),
                    "relative_change_of_means": (
                        None
                        if mean_reference <= 0.0
                        else float(np.mean(candidate_losses)) / mean_reference - 1.0
                    ),
                    "unit_wins": int(np.sum(candidate_losses < reference_losses)),
                    "unit_ties": int(np.sum(candidate_losses == reference_losses)),
                    "unit_losses": int(np.sum(candidate_losses > reference_losses)),
                    "per_unit_regression_vs_reference": _regression_summary(
                        candidate_losses, reference_losses, quantiles=quantiles
                    ),
                }
        if set(reference_by_unit) != set(unit_order):
            raise AssertionError("validated reference unit set diverged")

    return {
        "risk_score_order": "lower_is_safer",
        "selection_rule": (
            "accept the exact same count per method at each target coverage; sort "
            "by risk_score and break boundary ties deterministically by unit_id; "
            "rejected units use the common exact fallback"
        ),
        "target_coverages": list(coverages),
        "methods": curves,
    }


def _conditioned_summary(
    records: Sequence[EvidenceRecord], *, quantiles: tuple[float, ...]
) -> dict[str, object]:
    summary = _operational_summary(records, quantiles=quantiles)
    summary["mean_risk_score"] = _mean(
        np.asarray([record.risk_score for record in records], dtype=float)
    )
    return summary


def _reliability_conditioning(
    records: Sequence[EvidenceRecord],
    *,
    edges: tuple[float, ...],
    quantiles: tuple[float, ...],
) -> dict[str, object]:
    available = [record for record in records if record.reliability is not None]
    bins: list[dict[str, object]] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        last = index == len(edges) - 2
        selected = [
            record
            for record in available
            if record.reliability is not None
            and record.reliability >= lower
            and (record.reliability <= upper if last else record.reliability < upper)
        ]
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "upper_inclusive": last,
                "summary": _conditioned_summary(selected, quantiles=quantiles),
            }
        )
    return {
        "available_count": len(available),
        "missing_count": len(records) - len(available),
        "edges": list(edges),
        "bins": bins,
    }


def _rank_conditioning(
    records: Sequence[EvidenceRecord], *, quantiles: tuple[float, ...]
) -> dict[str, object]:
    available = [record for record in records if record.identifiable_rank is not None]
    ranks = sorted(
        {record.identifiable_rank for record in available if record.identifiable_rank is not None}
    )
    return {
        "available_count": len(available),
        "missing_count": len(records) - len(available),
        "by_rank": [
            {
                "rank": rank,
                "summary": _conditioned_summary(
                    [record for record in available if record.identifiable_rank == rank],
                    quantiles=quantiles,
                ),
            }
            for rank in ranks
        ],
    }


def _interval_calibration(records: Sequence[EvidenceRecord]) -> dict[str, object]:
    by_nominal: dict[float, list[IntervalObservation]] = {}
    for record in records:
        for interval in record.intervals:
            by_nominal.setdefault(interval.nominal_coverage, []).append(interval)
    return {
        "interval_observation_count": sum(len(value) for value in by_nominal.values()),
        "by_nominal_coverage": [
            {
                "nominal_coverage": nominal,
                "count": len(intervals),
                "empirical_coverage": float(
                    np.mean([interval.covered for interval in intervals])
                ),
                "coverage_error": float(
                    np.mean([interval.covered for interval in intervals]) - nominal
                ),
                "mean_width": float(np.mean([interval.width for interval in intervals])),
                "median_width": float(
                    np.median([interval.width for interval in intervals])
                ),
                "p90_width": float(
                    np.quantile([interval.width for interval in intervals], 0.90)
                ),
            }
            for nominal, intervals in sorted(by_nominal.items())
        ],
    }


def _horizon_summary(
    records: Sequence[EvidenceRecord], *, quantiles: tuple[float, ...]
) -> list[dict[str, object]]:
    horizons = sorted({record.horizon for record in records})
    return [
        {
            "horizon": horizon,
            "performance": _conditioned_summary(
                [record for record in records if record.horizon == horizon],
                quantiles=quantiles,
            ),
            "interval_calibration": _interval_calibration(
                [record for record in records if record.horizon == horizon]
            ),
        }
        for horizon in horizons
    ]


def analyze_decisive_evidence(
    payload: Mapping[str, object],
    *,
    target_coverages: Sequence[float] = DEFAULT_TARGET_COVERAGES,
    regression_quantiles: Sequence[float] = DEFAULT_REGRESSION_QUANTILES,
    reliability_edges: Sequence[float] = DEFAULT_RELIABILITY_EDGES,
    reference_method: str | None = None,
) -> dict[str, object]:
    """Build matched operational and risk-coverage evidence diagnostics."""

    bundle = parse_decisive_evidence(payload)
    coverages = _validated_probabilities(
        target_coverages, name="target_coverages", include_endpoints=True
    )
    quantiles = _validated_probabilities(
        regression_quantiles,
        name="regression_quantiles",
        include_endpoints=False,
    )
    edges = _validated_reliability_edges(reliability_edges)
    resolved_reference = reference_method or bundle.reference_method
    if reference_method is not None:
        resolved_reference = _text(reference_method, name="reference_method")

    records_by_metric: dict[str, list[EvidenceRecord]] = {}
    for record in bundle.records:
        records_by_metric.setdefault(record.metric, []).append(record)

    metric_summaries: dict[str, object] = {}
    for metric, metric_records in sorted(records_by_metric.items()):
        records_by_method: dict[str, list[EvidenceRecord]] = {}
        for record in metric_records:
            records_by_method.setdefault(record.method, []).append(record)
        for records in records_by_method.values():
            records.sort(key=lambda record: record.unit_id)
        if resolved_reference is not None and resolved_reference not in records_by_method:
            raise ValueError(
                f"reference method {resolved_reference!r} is absent for metric {metric!r}"
            )

        method_summaries: dict[str, object] = {}
        for method, records in sorted(records_by_method.items()):
            method_summary: dict[str, object] = {
                "operational_policy": _operational_summary(
                    records, quantiles=quantiles
                ),
                "performance_by_horizon": _horizon_summary(
                    records, quantiles=quantiles
                ),
                "performance_by_reliability": _reliability_conditioning(
                    records, edges=edges, quantiles=quantiles
                ),
                "performance_by_identifiable_rank": _rank_conditioning(
                    records, quantiles=quantiles
                ),
            }
            if resolved_reference is None or method == resolved_reference:
                method_summary["operational_vs_reference_method"] = None
                method_summary["raw_vs_reference_method"] = None
            else:
                reference_records = records_by_method[resolved_reference]
                method_summary["operational_vs_reference_method"] = _pairwise_summary(
                    records,
                    reference_records,
                    deployed=True,
                    quantiles=quantiles,
                )
                method_summary["raw_vs_reference_method"] = _pairwise_summary(
                    records,
                    reference_records,
                    deployed=False,
                    quantiles=quantiles,
                )
            method_summaries[method] = method_summary

        first_method = next(iter(records_by_method))
        metric_summaries[metric] = {
            "unit_count": len(records_by_method[first_method]),
            "group_count": len(
                {record.group_id for record in records_by_method[first_method]}
            ),
            "methods": method_summaries,
            "matched_risk_coverage": _risk_coverage_curves(
                records_by_method,
                coverages=coverages,
                quantiles=quantiles,
                reference_method=resolved_reference,
            ),
        }

    return {
        "schema_version": 1,
        "contract": DECISIVE_EVIDENCE_SUMMARY_CONTRACT,
        "source_contract": DECISIVE_EVIDENCE_INPUT_CONTRACT,
        "protocol_id": bundle.protocol_id,
        "statistical_unit": bundle.statistical_unit,
        "claim_boundary": bundle.claim_boundary,
        "reference_method": resolved_reference,
        "analysis_configuration": {
            "target_coverages": list(coverages),
            "regression_quantiles": list(quantiles),
            "reliability_edges": list(edges),
            "risk_score_order": "lower_is_safer",
            "matched_fallback": True,
        },
        "metrics": metric_summaries,
    }
