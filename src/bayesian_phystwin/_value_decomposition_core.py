"""Core arithmetic and invariants for Bayesian-value decomposition."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np

from .decisive_evidence import EvidenceRecord


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
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


def _integer(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _close(left: float, right: float, tolerance: float) -> bool:
    scale = 1.0 + max(abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


def _records_by_metric_method(
    records: Sequence[EvidenceRecord],
) -> dict[str, dict[str, dict[str, EvidenceRecord]]]:
    result: dict[str, dict[str, dict[str, EvidenceRecord]]] = {}
    for record in records:
        result.setdefault(record.metric, {}).setdefault(record.method, {})[
            record.unit_id
        ] = record
    return result


def _loss_vector(
    records: Mapping[str, EvidenceRecord],
    units: Sequence[str],
    *,
    deployed: bool,
) -> np.ndarray:
    attribute = "deployed_loss" if deployed else "loss"
    return np.asarray(
        [getattr(records[unit], attribute) for unit in units],
        dtype=np.float64,
    )


def _equal_group_vector(
    records: Mapping[str, EvidenceRecord],
    groups: Sequence[str],
    *,
    deployed: bool,
) -> np.ndarray:
    attribute = "deployed_loss" if deployed else "loss"
    by_group: dict[str, list[float]] = {group: [] for group in groups}
    for record in records.values():
        by_group[record.group_id].append(float(getattr(record, attribute)))
    return np.asarray(
        [float(np.mean(by_group[group])) for group in groups],
        dtype=np.float64,
    )


def _comparison(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    baseline_method: str,
    candidate_method: str,
    tolerance: float,
) -> dict[str, object]:
    _require(
        baseline.shape == candidate.shape and baseline.ndim == 1,
        "paired comparison vectors changed shape",
    )
    ties = np.asarray(
        [
            _close(float(candidate_value), float(baseline_value), tolerance)
            for candidate_value, baseline_value in zip(
                candidate,
                baseline,
                strict=True,
            )
        ],
        dtype=bool,
    )
    wins = (~ties) & (candidate < baseline)
    losses = (~ties) & (candidate > baseline)
    baseline_mean = float(np.mean(baseline))
    candidate_mean = float(np.mean(candidate))
    mean_difference = candidate_mean - baseline_mean
    return {
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "paired_count": int(len(baseline)),
        "baseline_mean_loss": baseline_mean,
        "candidate_mean_loss": candidate_mean,
        "mean_loss_difference": mean_difference,
        "mean_improvement": -mean_difference,
        "relative_change_of_means": (
            None
            if baseline_mean <= 0.0
            else candidate_mean / baseline_mean - 1.0
        ),
        "wins": int(np.sum(wins)),
        "ties": int(np.sum(ties)),
        "losses": int(np.sum(losses)),
    }


def _float_field(value: Mapping[str, object], name: str) -> float:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise AssertionError(f"comparison field {name!r} changed type")
    return float(item)


def _decomposition(
    vectors: Mapping[str, np.ndarray],
    *,
    deterministic_reference: str,
    guarded_reference: str,
    bayesian_mean: str,
    full_belief: str,
    tolerance: float,
) -> dict[str, object]:
    arm_order = (
        deterministic_reference,
        guarded_reference,
        bayesian_mean,
        full_belief,
    )
    roles = ("uncertainty_and_guard", "bayesian_mean", "full_belief")
    steps: list[dict[str, object]] = []
    for role, baseline_method, candidate_method in zip(
        roles,
        arm_order[:-1],
        arm_order[1:],
        strict=True,
    ):
        steps.append(
            {
                "role": role,
                **_comparison(
                    vectors[baseline_method],
                    vectors[candidate_method],
                    baseline_method=baseline_method,
                    candidate_method=candidate_method,
                    tolerance=tolerance,
                ),
            }
        )
    total = _comparison(
        vectors[deterministic_reference],
        vectors[full_belief],
        baseline_method=deterministic_reference,
        candidate_method=full_belief,
        tolerance=tolerance,
    )
    total_improvement = _float_field(total, "mean_improvement")
    for step in steps:
        improvement = _float_field(step, "mean_improvement")
        step["fraction_of_total_mean_improvement"] = (
            None
            if abs(total_improvement) <= tolerance
            else improvement / total_improvement
        )
    residual = (
        sum(_float_field(step, "mean_loss_difference") for step in steps)
        - _float_field(total, "mean_loss_difference")
    )
    _require(
        abs(residual) <= tolerance * (1.0 + abs(total_improvement)),
        "decomposition failed its telescoping identity",
    )
    return {
        "steps": steps,
        "total": total,
        "telescoping_mean_loss_difference_residual": residual,
    }


def _interval_widths_match(
    left: EvidenceRecord,
    right: EvidenceRecord,
    tolerance: float,
) -> bool:
    if len(left.intervals) != len(right.intervals):
        return False
    for left_interval, right_interval in zip(
        left.intervals,
        right.intervals,
        strict=True,
    ):
        if left_interval.nominal_coverage != right_interval.nominal_coverage:
            return False
        if not _close(left_interval.width, right_interval.width, tolerance):
            return False
    return True


__all__ = [
    "_canonical_json_sha256",
    "_decomposition",
    "_equal_group_vector",
    "_integer",
    "_interval_widths_match",
    "_loss_vector",
    "_number",
    "_records_by_metric_method",
    "_require",
    "_text",
]
