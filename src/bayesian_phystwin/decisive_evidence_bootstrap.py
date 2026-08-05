"""Equal-group paired bootstrap for decisive-evidence summaries."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from .decisive_evidence import EvidenceRecord, parse_decisive_evidence

GROUP_CLUSTERED_BOOTSTRAP_CONTRACT = (
    "bayesian-phystwin-group-clustered-paired-bootstrap-v1"
)
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260805
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95


def _integer(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be a finite number in (0, 1)")
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("confidence must be a finite number in (0, 1)")
    return result


def _metric_rng(seed: int, metric: str) -> np.random.Generator:
    digest = hashlib.sha256(metric.encode("utf-8")).digest()
    entropy = [
        seed,
        int.from_bytes(digest[:4], "big"),
        int.from_bytes(digest[4:8], "big"),
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _group_loss_vectors(
    records: Sequence[EvidenceRecord],
) -> dict[str, dict[str, float]]:
    by_group: dict[str, list[EvidenceRecord]] = {}
    for record in records:
        by_group.setdefault(record.group_id, []).append(record)
    return {
        group_id: {
            "raw": float(np.mean([record.loss for record in group_records])),
            "deployed": float(
                np.mean([record.deployed_loss for record in group_records])
            ),
            "fallback": float(
                np.mean([record.fallback_loss for record in group_records])
            ),
            "registered_unit_count": len(group_records),
        }
        for group_id, group_records in sorted(by_group.items())
    }


def _interval(samples: np.ndarray, confidence: float) -> dict[str, float]:
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha), method="linear")
    return {
        "confidence": confidence,
        "lower": float(lower),
        "upper": float(upper),
    }


def _comparison(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    sample_indices: np.ndarray,
    confidence: float,
) -> dict[str, object]:
    candidate_mean = float(np.mean(candidate))
    baseline_mean = float(np.mean(baseline))
    observed_difference = candidate_mean - baseline_mean
    observed_relative = (
        None if baseline_mean <= 0.0 else candidate_mean / baseline_mean - 1.0
    )
    observed = {
        "candidate_mean_loss": candidate_mean,
        "baseline_mean_loss": baseline_mean,
        "mean_loss_difference": observed_difference,
        "relative_change_of_means": observed_relative,
    }

    if len(candidate) < 2:
        return {
            "status": "insufficient_independent_groups",
            "observed": observed,
            "mean_loss_difference_interval": None,
            "relative_change_of_means_interval": None,
            "bootstrap_probability_candidate_better": None,
            "valid_relative_change_replicates": 0,
        }

    sampled_candidate = np.mean(candidate[sample_indices], axis=1)
    sampled_baseline = np.mean(baseline[sample_indices], axis=1)
    differences = sampled_candidate - sampled_baseline
    valid_relative = sampled_baseline > 0.0
    relative_samples = (
        sampled_candidate[valid_relative] / sampled_baseline[valid_relative] - 1.0
    )
    return {
        "status": "complete",
        "observed": observed,
        "mean_loss_difference_interval": _interval(differences, confidence),
        "relative_change_of_means_interval": (
            None
            if not len(relative_samples)
            else _interval(relative_samples, confidence)
        ),
        "bootstrap_probability_candidate_better": float(np.mean(differences < 0.0)),
        "valid_relative_change_replicates": int(len(relative_samples)),
    }


def group_clustered_paired_bootstrap(
    payload: Mapping[str, object],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
    reference_method: str | None = None,
) -> dict[str, object]:
    """Bootstrap equal-weight group means with paired group resampling.

    Each independent ``group_id`` receives equal weight. Multiple horizons or
    registered units inside one group are averaged before resampling, so frames,
    points, tracks, and repeated rows cannot inflate the effective sample size.
    The same sampled group indices are used for every method and comparison.
    """

    replicate_count = _integer(replicates, name="replicates", minimum=1)
    bootstrap_seed = _integer(seed, name="seed", minimum=0)
    interval_confidence = _confidence(confidence)
    bundle = parse_decisive_evidence(payload)
    resolved_reference = reference_method or bundle.reference_method
    if reference_method is not None:
        if not isinstance(reference_method, str) or not reference_method.strip():
            raise ValueError("reference_method must be a nonempty string")
        resolved_reference = reference_method.strip()

    records_by_metric: dict[str, list[EvidenceRecord]] = {}
    for record in bundle.records:
        records_by_metric.setdefault(record.metric, []).append(record)

    metrics: dict[str, object] = {}
    for metric, metric_records in sorted(records_by_metric.items()):
        records_by_method: dict[str, list[EvidenceRecord]] = {}
        for record in metric_records:
            records_by_method.setdefault(record.method, []).append(record)
        group_losses = {
            method: _group_loss_vectors(records)
            for method, records in sorted(records_by_method.items())
        }
        first_method = next(iter(group_losses))
        group_ids = tuple(group_losses[first_method])
        for method, losses in group_losses.items():
            if tuple(losses) != group_ids:
                raise AssertionError(
                    f"validated group sets diverged for {metric}/{method}"
                )
        if resolved_reference is not None and resolved_reference not in group_losses:
            raise ValueError(
                f"reference method {resolved_reference!r} is absent for "
                f"metric {metric!r}"
            )

        group_count = len(group_ids)
        rng = _metric_rng(bootstrap_seed, metric)
        sample_indices = rng.integers(
            0,
            group_count,
            size=(replicate_count, group_count),
            endpoint=False,
        )
        fallback = np.asarray(
            [
                group_losses[first_method][group_id]["fallback"]
                for group_id in group_ids
            ],
            dtype=float,
        )
        method_summaries: dict[str, object] = {}
        for method, losses in sorted(group_losses.items()):
            raw = np.asarray(
                [losses[group_id]["raw"] for group_id in group_ids], dtype=float
            )
            deployed = np.asarray(
                [losses[group_id]["deployed"] for group_id in group_ids],
                dtype=float,
            )
            if not np.array_equal(
                fallback,
                np.asarray(
                    [losses[group_id]["fallback"] for group_id in group_ids],
                    dtype=float,
                ),
            ):
                raise AssertionError("validated fallback group means diverged")
            summary: dict[str, object] = {
                "registered_unit_counts_by_group": {
                    group_id: int(losses[group_id]["registered_unit_count"])
                    for group_id in group_ids
                },
                "raw_vs_fallback": _comparison(
                    raw,
                    fallback,
                    sample_indices=sample_indices,
                    confidence=interval_confidence,
                ),
                "deployed_vs_fallback": _comparison(
                    deployed,
                    fallback,
                    sample_indices=sample_indices,
                    confidence=interval_confidence,
                ),
            }
            if resolved_reference is None or method == resolved_reference:
                summary["raw_vs_reference_method"] = None
                summary["deployed_vs_reference_method"] = None
            else:
                reference = group_losses[resolved_reference]
                reference_raw = np.asarray(
                    [reference[group_id]["raw"] for group_id in group_ids],
                    dtype=float,
                )
                reference_deployed = np.asarray(
                    [reference[group_id]["deployed"] for group_id in group_ids],
                    dtype=float,
                )
                summary["raw_vs_reference_method"] = {
                    "reference_method": resolved_reference,
                    **_comparison(
                        raw,
                        reference_raw,
                        sample_indices=sample_indices,
                        confidence=interval_confidence,
                    ),
                }
                summary["deployed_vs_reference_method"] = {
                    "reference_method": resolved_reference,
                    **_comparison(
                        deployed,
                        reference_deployed,
                        sample_indices=sample_indices,
                        confidence=interval_confidence,
                    ),
                }
            method_summaries[method] = summary

        metrics[metric] = {
            "group_count": group_count,
            "group_ids": list(group_ids),
            "minimum_independent_group_requirement_met": group_count >= 2,
            "methods": method_summaries,
        }

    return {
        "contract": GROUP_CLUSTERED_BOOTSTRAP_CONTRACT,
        "role": "paired_equal_group_uncertainty",
        "protocol_id": bundle.protocol_id,
        "statistical_unit": bundle.statistical_unit,
        "reference_method": resolved_reference,
        "replicates": replicate_count,
        "seed": bootstrap_seed,
        "confidence": interval_confidence,
        "resampling_unit": "group_id",
        "group_weighting": "equal",
        "within_group_aggregation": "mean_over_registered_units",
        "pairing": "shared_sampled_group_indices_across_methods",
        "metrics": metrics,
    }
