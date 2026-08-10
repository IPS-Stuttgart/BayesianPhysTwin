"""Paired group-bootstrap views for Bayesian-value decomposition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .decisive_evidence_bootstrap import group_clustered_paired_bootstrap


def _bootstrap_comparison(
    bootstrap: Mapping[str, object],
    *,
    metric: str,
    candidate_method: str,
    deployed: bool,
) -> object:
    metrics = bootstrap.get("metrics")
    if not isinstance(metrics, Mapping):
        raise AssertionError("bootstrap metrics changed type")
    metric_value = metrics.get(metric)
    if not isinstance(metric_value, Mapping):
        raise AssertionError("bootstrap metric changed type")
    methods = metric_value.get("methods")
    if not isinstance(methods, Mapping):
        raise AssertionError("bootstrap methods changed type")
    method_value = methods.get(candidate_method)
    if not isinstance(method_value, Mapping):
        raise AssertionError("bootstrap method changed type")
    key = (
        "deployed_vs_reference_method"
        if deployed
        else "raw_vs_reference_method"
    )
    return method_value.get(key)


def _bootstrap_decomposition(
    payload: Mapping[str, object],
    metrics: Sequence[str],
    *,
    deterministic_reference: str,
    guarded_reference: str,
    bayesian_mean: str,
    full_belief: str,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    by_reference = {
        deterministic_reference: group_clustered_paired_bootstrap(
            payload,
            replicates=replicates,
            seed=seed,
            confidence=confidence,
            reference_method=deterministic_reference,
        ),
        guarded_reference: group_clustered_paired_bootstrap(
            payload,
            replicates=replicates,
            seed=seed,
            confidence=confidence,
            reference_method=guarded_reference,
        ),
        bayesian_mean: group_clustered_paired_bootstrap(
            payload,
            replicates=replicates,
            seed=seed,
            confidence=confidence,
            reference_method=bayesian_mean,
        ),
    }
    steps = (
        (
            "uncertainty_and_guard",
            deterministic_reference,
            guarded_reference,
        ),
        ("bayesian_mean", guarded_reference, bayesian_mean),
        ("full_belief", bayesian_mean, full_belief),
    )
    output: dict[str, object] = {}
    for metric in metrics:
        views: dict[str, object] = {}
        for deployed in (False, True):
            label = "deployed" if deployed else "raw"
            views[label] = {
                "steps": [
                    {
                        "role": role,
                        "comparison": _bootstrap_comparison(
                            by_reference[reference],
                            metric=metric,
                            candidate_method=candidate,
                            deployed=deployed,
                        ),
                    }
                    for role, reference, candidate in steps
                ],
                "total": _bootstrap_comparison(
                    by_reference[deterministic_reference],
                    metric=metric,
                    candidate_method=full_belief,
                    deployed=deployed,
                ),
            }
        output[metric] = views
    return output


__all__ = ["_bootstrap_decomposition"]
