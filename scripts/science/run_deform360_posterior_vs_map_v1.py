#!/usr/bin/env python3
"""Compare a frozen Deform360 generalized-Bayes mixture with its MAP plug-in.

This is a retrospective mechanism ablation on the exact parent-bound 92-object
Deform360 cohort.  Candidate weights, predictors, source episodes, target
prefixes, recorded actions, query definitions, and target futures are fixed.
All bias, variance, and event-threshold calibration uses source episodes only.
No robot action is selected and no new data are opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-posterior-vs-map-result-v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-posterior-vs-map-protocol-v1"
ARMS = ("posterior_mixture", "posterior_mean_gaussian", "map_gaussian")
METRICS = (
    "nll",
    "crps",
    "brier",
    "log_loss",
    "decision_loss",
    "decision_regret",
    "acceptance",
    "harm_all",
    "harm_given_accept",
    "query_rmse",
)
EPS = 1e-12
LOG2PI = math.log(2.0 * math.pi)
SQRT2PI = math.sqrt(2.0 * math.pi)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(array.dtype.str.encode())
    result.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    result.update(array.tobytes())
    return result.hexdigest()


def mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def validate_protocol(protocol: Mapping[str, Any], data_root: Path, v6_path: Path) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("schema_version") != 1:
        raise ValueError("protocol identity changed")
    if protocol.get("status") != "frozen-before-retrospective-execution":
        raise ValueError("protocol is not frozen")
    if Path(str(protocol.get("dataset_root"))) != data_root:
        raise ValueError("dataset root changed")
    evaluation = mapping(protocol.get("evaluation"), "evaluation")
    if tuple(evaluation.get("arms", ())) != ARMS:
        raise ValueError("arm roster changed")
    if evaluation.get("inferential_unit") != "physical object":
        raise ValueError("inferential unit changed")
    if evaluation.get("source_only_calibration") is not True:
        raise ValueError("calibration boundary changed")
    bindings = mapping(protocol.get("bindings"), "bindings")
    if file_sha256(v6_path) != bindings.get("original_v6_runner_sha256"):
        raise ValueError("original v6 runner bytes changed")
    boundary = mapping(protocol.get("information_boundary"), "boundary")
    forbidden = (
        "new_measurements_collected",
        "robot_actions_selected",
        "target_outcomes_may_tune_protocol",
        "target_outcomes_may_select_arms",
        "camera_pixels_may_open",
        "geometry_or_point_cloud_may_open",
    )
    if any(boundary.get(name) is not False for name in forbidden):
        raise ValueError("protocol opens a forbidden boundary")
    if protocol.get("paper_claim_authorized") is not False:
        raise ValueError("protocol may not authorize a paper claim")


def normal_cdf(x: np.ndarray | float) -> np.ndarray:
    """Deterministic normal-CDF approximation with sub-1e-7 absolute error."""
    value = np.asarray(x, dtype=np.float64)
    z = np.abs(value)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    positive = 1.0 - np.exp(-0.5 * z * z) * poly / SQRT2PI
    return np.where(value >= 0.0, positive, 1.0 - positive)


def normal_pdf(x: np.ndarray | float) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * value * value) / SQRT2PI


def logsumexp(value: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(value, axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.sum(np.exp(value - maximum), axis=axis, keepdims=True)),
        axis=axis,
    )


def gaussian_nll(truth: np.ndarray, mean: np.ndarray, variance: float) -> np.ndarray:
    variance = max(float(variance), EPS)
    return 0.5 * (LOG2PI + math.log(variance) + (truth - mean) ** 2 / variance)


def mixture_nll(
    truth: np.ndarray, components: np.ndarray, weights: np.ndarray, variance: float
) -> np.ndarray:
    variance = max(float(variance), EPS)
    log_terms = np.log(np.maximum(weights, np.finfo(float).tiny))[:, None]
    log_terms = log_terms - 0.5 * (
        LOG2PI + math.log(variance) + (truth[None] - components) ** 2 / variance
    )
    return -logsumexp(log_terms, axis=0)


def absolute_normal(delta: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), math.sqrt(EPS))
    z = delta / sigma
    return 2.0 * sigma * normal_pdf(z) + delta * (2.0 * normal_cdf(z) - 1.0)


def gaussian_crps(truth: np.ndarray, mean: np.ndarray, variance: float) -> np.ndarray:
    sigma = math.sqrt(max(float(variance), EPS))
    return absolute_normal(truth - mean, sigma) - sigma / math.sqrt(math.pi)


def mixture_crps(
    truth: np.ndarray, components: np.ndarray, weights: np.ndarray, variance: float
) -> np.ndarray:
    sigma = math.sqrt(max(float(variance), EPS))
    first = np.einsum("k,kn->n", weights, absolute_normal(truth[None] - components, sigma))
    pairwise = components[:, None] - components[None, :]
    second = 0.5 * np.einsum(
        "k,l,kln->n", weights, weights, absolute_normal(pairwise, math.sqrt(2.0) * sigma)
    )
    return first - second


def gaussian_event(mean: np.ndarray, variance: float, threshold: float, event: str) -> np.ndarray:
    sigma = math.sqrt(max(float(variance), EPS))
    if event == "upper":
        return 1.0 - normal_cdf((threshold - mean) / sigma)
    if event == "absolute":
        return normal_cdf((-threshold - mean) / sigma) + 1.0 - normal_cdf(
            (threshold - mean) / sigma
        )
    raise ValueError(f"unknown event {event}")


def mixture_event(
    components: np.ndarray,
    weights: np.ndarray,
    variance: float,
    threshold: float,
    event: str,
) -> np.ndarray:
    return np.einsum("k,kn->n", weights, gaussian_event(components, variance, threshold, event))


def labels(truth: np.ndarray, threshold: float, event: str) -> np.ndarray:
    if event == "upper":
        return truth > threshold
    if event == "absolute":
        return np.abs(truth) > threshold
    raise ValueError(f"unknown event {event}")


def equal_episode_mean(values: Sequence[np.ndarray]) -> float:
    return float(np.mean([np.mean(np.asarray(value, dtype=np.float64)) for value in values]))


def fit_gaussian_variance(
    truths: Sequence[np.ndarray], means: Sequence[np.ndarray], floor: float
) -> float:
    return max(
        equal_episode_mean([(truth - mean) ** 2 for truth, mean in zip(truths, means, strict=True)]),
        floor,
    )


def fit_mixture_variance(
    truths: Sequence[np.ndarray],
    components: Sequence[np.ndarray],
    weights: np.ndarray,
    floor: float,
    grid_count: int,
) -> dict[str, Any]:
    means = [np.einsum("k,kn->n", weights, item) for item in components]
    mean_mse = fit_gaussian_variance(truths, means, floor)
    between = equal_episode_mean(
        [
            np.einsum("k,kn->n", weights, (item - mean[None]) ** 2)
            for item, mean in zip(components, means, strict=True)
        ]
    )
    start = max(mean_mse - between, floor)
    low = max(floor, min(start, mean_mse) * 1e-3)
    high = max(start, mean_mse, between, floor) * 1e3
    high = max(high, low * 1e6)

    def score(variance: float) -> float:
        return equal_episode_mean(
            [
                mixture_nll(truth, item, weights, variance)
                for truth, item in zip(truths, components, strict=True)
            ]
        )

    coarse = np.geomspace(low, high, grid_count)
    coarse_scores = np.asarray([score(float(value)) for value in coarse])
    index = int(np.argmin(coarse_scores))
    fine = np.geomspace(coarse[max(0, index - 1)], coarse[min(len(coarse) - 1, index + 1)], grid_count)
    fine_scores = np.asarray([score(float(value)) for value in fine])
    selected = int(np.argmin(fine_scores))
    return {
        "variance": float(fine[selected]),
        "source_nll": float(fine_scores[selected]),
        "mean_mse": mean_mse,
        "between_variance": between,
        "coarse_boundary": index in {0, len(coarse) - 1},
        "fine_boundary": selected in {0, len(fine) - 1},
    }


def score_distribution(
    truth: np.ndarray,
    mean: np.ndarray,
    variance: float,
    threshold: float,
    event: str,
    fallback_cost: float,
    probability_clip: float,
    components: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    observed = labels(truth, threshold, event)
    observed_float = observed.astype(np.float64)
    if components is None:
        nll = gaussian_nll(truth, mean, variance)
        crps = gaussian_crps(truth, mean, variance)
        probability = gaussian_event(mean, variance, threshold, event)
    else:
        if weights is None:
            raise ValueError("mixture weights missing")
        nll = mixture_nll(truth, components, weights, variance)
        crps = mixture_crps(truth, components, weights, variance)
        probability = mixture_event(components, weights, variance, threshold, event)
    probability = np.clip(probability, probability_clip, 1.0 - probability_clip)
    accept = probability <= fallback_cost
    loss = np.where(accept, observed_float, fallback_cost)
    oracle = np.where(observed, fallback_cost, 0.0)
    accepted = int(np.count_nonzero(accept))
    harmful = int(np.count_nonzero(accept & observed))
    return {
        "nll": float(np.mean(nll)),
        "crps": float(np.mean(crps)),
        "brier": float(np.mean((probability - observed_float) ** 2)),
        "log_loss": float(
            -np.mean(
                observed_float * np.log(probability)
                + (1.0 - observed_float) * np.log1p(-probability)
            )
        ),
        "decision_loss": float(np.mean(loss)),
        "decision_regret": float(np.mean(loss - oracle)),
        "acceptance": float(np.mean(accept)),
        "harm_all": float(harmful / len(observed)),
        "harm_given_accept": float(harmful / accepted if accepted else 0.0),
        "query_rmse": float(np.sqrt(np.mean((truth - mean) ** 2))),
        "event_rate": float(np.mean(observed_float)),
    }


def source_components(source: Sequence[Any], candidates: Sequence[Any], weights: np.ndarray) -> list[dict[str, Any]]:
    raw: dict[int, np.ndarray] = {}
    truth: dict[int, np.ndarray] = {}
    residual: dict[int, np.ndarray] = {}
    for episode in source:
        episode_id = int(episode.descriptor.episode_id)
        raw[episode_id] = np.stack([candidate.cv_predictions[episode_id] for candidate in candidates])
        truth[episode_id] = np.asarray(candidates[0].cv_truths[episode_id], dtype=np.float64)
        residual[episode_id] = truth[episode_id] - np.einsum("k,knd->nd", weights, raw[episode_id])
    rows = []
    for episode in source:
        episode_id = int(episode.descriptor.episode_id)
        donor = np.concatenate([value for key, value in residual.items() if key != episode_id]).mean(axis=0)
        adjusted = raw[episode_id] + donor[None, None]
        expected = truth[episode_id] - (residual[episode_id] - donor[None])
        rows.append(
            {
                "episode_id": episode_id,
                "truth": truth[episode_id],
                "raw": raw[episode_id],
                "adjusted": adjusted,
                "parity": float(np.max(np.abs(np.einsum("k,knd->nd", weights, adjusted) - expected))),
            }
        )
    return rows


def evaluate_with_components(v3: Any, v6: Any, descriptors: list[Any], development: dict[str, Any], base_protocol: dict[str, Any], rng: np.random.Generator):
    captured: dict[str, np.ndarray] = {}
    original = v3.target_candidate_prediction

    def wrapper(candidate: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        prediction = original(candidate, *args, **kwargs)
        captured[str(candidate.name)] = np.asarray(prediction, dtype=np.float64).copy()
        return prediction

    v3.target_candidate_prediction = wrapper
    try:
        row, capture, _, target_truth = v6.evaluate_object_with_capture(
            v3, descriptors, development, base_protocol, rng
        )
    finally:
        v3.target_candidate_prediction = original
    candidates = list(capture.candidates)
    weights = np.asarray(capture.weights, dtype=np.float64)
    names = [str(candidate.name) for candidate in candidates]
    if any(name not in captured for name in names):
        raise RuntimeError("target component capture incomplete")
    raw_target = np.stack([captured[name] for name in names])
    frozen_mean = np.asarray(target_truth) - np.asarray(capture.target_errors)
    raw_mean = np.einsum("k,knd->nd", weights, raw_target)
    adjusted_target = raw_target + (frozen_mean - raw_mean)[None]
    return row, capture, np.asarray(target_truth), frozen_mean, raw_target, adjusted_target, source_components(capture.source, candidates, weights)


def bootstrap(values: np.ndarray, repetitions: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    return [float(value) for value in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])]


def paired(rows: Sequence[Mapping[str, Any]], metric: str, left: str, right: str, repetitions: int, seed: int) -> dict[str, Any]:
    values = np.asarray([row["arm_summary"][left][metric] - row["arm_summary"][right][metric] for row in rows])
    interval = bootstrap(values, repetitions, seed)
    return {
        "metric": metric,
        "left": left,
        "right": right,
        "mean_difference": float(np.mean(values)),
        "median_difference": float(np.median(values)),
        "bootstrap_95": interval,
        "wins_ties_losses": [
            int(np.count_nonzero(values < 0.0)),
            int(np.count_nonzero(values == 0.0)),
            int(np.count_nonzero(values > 0.0)),
        ],
        "supported": bool(np.mean(values) < 0.0 and interval[1] < 0.0),
    }


def aggregate(rows: list[dict[str, Any]], protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation = mapping(protocol["evaluation"], "evaluation")
    repetitions = int(evaluation["bootstrap_repetitions"])
    seed = int(evaluation["random_seed"])
    arms = {
        arm: {metric: float(np.mean([row["arm_summary"][arm][metric] for row in rows])) for metric in METRICS}
        for arm in ARMS
    }
    comparisons: dict[str, Any] = {}
    index = 0
    for right in ("map_gaussian", "posterior_mean_gaussian"):
        comparisons[right] = {}
        for metric in ("nll", "crps", "brier", "decision_loss"):
            comparisons[right][metric] = paired(
                rows, metric, "posterior_mixture", right, repetitions, seed + index
            )
            index += 1
    field_values = np.asarray([row["point"]["posterior_mean_field_rmse"] - row["point"]["map_field_rmse"] for row in rows])
    field_interval = bootstrap(field_values, repetitions, seed + 100)
    field = {
        "posterior_mean": float(np.mean([row["point"]["posterior_mean_field_rmse"] for row in rows])),
        "map": float(np.mean([row["point"]["map_field_rmse"] for row in rows])),
        "mean_difference": float(np.mean(field_values)),
        "bootstrap_95": field_interval,
        "wins_ties_losses": [
            int(np.count_nonzero(field_values < 0.0)),
            int(np.count_nonzero(field_values == 0.0)),
            int(np.count_nonzero(field_values > 0.0)),
        ],
        "supported": bool(np.mean(field_values) < 0.0 and field_interval[1] < 0.0),
    }
    invariants = {
        "all_92_objects": len(rows) == 92,
        "parent_point_exact": all(row["parent_point_exact"] for row in rows),
        "weights_normalized": all(abs(row["weight_sum"] - 1.0) <= 1e-12 for row in rows),
        "target_mean_parity": max(row["target_mean_parity"] for row in rows) <= float(evaluation["mean_parity_tolerance"]),
        "source_mean_parity": max(row["source_mean_parity"] for row in rows) <= float(evaluation["mean_parity_tolerance"]),
    }
    mixture_map = comparisons["map_gaussian"]
    marginalization = bool(
        all(invariants.values())
        and mixture_map["nll"]["supported"]
        and mixture_map["crps"]["supported"]
    )
    decision_value = bool(
        all(invariants.values())
        and mixture_map["brier"]["supported"]
        and mixture_map["decision_loss"]["supported"]
    )
    same_mean = bool(
        all(invariants.values())
        and comparisons["posterior_mean_gaussian"]["nll"]["supported"]
        and comparisons["posterior_mean_gaussian"]["crps"]["supported"]
    )
    if not all(invariants.values()):
        classification = "invalid-invariant-failure"
    elif marginalization:
        classification = "positive-posterior-marginalization"
    elif any(item[metric]["supported"] for item in comparisons.values() for metric in item):
        classification = "mixed-posterior-marginalization"
    else:
        classification = "negative-posterior-marginalization"
    summary = {
        "object_count": len(rows),
        "query_count": len(rows[0]["queries"]),
        "arm_summary": arms,
        "comparisons": comparisons,
        "posterior_mean_vs_map_field_rmse": field,
        "mean_maximum_weight": float(np.mean([row["maximum_weight"] for row in rows])),
        "mean_effective_model_count": float(np.mean([row["effective_model_count"] for row in rows])),
        "maximum_target_mean_parity": float(max(row["target_mean_parity"] for row in rows)),
        "maximum_source_mean_parity": float(max(row["source_mean_parity"] for row in rows)),
    }
    decision = {
        "classification": classification,
        "invariants": invariants,
        "invariants_passed": all(invariants.values()),
        "posterior_marginalization_supported": marginalization,
        "decision_value_supported": decision_value,
        "same_mean_distribution_supported": same_mean,
        "model_averaging_mean_supported": field["supported"],
        "paper_claim_authorized": False,
        "fresh_confirmation_authorized": False,
        "robot_control_claim_authorized": False,
    }
    return summary, decision


def report(result: Mapping[str, Any]) -> str:
    summary = mapping(result["summary"], "summary")
    decision = mapping(result["decision"], "decision")
    lines = [
        "# Deform360 posterior predictive versus MAP v1",
        "",
        f"- Objects: **{summary['object_count']}**",
        f"- Queries per object: **{summary['query_count']}**",
        f"- Classification: **{decision['classification']}**",
        f"- Posterior marginalization supported: **{str(decision['posterior_marginalization_supported']).lower()}**",
        f"- Decision value supported: **{str(decision['decision_value_supported']).lower()}**",
        f"- Same-mean distribution value supported: **{str(decision['same_mean_distribution_supported']).lower()}**",
        "- Robot probing/action selection: **none**",
        "",
        "| Arm | NLL | CRPS | Brier | Decision loss | Query RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        value = summary["arm_summary"][arm]
        lines.append(
            f"| `{arm}` | {value['nll']:.6g} | {value['crps']:.6g} | {value['brier']:.6g} | {value['decision_loss']:.6g} | {value['query_rmse']:.6g} |"
        )
    lines += ["", "Negative paired differences favor the posterior mixture.", ""]
    for comparator, metrics in summary["comparisons"].items():
        lines += [f"## Mixture versus `{comparator}`", ""]
        for metric, value in metrics.items():
            ci = value["bootstrap_95"]
            w, t, loss = value["wins_ties_losses"]
            lines.append(
                f"- `{metric}`: {value['mean_difference']:.6g}, 95% object bootstrap [{ci[0]:.6g}, {ci[1]:.6g}], W/T/L {w}/{t}/{loss}."
            )
        lines.append("")
    lines += [
        "This is retrospective mechanism evidence for a generalized-Bayes model",
        "posterior. It is not fresh confirmation, a physical-parameter posterior,",
        "active robot improvement, unseen-object transfer, or deployment safety.",
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "object_id",
        "target_episode_id",
        "target_action_family",
        "maximum_weight",
        "effective_model_count",
        "map_candidate",
        "posterior_mean_field_rmse",
        "map_field_rmse",
    ] + [f"{arm}_{metric}" for arm in ARMS for metric in ("nll", "crps", "brier", "decision_loss", "query_rmse")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = {
                "object_id": row["object_id"],
                "target_episode_id": row["target_episode_id"],
                "target_action_family": row["target_action_family"],
                "maximum_weight": row["maximum_weight"],
                "effective_model_count": row["effective_model_count"],
                "map_candidate": row["map_candidate"],
                "posterior_mean_field_rmse": row["point"]["posterior_mean_field_rmse"],
                "map_field_rmse": row["point"]["map_field_rmse"],
            }
            for arm in ARMS:
                for metric in ("nll", "crps", "brier", "decision_loss", "query_rmse"):
                    value[f"{arm}_{metric}"] = row["arm_summary"][arm][metric]
            writer.writerow(value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve(strict=True)
    protocol = read_json(args.protocol)
    validate_protocol(protocol, data_root, args.original_v6_runner.resolve(strict=True))
    v6 = load_module(args.original_v6_runner.resolve(), "posterior_map_v6")
    recovery = load_module(args.recovery_runner.resolve(strict=True), "posterior_map_recovery")
    parent_control = args.parent_control_root.resolve(strict=True)
    frozen_root = args.frozen_root.resolve(strict=True)
    original_protocol = v6.read_json(args.original_v6_protocol)
    parent_protocol = v6.read_json(args.parent_protocol)
    parent_result = v6.read_json(args.parent_result)
    v6.validate_protocol(
        original_protocol,
        parent_control_root=parent_control,
        parent_protocol_path=args.parent_protocol,
        data_root=data_root,
    )
    parent_by_object = v6.validate_parent_result(parent_result, original_protocol, args.parent_result)
    binding = original_protocol["parent_confirmation"]
    v5 = v6.load_module(parent_control / binding["runner_path"], "posterior_map_parent")
    manifest = v5.verify_readiness(v6.read_json(args.readiness_json), parent_protocol, args.readiness_json)
    v3, development, base_protocol = v5.validate_frozen_method(frozen_root, parent_protocol)
    audit = v6.load_module(parent_control / binding["audit_path"], "posterior_map_audit")
    minimum = int(parent_protocol["selection"]["minimum_complete_episodes_per_object"])
    evaluation = protocol["evaluation"]
    rng = np.random.default_rng(int(development["statistics"]["random_seed"]))
    rows = []
    drift_rows = []
    for index, expected in enumerate(manifest, start=1):
        object_id = str(expected["object_id"])
        print(f"[{index}/{len(manifest)}] posterior-vs-MAP {object_id}", flush=True)
        parent_row = parent_by_object[object_id]
        descriptors, drift = recovery.build_bound_descriptors(
            v3=v3,
            v5=v5,
            audit=audit,
            data_root=data_root,
            expected=expected,
            parent_row=parent_row,
            minimum_episodes=minimum,
        )
        drift_rows.append(drift)
        row, capture, truth, posterior_mean, raw_target, components, sources = evaluate_with_components(
            v3, v6, descriptors, development, base_protocol, rng
        )
        exact = v6.point_projection(row) == v6.point_projection(parent_row)
        if not exact:
            raise RuntimeError(f"parent point result changed: {object_id}")
        weights = np.asarray(capture.weights, dtype=np.float64)
        if abs(float(np.sum(weights)) - 1.0) > 1e-12:
            raise RuntimeError(f"weights not normalized: {object_id}")
        map_index = int(np.argmax(weights))

        raw_map_residuals = {
            int(source["episode_id"]): source["truth"] - source["raw"][map_index]
            for source in sources
        }
        map_source_means = []
        for source in sources:
            episode_id = int(source["episode_id"])
            donor = np.concatenate(
                [value for key, value in raw_map_residuals.items() if key != episode_id]
            ).mean(axis=0)
            map_source_means.append(source["raw"][map_index] + donor[None])
        map_bias = np.concatenate(list(raw_map_residuals.values())).mean(axis=0)
        clip = float(base_protocol["model"]["normalized_feature_clip"])
        map_mean = np.clip(raw_target[map_index] + map_bias[None], 0.0, clip)

        queries = {}
        for query_name, (query_weight, event) in v6.query_bank(truth.shape[1]).items():
            target_truth = truth @ query_weight
            target_components = np.einsum("knd,d->kn", components, query_weight)
            target_posterior_mean = posterior_mean @ query_weight
            target_map_mean = map_mean @ query_weight
            source_truths = [source["truth"] @ query_weight for source in sources]
            source_components_list = [
                np.einsum("knd,d->kn", source["adjusted"], query_weight)
                for source in sources
            ]
            source_posterior_means = [
                np.einsum("k,kn->n", weights, item) for item in source_components_list
            ]
            source_map_means = [item @ query_weight for item in map_source_means]
            threshold_source = np.concatenate(
                [np.abs(item) if event == "absolute" else item for item in source_truths]
            )
            threshold = float(np.quantile(threshold_source, evaluation["event_threshold_quantile"]))
            mixture_fit = fit_mixture_variance(
                source_truths,
                source_components_list,
                weights,
                float(evaluation["minimum_variance"]),
                int(evaluation["variance_grid_count"]),
            )
            posterior_gaussian_variance = fit_gaussian_variance(
                source_truths, source_posterior_means, float(evaluation["minimum_variance"])
            )
            map_variance = fit_gaussian_variance(
                source_truths, source_map_means, float(evaluation["minimum_variance"])
            )
            common = {
                "truth": target_truth,
                "threshold": threshold,
                "event": event,
                "fallback_cost": float(evaluation["fallback_cost"]),
                "probability_clip": float(evaluation["probability_clip"]),
            }
            arm_values = {
                "posterior_mixture": score_distribution(
                    **common,
                    mean=target_posterior_mean,
                    variance=float(mixture_fit["variance"]),
                    components=target_components,
                    weights=weights,
                ),
                "posterior_mean_gaussian": score_distribution(
                    **common,
                    mean=target_posterior_mean,
                    variance=posterior_gaussian_variance,
                ),
                "map_gaussian": score_distribution(
                    **common,
                    mean=target_map_mean,
                    variance=map_variance,
                ),
            }
            queries[query_name] = {
                "event": event,
                "weight_sha256": v6.array_digest(query_weight),
                "threshold": threshold,
                "mixture_calibration": mixture_fit,
                "posterior_gaussian_variance": posterior_gaussian_variance,
                "map_gaussian_variance": map_variance,
                "arms": arm_values,
            }
        arm_summary = {
            arm: {
                metric: float(np.mean([value["arms"][arm][metric] for value in queries.values()]))
                for metric in METRICS
            }
            for arm in ARMS
        }
        target_parity = float(
            np.max(np.abs(np.einsum("k,knd->nd", weights, components) - posterior_mean))
        )
        rows.append(
            {
                "object_id": object_id,
                "target_episode_id": row["target_episode_id"],
                "target_action": row["target_action"],
                "target_action_family": row["target_action_family"],
                "component_count": len(weights),
                "candidate_names": [str(item.name) for item in capture.candidates],
                "weights": [float(value) for value in weights],
                "weight_sum": float(np.sum(weights)),
                "maximum_weight": float(weights[map_index]),
                "effective_model_count": float(1.0 / np.sum(weights * weights)),
                "map_candidate": str(capture.candidates[map_index].name),
                "parent_point_exact": exact,
                "target_mean_parity": target_parity,
                "source_mean_parity": float(max(source["parity"] for source in sources)),
                "posterior_mean_sha256": array_sha256(posterior_mean),
                "map_mean_sha256": array_sha256(map_mean),
                "component_sha256": array_sha256(components),
                "point": {
                    "posterior_mean_field_rmse": float(np.sqrt(np.mean((truth - posterior_mean) ** 2))),
                    "map_field_rmse": float(np.sqrt(np.mean((truth - map_mean) ** 2))),
                },
                "queries": queries,
                "arm_summary": arm_summary,
                "bound_carrier_recovery": drift,
            }
        )
    summary, decision = aggregate(rows, protocol)
    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "dataset_root": str(data_root),
        "information_boundary": {
            "retrospective_target_reuse": True,
            "exact_parent_bound_carriers_reused": True,
            "exact_frozen_point_result_reproduced": all(row["parent_point_exact"] for row in rows),
            "source_only_calibration": True,
            "same_recorded_action_and_future_for_every_arm": True,
            "target_outcomes_used_for_tuning_or_arm_selection": False,
            "unbound_numeric_payloads_opened": False,
            "robot_probe_or_action_selection": False,
            "camera_pixels_opened": False,
            "geometry_or_point_cloud_opened": False,
            "new_measurements_collected": False,
        },
        "summary": summary,
        "decision": decision,
        "objects": rows,
        "carrier_drift": drift_rows,
        "protocol": protocol,
    }
    result["result_sha256"] = digest(result)
    return result


def self_test() -> None:
    weights = np.asarray([0.4, 0.6])
    components = np.asarray([[-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]])
    truth = np.asarray([-1.0, -1.0, 1.0])
    mean = np.einsum("k,kn->n", weights, components)
    variance = 0.04
    identical = np.stack([mean, mean])
    assert np.allclose(
        mixture_nll(truth, identical, weights, variance),
        gaussian_nll(truth, mean, variance),
        atol=1e-10,
    )
    assert np.allclose(
        mixture_crps(truth, identical, weights, variance),
        gaussian_crps(truth, mean, variance),
        atol=2e-7,
    )
    fit = fit_mixture_variance(
        [truth[:2], truth[2:]],
        [components[:, :2], components[:, 2:]],
        weights,
        1e-10,
        41,
    )
    assert math.isfinite(float(fit["variance"])) and float(fit["variance"]) > 0.0
    print("posterior-vs-MAP numerical contracts passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--original-v6-runner", type=Path)
    parser.add_argument("--original-v6-protocol", type=Path)
    parser.add_argument("--recovery-runner", type=Path)
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--parent-result", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--parent-control-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = [name for name, value in vars(args).items() if name != "self_test" and value is None]
    if required:
        parser.error("missing: " + ", ".join(required))
    result = run(args)
    write_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(report(result), encoding="utf-8")
    write_csv(args.output_csv, result["objects"])
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
