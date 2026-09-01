"""Matched-coverage heuristic gates and support audit for the DEFORM certificate.

The primary certificate and its held result are unchanged.  This follow-up fits
all heuristic gate thresholds on the already registered source-test partition,
then applies those frozen thresholds to the official DLO4/DLO5 evaluation
trajectories.  A secondary outcome-free target-covariate ranking matches the
certificate's target update count exactly and is reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

import numpy as np
import numpy.typing as npt

from bayesian_phystwin.query_decision_certificate_v1 import (
    query_decision_certificate,
)

from ._common import (
    ATOL,
    DLOS,
    INTERNAL,
    Decision,
    FloatArray,
    Model,
    Protocol,
    canonical_sha256,
    extract_observation,
    load_protocol,
    load_trajectory,
    partition_names,
    read_json,
    sha256_file,
    trajectory_paths,
    window_starts,
    write_json,
)
from ._evaluation import bootstrap_interval, load_models
from ._model import (
    build_pool,
    class_ambiguity,
    decide,
    fit_model,
    unsupported_specificity,
)

AUDIT_CONTRACT = "deform-dlo45-decision-gate-audit-v1"
AUDIT_REQUEST_CONTRACT = "deform-dlo45-decision-gate-audit-request-v1"
PARENT_CONTRACT = "deform-dlo45-decision-identifiability-v1"
HEURISTICS = (
    "quotient_concentration",
    "maximum_quotient_mass",
    "maximum_kernel_weight",
    "expected_fallback_advantage",
    "expected_action_gap",
    "hypothesis_action_agreement",
    "negative_residual_disagreement",
    "negative_unsupported_specificity",
    "deterministic_random",
)

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class AuditProtocol:
    parent_workflow_run_id: int
    parent_source_artifact: str
    parent_source_artifact_digest: str
    dataset_repository: str
    dataset_commit: str
    heuristics: tuple[str, ...]
    source_coverage_grid: tuple[float, ...]
    bootstrap_replicates: int
    bootstrap_seed: int
    claim_boundary: str


class DiagnosticDecision(NamedTuple):
    decision: Decision
    selected_indices: IntArray
    selected_distances: FloatArray
    selected_residuals: FloatArray
    kernel_weights: FloatArray
    jeffrey_weights: FloatArray
    quotient: FloatArray
    local_classes: IntArray
    relative_losses: FloatArray
    expected_losses: FloatArray
    scores: dict[str, float]


@dataclass(frozen=True)
class WindowRecord:
    stable_id: str
    dlo: str
    trajectory: str
    current_frame: int
    decision: DiagnosticDecision
    physical_mse: FloatArray
    normalized_regret: FloatArray
    fallback_mse: float
    candidate_action: int
    oracle_action: int
    nearest_selected_residual_rmse: float
    nearest_global_residual_rmse: float
    certificate_source_regret_bound: float
    certificate_realized_regret: float
    certificate_regret_excess: float
    certificate_harmful_vs_fallback: bool


def load_audit_protocol(path: Path) -> AuditProtocol:
    value = read_json(path)
    bootstrap = value.get("bootstrap")
    support = value.get("support_audit")
    if (
        value.get("contract") != AUDIT_CONTRACT
        or value.get("schema_version") != 1
        or value.get("parent_contract") != PARENT_CONTRACT
        or value.get("candidate_action")
        != "jeffrey_expected_loss_minimizer"
        or value.get("primary_threshold_rule")
        != "source_test_match_certificate_nonfallback_count"
        or value.get("primary_target_use")
        != "apply_source_frozen_scalar_threshold_without_target_adjustment"
        or value.get("secondary_target_use")
        != "rank_target_covariates_only_to_match_certificate_nonfallback_count_exactly"
        or tuple(value.get("heuristics", ())) != HEURISTICS
        or value.get("target_tuning") is not False
        or value.get("target_outcomes_used_for_thresholds") is not False
        or value.get("target_retries") is not False
        or not isinstance(bootstrap, dict)
        or not isinstance(support, dict)
        or support.get("nearest_selected_residual") is not True
        or support.get("nearest_global_residual") is not True
        or support.get(
            "realized_regret_excess_over_registered_support_bound"
        )
        is not True
    ):
        raise ValueError("invalid gate-audit protocol")
    grid = tuple(float(item) for item in value.get("source_coverage_grid", ()))
    if not grid or any(not 0.0 < item < 1.0 for item in grid):
        raise ValueError("invalid source coverage grid")
    return AuditProtocol(
        parent_workflow_run_id=int(value["parent_workflow_run_id"]),
        parent_source_artifact=str(value["parent_source_artifact"]),
        parent_source_artifact_digest=str(value["parent_source_artifact_digest"]),
        dataset_repository=str(value["dataset_repository"]),
        dataset_commit=str(value["dataset_commit"]),
        heuristics=HEURISTICS,
        source_coverage_grid=grid,
        bootstrap_replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
        claim_boundary=str(value["claim_boundary"]),
    )


def validate_audit_request(path: Path, audit: AuditProtocol) -> dict[str, object]:
    value = read_json(path)
    if (
        value.get("contract") != AUDIT_REQUEST_CONTRACT
        or value.get("schema_version") != 1
        or value.get("status") != "authorized"
        or value.get("parent_workflow_run_id")
        != audit.parent_workflow_run_id
        or value.get("source_only_threshold_selection") is not True
        or value.get("target_tuning") is not False
        or value.get("target_retries") is not False
        or value.get("publish_per_decision_diagnostics") is not True
        or not isinstance(value.get("run_key"), str)
        or not str(value["run_key"]).strip()
    ):
        raise ValueError("invalid gate-audit request")
    return value


def deterministic_score(stable_id: str, heuristic: str) -> float:
    payload = (
        "bayesian-phystwin/deform-dlo45-gate-audit-v1\0"
        f"{heuristic}\0{stable_id}"
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64)


def quotient_concentration(quotient: FloatArray) -> float:
    positive = quotient[quotient > 0.0]
    if len(positive) <= 1:
        return 1.0
    entropy = -float(np.sum(positive * np.log(positive)))
    maximum = math.log(len(positive))
    return 1.0 - entropy / maximum


def diagnose(feature: FloatArray, model: Model, protocol: Protocol) -> DiagnosticDecision:
    """Recompute the registered decision and expose outcome-free gate scores."""

    query = (feature - model.feature_mean) / model.feature_scale
    pool = (model.features - model.feature_mean) / model.feature_scale
    distance = np.mean(np.square(pool - query[None, :]), axis=1)
    neighbor_count = min(model.neighbors, len(distance))
    selected = np.argpartition(distance, neighbor_count - 1)[:neighbor_count]
    selected = selected[np.lexsort((selected, distance[selected]))]
    selected_distance = distance[selected]
    positive_distance = selected_distance[selected_distance > 0.0]
    base_bandwidth = (
        float(np.median(positive_distance))
        if len(positive_distance)
        else max(float(np.mean(selected_distance)), 1e-12)
    )
    bandwidth = max(base_bandwidth * model.temperature_scale, 1e-12)
    logits = -(selected_distance - float(np.min(selected_distance))) / bandwidth
    kernel_weights = np.exp(logits)
    kernel_weights /= np.sum(kernel_weights)

    global_classes = model.class_labels[selected]
    unique_classes = np.unique(global_classes)
    remap = {int(value): index for index, value in enumerate(unique_classes)}
    classes = np.asarray(
        [remap[int(value)] for value in global_classes], dtype=np.int64
    )
    class_count = len(unique_classes)
    quotient = np.bincount(
        classes,
        weights=kernel_weights,
        minlength=class_count,
    ).astype(np.float64)
    class_sizes = np.bincount(classes, minlength=class_count).astype(np.float64)
    jeffrey_weights = quotient[classes] / class_sizes[classes]
    selected_residuals = model.residuals[selected]
    correction = np.einsum("i,id->d", jeffrey_weights, selected_residuals)
    actions = model.action_scales[:, None] * correction[None, :]
    raw_losses = np.mean(
        np.square(selected_residuals[:, None, :] - actions[None, :, :]),
        axis=2,
    )
    relative_losses = raw_losses / (raw_losses[:, :1] + model.loss_floor)
    prior = np.full(neighbor_count, 1.0 / neighbor_count)
    certificate = query_decision_certificate(
        prior,
        quotient,
        classes,
        relative_losses,
        regret_tolerance=model.regret_tolerance,
    )
    certificate_action = (
        certificate.minimax_action_index
        if certificate.minimax_worst_case_regret
        <= model.regret_tolerance + ATOL
        else 0
    )
    expected_losses = np.einsum("i,ia->a", jeffrey_weights, relative_losses)
    jeffrey_action = int(np.argmin(expected_losses))
    kernel_expected_losses = np.einsum(
        "i,ia->a", kernel_weights, relative_losses
    )
    kernel_action = int(np.argmin(kernel_expected_losses))
    map_action = int(np.argmin(relative_losses[0]))
    specificity = unsupported_specificity(kernel_weights, classes, quotient)
    registered = Decision(
        certificate_action=certificate_action,
        jeffrey_action=jeffrey_action,
        kernel_action=kernel_action,
        map_action=map_action,
        correction=correction,
        worst_case_regret=certificate.worst_case_regret,
        minimax_regret=certificate.minimax_worst_case_regret,
        robust_mask=certificate.robustly_optimal_action_mask,
        tolerance_mask=certificate.tolerance_admissible_action_mask,
        ambiguity_width=class_ambiguity(
            selected_residuals,
            classes,
            quotient,
            protocol,
        ),
        unsupported_specificity_nats=specificity,
        neighbor_count=neighbor_count,
        quotient_class_count=class_count,
    )
    reference = decide(feature, model, protocol)
    if (
        registered.certificate_action != reference.certificate_action
        or registered.jeffrey_action != reference.jeffrey_action
        or registered.kernel_action != reference.kernel_action
        or registered.map_action != reference.map_action
        or not np.allclose(registered.correction, reference.correction)
        or not np.allclose(
            registered.worst_case_regret, reference.worst_case_regret
        )
        or not np.array_equal(registered.robust_mask, reference.robust_mask)
        or not np.array_equal(
            registered.tolerance_mask, reference.tolerance_mask
        )
    ):
        raise RuntimeError("gate diagnostics diverged from registered decision")

    sorted_expected = np.sort(expected_losses)
    nonfallback_best = (
        float(np.min(expected_losses[1:]))
        if len(expected_losses) > 1
        else float(expected_losses[0])
    )
    per_hypothesis_action = np.argmin(relative_losses, axis=1)
    agreement = float(
        np.sum(kernel_weights[per_hypothesis_action == jeffrey_action])
    )
    kernel_mean = np.einsum("i,id->d", kernel_weights, selected_residuals)
    disagreement = math.sqrt(
        float(
            np.einsum(
                "i,id,id->",
                kernel_weights,
                selected_residuals - kernel_mean[None, :],
                selected_residuals - kernel_mean[None, :],
            )
        )
        / selected_residuals.shape[1]
    )
    scores = {
        "quotient_concentration": quotient_concentration(quotient),
        "maximum_quotient_mass": float(np.max(quotient)),
        "maximum_kernel_weight": float(np.max(kernel_weights)),
        "expected_fallback_advantage": float(
            expected_losses[0] - nonfallback_best
        ),
        "expected_action_gap": float(sorted_expected[1] - sorted_expected[0]),
        "hypothesis_action_agreement": agreement,
        "negative_residual_disagreement": -disagreement,
        "negative_unsupported_specificity": -specificity,
    }
    return DiagnosticDecision(
        decision=registered,
        selected_indices=selected.astype(np.int64, copy=False),
        selected_distances=selected_distance.astype(np.float64, copy=False),
        selected_residuals=selected_residuals,
        kernel_weights=kernel_weights,
        jeffrey_weights=jeffrey_weights,
        quotient=quotient,
        local_classes=classes,
        relative_losses=relative_losses,
        expected_losses=expected_losses,
        scores=scores,
    )


def _window_records(
    paths: Iterable[Path],
    model: Model,
    protocol: Protocol,
    dlo: str,
) -> list[WindowRecord]:
    records: list[WindowRecord] = []
    for path in paths:
        trajectory = load_trajectory(path)
        for current in window_starts(protocol):
            observation = extract_observation(trajectory, current, protocol)
            diagnostic = diagnose(observation.feature, model, protocol)
            stable_id = f"{dlo}/{path.name}/{current}"
            scores = dict(diagnostic.scores)
            scores["deterministic_random"] = deterministic_score(
                stable_id, "deterministic_random"
            )
            diagnostic = diagnostic._replace(scores=scores)
            truth = trajectory[
                current + 1 : current + 1 + protocol.horizon_frames,
                INTERNAL,
                :,
            ].copy()
            actual_residual = (truth - observation.baseline).reshape(
                -1
            ) / observation.length_scale
            actions = (
                model.action_scales[:, None]
                * diagnostic.decision.correction[None, :]
            )
            normalized_mse = np.mean(
                np.square(actual_residual[None, :] - actions), axis=1
            )
            physical_mse = normalized_mse * observation.length_scale**2
            best = float(np.min(normalized_mse))
            denominator = max(float(normalized_mse[0]), model.loss_floor)
            normalized_regret = (normalized_mse - best) / denominator
            candidate_action = diagnostic.decision.jeffrey_action
            oracle_action = int(np.argmin(physical_mse))
            selected_difference = (
                diagnostic.selected_residuals - actual_residual[None, :]
            )
            global_difference = model.residuals - actual_residual[None, :]
            nearest_selected = math.sqrt(
                float(
                    np.min(np.mean(np.square(selected_difference), axis=1))
                )
            )
            nearest_global = math.sqrt(
                float(np.min(np.mean(np.square(global_difference), axis=1)))
            )
            certificate_action = diagnostic.decision.certificate_action
            source_bound = float(
                diagnostic.decision.worst_case_regret[certificate_action]
            )
            realized_regret = float(normalized_regret[certificate_action])
            fallback_mse = float(physical_mse[0])
            records.append(
                WindowRecord(
                    stable_id=stable_id,
                    dlo=dlo,
                    trajectory=path.name,
                    current_frame=current,
                    decision=diagnostic,
                    physical_mse=physical_mse,
                    normalized_regret=normalized_regret,
                    fallback_mse=fallback_mse,
                    candidate_action=candidate_action,
                    oracle_action=oracle_action,
                    nearest_selected_residual_rmse=nearest_selected,
                    nearest_global_residual_rmse=nearest_global,
                    certificate_source_regret_bound=source_bound,
                    certificate_realized_regret=realized_regret,
                    certificate_regret_excess=realized_regret - source_bound,
                    certificate_harmful_vs_fallback=bool(
                        physical_mse[certificate_action]
                        > fallback_mse + ATOL
                    ),
                )
            )
    return records


def _rank_key(record: WindowRecord, heuristic: str) -> tuple[float, float]:
    return (
        float(record.decision.scores[heuristic]),
        deterministic_score(record.stable_id, heuristic),
    )


def fit_rank_threshold(
    records: Sequence[WindowRecord], heuristic: str, selected_count: int
) -> dict[str, object]:
    eligible = [record for record in records if record.candidate_action != 0]
    if selected_count < 0 or selected_count > len(eligible):
        raise ValueError("matched coverage exceeds eligible source decisions")
    ranked = sorted(
        eligible,
        key=lambda record: _rank_key(record, heuristic),
        reverse=True,
    )
    if selected_count == 0:
        return {
            "mode": "select_none",
            "selected_count": 0,
            "eligible_count": len(eligible),
        }
    if selected_count == len(ranked):
        return {
            "mode": "select_all_eligible",
            "selected_count": selected_count,
            "eligible_count": len(eligible),
        }
    boundary = ranked[selected_count - 1]
    score, tie = _rank_key(boundary, heuristic)
    return {
        "mode": "rank_boundary",
        "score": score,
        "tie_break": tie,
        "selected_count": selected_count,
        "eligible_count": len(eligible),
        "boundary_stable_id": boundary.stable_id,
    }


def threshold_passes(
    record: WindowRecord, heuristic: str, threshold: Mapping[str, object]
) -> bool:
    if record.candidate_action == 0:
        return False
    mode = threshold.get("mode")
    if mode == "select_none":
        return False
    if mode == "select_all_eligible":
        return True
    if mode != "rank_boundary":
        raise ValueError("unknown threshold mode")
    score, tie = _rank_key(record, heuristic)
    boundary_score = float(threshold["score"])
    boundary_tie = float(threshold["tie_break"])
    return score > boundary_score or (
        score == boundary_score and tie >= boundary_tie
    )


def exact_covariate_matched_actions(
    records: Sequence[WindowRecord], heuristic: str, selected_count: int
) -> list[int]:
    eligible = [record for record in records if record.candidate_action != 0]
    selected_count = min(selected_count, len(eligible))
    ranked = sorted(
        eligible,
        key=lambda record: _rank_key(record, heuristic),
        reverse=True,
    )
    selected_ids = {record.stable_id for record in ranked[:selected_count]}
    return [
        record.candidate_action if record.stable_id in selected_ids else 0
        for record in records
    ]


def threshold_actions(
    records: Sequence[WindowRecord],
    heuristic: str,
    threshold: Mapping[str, object],
) -> list[int]:
    return [
        record.candidate_action
        if threshold_passes(record, heuristic, threshold)
        else 0
        for record in records
    ]


def certificate_actions(records: Sequence[WindowRecord]) -> list[int]:
    return [record.decision.decision.certificate_action for record in records]


def point_actions(records: Sequence[WindowRecord], name: str) -> list[int]:
    if name == "jeffrey_point":
        return [record.decision.decision.jeffrey_action for record in records]
    if name == "kernel_point":
        return [record.decision.decision.kernel_action for record in records]
    if name == "map_point":
        return [record.decision.decision.map_action for record in records]
    if name == "oracle":
        return [record.oracle_action for record in records]
    if name == "fallback":
        return [0 for _ in records]
    raise ValueError(f"unknown point action method: {name}")


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z**2 / (4.0 * total**2)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _trajectory_summary(
    records: Sequence[WindowRecord], actions: Sequence[int]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault((record.dlo, record.trajectory), []).append(index)
    result: list[dict[str, object]] = []
    for (dlo, trajectory), indices in sorted(grouped.items()):
        fallback = math.sqrt(
            float(np.mean([records[index].fallback_mse for index in indices]))
        )
        selected = math.sqrt(
            float(
                np.mean(
                    [
                        records[index].physical_mse[actions[index]]
                        for index in indices
                    ]
                )
            )
        )
        result.append(
            {
                "dlo": dlo,
                "trajectory": trajectory,
                "decision_count": len(indices),
                "fallback_rmse_mm": 1000.0 * fallback,
                "selected_rmse_mm": 1000.0 * selected,
                "ratio": selected / max(fallback, 1e-12),
                "improvement": 1.0 - selected / max(fallback, 1e-12),
            }
        )
    return result


def summarize_method(
    records: Sequence[WindowRecord], actions: Sequence[int]
) -> dict[str, object]:
    if len(records) != len(actions):
        raise ValueError("action roster length mismatch")
    selected_mse = np.asarray(
        [record.physical_mse[action] for record, action in zip(records, actions)],
        dtype=np.float64,
    )
    fallback_mse = np.asarray(
        [record.fallback_mse for record in records], dtype=np.float64
    )
    regrets = np.asarray(
        [record.normalized_regret[action] for record, action in zip(records, actions)],
        dtype=np.float64,
    )
    action_array = np.asarray(actions, dtype=np.int64)
    nonfallback = action_array != 0
    harmful = selected_mse > fallback_mse + ATOL
    fallback_rmse = math.sqrt(float(np.mean(fallback_mse)))
    selected_rmse = math.sqrt(float(np.mean(selected_mse)))
    harmful_nonfallback = int(np.count_nonzero(harmful & nonfallback))
    nonfallback_count = int(np.count_nonzero(nonfallback))
    interval = _wilson_interval(harmful_nonfallback, nonfallback_count)
    trajectories = _trajectory_summary(records, actions)
    ratios = np.asarray([float(item["ratio"]) for item in trajectories])
    return {
        "decision_count": len(records),
        "nonfallback_count": nonfallback_count,
        "nonfallback_fraction": nonfallback_count / max(len(records), 1),
        "rmse_mm": 1000.0 * selected_rmse,
        "rmse_ratio_to_fallback": selected_rmse / max(fallback_rmse, 1e-12),
        "mean_normalized_regret": float(np.mean(regrets)),
        "p95_normalized_regret": float(np.quantile(regrets, 0.95)),
        "p99_normalized_regret": float(np.quantile(regrets, 0.99)),
        "maximum_normalized_regret": float(np.max(regrets)),
        "harmful_decision_count": int(np.count_nonzero(harmful)),
        "harmful_fraction_all": float(np.mean(harmful)),
        "harmful_nonfallback_count": harmful_nonfallback,
        "harmful_fraction_nonfallback": (
            harmful_nonfallback / nonfallback_count
            if nonfallback_count
            else 0.0
        ),
        "harmful_nonfallback_wilson95": list(interval),
        "action_counts": np.bincount(
            action_array, minlength=len(records[0].physical_mse)
        ).tolist(),
        "trajectory_wins_ties_losses": [
            int(np.count_nonzero(ratios < 1.0 - ATOL)),
            int(np.count_nonzero(np.abs(ratios - 1.0) <= ATOL)),
            int(np.count_nonzero(ratios > 1.0 + ATOL)),
        ],
        "maximum_trajectory_ratio": float(np.max(ratios)),
        "per_trajectory": trajectories,
    }


def _stratified_bootstrap(
    trajectories: Sequence[Mapping[str, object]],
    replicates: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[float]] = {dlo: [] for dlo in DLOS}
    for item in trajectories:
        grouped[str(item["dlo"])].append(float(item["improvement"]))
    if any(not grouped[dlo] for dlo in DLOS):
        raise ValueError("bootstrap requires both DLO strata")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        values: list[float] = []
        for dlo in DLOS:
            array = np.asarray(grouped[dlo], dtype=np.float64)
            values.extend(
                array[rng.integers(0, len(array), size=len(array))].tolist()
            )
        estimates[index] = float(np.mean(values))
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _rankdata(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if len(x_array) < 2:
        return 0.0
    x_rank = _rankdata(x_array)
    y_rank = _rankdata(y_array)
    if float(np.std(x_rank)) <= 0.0 or float(np.std(y_rank)) <= 0.0:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=bool)
    positives = int(np.count_nonzero(label_array))
    negatives = len(label_array) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _rankdata(score_array)
    positive_rank_sum = float(np.sum(ranks[label_array]))
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def support_audit(records: Sequence[WindowRecord]) -> dict[str, object]:
    nonfallback = [
        record
        for record in records
        if record.decision.decision.certificate_action != 0
    ]
    harmful = [
        record for record in nonfallback if record.certificate_harmful_vs_fallback
    ]
    safe = [
        record
        for record in nonfallback
        if not record.certificate_harmful_vs_fallback
    ]

    def summary(items: Sequence[WindowRecord]) -> dict[str, object]:
        if not items:
            return {"count": 0}
        selected = np.asarray(
            [item.nearest_selected_residual_rmse for item in items]
        )
        global_distance = np.asarray(
            [item.nearest_global_residual_rmse for item in items]
        )
        excess = np.asarray([item.certificate_regret_excess for item in items])
        return {
            "count": len(items),
            "nearest_selected_residual_rmse_mean": float(np.mean(selected)),
            "nearest_selected_residual_rmse_median": float(np.median(selected)),
            "nearest_selected_residual_rmse_max": float(np.max(selected)),
            "nearest_global_residual_rmse_mean": float(np.mean(global_distance)),
            "nearest_global_residual_rmse_median": float(
                np.median(global_distance)
            ),
            "nearest_global_residual_rmse_max": float(np.max(global_distance)),
            "regret_excess_mean": float(np.mean(excess)),
            "regret_excess_median": float(np.median(excess)),
            "regret_excess_max": float(np.max(excess)),
            "support_bound_exceeded_fraction": float(np.mean(excess > ATOL)),
        }

    distances = [item.nearest_selected_residual_rmse for item in nonfallback]
    realized = [item.certificate_realized_regret for item in nonfallback]
    excesses = [item.certificate_regret_excess for item in nonfallback]
    labels = [item.certificate_harmful_vs_fallback for item in nonfallback]
    harmful_cases = [
        {
            "stable_id": item.stable_id,
            "dlo": item.dlo,
            "trajectory": item.trajectory,
            "current_frame": item.current_frame,
            "certificate_action": item.decision.decision.certificate_action,
            "source_regret_bound": item.certificate_source_regret_bound,
            "realized_regret": item.certificate_realized_regret,
            "regret_excess": item.certificate_regret_excess,
            "nearest_selected_residual_rmse": (
                item.nearest_selected_residual_rmse
            ),
            "nearest_global_residual_rmse": item.nearest_global_residual_rmse,
        }
        for item in harmful
    ]
    return {
        "certificate_nonfallback": summary(nonfallback),
        "harmful_certificate_nonfallback": summary(harmful),
        "safe_certificate_nonfallback": summary(safe),
        "spearman_nearest_selected_vs_realized_regret": _spearman(
            distances, realized
        ),
        "spearman_nearest_selected_vs_regret_excess": _spearman(
            distances, excesses
        ),
        "nearest_selected_distance_auc_for_harm": _auc(distances, labels),
        "regret_excess_auc_for_harm": _auc(excesses, labels),
        "harmful_cases": harmful_cases,
    }


def _source_model(
    train_paths: tuple[Path, ...],
    dlo: str,
    source_result: Mapping[str, object],
    protocol: Protocol,
) -> tuple[Model, tuple[Path, ...]]:
    dlo_results = source_result.get("dlos")
    if not isinstance(dlo_results, dict):
        raise ValueError("source result lacks DLO records")
    record = dlo_results.get(dlo)
    if not isinstance(record, dict):
        raise ValueError(f"source result lacks {dlo}")
    selected = record.get("selected_settings")
    partition = record.get("partition")
    if not isinstance(selected, dict) or not isinstance(partition, dict):
        raise ValueError("source result settings/partition malformed")
    fit_names = tuple(str(item) for item in partition["fit"])
    calibration_names = tuple(str(item) for item in partition["calibration"])
    source_test_names = tuple(str(item) for item in partition["source_test"])
    names = tuple(path.name for path in train_paths)
    expected = partition_names(names, dlo, protocol)
    if (
        expected["fit"] != fit_names
        or expected["calibration"] != calibration_names
        or expected["source_test"] != source_test_names
    ):
        raise ValueError("source partition differs from retained result")
    refit_names = fit_names + calibration_names
    features, residuals, _ = build_pool(train_paths, refit_names, protocol)
    model = fit_model(
        features,
        residuals,
        cluster_count=int(selected["cluster_count"]),
        neighbors=int(selected["neighbors"]),
        temperature_scale=float(selected["temperature_scale"]),
        regret_tolerance=float(selected["regret_tolerance"]),
        protocol=protocol,
    )
    test_set = set(source_test_names)
    test_paths = tuple(path for path in train_paths if path.name in test_set)
    return model, test_paths


def source_command(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    audit = load_audit_protocol(args.audit_protocol)
    request = validate_audit_request(args.request, audit)
    source_result = read_json(args.parent_source_result)
    source_seal = read_json(args.parent_source_seal)
    if (
        source_result.get("contract") != PARENT_CONTRACT
        or source_result.get("stage") != "source"
        or source_seal.get("contract") != PARENT_CONTRACT
        or source_seal.get("stage") != "source-seal"
        or source_seal.get("source_result_sha256")
        != sha256_file(args.parent_source_result)
        or source_seal.get("source_model_sha256")
        != sha256_file(args.parent_source_model)
    ):
        raise ValueError("retained parent source artifact does not verify")

    threshold_result: dict[str, object] = {}
    source_summaries: dict[str, object] = {}
    for dlo in DLOS:
        train_paths = trajectory_paths(args.dataset_root, dlo, "train")
        model, source_test_paths = _source_model(
            train_paths, dlo, source_result, protocol
        )
        records = _window_records(source_test_paths, model, protocol, dlo)
        certificate = certificate_actions(records)
        selected_count = int(np.count_nonzero(np.asarray(certificate) != 0))
        retained_dlo = source_result["dlos"][dlo]
        assert isinstance(retained_dlo, dict)
        retained_test = retained_dlo["source_test"]
        assert isinstance(retained_test, dict)
        expected_fraction = float(retained_test["certificate_nonfallback_fraction"])
        if not math.isclose(
            selected_count / len(records), expected_fraction, abs_tol=1e-12
        ):
            raise ValueError("recomputed source certificate coverage changed")
        primary = {
            heuristic: fit_rank_threshold(records, heuristic, selected_count)
            for heuristic in audit.heuristics
        }
        grid: dict[str, object] = {}
        for heuristic in audit.heuristics:
            grid[heuristic] = {
                f"{fraction:.2f}": fit_rank_threshold(
                    records,
                    heuristic,
                    min(
                        int(round(fraction * len(records))),
                        sum(record.candidate_action != 0 for record in records),
                    ),
                )
                for fraction in audit.source_coverage_grid
            }
        threshold_result[dlo] = {
            "source_test_decision_count": len(records),
            "source_certificate_nonfallback_count": selected_count,
            "source_certificate_nonfallback_fraction": selected_count
            / len(records),
            "primary": primary,
            "source_coverage_grid": grid,
        }
        source_summaries[dlo] = {
            "certificate": summarize_method(records, certificate),
            "heuristics": {
                heuristic: summarize_method(
                    records,
                    threshold_actions(records, heuristic, primary[heuristic]),
                )
                for heuristic in audit.heuristics
            },
        }

    output = args.output_root.resolve()
    thresholds = {
        "contract": AUDIT_CONTRACT,
        "schema_version": 1,
        "stage": "source-thresholds",
        "run_key": request["run_key"],
        "parent_protocol_sha256": sha256_file(args.protocol),
        "audit_protocol_sha256": sha256_file(args.audit_protocol),
        "request_sha256": sha256_file(args.request),
        "parent_source_result_sha256": sha256_file(args.parent_source_result),
        "parent_source_seal_sha256": sha256_file(args.parent_source_seal),
        "parent_source_model_sha256": sha256_file(args.parent_source_model),
        "dlos": threshold_result,
        "target_data_read": False,
        "target_outcomes_used": False,
        "claim_boundary": audit.claim_boundary,
    }
    write_json(output / "thresholds.json", thresholds)
    source_audit = {
        "contract": AUDIT_CONTRACT,
        "schema_version": 1,
        "stage": "source-audit",
        "run_key": request["run_key"],
        "dlos": source_summaries,
        "target_data_read": False,
        "target_outcomes_used": False,
    }
    write_json(output / "source_audit.json", source_audit)
    seal = {
        "contract": AUDIT_CONTRACT,
        "schema_version": 1,
        "stage": "source-seal",
        "run_key": request["run_key"],
        "thresholds_sha256": sha256_file(output / "thresholds.json"),
        "source_audit_sha256": sha256_file(output / "source_audit.json"),
        "parent_source_model_sha256": sha256_file(args.parent_source_model),
        "canonical_thresholds_sha256": canonical_sha256(thresholds),
    }
    write_json(output / "source_seal.json", seal)
    return 0


def _combine_method(
    by_dlo: Mapping[str, dict[str, object]],
    audit: AuditProtocol,
    seed_offset: int,
) -> dict[str, object]:
    decision_count = sum(int(by_dlo[dlo]["decision_count"]) for dlo in DLOS)
    total_fallback_sse = 0.0
    total_selected_sse = 0.0
    trajectories: list[Mapping[str, object]] = []
    nonfallback_count = 0
    harmful_nonfallback = 0
    harmful_all = 0
    regret_values: list[float] = []
    for dlo in DLOS:
        item = by_dlo[dlo]
        count = int(item["decision_count"])
        fallback_rmse = float(item["fallback_rmse_mm"]) / 1000.0
        selected_rmse = float(item["rmse_mm"]) / 1000.0
        total_fallback_sse += count * fallback_rmse**2
        total_selected_sse += count * selected_rmse**2
        nonfallback_count += int(item["nonfallback_count"])
        harmful_nonfallback += int(item["harmful_nonfallback_count"])
        harmful_all += int(item["harmful_decision_count"])
        trajectories.extend(item["per_trajectory"])
        regret_values.extend(float(value) for value in item["regret_values"])
    ratio = math.sqrt(total_selected_sse / total_fallback_sse)
    trajectory_improvements = [float(item["improvement"]) for item in trajectories]
    interval = _stratified_bootstrap(
        trajectories,
        audit.bootstrap_replicates,
        audit.bootstrap_seed + seed_offset,
    )
    wilson = _wilson_interval(harmful_nonfallback, nonfallback_count)
    ratios = np.asarray([float(item["ratio"]) for item in trajectories])
    regrets = np.asarray(regret_values, dtype=np.float64)
    return {
        "decision_count": decision_count,
        "nonfallback_count": nonfallback_count,
        "nonfallback_fraction": nonfallback_count / decision_count,
        "rmse_ratio_to_fallback": ratio,
        "rmse_reduction": 1.0 - ratio,
        "mean_paired_trajectory_improvement": float(
            np.mean(trajectory_improvements)
        ),
        "trajectory_bootstrap_95_interval": interval,
        "trajectory_wins_ties_losses": [
            int(np.count_nonzero(ratios < 1.0 - ATOL)),
            int(np.count_nonzero(np.abs(ratios - 1.0) <= ATOL)),
            int(np.count_nonzero(ratios > 1.0 + ATOL)),
        ],
        "maximum_trajectory_ratio": float(np.max(ratios)),
        "harmful_decision_count": harmful_all,
        "harmful_nonfallback_count": harmful_nonfallback,
        "harmful_fraction_nonfallback": (
            harmful_nonfallback / nonfallback_count
            if nonfallback_count
            else 0.0
        ),
        "harmful_nonfallback_wilson95": list(wilson),
        "mean_normalized_regret": float(np.mean(regrets)),
        "p95_normalized_regret": float(np.quantile(regrets, 0.95)),
        "p99_normalized_regret": float(np.quantile(regrets, 0.99)),
    }


def _augment_for_combine(summary: dict[str, object], records: Sequence[WindowRecord], actions: Sequence[int]) -> dict[str, object]:
    result = dict(summary)
    fallback_rmse = math.sqrt(float(np.mean([record.fallback_mse for record in records])))
    result["fallback_rmse_mm"] = 1000.0 * fallback_rmse
    result["regret_values"] = [
        float(record.normalized_regret[action])
        for record, action in zip(records, actions)
    ]
    return result


def target_command(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    audit = load_audit_protocol(args.audit_protocol)
    request = validate_audit_request(args.request, audit)
    thresholds = read_json(args.thresholds)
    threshold_seal = read_json(args.threshold_seal)
    parent_seal = read_json(args.parent_source_seal)
    if (
        thresholds.get("contract") != AUDIT_CONTRACT
        or thresholds.get("stage") != "source-thresholds"
        or thresholds.get("run_key") != request["run_key"]
        or threshold_seal.get("contract") != AUDIT_CONTRACT
        or threshold_seal.get("stage") != "source-seal"
        or threshold_seal.get("thresholds_sha256")
        != sha256_file(args.thresholds)
        or parent_seal.get("source_model_sha256")
        != sha256_file(args.parent_source_model)
        or thresholds.get("parent_source_model_sha256")
        != sha256_file(args.parent_source_model)
    ):
        raise ValueError("source-frozen threshold/model seal does not verify")
    models = load_models(args.parent_source_model)

    records_by_dlo: dict[str, list[WindowRecord]] = {}
    certificate_by_dlo: dict[str, dict[str, object]] = {}
    source_gate_by_dlo: dict[str, dict[str, dict[str, object]]] = {}
    matched_gate_by_dlo: dict[str, dict[str, dict[str, object]]] = {}
    curves_by_dlo: dict[str, dict[str, object]] = {}
    point_by_dlo: dict[str, dict[str, dict[str, object]]] = {}
    all_records: list[WindowRecord] = []

    threshold_dlos = thresholds.get("dlos")
    if not isinstance(threshold_dlos, dict):
        raise ValueError("threshold DLO records missing")
    for dlo in DLOS:
        eval_paths = trajectory_paths(args.dataset_root, dlo, "eval")
        records = _window_records(eval_paths, models[dlo], protocol, dlo)
        records_by_dlo[dlo] = records
        all_records.extend(records)
        certificate = certificate_actions(records)
        certificate_summary = summarize_method(records, certificate)
        certificate_by_dlo[dlo] = _augment_for_combine(
            certificate_summary, records, certificate
        )
        dlo_thresholds = threshold_dlos[dlo]
        assert isinstance(dlo_thresholds, dict)
        primary_thresholds = dlo_thresholds["primary"]
        assert isinstance(primary_thresholds, dict)
        source_gate_by_dlo[dlo] = {}
        matched_gate_by_dlo[dlo] = {}
        target_nonfallback = int(certificate_summary["nonfallback_count"])
        for heuristic in audit.heuristics:
            threshold = primary_thresholds[heuristic]
            assert isinstance(threshold, dict)
            actions = threshold_actions(records, heuristic, threshold)
            summary = summarize_method(records, actions)
            source_gate_by_dlo[dlo][heuristic] = _augment_for_combine(
                summary, records, actions
            )
            matched_actions = exact_covariate_matched_actions(
                records, heuristic, target_nonfallback
            )
            matched_summary = summarize_method(records, matched_actions)
            matched_gate_by_dlo[dlo][heuristic] = _augment_for_combine(
                matched_summary, records, matched_actions
            )
        grid = dlo_thresholds["source_coverage_grid"]
        assert isinstance(grid, dict)
        curves_by_dlo[dlo] = {}
        for heuristic in audit.heuristics:
            heuristic_grid = grid[heuristic]
            assert isinstance(heuristic_grid, dict)
            curves_by_dlo[dlo][heuristic] = {
                fraction: summarize_method(
                    records,
                    threshold_actions(records, heuristic, threshold),
                )
                for fraction, threshold in heuristic_grid.items()
                if isinstance(threshold, dict)
            }
        point_by_dlo[dlo] = {}
        for name in ("fallback", "jeffrey_point", "kernel_point", "map_point", "oracle"):
            actions = point_actions(records, name)
            point_by_dlo[dlo][name] = _augment_for_combine(
                summarize_method(records, actions), records, actions
            )

    combined_certificate = _combine_method(
        certificate_by_dlo, audit, seed_offset=0
    )
    combined_source_gates: dict[str, object] = {}
    combined_matched_gates: dict[str, object] = {}
    for index, heuristic in enumerate(audit.heuristics):
        combined_source_gates[heuristic] = _combine_method(
            {dlo: source_gate_by_dlo[dlo][heuristic] for dlo in DLOS},
            audit,
            seed_offset=100 + index,
        )
        combined_matched_gates[heuristic] = _combine_method(
            {dlo: matched_gate_by_dlo[dlo][heuristic] for dlo in DLOS},
            audit,
            seed_offset=200 + index,
        )
    combined_points = {
        name: _combine_method(
            {dlo: point_by_dlo[dlo][name] for dlo in DLOS},
            audit,
            seed_offset=300 + index,
        )
        for index, name in enumerate(
            ("fallback", "jeffrey_point", "kernel_point", "map_point", "oracle")
        )
    }
    support = support_audit(all_records)
    matched_better_harm = [
        heuristic
        for heuristic, item in combined_matched_gates.items()
        if isinstance(item, dict)
        and combined_certificate["harmful_fraction_nonfallback"]
        < item["harmful_fraction_nonfallback"] - ATOL
    ]
    matched_no_worse_tail = [
        heuristic
        for heuristic, item in combined_matched_gates.items()
        if isinstance(item, dict)
        and combined_certificate["p95_normalized_regret"]
        <= item["p95_normalized_regret"] + ATOL
    ]
    result = {
        "contract": AUDIT_CONTRACT,
        "schema_version": 1,
        "stage": "target-audit",
        "run_key": request["run_key"],
        "parent_protocol_sha256": sha256_file(args.protocol),
        "audit_protocol_sha256": sha256_file(args.audit_protocol),
        "request_sha256": sha256_file(args.request),
        "parent_source_model_sha256": sha256_file(args.parent_source_model),
        "thresholds_sha256": sha256_file(args.thresholds),
        "threshold_seal_sha256": sha256_file(args.threshold_seal),
        "dlos": {
            dlo: {
                "certificate": certificate_by_dlo[dlo],
                "source_calibrated_gates": source_gate_by_dlo[dlo],
                "target_covariate_matched_gates": matched_gate_by_dlo[dlo],
                "source_calibrated_coverage_curves": curves_by_dlo[dlo],
                "point_diagnostics": point_by_dlo[dlo],
            }
            for dlo in DLOS
        },
        "combined": {
            "certificate": combined_certificate,
            "source_calibrated_gates": combined_source_gates,
            "target_covariate_matched_gates": combined_matched_gates,
            "point_diagnostics": combined_points,
            "certificate_strictly_lower_harm_than_matched_heuristics": matched_better_harm,
            "certificate_no_worse_p95_regret_than_matched_heuristics": matched_no_worse_tail,
        },
        "support_mismatch_audit": support,
        "target_threshold_tuning": False,
        "target_outcomes_used_for_thresholds": False,
        "target_covariate_matching_uses_outcomes": False,
        "target_retries": False,
        "claim_boundary": audit.claim_boundary,
    }
    output = args.output_root.resolve()
    write_json(output / "target_audit.json", result)
    with (output / "per_decision.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_records:
            payload = {
                "stable_id": record.stable_id,
                "dlo": record.dlo,
                "trajectory": record.trajectory,
                "current_frame": record.current_frame,
                "certificate_action": record.decision.decision.certificate_action,
                "candidate_action": record.candidate_action,
                "oracle_action": record.oracle_action,
                "scores": record.decision.scores,
                "certificate_harmful_vs_fallback": record.certificate_harmful_vs_fallback,
                "certificate_source_regret_bound": record.certificate_source_regret_bound,
                "certificate_realized_regret": record.certificate_realized_regret,
                "certificate_regret_excess": record.certificate_regret_excess,
                "nearest_selected_residual_rmse": record.nearest_selected_residual_rmse,
                "nearest_global_residual_rmse": record.nearest_global_residual_rmse,
            }
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    (output / "summary.md").write_text(
        render_summary(result), encoding="utf-8"
    )
    return 0


def render_summary(result: Mapping[str, object]) -> str:
    combined = result["combined"]
    assert isinstance(combined, dict)
    certificate = combined["certificate"]
    source_gates = combined["source_calibrated_gates"]
    matched_gates = combined["target_covariate_matched_gates"]
    assert isinstance(certificate, dict)
    assert isinstance(source_gates, dict)
    assert isinstance(matched_gates, dict)
    lines = [
        "# DEFORM DLO4/DLO5 matched-coverage gate and support audit",
        "",
        "The primary certificate is unchanged. Heuristic thresholds are fitted ",
        "only on the pre-existing source-test partitions.",
        "",
        "## Frozen certificate",
        "",
        f"- nonfallback: **{int(certificate['nonfallback_count'])} / {int(certificate['decision_count'])}**",
        f"- RMSE reduction: **{100.0 * float(certificate['rmse_reduction']):.2f}%**",
        f"- harmful nonfallback: **{int(certificate['harmful_nonfallback_count'])}**",
        f"- p95 normalized regret: **{float(certificate['p95_normalized_regret']):.4f}**",
        "",
        "## Source-calibrated gates",
        "",
        "| Gate | Coverage | RMSE reduction | Harm/nonfallback | p95 regret |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for heuristic in HEURISTICS:
        item = source_gates[heuristic]
        assert isinstance(item, dict)
        lines.append(
            f"| `{heuristic}` | {100.0 * float(item['nonfallback_fraction']):.1f}% | "
            f"{100.0 * float(item['rmse_reduction']):.2f}% | "
            f"{int(item['harmful_nonfallback_count'])}/{int(item['nonfallback_count'])} | "
            f"{float(item['p95_normalized_regret']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Exact target-covariate coverage match (outcome-free secondary)",
            "",
            "| Gate | RMSE reduction | Harm/nonfallback | p95 regret |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for heuristic in HEURISTICS:
        item = matched_gates[heuristic]
        assert isinstance(item, dict)
        lines.append(
            f"| `{heuristic}` | {100.0 * float(item['rmse_reduction']):.2f}% | "
            f"{int(item['harmful_nonfallback_count'])}/{int(item['nonfallback_count'])} | "
            f"{float(item['p95_normalized_regret']):.4f} |"
        )
    support = result["support_mismatch_audit"]
    assert isinstance(support, dict)
    harmful = support["harmful_certificate_nonfallback"]
    safe = support["safe_certificate_nonfallback"]
    assert isinstance(harmful, dict)
    assert isinstance(safe, dict)
    lines.extend(
        [
            "",
            "## Registered-support audit",
            "",
            f"- harmful certified decisions: **{int(harmful.get('count', 0))}**",
            f"- safe certified decisions: **{int(safe.get('count', 0))}**",
            f"- support-distance AUC for harm: **{support.get('nearest_selected_distance_auc_for_harm')}**",
            f"- regret-excess AUC for harm: **{support.get('regret_excess_auc_for_harm')}**",
            "",
            "## Claim boundary",
            "",
            str(result["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("source", "target"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit-protocol", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-source-model", type=Path, required=True)
    parser.add_argument("--parent-source-result", type=Path)
    parser.add_argument("--parent-source-seal", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--threshold-seal", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "source":
        if args.parent_source_result is None:
            raise ValueError("source command requires parent source result")
        return source_command(args)
    if args.thresholds is None or args.threshold_seal is None:
        raise ValueError("target command requires thresholds and threshold seal")
    return target_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
