#!/usr/bin/env python3
"""Evaluate minimum-rank decision-sufficient covariance compression on Deform360.

The experiment reuses the exact frozen 92-object same-mean dependence study.
For a Gaussian covariance ``D + U U^T`` and fixed linear query matrix ``Q``, it
projects the shared factor onto ``range(U^T Q^T)``.  This is the minimum-rank
member of the fixed orthogonal factor-projection family that preserves the
complete joint distribution of ``Q X``.  Consequently every registered query
score and execute-versus-fallback decision must remain identical to the full
factor.  A matched-rank spectral projection is retained as a non-query-aware
control.

The target episodes are retrospective: they were already opened by the parent
Deform360 study.  No target outcome is used to construct either projection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "bayesian-phystwin/deform360-decision-sufficient-compression-result-v1"
PROTOCOL_SCHEMA = (
    "bayesian-phystwin/deform360-decision-sufficient-compression-protocol-v1"
)
REPRESENTATIONS = (
    "full_low_rank",
    "portfolio_sufficient",
    "per_query_sufficient",
    "spectral_rank_matched",
)
METRICS = (
    "target_query_nanees",
    "target_90_coverage",
    "mean_90_interval_width",
    "query_nll",
    "event_brier",
    "event_log_loss",
    "decision_loss",
    "decision_regret",
    "acceptance_fraction",
    "harmful_accept_fraction_all",
    "harmful_accept_rate_given_accept",
)
_EPS = 1e-12


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_digest(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=True,
    ).strip()


def validate_protocol(
    protocol: dict[str, Any],
    *,
    data_root: Path,
    original_v6_root: Path,
    parent_control_root: Path,
    frozen_root: Path,
) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected compression protocol schema")
    if protocol.get("schema_version") != 1:
        raise ValueError("unexpected compression protocol version")
    if protocol.get("status") != "frozen-before-compression-evaluation":
        raise ValueError("compression protocol is not frozen")
    if Path(str(protocol.get("dataset_root"))) != data_root:
        raise ValueError("dataset root changed")
    binding = protocol.get("parent_dependence_study")
    if not isinstance(binding, Mapping):
        raise ValueError("parent dependence binding is absent")
    expected_revisions = (
        (original_v6_root, "original_v6_revision"),
        (parent_control_root, "parent_confirmation_control_revision"),
        (frozen_root, "frozen_point_revision"),
    )
    for root, key in expected_revisions:
        if git_output(root, "rev-parse", "HEAD") != str(binding[key]):
            raise ValueError(f"bound revision changed: {key}")
    evaluation = protocol.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation contract is absent")
    if tuple(evaluation.get("representations", ())) != REPRESENTATIONS:
        raise ValueError("representation roster changed")
    if evaluation.get("same_predictive_mean_required") is not True:
        raise ValueError("same-mean contract must be enabled")
    if evaluation.get("source_only_query_calibration_required") is not True:
        raise ValueError("query calibration must remain source-only")
    boundary = protocol.get("information_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("information boundary is absent")
    for key in (
        "point_predictor_may_change",
        "query_bank_may_change",
        "covariance_estimator_may_change",
        "source_calibration_may_change",
        "decision_rule_may_change",
        "target_outcomes_may_select_projection",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"forbidden change enabled: {key}")
    if boundary.get("projection_uses_only_covariance_factor_and_registered_queries") is not True:
        raise ValueError("projection information boundary changed")
    if protocol.get("paper_claim_authorized") is not False:
        raise ValueError("protocol may not self-authorize a paper claim")


def orthonormal_range(
    matrix: object,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("range matrix must be a finite matrix")
    if values.shape[0] == 0 or values.shape[1] == 0:
        return np.empty((values.shape[0], 0), dtype=np.float64), np.empty(0)
    left, singular, _ = np.linalg.svd(values, full_matrices=False)
    reference = max(float(singular[0]) if len(singular) else 0.0, _EPS)
    rank = int(np.count_nonzero(singular > relative_tolerance * reference))
    basis = np.asarray(left[:, :rank], dtype=np.float64)
    if rank:
        np.testing.assert_allclose(
            basis.T @ basis,
            np.eye(rank),
            rtol=1e-11,
            atol=1e-11,
        )
    return basis, singular


def spectral_basis(
    factor: object,
    rank: int,
) -> np.ndarray:
    values = np.asarray(factor, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("factor must be a finite matrix")
    if rank < 0 or rank > values.shape[1]:
        raise ValueError("invalid spectral rank")
    if rank == 0:
        return np.empty((values.shape[1], 0), dtype=np.float64)
    _, _, right = np.linalg.svd(values, full_matrices=False)
    if rank > right.shape[0]:
        raise ValueError("requested spectral rank exceeds numerical factor rank")
    return np.asarray(right[:rank].T, dtype=np.float64)


def projected_model(base: Any, covariance: Any, basis: np.ndarray) -> Any:
    factor = np.asarray(covariance.factor, dtype=np.float64)
    if basis.shape[0] != factor.shape[1]:
        raise ValueError("projection basis does not match factor columns")
    projected = factor @ basis
    return base.CovarianceModel(
        np.asarray(covariance.mean_error, dtype=np.float64).copy(),
        np.asarray(covariance.diagonal, dtype=np.float64).copy(),
        projected,
        float(covariance.multiplier),
        float(covariance.marginal_z),
        float(covariance.source_marginal_coverage),
        float(covariance.source_joint_nanees),
    )


def query_covariance(model: Any, query_matrix: object) -> np.ndarray:
    query = np.asarray(query_matrix, dtype=np.float64)
    diagonal = float(model.multiplier) * np.asarray(model.diagonal, dtype=np.float64)
    factor = math.sqrt(float(model.multiplier)) * np.asarray(
        model.factor,
        dtype=np.float64,
    )
    result = (query * diagonal[None, :]) @ query.T
    if factor.shape[1]:
        projected = query @ factor
        result += projected @ projected.T
    return np.asarray(result, dtype=np.float64)


def maximum_metric_difference(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> float:
    differences = []
    for metric in METRICS:
        differences.append(abs(float(reference[metric]) - float(candidate[metric])))
    return float(max(differences, default=0.0))


def bootstrap_interval(
    values: object,
    repetitions: int,
    seed: int,
) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or len(vector) == 0:
        raise ValueError("bootstrap values must be a nonempty vector")
    if len(vector) == 1:
        return [float(vector[0]), float(vector[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vector), size=(repetitions, len(vector)))
    means = vector[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def filter_bound_descriptors(
    descriptors: Sequence[Any],
    expected_ids: Sequence[int],
) -> list[Any]:
    expected = [int(value) for value in expected_ids]
    by_id = {int(item.episode_id): item for item in descriptors}
    if len(by_id) != len(descriptors):
        raise ValueError("live descriptor roster contains duplicate episode IDs")
    missing = [episode_id for episode_id in expected if episode_id not in by_id]
    if missing:
        raise ValueError(f"bound episodes are unavailable: {missing}")
    result = [by_id[episode_id] for episode_id in expected]
    if [int(item.episode_id) for item in result] != expected:
        raise RuntimeError("bound descriptor ordering changed")
    return result


def object_average(row: Mapping[str, Any], representation: str, metric: str) -> float:
    return float(
        np.mean(
            [
                float(query["representations"][representation][metric])
                for query in row["queries"].values()
            ]
        )
    )


def paired_spectral_summary(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    differences = np.asarray(
        [
            object_average(row, "full_low_rank", metric)
            - object_average(row, "spectral_rank_matched", metric)
            for row in rows
        ],
        dtype=np.float64,
    )
    return {
        "metric": metric,
        "full_minus_spectral_mean": float(np.mean(differences)),
        "full_minus_spectral_median": float(np.median(differences)),
        "object_bootstrap_95_interval": bootstrap_interval(
            differences,
            repetitions,
            seed,
        ),
        "full_wins": int(np.count_nonzero(differences < 0.0)),
        "ties": int(np.count_nonzero(differences == 0.0)),
        "full_losses": int(np.count_nonzero(differences > 0.0)),
    }


def run(
    *,
    protocol_path: Path,
    original_v6_runner_path: Path,
    original_v6_protocol_path: Path,
    parent_protocol_path: Path,
    parent_result_path: Path,
    readiness_path: Path,
    data_root: Path,
    original_v6_root: Path,
    parent_control_root: Path,
    frozen_root: Path,
) -> dict[str, Any]:
    data_root = data_root.resolve(strict=True)
    original_v6_root = original_v6_root.resolve(strict=True)
    parent_control_root = parent_control_root.resolve(strict=True)
    frozen_root = frozen_root.resolve(strict=True)
    protocol = read_json(protocol_path)
    validate_protocol(
        protocol,
        data_root=data_root,
        original_v6_root=original_v6_root,
        parent_control_root=parent_control_root,
        frozen_root=frozen_root,
    )

    v6 = load_module(original_v6_runner_path, "decision_sufficient_original_v6")
    original_protocol = read_json(original_v6_protocol_path)
    parent_protocol = read_json(parent_protocol_path)
    v6.validate_protocol(
        original_protocol,
        parent_control_root=parent_control_root,
        parent_protocol_path=parent_protocol_path,
        data_root=data_root,
    )
    parent_result = read_json(parent_result_path)
    parent_by_object = v6.validate_parent_result(
        parent_result,
        original_protocol,
        parent_result_path,
    )

    binding = original_protocol["parent_confirmation"]
    v5_path = parent_control_root / str(binding["runner_path"])
    v5 = load_module(v5_path, "decision_sufficient_parent_v5")
    readiness = read_json(readiness_path)
    manifest = v5.verify_readiness(
        readiness,
        parent_protocol,
        readiness_path,
    )
    v3, development, base_protocol = v5.validate_frozen_method(
        frozen_root,
        parent_protocol,
    )
    minimum = int(parent_protocol["selection"]["minimum_complete_episodes_per_object"])
    evaluation = protocol["evaluation"]
    relative_tolerance = float(evaluation["rank_tolerance_relative"])
    original_evaluation = original_protocol["evaluation"]
    point_rng = np.random.default_rng(int(development["statistics"]["random_seed"]))

    rows: list[dict[str, Any]] = []
    for index, expected in enumerate(manifest, start=1):
        object_id = str(expected["object_id"])
        print(
            f"[{index}/{len(manifest)}] decision-sufficient compression {object_id}",
            flush=True,
        )
        live_descriptors = v3.base.discover_object(data_root, object_id, minimum)
        descriptors = filter_bound_descriptors(
            live_descriptors,
            expected["complete_episode_ids"],
        )
        row, capture, source_truth, target_truth = v6.evaluate_object_with_capture(
            v3,
            descriptors,
            development,
            base_protocol,
            point_rng,
        )
        parent_row = parent_by_object[object_id]
        if v6.point_projection(row) != v6.point_projection(parent_row):
            raise RuntimeError(f"frozen point result changed: {object_id}")

        covariance = capture.covariance
        factor = np.asarray(covariance.factor, dtype=np.float64)
        source_errors = np.asarray(capture.source_residuals, dtype=np.float64)
        target_errors = np.asarray(capture.target_errors, dtype=np.float64)
        centered_source_errors = source_errors - source_errors.mean(
            axis=0,
            keepdims=True,
        )
        bank = v6.query_bank(target_truth.shape[1])
        query_names = tuple(bank)
        if query_names != tuple(evaluation["query_names"]):
            raise RuntimeError("registered query order changed")
        query_matrix = np.stack([bank[name][0] for name in query_names], axis=0)

        portfolio_basis, portfolio_singular = orthonormal_range(
            factor.T @ query_matrix.T,
            relative_tolerance,
        )
        portfolio_model = projected_model(v3.base, covariance, portfolio_basis)
        portfolio_rank = int(portfolio_basis.shape[1])
        spectral_model = projected_model(
            v3.base,
            covariance,
            spectral_basis(factor, portfolio_rank),
        )
        full_query_covariance = query_covariance(covariance, query_matrix)
        portfolio_query_covariance = query_covariance(
            portfolio_model,
            query_matrix,
        )
        portfolio_covariance_error = float(
            np.max(np.abs(full_query_covariance - portfolio_query_covariance))
        )

        original_arms = v6.covariance_arms(
            v3.base,
            covariance,
            seed=v6.stable_seed(
                int(original_evaluation["random_seed"]),
                object_id,
                "scrambled-factor",
            ),
        )
        query_rows: dict[str, Any] = {}
        object_metric_parity = 0.0
        object_variance_parity = 0.0
        per_query_ranks: list[int] = []
        per_query_reductions: list[float] = []
        for query_name in query_names:
            weight, event = bank[query_name]
            raw_variances = {
                arm_name: v6.covariance_query_variance(model, weight)
                for arm_name, model in original_arms.items()
            }
            calibration = v6.source_query_calibration(
                centered_source_errors,
                source_truth,
                weight,
                raw_variances,
                event=event,
                probability=float(original_evaluation["coverage_probability"]),
                event_quantile=float(
                    original_evaluation["event_threshold_quantile"]
                ),
            )
            scalar_basis, scalar_singular = orthonormal_range(
                factor.T @ np.asarray(weight, dtype=np.float64)[:, None],
                relative_tolerance,
            )
            scalar_model = projected_model(v3.base, covariance, scalar_basis)
            per_query_rank = int(scalar_basis.shape[1])
            per_query_ranks.append(per_query_rank)
            per_query_reductions.append(
                float(factor.shape[1] / max(per_query_rank, 1))
            )

            models = {
                "full_low_rank": covariance,
                "portfolio_sufficient": portfolio_model,
                "per_query_sufficient": scalar_model,
                "spectral_rank_matched": spectral_model,
            }
            representation_metrics = {
                name: v6.query_metrics(
                    centered_source_errors=centered_source_errors,
                    target_truth=target_truth,
                    target_errors=target_errors,
                    weight=weight,
                    event=event,
                    model=model,
                    calibration=calibration,
                    fallback_cost=float(original_evaluation["fallback_cost"]),
                    probability_clip=float(original_evaluation["probability_clip"]),
                )
                for name, model in models.items()
            }
            full_variance = v6.covariance_query_variance(covariance, weight)
            portfolio_variance = v6.covariance_query_variance(portfolio_model, weight)
            scalar_variance = v6.covariance_query_variance(scalar_model, weight)
            variance_error = max(
                abs(full_variance - portfolio_variance),
                abs(full_variance - scalar_variance),
            )
            metric_error = max(
                maximum_metric_difference(
                    representation_metrics["full_low_rank"],
                    representation_metrics["portfolio_sufficient"],
                ),
                maximum_metric_difference(
                    representation_metrics["full_low_rank"],
                    representation_metrics["per_query_sufficient"],
                ),
            )
            object_variance_parity = max(object_variance_parity, variance_error)
            object_metric_parity = max(object_metric_parity, metric_error)
            query_rows[query_name] = {
                "event": event,
                "weight_sha256": array_digest(weight),
                "full_query_variance": float(full_variance),
                "portfolio_query_variance": float(portfolio_variance),
                "per_query_variance": float(scalar_variance),
                "spectral_query_variance": float(
                    v6.covariance_query_variance(spectral_model, weight)
                ),
                "per_query_rank": per_query_rank,
                "per_query_singular_values": [
                    float(value) for value in scalar_singular
                ],
                "factor_entry_reduction": per_query_reductions[-1],
                "calibration": calibration,
                "representations": representation_metrics,
            }

        original_rank = int(factor.shape[1])
        rows.append(
            {
                "object_id": object_id,
                "source_episode_ids": row["source_episode_ids"],
                "target_episode_id": row["target_episode_id"],
                "target_action": row["target_action"],
                "target_action_family": row["target_action_family"],
                "field_dimension": int(target_truth.shape[1]),
                "window_count": int(target_truth.shape[0]),
                "parent_point_result_exact": True,
                "predictive_mean_sha256": array_digest(target_truth - target_errors),
                "query_matrix_sha256": array_digest(query_matrix),
                "original_factor_rank": original_rank,
                "portfolio_sufficient_rank": portfolio_rank,
                "portfolio_singular_values": [
                    float(value) for value in portfolio_singular
                ],
                "portfolio_factor_entry_reduction": float(
                    original_rank / max(portfolio_rank, 1)
                ),
                "per_query_ranks": per_query_ranks,
                "per_query_factor_entry_reductions": per_query_reductions,
                "portfolio_query_covariance_max_abs_error": (
                    portfolio_covariance_error
                ),
                "per_query_variance_max_abs_error": object_variance_parity,
                "exact_metric_parity_max_abs_error": object_metric_parity,
                "queries": query_rows,
            }
        )

    if len(rows) != int(evaluation["object_count"]):
        raise RuntimeError("complete registered object roster was not evaluated")

    all_original_ranks = np.asarray(
        [row["original_factor_rank"] for row in rows],
        dtype=np.float64,
    )
    all_portfolio_ranks = np.asarray(
        [row["portfolio_sufficient_rank"] for row in rows],
        dtype=np.float64,
    )
    all_per_query_ranks = np.asarray(
        [rank for row in rows for rank in row["per_query_ranks"]],
        dtype=np.float64,
    )
    all_per_query_reductions = np.asarray(
        [
            reduction
            for row in rows
            for reduction in row["per_query_factor_entry_reductions"]
        ],
        dtype=np.float64,
    )
    strict_reduction_fraction = float(
        np.mean(
            [
                rank < row["original_factor_rank"]
                for row in rows
                for rank in row["per_query_ranks"]
            ]
        )
    )
    max_portfolio_covariance_error = float(
        max(row["portfolio_query_covariance_max_abs_error"] for row in rows)
    )
    max_query_variance_error = float(
        max(row["per_query_variance_max_abs_error"] for row in rows)
    )
    max_metric_error = float(
        max(row["exact_metric_parity_max_abs_error"] for row in rows)
    )

    representation_summary = {
        representation: {
            metric: float(
                np.mean(
                    [
                        object_average(row, representation, metric)
                        for row in rows
                    ]
                )
            )
            for metric in METRICS
        }
        for representation in REPRESENTATIONS
    }
    repetitions = int(evaluation["bootstrap_repetitions"])
    seed = int(evaluation["bootstrap_seed"])
    spectral_comparisons = {
        metric: paired_spectral_summary(
            rows,
            metric,
            repetitions=repetitions,
            seed=seed + index,
        )
        for index, metric in enumerate(
            ("decision_loss", "event_brier", "query_nll")
        )
    }

    gates = {
        "complete_92_object_roster": len(rows) == int(evaluation["object_count"]),
        "exact_parent_point_result_reproduced": all(
            bool(row["parent_point_result_exact"]) for row in rows
        ),
        "portfolio_joint_query_covariance_exact": (
            max_portfolio_covariance_error
            <= float(evaluation["portfolio_query_covariance_tolerance"])
        ),
        "all_scalar_query_variances_exact": (
            max_query_variance_error
            <= float(evaluation["per_query_variance_tolerance"])
        ),
        "all_registered_scores_and_decisions_exact": (
            max_metric_error <= float(evaluation["metric_parity_tolerance"])
        ),
        "portfolio_rank_at_most_query_count": bool(
            np.all(all_portfolio_ranks <= len(evaluation["query_names"]))
        ),
        "scalar_query_rank_at_most_one": bool(np.all(all_per_query_ranks <= 1.0)),
        "strict_per_query_rank_reduction_is_widespread": (
            strict_reduction_fraction
            >= float(
                evaluation[
                    "minimum_objects_with_strict_per_query_rank_reduction_fraction"
                ]
            )
        ),
        "median_per_query_factor_reduction_reaches_target": (
            float(np.median(all_per_query_reductions))
            >= float(evaluation["minimum_median_per_query_factor_entry_reduction"])
        ),
    }
    contribution_supported = all(gates.values())
    summary = {
        "object_count": len(rows),
        "query_count": len(evaluation["query_names"]),
        "object_query_count": int(len(rows) * len(evaluation["query_names"])),
        "original_factor_rank": {
            "minimum": int(np.min(all_original_ranks)),
            "median": float(np.median(all_original_ranks)),
            "maximum": int(np.max(all_original_ranks)),
        },
        "portfolio_sufficient_rank": {
            "minimum": int(np.min(all_portfolio_ranks)),
            "median": float(np.median(all_portfolio_ranks)),
            "maximum": int(np.max(all_portfolio_ranks)),
        },
        "per_query_sufficient_rank": {
            "minimum": int(np.min(all_per_query_ranks)),
            "median": float(np.median(all_per_query_ranks)),
            "maximum": int(np.max(all_per_query_ranks)),
        },
        "strict_per_query_rank_reduction_fraction": strict_reduction_fraction,
        "per_query_factor_entry_reduction": {
            "minimum": float(np.min(all_per_query_reductions)),
            "median": float(np.median(all_per_query_reductions)),
            "maximum": float(np.max(all_per_query_reductions)),
        },
        "portfolio_query_covariance_max_abs_error": (
            max_portfolio_covariance_error
        ),
        "per_query_variance_max_abs_error": max_query_variance_error,
        "registered_metric_max_abs_error": max_metric_error,
        "representation_summary": representation_summary,
        "full_vs_spectral_rank_matched": spectral_comparisons,
    }
    decision = {
        "gates": gates,
        "decision_sufficient_compression_supported": contribution_supported,
        "minimum_rank_claim_scope": (
            "fixed supplied decomposition D + U U^T, fixed D, fixed linear query "
            "portfolio, and orthogonal latent-factor projections U -> U V"
        ),
        "exact_registered_decision_preservation_supported": (
            gates["all_registered_scores_and_decisions_exact"]
        ),
        "global_field_covariance_preservation_supported": False,
        "arbitrary_future_query_preservation_supported": False,
        "calibration_supported": False,
        "closed_loop_robot_control_supported": False,
        "paper_claim_authorized": False,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "dataset_root": str(data_root),
        "information_boundary": {
            "retrospective_target_reuse": True,
            "exact_frozen_point_predictor_reused": True,
            "exact_parent_point_result_reproduced": True,
            "same_predictive_mean_for_all_representations": True,
            "query_projection_uses_target_outcomes": False,
            "query_projection_uses_only_registered_queries_and_covariance_factor": True,
            "source_calibration_changed": False,
            "decision_rule_changed": False,
            "new_measurements_collected": False,
        },
        "theorem_contract": {
            "condition": "range(U^T Q^T) subseteq range(V)",
            "minimum_rank": "rank(Q U)",
            "preserved_distribution": "Q X",
            "decision_corollary": (
                "Every measurable decision rule and loss depending only on the "
                "registered Gaussian query vector has identical Bayes risk."
            ),
        },
        "summary": summary,
        "decision": decision,
        "objects": rows,
        "protocol": protocol,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "object_id",
        "target_episode_id",
        "target_action_family",
        "original_factor_rank",
        "portfolio_sufficient_rank",
        "portfolio_factor_entry_reduction",
        "maximum_per_query_rank",
        "median_per_query_factor_entry_reduction",
        "portfolio_query_covariance_max_abs_error",
        "per_query_variance_max_abs_error",
        "exact_metric_parity_max_abs_error",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "object_id": row["object_id"],
                    "target_episode_id": row["target_episode_id"],
                    "target_action_family": row["target_action_family"],
                    "original_factor_rank": row["original_factor_rank"],
                    "portfolio_sufficient_rank": row[
                        "portfolio_sufficient_rank"
                    ],
                    "portfolio_factor_entry_reduction": row[
                        "portfolio_factor_entry_reduction"
                    ],
                    "maximum_per_query_rank": max(row["per_query_ranks"]),
                    "median_per_query_factor_entry_reduction": float(
                        np.median(row["per_query_factor_entry_reductions"])
                    ),
                    "portfolio_query_covariance_max_abs_error": row[
                        "portfolio_query_covariance_max_abs_error"
                    ],
                    "per_query_variance_max_abs_error": row[
                        "per_query_variance_max_abs_error"
                    ],
                    "exact_metric_parity_max_abs_error": row[
                        "exact_metric_parity_max_abs_error"
                    ],
                }
            )


def make_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    decision = result["decision"]
    full = summary["representation_summary"]["full_low_rank"]
    portfolio = summary["representation_summary"]["portfolio_sufficient"]
    per_query = summary["representation_summary"]["per_query_sufficient"]
    spectral = summary["representation_summary"]["spectral_rank_matched"]
    lines = [
        "# Deform360 decision-sufficient covariance compression v1",
        "",
        f"- Objects: **{summary['object_count']}**",
        f"- Registered scalar queries: **{summary['query_count']}**",
        f"- Object-query evaluations: **{summary['object_query_count']}**",
        "- Exact registered decision preservation: "
        f"**{str(decision['exact_registered_decision_preservation_supported']).lower()}**",
        "- Decision-sufficient compression supported: "
        f"**{str(decision['decision_sufficient_compression_supported']).lower()}**",
        "",
        "## Rank and exactness",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        "| Original factor rank, min/median/max | "
        f"{summary['original_factor_rank']['minimum']}/"
        f"{summary['original_factor_rank']['median']:.3g}/"
        f"{summary['original_factor_rank']['maximum']} |",
        "| Five-query sufficient rank, min/median/max | "
        f"{summary['portfolio_sufficient_rank']['minimum']}/"
        f"{summary['portfolio_sufficient_rank']['median']:.3g}/"
        f"{summary['portfolio_sufficient_rank']['maximum']} |",
        "| Scalar-query sufficient rank, min/median/max | "
        f"{summary['per_query_sufficient_rank']['minimum']}/"
        f"{summary['per_query_sufficient_rank']['median']:.3g}/"
        f"{summary['per_query_sufficient_rank']['maximum']} |",
        "| Median scalar-query factor-entry reduction | "
        f"{summary['per_query_factor_entry_reduction']['median']:.3g}x |",
        "| Maximum five-query covariance error | "
        f"{summary['portfolio_query_covariance_max_abs_error']:.3e} |",
        "| Maximum scalar-query variance error | "
        f"{summary['per_query_variance_max_abs_error']:.3e} |",
        "| Maximum registered metric difference | "
        f"{summary['registered_metric_max_abs_error']:.3e} |",
        "",
        "## Registered query and decision results",
        "",
        "| Representation | Decision loss | Brier | Query NLL | 90% coverage | nANEES |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in (
        ("Full", full),
        ("Portfolio sufficient", portfolio),
        ("Per-query sufficient", per_query),
        ("Spectral rank matched", spectral),
    ):
        lines.append(
            f"| {name} | {values['decision_loss']:.8g} | "
            f"{values['event_brier']:.8g} | {values['query_nll']:.8g} | "
            f"{values['target_90_coverage']:.3%} | "
            f"{values['target_query_nanees']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Frozen gates",
            "",
            "| Gate | Passed |",
            "|---|---:|",
        ]
    )
    for name, passed in decision["gates"].items():
        lines.append(f"| `{name}` | {str(bool(passed)).lower()} |")
    lines.extend(
        [
            "",
            "The full and sufficient representations use the same predictive mean,",
            "the same diagonal covariance block, the same source-only calibration,",
            "and the same decision rule. The sufficient factor is computed without",
            "target outcomes. It preserves the complete joint Gaussian distribution",
            "of the registered query portfolio, not the unrestricted field covariance",
            "or arbitrary future queries.",
            "",
            "Absolute calibration remains outside the supported claim because the",
            "parent Deform360 query study is underdispersed. This experiment tests",
            "lossless decision-facing representation, not calibration repair.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    rng = np.random.default_rng(19)
    dimension = 24
    latent_rank = 8
    query_count = 4
    factor = rng.normal(size=(dimension, latent_rank))
    query = rng.normal(size=(query_count, dimension))
    basis, _ = orthonormal_range(factor.T @ query.T, 1e-12)
    compressed = factor @ basis
    expected = (query @ factor) @ (query @ factor).T
    actual = (query @ compressed) @ (query @ compressed).T
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    if basis.shape[1] != min(query_count, latent_rank):
        raise RuntimeError("unexpected generic portfolio rank")
    for weight in query:
        scalar_basis, _ = orthonormal_range(factor.T @ weight[:, None], 1e-12)
        if scalar_basis.shape[1] > 1:
            raise RuntimeError("a scalar query required rank greater than one")
        scalar_factor = factor @ scalar_basis
        full_variance = float(np.sum((weight @ factor) ** 2))
        compressed_variance = float(np.sum((weight @ scalar_factor) ** 2))
        if not math.isclose(
            full_variance,
            compressed_variance,
            rel_tol=1e-11,
            abs_tol=1e-11,
        ):
            raise RuntimeError("scalar query variance was not preserved")
    print("decision-sufficient compression self-test passed", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--original-v6-runner", type=Path)
    parser.add_argument("--original-v6-protocol", type=Path)
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--parent-result", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--original-v6-root", type=Path)
    parser.add_argument("--parent-control-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (
        "protocol",
        "original_v6_runner",
        "original_v6_protocol",
        "parent_protocol",
        "parent_result",
        "readiness_json",
        "data_root",
        "original_v6_root",
        "parent_control_root",
        "frozen_root",
        "output_json",
        "output_report",
        "output_csv",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    result = run(
        protocol_path=args.protocol,
        original_v6_runner_path=args.original_v6_runner,
        original_v6_protocol_path=args.original_v6_protocol,
        parent_protocol_path=args.parent_protocol,
        parent_result_path=args.parent_result,
        readiness_path=args.readiness_json,
        data_root=args.data_root,
        original_v6_root=args.original_v6_root,
        parent_control_root=args.parent_control_root,
        frozen_root=args.frozen_root,
    )
    write_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(make_report(result), encoding="utf-8")
    write_csv(args.output_csv, result["objects"])
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    print(json.dumps(result["decision"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
