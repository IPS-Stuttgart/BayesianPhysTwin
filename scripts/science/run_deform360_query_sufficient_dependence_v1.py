#!/usr/bin/env python3
"""Test exact query-sufficient dependence compression on public Deform360.

This retrospective extension reproduces the frozen 92-object Deform360
same-mean dependence study from its exact bound carriers, then adds two
rank-matched factor representations:

* a query-sufficient latent projection that preserves the complete covariance
  of the registered five-query portfolio; and
* a leading-observation-energy projection at the same retained rank.

The query-sufficient arm is computed independently from the range of ``W U``
and through the pinned Prob4D query-compression implementation.  Both routes
must agree.  The point predictor, full covariance, query definitions, source
calibration, event thresholds, decisions, bootstrap, and original dependence
controls remain frozen.  No new target, camera, geometry, or measurement is
opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
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

SCHEMA = "bayesian-phystwin/deform360-query-sufficient-dependence-result-v1"
PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-query-sufficient-dependence-protocol-v1"
REFERENCE_SCHEMA = (
    "bayesian-phystwin/deform360-dependence-query-result-v6-bound-carrier-recovery-v1"
)
ARMS = (
    "full_low_rank",
    "query_sufficient_portfolio",
    "leading_energy_matched_rank",
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
    "calibration_log_error",
    "coverage_absolute_error",
)
PARITY_METRICS = (
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
_EPS = 1e-15


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


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=True,
    ).strip()


def _finite_matrix(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite matrix")
    return array


def canonical_projector_basis(vectors: np.ndarray) -> np.ndarray:
    """Return a deterministic orthonormal basis for a supplied subspace."""

    matrix = _finite_matrix(vectors, name="vectors")
    dimension, rank = matrix.shape
    if rank == 0:
        return np.empty((dimension, 0), dtype=np.float64)
    projector = matrix @ matrix.T
    basis: list[np.ndarray] = []
    tolerance = 512.0 * np.finfo(np.float64).eps * max(1, dimension)
    for axis in range(dimension):
        candidate = projector[:, axis].copy()
        for _ in range(2):
            for previous in basis:
                candidate -= previous * float(previous @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm <= tolerance:
            continue
        candidate /= norm
        pivot = int(np.argmax(np.abs(candidate)))
        if candidate[pivot] < 0.0:
            candidate *= -1.0
        basis.append(candidate)
        if len(basis) == rank:
            break
    if len(basis) != rank:
        raise RuntimeError("failed to construct canonical projector basis")
    result = np.column_stack(basis)
    if not np.allclose(
        result.T @ result,
        np.eye(rank),
        atol=1e-11,
        rtol=1e-11,
    ):
        raise RuntimeError("canonical projector basis is not orthonormal")
    return result


def exact_query_projection(
    factor: object,
    query_matrix: object,
    *,
    relative_rank_tolerance: float,
    absolute_rank_tolerance: float,
) -> dict[str, Any]:
    """Compute the minimum latent projector preserving ``W U U^T W^T``."""

    shared = _finite_matrix(factor, name="factor")
    queries = _finite_matrix(query_matrix, name="query_matrix")
    if queries.shape[1] != shared.shape[0]:
        raise ValueError("query_matrix and factor row dimensions disagree")
    if relative_rank_tolerance < 0.0 or absolute_rank_tolerance < 0.0:
        raise ValueError("rank tolerances must be nonnegative")
    projected = queries @ shared
    _, singular, right = np.linalg.svd(projected, full_matrices=False)
    scale = float(singular[0]) if len(singular) else 0.0
    threshold = max(float(absolute_rank_tolerance), relative_rank_tolerance * scale)
    rank = int(np.count_nonzero(singular > threshold))
    preliminary = right[:rank].T if rank else np.empty((shared.shape[1], 0))
    basis = canonical_projector_basis(preliminary)
    reduced = shared @ basis
    full_query_factor = projected
    reduced_query_factor = queries @ reduced
    full_covariance = full_query_factor @ full_query_factor.T
    reduced_covariance = reduced_query_factor @ reduced_query_factor.T
    difference = reduced_covariance - full_covariance
    max_abs = float(np.max(np.abs(difference), initial=0.0))
    relative = float(
        np.linalg.norm(difference, ord="fro")
        / max(np.linalg.norm(full_covariance, ord="fro"), _EPS)
    )
    residual = float(
        np.linalg.norm(projected - projected @ basis @ basis.T, ord="fro")
        / max(np.linalg.norm(projected, ord="fro"), _EPS)
    )
    return {
        "projection": basis,
        "compressed_factor": reduced,
        "singular_values": singular,
        "rank_threshold": threshold,
        "retained_rank": rank,
        "query_factor_residual_relative": residual,
        "query_covariance_max_abs_error": max_abs,
        "query_covariance_relative_frobenius_error": relative,
    }


def leading_energy_projection(factor: object, rank: int) -> dict[str, Any]:
    shared = _finite_matrix(factor, name="factor")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise TypeError("rank must be an integer")
    retained = int(rank)
    if not 0 <= retained <= shared.shape[1]:
        raise ValueError("rank is outside the factor rank")
    gram = shared.T @ shared
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    preliminary = eigenvectors[:, order[:retained]]
    basis = canonical_projector_basis(preliminary)
    reduced = shared @ basis
    full_trace = float(np.trace(gram))
    retained_trace = float(np.sum(reduced * reduced))
    return {
        "projection": basis,
        "compressed_factor": reduced,
        "eigenvalues": eigenvalues,
        "observation_trace_fraction": (
            1.0 if full_trace <= 0.0 else retained_trace / full_trace
        ),
    }


def query_covariance(
    model: Any,
    query_matrix: np.ndarray,
) -> np.ndarray:
    weights = _finite_matrix(query_matrix, name="query_matrix")
    diagonal = float(model.multiplier) * np.asarray(model.diagonal, dtype=np.float64)
    factor = math.sqrt(float(model.multiplier)) * np.asarray(
        model.factor,
        dtype=np.float64,
    )
    covariance = (weights * diagonal[None, :]) @ weights.T
    if factor.shape[1]:
        projected = weights @ factor
        covariance += projected @ projected.T
    return 0.5 * (covariance + covariance.T)


def covariance_model_with_factor(base: Any, source: Any, factor: np.ndarray) -> Any:
    return base.CovarianceModel(
        np.asarray(source.mean_error, dtype=np.float64).copy(),
        np.asarray(source.diagonal, dtype=np.float64).copy(),
        np.asarray(factor, dtype=np.float64).copy(),
        float(source.multiplier),
        float(source.marginal_z),
        float(source.source_marginal_coverage),
        float(source.source_joint_nanees),
    )


def relative_matrix_error(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    difference = np.asarray(first) - np.asarray(second)
    maximum = float(np.max(np.abs(difference), initial=0.0))
    relative = float(
        np.linalg.norm(difference, ord="fro")
        / max(np.linalg.norm(np.asarray(second), ord="fro"), _EPS)
    )
    return maximum, relative


def recursive_max_abs_difference(
    first: Any, second: Any, *, path: str = "root"
) -> float:
    """Return the largest scalar difference; reject structural mismatches."""

    if isinstance(first, Mapping) and isinstance(second, Mapping):
        if set(first) != set(second):
            raise ValueError(f"mapping keys differ at {path}")
        return max(
            (
                recursive_max_abs_difference(
                    first[key], second[key], path=f"{path}.{key}"
                )
                for key in first
            ),
            default=0.0,
        )
    if (
        isinstance(first, Sequence)
        and isinstance(second, Sequence)
        and not isinstance(first, (str, bytes, bytearray))
        and not isinstance(second, (str, bytes, bytearray))
    ):
        if len(first) != len(second):
            raise ValueError(f"sequence lengths differ at {path}")
        return max(
            (
                recursive_max_abs_difference(left, right, path=f"{path}[{index}]")
                for index, (left, right) in enumerate(zip(first, second, strict=True))
            ),
            default=0.0,
        )
    if isinstance(first, bool) or isinstance(second, bool):
        if first is not second:
            raise ValueError(f"boolean values differ at {path}")
        return 0.0
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        left = float(first)
        right = float(second)
        if not math.isfinite(left) or not math.isfinite(right):
            if left != right:
                raise ValueError(f"nonfinite values differ at {path}")
            return 0.0
        return abs(left - right)
    if first != second:
        raise ValueError(f"values differ at {path}: {first!r} != {second!r}")
    return 0.0


def reference_object_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "object_id",
        "source_episode_ids",
        "target_episode_id",
        "target_action",
        "target_action_family",
        "dimension",
        "window_count",
        "predictive_mean_sha256",
        "same_mean_by_construction",
        "parent_point_result_exact",
        "coordinate_marginal_parity_max_abs",
        "query_bank_sha256",
        "queries",
        "arm_summary",
        "joint_metrics",
    )
    return {key: row[key] for key in keys}


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    data_root: Path,
    original_v6_root: Path,
    recovery_root: Path,
    prob4d_root: Path,
) -> None:
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("schema_version") != 1:
        raise ValueError("unexpected query-sufficient protocol schema")
    if protocol.get("status") != "frozen-before-execution":
        raise ValueError("query-sufficient protocol is not frozen")
    allowed_roots = {Path(str(protocol.get("dataset_root")))}
    mirror_roots = protocol.get("exact_bound_mirror_roots", [])
    if not isinstance(mirror_roots, list) or not all(
        isinstance(value, str) for value in mirror_roots
    ):
        raise ValueError("exact-bound mirror roots are invalid")
    allowed_roots.update(Path(value) for value in mirror_roots)
    if data_root not in allowed_roots:
        raise ValueError("dataset root is not an authorized exact-bound carrier root")
    if protocol.get("paper_claim_authorized") is not False:
        raise ValueError("protocol may not self-authorize a paper claim")
    if protocol.get("fresh_confirmation_authorized") is not False:
        raise ValueError("protocol may not self-authorize fresh confirmation")
    if protocol.get("new_measurements_collected") is not False:
        raise ValueError("new measurements are forbidden")

    source = protocol.get("source_bindings")
    if not isinstance(source, Mapping):
        raise ValueError("source bindings are absent")
    if git_output(original_v6_root, "rev-parse", "HEAD") != source.get(
        "original_v6_revision"
    ):
        raise ValueError("original v6 revision changed")
    if git_output(
        original_v6_root,
        "hash-object",
        str(source["original_v6_runner_path"]),
    ) != source.get("original_v6_runner_git_blob_sha1"):
        raise ValueError("original v6 runner changed")
    if git_output(
        recovery_root, "hash-object", str(source["recovery_runner_path"])
    ) != source.get("recovery_runner_git_blob_sha1"):
        raise ValueError("recovery runner changed")
    if git_output(prob4d_root, "rev-parse", "HEAD") != source.get("prob4d_revision"):
        raise ValueError("Prob4D revision changed")
    if git_output(
        prob4d_root, "hash-object", str(source["prob4d_kernel_path"])
    ) != source.get("prob4d_kernel_git_blob_sha1"):
        raise ValueError("Prob4D query compressor changed")

    evaluation = protocol.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation contract is absent")
    query_names = tuple(item.get("name") for item in evaluation.get("query_bank", ()))
    if query_names != (
        "total_load",
        "sensor_imbalance",
        "horizontal_balance",
        "vertical_balance",
        "center_periphery",
    ):
        raise ValueError("registered query portfolio changed")
    if evaluation.get("maximum_portfolio_rank") != len(query_names):
        raise ValueError("maximum portfolio rank must equal query dimension")
    for key in (
        "same_point_predictor_required",
        "same_full_covariance_required",
        "same_source_calibration_required",
        "same_decision_rule_required",
        "complete_query_covariance_parity_required",
        "prob4d_and_direct_projection_agreement_required",
        "matched_rank_energy_control_required",
    ):
        if evaluation.get(key) is not True:
            raise ValueError(f"required evaluation invariant disabled: {key}")
    boundary = protocol.get("information_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("information boundary is absent")
    for key in (
        "target_outcomes_may_tune_projection",
        "target_outcomes_may_select_rank",
        "target_outcomes_may_select_queries",
        "camera_pixels_may_open",
        "geometry_or_point_cloud_may_open",
        "unbound_numeric_payloads_may_open",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"forbidden information flow enabled: {key}")


def validate_reference_result(
    result: Mapping[str, Any],
    path: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    binding = protocol["reference_result"]
    if sha256_file(path) != binding["result_json_sha256"]:
        raise ValueError("reference result file bytes changed")
    if result.get("schema") != REFERENCE_SCHEMA or result.get("status") != "complete":
        raise ValueError("reference dependence result is incomplete or unexpected")
    unsigned = dict(result)
    supplied = unsigned.pop("result_sha256", None)
    if canonical_digest(unsigned) != supplied:
        raise ValueError("reference result internal digest is invalid")
    if supplied != binding["result_sha256"]:
        raise ValueError("reference result identity changed")
    if result.get("github_sha") != binding["execution_revision"]:
        raise ValueError("reference execution revision changed")
    if result["summary"]["object_count"] != 92:
        raise ValueError("reference result does not contain 92 objects")
    if result["decision"]["dependence_value_supported"] is not True:
        raise ValueError("reference dependence-value result is not positive")
    rows = result.get("objects")
    if not isinstance(rows, list):
        raise ValueError("reference object results are absent")
    by_object = {str(row["object_id"]): row for row in rows}
    if len(by_object) != 92:
        raise ValueError("reference object rows are incomplete or duplicated")
    return by_object


def load_prob4d_compressor(prob4d_root: Path) -> Any:
    source = str((prob4d_root / "src").resolve(strict=True))
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module("prob4d.query_preserving_compression")


def prob4d_compress(
    qpc: Any,
    factor: np.ndarray,
    bank: Mapping[str, tuple[np.ndarray, str]],
    protocol: Mapping[str, Any],
) -> Any:
    if factor.shape[0] % 3:
        raise ValueError(
            "Prob4D block compressor requires a row count divisible by three"
        )
    evaluation = protocol["evaluation"]
    factor_3d = np.asarray(factor, dtype=np.float64).reshape(
        factor.shape[0] // 3,
        3,
        factor.shape[1],
    )
    queries = {
        name: np.asarray(weight, dtype=np.float64).reshape(factor.shape[0] // 3, 3)
        for name, (weight, _) in bank.items()
    }
    policy = qpc.QueryPreservingCompressionPolicyV1(
        minimum_observation_trace_fraction=0.0,
        maximum_query_trace_loss_fraction=float(
            evaluation["maximum_query_loss_fraction"]
        ),
        maximum_query_spectral_loss_fraction=float(
            evaluation["maximum_query_loss_fraction"]
        ),
        maximum_rank=int(evaluation["maximum_portfolio_rank"]),
        minimum_rank=0,
        observation_weight=0.0,
        query_weights={name: 1.0 for name in queries},
        relative_eigenspace_tolerance=float(
            evaluation["relative_eigenspace_tolerance"]
        ),
    )
    return qpc.compress_shared_factor_for_queries(
        factor_3d,
        queries,
        policy=policy,
    )


def summarize_query_metrics(values: list[Mapping[str, float]]) -> dict[str, float]:
    summary = {
        metric: float(np.mean([float(value[metric]) for value in values]))
        for metric in PARITY_METRICS
    }
    summary["calibration_log_error"] = float(
        np.mean(
            [
                abs(math.log(max(float(value["target_query_nanees"]), _EPS)))
                for value in values
            ]
        )
    )
    return summary


def add_coverage_error(
    summary: dict[str, float],
    values: list[Mapping[str, float]],
    coverage_probability: float,
) -> None:
    summary["coverage_absolute_error"] = float(
        np.mean(
            [
                abs(float(value["target_90_coverage"]) - coverage_probability)
                for value in values
            ]
        )
    )


def bootstrap_interval(values: np.ndarray, repetitions: int, seed: int) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    if len(vector) == 1:
        return [float(vector[0]), float(vector[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vector), size=(repetitions, len(vector)))
    return [
        float(value)
        for value in np.quantile(vector[indices].mean(axis=1), [0.025, 0.975])
    ]


def aggregate_extended(
    rows: list[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation = protocol["evaluation"]
    repetitions = int(evaluation["bootstrap_repetitions"])
    seed = int(evaluation["random_seed"])
    arm_summary = {
        arm: {
            metric: float(np.mean([row["arm_summary"][arm][metric] for row in rows]))
            for metric in METRICS
        }
        for arm in ARMS
    }

    comparisons: dict[str, Any] = {}
    for arm_index, arm in enumerate(ARMS[1:], start=1):
        comparisons[arm] = {}
        for metric_index, metric in enumerate(
            ("decision_loss", "event_brier", "query_nll", "target_90_coverage")
        ):
            differences = np.asarray(
                [
                    row["arm_summary"][arm][metric]
                    - row["arm_summary"]["full_low_rank"][metric]
                    for row in rows
                ],
                dtype=np.float64,
            )
            tolerance = float(evaluation["metric_parity_absolute_tolerance"])
            comparisons[arm][metric] = {
                "mean_difference": float(np.mean(differences)),
                "maximum_absolute_difference": float(np.max(np.abs(differences))),
                "object_bootstrap_95_interval": bootstrap_interval(
                    differences,
                    repetitions,
                    seed + arm_index * 1000 + metric_index,
                ),
                "object_wins": int(np.count_nonzero(differences < -tolerance)),
                "object_ties": int(np.count_nonzero(np.abs(differences) <= tolerance)),
                "object_losses": int(np.count_nonzero(differences > tolerance)),
            }

    original_ranks = np.asarray([row["compression"]["original_rank"] for row in rows])
    retained_ranks = np.asarray([row["compression"]["retained_rank"] for row in rows])
    full_bytes = np.asarray([row["storage"]["full_factor_bytes"] for row in rows])
    reduced_bytes = np.asarray(
        [row["storage"]["query_sufficient_factor_bytes"] for row in rows]
    )
    per_query_ranks: dict[str, list[int]] = {
        name: [int(row["per_query_ranks"][name]) for row in rows]
        for name in rows[0]["per_query_ranks"]
    }
    scalar_factor_bytes = np.asarray(
        [row["storage"]["separate_scalar_query_factor_bytes"] for row in rows]
    )

    parity = {
        "maximum_query_covariance_abs_error": float(
            max(row["compression"]["query_covariance_max_abs_error"] for row in rows)
        ),
        "maximum_query_covariance_relative_frobenius_error": float(
            max(
                row["compression"]["query_covariance_relative_frobenius_error"]
                for row in rows
            )
        ),
        "maximum_query_factor_residual_relative": float(
            max(row["compression"]["query_factor_residual_relative"] for row in rows)
        ),
        "maximum_prob4d_projector_difference": float(
            max(row["compression"]["prob4d_projector_difference"] for row in rows)
        ),
        "maximum_prob4d_query_covariance_abs_error": float(
            max(
                row["compression"]["prob4d_query_covariance_max_abs_error"]
                for row in rows
            )
        ),
        "maximum_full_metric_absolute_difference": float(
            max(
                row["compression"]["maximum_metric_absolute_difference"] for row in rows
            )
        ),
        "maximum_reference_reproduction_difference": float(
            max(row["reference_reproduction_max_abs_difference"] for row in rows)
        ),
    }
    rank_summary = {
        "original_min": int(np.min(original_ranks)),
        "original_median": float(np.median(original_ranks)),
        "original_max": int(np.max(original_ranks)),
        "retained_min": int(np.min(retained_ranks)),
        "retained_median": float(np.median(retained_ranks)),
        "retained_max": int(np.max(retained_ranks)),
        "strict_reduction_objects": int(
            np.count_nonzero(retained_ranks < original_ranks)
        ),
        "full_rank_objects": int(np.count_nonzero(retained_ranks == original_ranks)),
        "rank_histogram": {
            str(rank): int(np.count_nonzero(retained_ranks == rank))
            for rank in sorted(set(map(int, retained_ranks)))
        },
        "per_scalar_query": {
            name: {
                "minimum": int(np.min(values)),
                "maximum": int(np.max(values)),
                "mean": float(np.mean(values)),
            }
            for name, values in per_query_ranks.items()
        },
    }
    storage = {
        "total_full_factor_bytes": int(np.sum(full_bytes)),
        "total_query_sufficient_factor_bytes": int(np.sum(reduced_bytes)),
        "aggregate_factor_payload_reduction": float(
            np.sum(full_bytes) / max(np.sum(reduced_bytes), 1)
        ),
        "median_object_factor_payload_reduction": float(
            np.median(full_bytes / np.maximum(reduced_bytes, 1))
        ),
        "total_separate_scalar_query_factor_bytes": int(np.sum(scalar_factor_bytes)),
        "portfolio_covariance_cache_bytes_full_matrix": 5 * 5 * 8,
        "portfolio_covariance_cache_bytes_symmetric": 5 * 6 // 2 * 8,
        "five_scalar_variance_cache_bytes": 5 * 8,
        "boundary": (
            "Factor payload excludes the unchanged diagonal and means. For an immutable "
            "query portfolio, directly caching its covariance or five scalar variances "
            "is smaller than retaining a factor and remains the required simple baseline."
        ),
    }
    summary = {
        "object_count": len(rows),
        "query_count": 5,
        "arm_summary": arm_summary,
        "comparisons_against_full": comparisons,
        "rank_summary": rank_summary,
        "storage": storage,
        "parity": parity,
        "energy_control": {
            "mean_observation_trace_fraction": float(
                np.mean(
                    [
                        row["compression"]["leading_energy_observation_trace_fraction"]
                        for row in rows
                    ]
                )
            ),
            "maximum_query_covariance_abs_error": float(
                max(
                    row["compression"]["leading_energy_query_covariance_max_abs_error"]
                    for row in rows
                )
            ),
            "maximum_query_covariance_relative_frobenius_error": float(
                max(
                    row["compression"][
                        "leading_energy_query_covariance_relative_frobenius_error"
                    ]
                    for row in rows
                )
            ),
        },
        "reference_dependence_result": {
            "dependence_value_supported": reference["decision"][
                "dependence_value_supported"
            ],
            "query_calibration_supported": reference["decision"][
                "query_calibration_supported"
            ],
            "full_vs_diagonal": reference["summary"]["comparisons"][
                "diagonal_marginal_matched"
            ],
            "full_vs_scrambled": reference["summary"]["comparisons"][
                "scrambled_marginal_matched"
            ],
        },
    }

    tolerances = evaluation
    gates = {
        "complete_92_object_roster": len(rows) == 92,
        "reference_full_study_reproduced_exactly": parity[
            "maximum_reference_reproduction_difference"
        ]
        == 0.0,
        "original_dependence_value_supported": reference["decision"][
            "dependence_value_supported"
        ]
        is True,
        "all_retained_ranks_no_larger_than_query_dimension": rank_summary[
            "retained_max"
        ]
        <= int(evaluation["maximum_portfolio_rank"]),
        "prob4d_rank_matches_direct_minimum": all(
            row["compression"]["prob4d_retained_rank"]
            == row["compression"]["retained_rank"]
            for row in rows
        ),
        "prob4d_and_direct_subspaces_match": parity[
            "maximum_prob4d_projector_difference"
        ]
        <= float(tolerances["projector_absolute_tolerance"]),
        "complete_query_covariance_preserved": (
            parity["maximum_query_covariance_abs_error"]
            <= float(tolerances["query_covariance_absolute_tolerance"])
            and parity["maximum_query_covariance_relative_frobenius_error"]
            <= float(tolerances["query_covariance_relative_tolerance"])
            and parity["maximum_prob4d_query_covariance_abs_error"]
            <= float(tolerances["query_covariance_absolute_tolerance"])
        ),
        "all_frozen_query_metrics_preserved": parity[
            "maximum_full_metric_absolute_difference"
        ]
        <= float(tolerances["metric_parity_absolute_tolerance"]),
        "all_bound_carriers_and_actions_preserved": all(
            row["bound_carrier_recovery"]["bound_numeric_fingerprints_equal"]
            and row["bound_carrier_recovery"]["bound_episode_actions_equal"]
            and not row["bound_carrier_recovery"]["unbound_numeric_payloads_opened"]
            for row in rows
        ),
    }
    decision = {
        "gates": gates,
        "query_sufficient_representation_supported": all(gates.values()),
        "strict_factor_reduction_on_every_object": (
            rank_summary["strict_reduction_objects"] == len(rows)
        ),
        "existing_decision_advantage_retained_exactly": (
            all(gates.values())
            and reference["decision"]["dependence_value_supported"] is True
        ),
        "query_calibration_supported": False,
        "fresh_confirmation_authorized": False,
        "paper_claim_authorized": False,
        "deployment_safety_authorized": False,
    }
    return summary, decision


def make_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    decision = result["decision"]
    rank = summary["rank_summary"]
    storage = summary["storage"]
    parity = summary["parity"]
    lines = [
        "# Deform360 query-sufficient dependence compression v1",
        "",
        f"- Objects: **{summary['object_count']}**",
        f"- Registered queries: **{summary['query_count']}**",
        f"- Original factor rank: **{rank['original_min']}--{rank['original_max']}**",
        f"- Query-sufficient rank: **{rank['retained_min']}--{rank['retained_max']}**",
        f"- Strict reductions: **{rank['strict_reduction_objects']}/{summary['object_count']}**",
        "- Aggregate factor-only payload reduction: "
        f"**{storage['aggregate_factor_payload_reduction']:.3f}x**",
        "- Maximum five-query covariance error: "
        f"**{parity['maximum_query_covariance_abs_error']:.3e}**",
        "- Maximum frozen metric error: "
        f"**{parity['maximum_full_metric_absolute_difference']:.3e}**",
        "- Existing dependence value retained exactly: "
        f"**{str(decision['existing_decision_advantage_retained_exactly']).lower()}**",
        "- Query calibration supported: **false**",
        "",
        "## Object-balanced query metrics",
        "",
        "| Arm | Query nANEES | 90% coverage | NLL | Brier | Decision loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = summary["arm_summary"][arm]
        lines.append(
            f"| `{arm}` | {values['target_query_nanees']:.6g} | "
            f"{values['target_90_coverage']:.3%} | {values['query_nll']:.6g} | "
            f"{values['event_brier']:.6g} | {values['decision_loss']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Rank and representation boundary",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| Full factor bytes, all objects | {storage['total_full_factor_bytes']:,} |",
            "| Query-sufficient factor bytes, all objects | "
            f"{storage['total_query_sufficient_factor_bytes']:,} |",
            "| Five-query full covariance cache | "
            f"{storage['portfolio_covariance_cache_bytes_full_matrix']} bytes/object |",
            "| Five scalar variances only | "
            f"{storage['five_scalar_variance_cache_bytes']} bytes/object |",
            "",
            "The factor comparison excludes the unchanged diagonal and mean. A fixed-query",
            "cache is smaller and is reported as the appropriate simple baseline; the",
            "factor representation is useful only when the downstream interface consumes",
            "a structured covariance factor.",
            "",
            "## Registered gates",
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
            "This is retrospective representation evidence on the already opened, bound",
            "Deform360 cohort. It preserves the five registered predictive-query",
            "covariances and their frozen decisions; it does not preserve the full tactile",
            "field covariance, establish calibration, open a fresh cohort, or authorize",
            "deployment or a paper claim automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def write_object_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = [
        "object_id",
        "original_rank",
        "retained_rank",
        "prob4d_retained_rank",
        "full_factor_bytes",
        "query_sufficient_factor_bytes",
        "query_covariance_max_abs_error",
        "query_covariance_relative_frobenius_error",
        "maximum_metric_absolute_difference",
        "leading_energy_observation_trace_fraction",
        "leading_energy_query_covariance_max_abs_error",
        "full_decision_loss",
        "compressed_decision_loss",
        "energy_decision_loss",
        "full_brier",
        "compressed_brier",
        "energy_brier",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            compression = row["compression"]
            storage = row["storage"]
            writer.writerow(
                {
                    "object_id": row["object_id"],
                    "original_rank": compression["original_rank"],
                    "retained_rank": compression["retained_rank"],
                    "prob4d_retained_rank": compression["prob4d_retained_rank"],
                    "full_factor_bytes": storage["full_factor_bytes"],
                    "query_sufficient_factor_bytes": storage[
                        "query_sufficient_factor_bytes"
                    ],
                    "query_covariance_max_abs_error": compression[
                        "query_covariance_max_abs_error"
                    ],
                    "query_covariance_relative_frobenius_error": compression[
                        "query_covariance_relative_frobenius_error"
                    ],
                    "maximum_metric_absolute_difference": compression[
                        "maximum_metric_absolute_difference"
                    ],
                    "leading_energy_observation_trace_fraction": compression[
                        "leading_energy_observation_trace_fraction"
                    ],
                    "leading_energy_query_covariance_max_abs_error": compression[
                        "leading_energy_query_covariance_max_abs_error"
                    ],
                    "full_decision_loss": row["arm_summary"]["full_low_rank"][
                        "decision_loss"
                    ],
                    "compressed_decision_loss": row["arm_summary"][
                        "query_sufficient_portfolio"
                    ]["decision_loss"],
                    "energy_decision_loss": row["arm_summary"][
                        "leading_energy_matched_rank"
                    ]["decision_loss"],
                    "full_brier": row["arm_summary"]["full_low_rank"]["event_brier"],
                    "compressed_brier": row["arm_summary"][
                        "query_sufficient_portfolio"
                    ]["event_brier"],
                    "energy_brier": row["arm_summary"]["leading_energy_matched_rank"][
                        "event_brier"
                    ],
                }
            )


def run(
    *,
    protocol_path: Path,
    reference_result_path: Path,
    reference_protocol_path: Path,
    original_v6_runner_path: Path,
    recovery_runner_path: Path,
    parent_protocol_path: Path,
    parent_result_path: Path,
    readiness_path: Path,
    data_root: Path,
    original_v6_root: Path,
    recovery_root: Path,
    parent_control_root: Path,
    frozen_root: Path,
    prob4d_root: Path,
) -> dict[str, Any]:
    data_root = data_root.resolve(strict=True)
    original_v6_root = original_v6_root.resolve(strict=True)
    recovery_root = recovery_root.resolve(strict=True)
    parent_control_root = parent_control_root.resolve(strict=True)
    frozen_root = frozen_root.resolve(strict=True)
    prob4d_root = prob4d_root.resolve(strict=True)
    protocol = read_json(protocol_path)
    validate_protocol(
        protocol,
        data_root=data_root,
        original_v6_root=original_v6_root,
        recovery_root=recovery_root,
        prob4d_root=prob4d_root,
    )
    reference = read_json(reference_result_path)
    reference_by_object = validate_reference_result(
        reference,
        reference_result_path,
        protocol,
    )

    v6 = load_module(
        original_v6_runner_path.resolve(strict=True),
        "deform360_dependence_query_v6_for_query_sufficient_v1",
    )
    recovery = load_module(
        recovery_runner_path.resolve(strict=True),
        "deform360_bound_recovery_for_query_sufficient_v1",
    )
    qpc = load_prob4d_compressor(prob4d_root)
    reference_protocol = v6.read_json(reference_protocol_path)
    validation_protocol = json.loads(json.dumps(reference_protocol))
    validation_protocol["dataset_root"] = str(data_root)
    parent_result = v6.read_json(parent_result_path)
    parent_protocol = v6.read_json(parent_protocol_path)
    v6.validate_protocol(
        validation_protocol,
        parent_control_root=parent_control_root,
        parent_protocol_path=parent_protocol_path,
        data_root=data_root,
    )
    parent_by_object = v6.validate_parent_result(
        parent_result,
        reference_protocol,
        parent_result_path,
    )

    parent_binding = reference_protocol["parent_confirmation"]
    v5 = v6.load_module(
        parent_control_root / str(parent_binding["runner_path"]),
        "deform360_v5_parent_for_query_sufficient_v1",
    )
    manifest = v5.verify_readiness(
        v6.read_json(readiness_path),
        parent_protocol,
        readiness_path,
    )
    v3, development, base_protocol = v5.validate_frozen_method(
        frozen_root,
        parent_protocol,
    )
    audit = v6.load_module(
        parent_control_root / str(parent_binding["audit_path"]),
        "deform360_v5_audit_for_query_sufficient_v1",
    )
    minimum = int(parent_protocol["selection"]["minimum_complete_episodes_per_object"])

    evaluation = protocol["evaluation"]
    point_rng = np.random.default_rng(int(development["statistics"]["random_seed"]))
    rows: list[dict[str, Any]] = []
    reference_rows_current: list[dict[str, Any]] = []
    carrier_drift: list[dict[str, Any]] = []

    for index, expected_value in enumerate(manifest, start=1):
        if not isinstance(expected_value, Mapping):
            raise ValueError("readiness manifest row is not a mapping")
        expected = dict(expected_value)
        object_id = str(expected["object_id"])
        print(
            f"[{index}/{len(manifest)}] query-sufficient dependence {object_id}",
            flush=True,
        )
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
        carrier_drift.append(drift)
        point_row, capture, source_truth, target_truth = (
            v6.evaluate_object_with_capture(
                v3,
                descriptors,
                development,
                base_protocol,
                point_rng,
            )
        )
        exact_point = v6.point_projection(point_row) == v6.point_projection(parent_row)
        if not exact_point:
            raise RuntimeError(
                f"exact parent point result did not reproduce: {object_id}"
            )

        target_errors = np.asarray(capture.target_errors, dtype=np.float64)
        source_errors = np.asarray(capture.source_residuals, dtype=np.float64)
        predicted_mean = target_truth - target_errors
        original_arms = v6.covariance_arms(
            v3.base,
            capture.covariance,
            seed=v6.stable_seed(
                int(reference_protocol["evaluation"]["random_seed"]),
                object_id,
                "scrambled-factor",
            ),
        )
        full_model = original_arms["full_low_rank"]
        bank = v6.query_bank(target_truth.shape[1])
        query_matrix = np.stack([bank[name][0] for name, _ in v6.QUERY_SPECS])
        direct = exact_query_projection(
            np.asarray(full_model.factor, dtype=np.float64),
            query_matrix,
            relative_rank_tolerance=float(evaluation["relative_rank_tolerance"]),
            absolute_rank_tolerance=float(evaluation["absolute_rank_tolerance"]),
        )
        if direct["retained_rank"] > int(evaluation["maximum_portfolio_rank"]):
            raise RuntimeError(
                f"query rank exceeds registered portfolio size: {object_id}"
            )

        prob4d_result = prob4d_compress(
            qpc,
            np.asarray(full_model.factor, dtype=np.float64),
            bank,
            protocol,
        )
        prob4d_factor = np.asarray(prob4d_result.compressed_factor_m).reshape(
            target_truth.shape[1],
            prob4d_result.retained_rank,
        )
        direct_projector = direct["projection"] @ direct["projection"].T
        prob4d_projector = (
            np.asarray(prob4d_result.latent_projection)
            @ np.asarray(prob4d_result.latent_projection).T
        )
        projector_difference = float(
            np.linalg.norm(direct_projector - prob4d_projector, ord=2)
        )
        if prob4d_result.retained_rank != direct["retained_rank"]:
            raise RuntimeError(f"Prob4D retained a nonminimum rank: {object_id}")

        compressed_model = covariance_model_with_factor(
            v3.base,
            full_model,
            prob4d_factor,
        )
        energy = leading_energy_projection(
            np.asarray(full_model.factor, dtype=np.float64),
            int(direct["retained_rank"]),
        )
        energy_model = covariance_model_with_factor(
            v3.base,
            full_model,
            energy["compressed_factor"],
        )
        full_query_covariance = query_covariance(full_model, query_matrix)
        compressed_query_covariance = query_covariance(compressed_model, query_matrix)
        energy_query_covariance = query_covariance(energy_model, query_matrix)
        query_covariance_max, query_covariance_relative = relative_matrix_error(
            compressed_query_covariance,
            full_query_covariance,
        )
        energy_query_covariance_max, energy_query_covariance_relative = (
            relative_matrix_error(energy_query_covariance, full_query_covariance)
        )
        direct_model = covariance_model_with_factor(
            v3.base,
            full_model,
            direct["compressed_factor"],
        )
        prob4d_query_covariance_max, _ = relative_matrix_error(
            query_covariance(compressed_model, query_matrix),
            query_covariance(direct_model, query_matrix),
        )

        centered_source_errors = source_errors - source_errors.mean(
            axis=0,
            keepdims=True,
        )
        queries: dict[str, Any] = {}
        original_queries: dict[str, Any] = {}
        per_query_ranks: dict[str, int] = {}
        maximum_metric_difference = 0.0
        for query_name, (weight, event) in bank.items():
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
                probability=float(
                    reference_protocol["evaluation"]["coverage_probability"]
                ),
                event_quantile=float(
                    reference_protocol["evaluation"]["event_threshold_quantile"]
                ),
            )
            original_query = {
                "event": event,
                "weight_sha256": v6.array_digest(weight),
                "calibration": calibration,
                "arms": {},
            }
            for arm_name, model in original_arms.items():
                original_query["arms"][arm_name] = v6.query_metrics(
                    centered_source_errors=centered_source_errors,
                    target_truth=target_truth,
                    target_errors=target_errors,
                    weight=weight,
                    event=event,
                    model=model,
                    calibration=calibration,
                    fallback_cost=float(
                        reference_protocol["evaluation"]["fallback_cost"]
                    ),
                    probability_clip=float(
                        reference_protocol["evaluation"]["probability_clip"]
                    ),
                )
            full_metrics = original_query["arms"]["full_low_rank"]
            compressed_metrics = v6.query_metrics(
                centered_source_errors=centered_source_errors,
                target_truth=target_truth,
                target_errors=target_errors,
                weight=weight,
                event=event,
                model=compressed_model,
                calibration=calibration,
                fallback_cost=float(reference_protocol["evaluation"]["fallback_cost"]),
                probability_clip=float(
                    reference_protocol["evaluation"]["probability_clip"]
                ),
            )
            energy_metrics = v6.query_metrics(
                centered_source_errors=centered_source_errors,
                target_truth=target_truth,
                target_errors=target_errors,
                weight=weight,
                event=event,
                model=energy_model,
                calibration=calibration,
                fallback_cost=float(reference_protocol["evaluation"]["fallback_cost"]),
                probability_clip=float(
                    reference_protocol["evaluation"]["probability_clip"]
                ),
            )
            metric_difference = max(
                abs(float(compressed_metrics[name]) - float(full_metrics[name]))
                for name in PARITY_METRICS
            )
            maximum_metric_difference = max(
                maximum_metric_difference, metric_difference
            )
            scalar_projection = exact_query_projection(
                np.asarray(full_model.factor, dtype=np.float64),
                np.asarray(weight, dtype=np.float64)[None, :],
                relative_rank_tolerance=float(evaluation["relative_rank_tolerance"]),
                absolute_rank_tolerance=float(evaluation["absolute_rank_tolerance"]),
            )
            per_query_ranks[query_name] = int(scalar_projection["retained_rank"])
            queries[query_name] = {
                "event": event,
                "weight_sha256": v6.array_digest(weight),
                "calibration": calibration,
                "full": full_metrics,
                "query_sufficient_portfolio": compressed_metrics,
                "leading_energy_matched_rank": energy_metrics,
                "scalar_minimum_rank": int(scalar_projection["retained_rank"]),
                "scalar_query_covariance_max_abs_error": scalar_projection[
                    "query_covariance_max_abs_error"
                ],
            }
            original_queries[query_name] = original_query

        original_arm_summary: dict[str, dict[str, float]] = {}
        for arm_name in v6.COVARIANCE_ARMS:
            values = [
                original_queries[query_name]["arms"][arm_name]
                for query_name, _ in v6.QUERY_SPECS
            ]
            original_arm_summary[arm_name] = summarize_query_metrics(values)
            add_coverage_error(
                original_arm_summary[arm_name],
                values,
                float(reference_protocol["evaluation"]["coverage_probability"]),
            )

        arm_summary: dict[str, dict[str, float]] = {
            "full_low_rank": original_arm_summary["full_low_rank"]
        }
        for arm_name in ARMS[1:]:
            values = [queries[name][arm_name] for name, _ in v6.QUERY_SPECS]
            arm_summary[arm_name] = summarize_query_metrics(values)
            add_coverage_error(
                arm_summary[arm_name],
                values,
                float(reference_protocol["evaluation"]["coverage_probability"]),
            )

        original_result_row = {
            "object_id": object_id,
            "source_episode_ids": point_row["source_episode_ids"],
            "target_episode_id": point_row["target_episode_id"],
            "target_action": point_row["target_action"],
            "target_action_family": point_row["target_action_family"],
            "dimension": int(target_truth.shape[1]),
            "window_count": int(target_truth.shape[0]),
            "predictive_mean_sha256": v6.array_digest(predicted_mean),
            "same_mean_by_construction": True,
            "parent_point_result_exact": exact_point,
            "coordinate_marginal_parity_max_abs": float(
                max(
                    np.max(
                        np.abs(
                            v6.marginal_variance(model)
                            - v6.marginal_variance(full_model)
                        )
                    )
                    for model in original_arms.values()
                )
            ),
            "query_bank_sha256": v6.canonical_digest(
                {
                    name: {
                        "event": event,
                        "weight_sha256": original_queries[name]["weight_sha256"],
                    }
                    for name, event in v6.QUERY_SPECS
                }
            ),
            "queries": original_queries,
            "arm_summary": original_arm_summary,
            "joint_metrics": {
                name: v6.joint_metrics(
                    v3.base,
                    target_errors,
                    model,
                    float(reference_protocol["evaluation"]["coverage_probability"]),
                )
                for name, model in original_arms.items()
            },
        }
        reference_difference = recursive_max_abs_difference(
            reference_object_projection(original_result_row),
            reference_object_projection(reference_by_object[object_id]),
            path=f"objects.{object_id}",
        )
        reference_rows_current.append(original_result_row)

        original_rank = int(np.asarray(full_model.factor).shape[1])
        retained_rank = int(prob4d_result.retained_rank)
        dimension = int(target_truth.shape[1])
        rows.append(
            {
                "object_id": object_id,
                "target_episode_id": point_row["target_episode_id"],
                "target_action_family": point_row["target_action_family"],
                "dimension": dimension,
                "window_count": int(target_truth.shape[0]),
                "reference_reproduction_max_abs_difference": reference_difference,
                "per_query_ranks": per_query_ranks,
                "queries": queries,
                "arm_summary": arm_summary,
                "compression": {
                    "original_rank": original_rank,
                    "retained_rank": retained_rank,
                    "direct_minimum_rank": int(direct["retained_rank"]),
                    "prob4d_retained_rank": int(prob4d_result.retained_rank),
                    "prob4d_compression_applied": bool(
                        prob4d_result.compression_applied
                    ),
                    "prob4d_fallback_reason": prob4d_result.fallback_reason,
                    "prob4d_summary": prob4d_result.summary(),
                    "prob4d_projector_difference": projector_difference,
                    "query_factor_residual_relative": direct[
                        "query_factor_residual_relative"
                    ],
                    "query_covariance_max_abs_error": query_covariance_max,
                    "query_covariance_relative_frobenius_error": (
                        query_covariance_relative
                    ),
                    "prob4d_query_covariance_max_abs_error": (
                        prob4d_query_covariance_max
                    ),
                    "maximum_metric_absolute_difference": maximum_metric_difference,
                    "leading_energy_observation_trace_fraction": energy[
                        "observation_trace_fraction"
                    ],
                    "leading_energy_query_covariance_max_abs_error": (
                        energy_query_covariance_max
                    ),
                    "leading_energy_query_covariance_relative_frobenius_error": (
                        energy_query_covariance_relative
                    ),
                    "direct_singular_values": [
                        float(value) for value in direct["singular_values"]
                    ],
                    "direct_rank_threshold": float(direct["rank_threshold"]),
                },
                "storage": {
                    "full_factor_bytes": dimension * original_rank * 8,
                    "query_sufficient_factor_bytes": dimension * retained_rank * 8,
                    "leading_energy_factor_bytes": dimension * retained_rank * 8,
                    "separate_scalar_query_factor_bytes": (
                        dimension * sum(per_query_ranks.values()) * 8
                    ),
                },
                "bound_carrier_recovery": drift,
            }
        )

    current_summary, current_decision = v6.aggregate(
        reference_rows_current,
        reference_protocol,
    )
    reference_summary_difference = recursive_max_abs_difference(
        current_summary,
        reference["summary"],
        path="summary",
    )
    reference_decision_difference = recursive_max_abs_difference(
        current_decision,
        reference["decision"],
        path="decision",
    )
    if reference_summary_difference != 0.0 or reference_decision_difference != 0.0:
        raise RuntimeError(
            "the frozen reference dependence result did not reproduce exactly"
        )

    summary, decision = aggregate_extended(rows, protocol, reference)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "dataset_root": str(data_root),
        "source_bindings": protocol["source_bindings"],
        "reference_result": protocol["reference_result"],
        "information_boundary": {
            "official_dataset_root": str(protocol["dataset_root"]),
            "operational_dataset_root": str(data_root),
            "exact_bound_mirror_used": (
                str(data_root) != str(protocol["dataset_root"])
            ),
            "retrospective_target_reuse": True,
            "exact_reference_study_reproduced": True,
            "exact_parent_bound_numeric_carriers_reused": True,
            "point_predictor_changed": False,
            "full_covariance_changed_before_compression": False,
            "query_bank_changed": False,
            "source_calibration_changed": False,
            "decision_rule_changed": False,
            "target_outcomes_used_to_select_projection_or_rank": False,
            "unbound_numeric_payloads_opened": False,
            "camera_pixels_opened": False,
            "geometry_or_point_cloud_opened": False,
            "new_measurements_collected": False,
            "full_field_covariance_preserved": False,
            "registered_query_portfolio_covariance_preserved": True,
        },
        "carrier_drift_summary": {
            "object_count": len(carrier_drift),
            "all_bound_numeric_fingerprints_equal": all(
                row["bound_numeric_fingerprints_equal"] for row in carrier_drift
            ),
            "all_bound_episode_actions_equal": all(
                row["bound_episode_actions_equal"] for row in carrier_drift
            ),
            "unbound_numeric_payloads_opened": False,
            "carrier_drift_sha256": canonical_digest(carrier_drift),
        },
        "summary": summary,
        "decision": decision,
        "objects": rows,
        "protocol": protocol,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def self_test() -> None:
    rng = np.random.default_rng(7)
    factor = rng.normal(size=(24, 8))
    query_matrix = rng.normal(size=(5, 24))
    direct = exact_query_projection(
        factor,
        query_matrix,
        relative_rank_tolerance=1e-12,
        absolute_rank_tolerance=1e-14,
    )
    assert direct["retained_rank"] <= 5
    assert direct["query_covariance_relative_frobenius_error"] < 1e-12
    energy = leading_energy_projection(factor, direct["retained_rank"])
    full = (query_matrix @ factor) @ (query_matrix @ factor).T
    reduced = (query_matrix @ energy["compressed_factor"]) @ (
        query_matrix @ energy["compressed_factor"]
    ).T
    assert np.linalg.norm(reduced - full) > 1e-8

    zero_query = np.zeros((2, 24))
    zero = exact_query_projection(
        factor,
        zero_query,
        relative_rank_tolerance=1e-12,
        absolute_rank_tolerance=1e-14,
    )
    assert zero["retained_rank"] == 0
    assert zero["compressed_factor"].shape == (24, 0)

    vectors = rng.normal(size=(8, 3))
    orthonormal, _ = np.linalg.qr(vectors)
    canonical = canonical_projector_basis(orthonormal)
    assert np.allclose(
        canonical @ canonical.T,
        orthonormal @ orthonormal.T,
        atol=1e-11,
        rtol=1e-11,
    )
    print("query-sufficient dependence self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--reference-protocol", type=Path)
    parser.add_argument("--original-v6-runner", type=Path)
    parser.add_argument("--recovery-runner", type=Path)
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--parent-result", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--original-v6-root", type=Path)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--parent-control-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--prob4d-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (
        "protocol",
        "reference_result",
        "reference_protocol",
        "original_v6_runner",
        "recovery_runner",
        "parent_protocol",
        "parent_result",
        "readiness_json",
        "data_root",
        "original_v6_root",
        "recovery_root",
        "parent_control_root",
        "frozen_root",
        "prob4d_root",
        "output_json",
        "output_report",
        "output_csv",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    result = run(
        protocol_path=args.protocol,
        reference_result_path=args.reference_result,
        reference_protocol_path=args.reference_protocol,
        original_v6_runner_path=args.original_v6_runner,
        recovery_runner_path=args.recovery_runner,
        parent_protocol_path=args.parent_protocol,
        parent_result_path=args.parent_result,
        readiness_path=args.readiness_json,
        data_root=args.data_root,
        original_v6_root=args.original_v6_root,
        recovery_root=args.recovery_root,
        parent_control_root=args.parent_control_root,
        frozen_root=args.frozen_root,
        prob4d_root=args.prob4d_root,
    )
    write_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(make_report(result), encoding="utf-8")
    write_object_csv(args.output_csv, result["objects"])
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    print(json.dumps(result["decision"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
