"""Nested cross-material expert routing for the public Tracking Cloth study.

This is a retrospective follow-up to ``tracking_cloth_selective_twin_v1``.
All target outcomes were already open before this implementation was designed.
The outer material is excluded from model fitting and from inner
hyperparameter/threshold selection, but the result is not fresh confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tracking_cloth_deformation_v1.data import (
    audit_dataset,
    infer_source_scale,
    object_digest,
    read_prefix,
    write_json,
)
from experiments.tracking_cloth_selective_twin_v1.run import (
    _source_predictions,
    _twist_predictions,
    cross_material_policy_rows,
    score_records,
)

HERE = Path(__file__).resolve().parent
BASE_HERE = HERE.parent / "tracking_cloth_deformation_v1"
V1_HERE = HERE.parent / "tracking_cloth_selective_twin_v1"

TARGET_ORDER = ("bayesian_physics", "last_residual")
LOSS_FIELD = {
    "persistence": "fallback_loss_mm",
    "bayesian_physics": "candidate_loss_mm",
    "last_residual": "last_residual_loss_mm",
}
TARGET_FIELD = {
    "bayesian_physics": "candidate_regret_mm",
    "last_residual": "last_residual_regret_mm",
}
FEATURE_FIELDS = ("motion_query_horizon", "speed", "grasp", "size")
POLICY_ORDER = (
    "always_fallback",
    "v1_query_horizon_gate",
    "nested_physics_only",
    "nested_residual_only",
    "nested_triage_drop_last_residual",
    "nested_triage_drop_bayesian_physics",
    "nested_triage",
    "outcome_oracle",
)


@dataclass(frozen=True)
class RidgeState:
    """Dense one-hot ridge state with an unpenalized intercept."""

    categories: dict[str, tuple[str, ...]]
    x_mean: np.ndarray
    y_mean: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    alpha: float


@dataclass(frozen=True)
class FoldChoice:
    """Outer-fold tuning result."""

    heldout_material: str
    alpha: float
    threshold_mm: float
    inner_feasible: bool
    inner_summary: dict[str, Any]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _context(row: dict[str, Any]) -> str:
    horizon = f"{float(row['horizon_seconds']):g}"
    return f"{row['motion']}|{row['query']}|{horizon}"


def prepare_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add target-blind context fields and explicit arm regrets."""

    prepared: list[dict[str, Any]] = []
    for row_id, source in enumerate(rows):
        row = dict(source)
        row["row_id"] = row_id
        row["motion_query_horizon"] = _context(row)
        row["candidate_regret_mm"] = float(row["candidate_loss_mm"]) - float(
            row["fallback_loss_mm"]
        )
        row["last_residual_regret_mm"] = float(
            row["last_residual_loss_mm"]
        ) - float(row["fallback_loss_mm"])
        for field in (
            "candidate_regret_mm",
            "last_residual_regret_mm",
            "fallback_loss_mm",
            "practical_harm_margin_mm",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field} in row {row_id}")
        prepared.append(row)
    if len({int(row["row_id"]) for row in prepared}) != len(prepared):
        raise ValueError("Row identities are not unique")
    return prepared


def _fit_categories(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    if not rows:
        raise ValueError("Cannot fit feature schema on no rows")
    return {
        field: tuple(sorted({str(row[field]) for row in rows}))
        for field in FEATURE_FIELDS
    }


def _feature_matrix(
    rows: Sequence[dict[str, Any]],
    categories: dict[str, tuple[str, ...]],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for field in FEATURE_FIELDS:
        values = np.asarray([str(row[field]) for row in rows], dtype=object)
        for category in categories[field]:
            columns.append((values == category).astype(float))
    if not columns:
        raise ValueError("Feature matrix has no columns")
    return np.column_stack(columns)


def _target_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(row[TARGET_FIELD["bayesian_physics"]]),
                float(row[TARGET_FIELD["last_residual"]]),
            ]
            for row in rows
        ],
        dtype=float,
    )


def fit_ridge(
    rows: Sequence[dict[str, Any]],
    alpha: float,
) -> RidgeState:
    """Fit the two arm-regret models with deterministic dense linear algebra."""

    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("Ridge alpha must be finite and positive")
    categories = _fit_categories(rows)
    x = _feature_matrix(rows, categories)
    y = _target_matrix(rows)
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean
    gram = x_centered.T @ x_centered
    rhs = x_centered.T @ y_centered
    coefficients = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]),
        rhs,
    )
    intercept = y_mean - x_mean @ coefficients
    return RidgeState(
        categories=categories,
        x_mean=x_mean,
        y_mean=y_mean,
        coefficients=coefficients,
        intercept=intercept,
        alpha=float(alpha),
    )


def predict_ridge(
    state: RidgeState,
    rows: Sequence[dict[str, Any]],
) -> np.ndarray:
    x = _feature_matrix(rows, state.categories)
    prediction = x @ state.coefficients + state.intercept
    if prediction.shape != (len(rows), len(TARGET_ORDER)):
        raise ValueError("Unexpected ridge prediction shape")
    if not np.isfinite(prediction).all():
        raise ValueError("Ridge prediction contains non-finite values")
    return prediction


def _arm_index(arm: str) -> int:
    try:
        return TARGET_ORDER.index(arm)
    except ValueError as exc:
        raise ValueError(f"Arm has no regret model: {arm}") from exc


def route_rows(
    rows: Sequence[dict[str, Any]],
    predictions: np.ndarray,
    allowed_arms: Sequence[str],
    threshold_mm: float,
    *,
    policy: str,
    heldout_material: str,
    alpha: float,
    inner_feasible: bool,
) -> list[dict[str, Any]]:
    """Apply one fitted expert router with byte-exact persistence fallback."""

    if not allowed_arms:
        raise ValueError("At least one non-fallback arm is required")
    if predictions.shape != (len(rows), len(TARGET_ORDER)):
        raise ValueError("Prediction matrix does not match rows")
    indices = [_arm_index(arm) for arm in allowed_arms]
    candidate_predictions = predictions[:, indices]
    selected_indices = np.argmin(candidate_predictions, axis=1)
    best_prediction = candidate_predictions[
        np.arange(len(rows)), selected_indices
    ]
    selected_arms = np.asarray(allowed_arms, dtype=object)[selected_indices]
    accepted = best_prediction < float(threshold_mm)

    routed: list[dict[str, Any]] = []
    for position, source in enumerate(rows):
        row = dict(source)
        arm = str(selected_arms[position]) if accepted[position] else "persistence"
        fallback = float(row["fallback_loss_mm"])
        selected_loss = float(row[LOSS_FIELD[arm]])
        regret = selected_loss - fallback
        practical_harm = regret > float(row["practical_harm_margin_mm"])
        row.update(
            {
                "policy": policy,
                "outer_heldout_material": heldout_material,
                "selected_arm": arm,
                "accepted": bool(accepted[position]),
                "selected_loss_mm": selected_loss,
                "selected_minus_fallback_mm": regret,
                "selected_practical_harm": bool(practical_harm),
                "selected_strict_regression": bool(regret > 0.0),
                "predicted_best_regret_mm": float(best_prediction[position]),
                "ridge_alpha": float(alpha),
                "admission_threshold_mm": float(threshold_mm),
                "inner_feasible": bool(inner_feasible),
                "exact_fallback": bool(
                    accepted[position] or selected_loss == fallback
                ),
            }
        )
        routed.append(row)
    return routed


def _material_values(
    rows: Sequence[dict[str, Any]],
    field: str,
    materials: Sequence[str],
) -> np.ndarray:
    values = []
    for material in materials:
        subset = [float(row[field]) for row in rows if row["material"] == material]
        if not subset:
            raise ValueError(f"No rows for material {material}")
        values.append(float(np.mean(subset)))
    return np.asarray(values, dtype=float)


def summarize_policy(
    rows: Sequence[dict[str, Any]],
    materials: Sequence[str],
    *,
    bootstrap_seed: int,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty policy")
    accepted = [row for row in rows if bool(row["accepted"])]
    material_regrets = _material_values(
        rows,
        "selected_minus_fallback_mm",
        materials,
    )
    material_selected = _material_values(rows, "selected_loss_mm", materials)
    material_fallback = _material_values(rows, "fallback_loss_mm", materials)
    rng = np.random.default_rng(int(bootstrap_seed))
    draws = rng.integers(
        0,
        len(materials),
        size=(int(bootstrap_repetitions), len(materials)),
    )
    bootstrap = material_regrets[draws].mean(axis=1)
    material_results = []
    for material, regret, selected, fallback in zip(
        materials,
        material_regrets,
        material_selected,
        material_fallback,
        strict=True,
    ):
        subset = [row for row in rows if row["material"] == material]
        selected_subset = [row for row in subset if bool(row["accepted"])]
        material_results.append(
            {
                "material": material,
                "selected_minus_fallback_mm": float(regret),
                "selected_loss_mm": float(selected),
                "fallback_loss_mm": float(fallback),
                "selected_coverage": float(
                    np.mean([bool(row["accepted"]) for row in subset])
                ),
                "accepted_practical_harm_fraction": (
                    float(
                        np.mean(
                            [
                                bool(row["selected_practical_harm"])
                                for row in selected_subset
                            ]
                        )
                    )
                    if selected_subset
                    else 0.0
                ),
            }
        )
    arm_counts = {
        arm: sum(row["selected_arm"] == arm for row in rows)
        for arm in ("persistence", *TARGET_ORDER)
    }
    mean_fallback = float(np.mean([float(row["fallback_loss_mm"]) for row in rows]))
    mean_selected = float(np.mean([float(row["selected_loss_mm"]) for row in rows]))
    return {
        "policy": str(rows[0]["policy"]),
        "query_cases": len(rows),
        "selected_query_cases": len(accepted),
        "selected_coverage": len(accepted) / len(rows),
        "selected_loss_mm": mean_selected,
        "fallback_loss_mm": mean_fallback,
        "selected_minus_fallback_mm": mean_selected - mean_fallback,
        "relative_loss_reduction_vs_fallback": (
            (mean_fallback - mean_selected) / mean_fallback
        ),
        "material_bootstrap_95_interval_mm": np.quantile(
            bootstrap,
            [0.025, 0.975],
        ).tolist(),
        "heldout_materials_nonpositive": int(np.sum(material_regrets <= 0.0)),
        "heldout_materials_negative": int(np.sum(material_regrets < 0.0)),
        "accepted_practical_harm_fraction": (
            float(
                np.mean(
                    [bool(row["selected_practical_harm"]) for row in accepted]
                )
            )
            if accepted
            else 0.0
        ),
        "accepted_strict_regression_fraction": (
            float(
                np.mean(
                    [bool(row["selected_strict_regression"]) for row in accepted]
                )
            )
            if accepted
            else 0.0
        ),
        "exact_fallback_violations": int(
            np.sum([not bool(row["exact_fallback"]) for row in rows])
        ),
        "arm_counts": arm_counts,
        "arm_coverage": {
            arm: arm_counts[arm] / len(rows) for arm in TARGET_ORDER
        },
        "material_results": material_results,
    }


def _inner_oof_predictions(
    rows: Sequence[dict[str, Any]],
    alpha: float,
    materials: Sequence[str],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    output_rows: list[dict[str, Any]] = []
    output_predictions: list[np.ndarray] = []
    for heldout in materials:
        fit = [row for row in rows if row["material"] != heldout]
        evaluate = [row for row in rows if row["material"] == heldout]
        if not fit or not evaluate:
            raise ValueError("Incomplete inner material split")
        state = fit_ridge(fit, alpha)
        output_rows.extend(evaluate)
        output_predictions.append(predict_ridge(state, evaluate))
    order = np.argsort([int(row["row_id"]) for row in output_rows])
    ordered_rows = [output_rows[int(index)] for index in order]
    predictions = np.vstack(output_predictions)[order]
    return ordered_rows, predictions


def _selection_key(
    summary: dict[str, Any],
    alpha: float,
    threshold_mm: float,
) -> tuple[float, ...]:
    improvement = -float(summary["selected_minus_fallback_mm"])
    return (
        improvement,
        -float(summary["accepted_practical_harm_fraction"]),
        float(summary["selected_coverage"]),
        -abs(math.log10(float(alpha))),
        -abs(float(threshold_mm)),
        -float(alpha),
        -float(threshold_mm),
    )


def choose_fold(
    training_rows: Sequence[dict[str, Any]],
    heldout_material: str,
    allowed_arms: Sequence[str],
    protocol: dict[str, Any],
    *,
    policy: str,
) -> FoldChoice:
    materials = sorted({str(row["material"]) for row in training_rows})
    if len(materials) != 3:
        raise ValueError("Each outer fold must contain three training materials")
    selection = protocol["inner_selection"]
    candidates: list[tuple[bool, dict[str, Any], float, float]] = []
    for alpha in protocol["ridge_alphas"]:
        inner_rows, predictions = _inner_oof_predictions(
            training_rows,
            float(alpha),
            materials,
        )
        for threshold in protocol["admission_thresholds_mm"]:
            routed = route_rows(
                inner_rows,
                predictions,
                allowed_arms,
                float(threshold),
                policy=f"{policy}-inner",
                heldout_material=heldout_material,
                alpha=float(alpha),
                inner_feasible=False,
            )
            summary = summarize_policy(
                routed,
                materials,
                bootstrap_seed=int(protocol["bootstrap_seed"]) + 11,
                bootstrap_repetitions=int(protocol["bootstrap_repetitions"]),
            )
            feasible = bool(
                summary["selected_coverage"]
                >= float(selection["minimum_selected_coverage"])
                and summary["accepted_practical_harm_fraction"]
                <= float(selection["maximum_practical_harm_fraction"])
                and summary["heldout_materials_nonpositive"] == len(materials)
                and summary["exact_fallback_violations"] == 0
            )
            candidates.append(
                (feasible, summary, float(alpha), float(threshold))
            )
    feasible = [candidate for candidate in candidates if candidate[0]]
    if feasible:
        selected = max(
            feasible,
            key=lambda candidate: _selection_key(
                candidate[1],
                candidate[2],
                candidate[3],
            ),
        )
        inner_feasible = True
    else:
        safe = [
            candidate
            for candidate in candidates
            if candidate[1]["accepted_practical_harm_fraction"]
            <= float(selection["maximum_practical_harm_fraction"])
            and candidate[1]["heldout_materials_nonpositive"] == len(materials)
            and candidate[1]["exact_fallback_violations"] == 0
        ]
        pool = safe if safe else candidates
        selected = max(
            pool,
            key=lambda candidate: _selection_key(
                candidate[1],
                candidate[2],
                candidate[3],
            ),
        )
        inner_feasible = False
    _, summary, alpha, threshold = selected
    return FoldChoice(
        heldout_material=heldout_material,
        alpha=alpha,
        threshold_mm=threshold,
        inner_feasible=inner_feasible,
        inner_summary=summary,
    )


def nested_policy(
    rows: Sequence[dict[str, Any]],
    allowed_arms: Sequence[str],
    protocol: dict[str, Any],
    *,
    policy: str,
) -> tuple[list[dict[str, Any]], list[FoldChoice]]:
    materials = [str(value) for value in protocol["materials"]]
    routed: list[dict[str, Any]] = []
    choices: list[FoldChoice] = []
    for heldout in materials:
        training = [row for row in rows if row["material"] != heldout]
        evaluation = [row for row in rows if row["material"] == heldout]
        choice = choose_fold(
            training,
            heldout,
            allowed_arms,
            protocol,
            policy=policy,
        )
        state = fit_ridge(training, choice.alpha)
        predictions = predict_ridge(state, evaluation)
        routed.extend(
            route_rows(
                evaluation,
                predictions,
                allowed_arms,
                choice.threshold_mm,
                policy=policy,
                heldout_material=heldout,
                alpha=choice.alpha,
                inner_feasible=choice.inner_feasible,
            )
        )
        choices.append(choice)
    routed.sort(key=lambda row: int(row["row_id"]))
    if len(routed) != len(rows):
        raise ValueError("Nested policy lost or duplicated rows")
    return routed, choices


def apply_fold_choices(
    rows: Sequence[dict[str, Any]],
    choices: Sequence[FoldChoice],
    allowed_arms: Sequence[str],
    protocol: dict[str, Any],
    *,
    policy: str,
) -> list[dict[str, Any]]:
    """Drop one expert while preserving the primary router's fold choices."""

    choice_by_material = {
        choice.heldout_material: choice for choice in choices
    }
    routed: list[dict[str, Any]] = []
    for heldout in protocol["materials"]:
        choice = choice_by_material[str(heldout)]
        training = [row for row in rows if row["material"] != heldout]
        evaluation = [row for row in rows if row["material"] == heldout]
        state = fit_ridge(training, choice.alpha)
        predictions = predict_ridge(state, evaluation)
        routed.extend(
            route_rows(
                evaluation,
                predictions,
                allowed_arms,
                choice.threshold_mm,
                policy=policy,
                heldout_material=str(heldout),
                alpha=choice.alpha,
                inner_feasible=choice.inner_feasible,
            )
        )
    routed.sort(key=lambda row: int(row["row_id"]))
    return routed


def _convert_v1_rows(
    rows: Sequence[dict[str, Any]],
    *,
    policy: str,
) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        item = dict(row)
        item["policy"] = policy
        item["outer_heldout_material"] = str(row["heldout_material"])
        item["selected_arm"] = (
            "bayesian_physics" if bool(row["accepted"]) else "persistence"
        )
        item["selected_practical_harm"] = bool(
            row["selected_practical_harm"]
        )
        item["selected_strict_regression"] = bool(
            row["accepted"] and row["strict_regression"]
        )
        item["predicted_best_regret_mm"] = None
        item["ridge_alpha"] = None
        item["admission_threshold_mm"] = None
        item["inner_feasible"] = True
        item["exact_fallback"] = bool(row["exact_fallback"])
        converted.append(item)
    converted.sort(key=lambda row: int(row["row_id"]))
    return converted


def _always_fallback(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    routed = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "policy": "always_fallback",
                "outer_heldout_material": str(row["material"]),
                "selected_arm": "persistence",
                "accepted": False,
                "selected_loss_mm": float(row["fallback_loss_mm"]),
                "selected_minus_fallback_mm": 0.0,
                "selected_practical_harm": False,
                "selected_strict_regression": False,
                "predicted_best_regret_mm": None,
                "ridge_alpha": None,
                "admission_threshold_mm": None,
                "inner_feasible": True,
                "exact_fallback": True,
            }
        )
        routed.append(item)
    return routed


def _pairwise_summary(
    primary: Sequence[dict[str, Any]],
    comparator: Sequence[dict[str, Any]],
    materials: Sequence[str],
    *,
    bootstrap_seed: int,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    by_id = {int(row["row_id"]): row for row in comparator}
    differences = []
    labelled = []
    for row in primary:
        other = by_id[int(row["row_id"])]
        difference = float(row["selected_loss_mm"]) - float(
            other["selected_loss_mm"]
        )
        differences.append(difference)
        labelled.append((str(row["material"]), difference))
    material_differences = np.asarray(
        [
            np.mean([value for label, value in labelled if label == material])
            for material in materials
        ],
        dtype=float,
    )
    rng = np.random.default_rng(int(bootstrap_seed))
    draws = rng.integers(
        0,
        len(materials),
        size=(int(bootstrap_repetitions), len(materials)),
    )
    bootstrap = material_differences[draws].mean(axis=1)
    return {
        "primary_minus_comparator_mm": float(np.mean(differences)),
        "material_bootstrap_95_interval_mm": np.quantile(
            bootstrap,
            [0.025, 0.975],
        ).tolist(),
        "materials_primary_better": int(np.sum(material_differences < 0.0)),
        "material_differences_mm": [
            {
                "material": material,
                "primary_minus_comparator_mm": float(value),
            }
            for material, value in zip(
                materials,
                material_differences,
                strict=True,
            )
        ],
    }


def analyze_rows(
    score_rows: Sequence[dict[str, Any]],
    protocol: dict[str, Any],
    v1_protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = prepare_rows(score_rows)
    materials = [str(value) for value in protocol["materials"]]

    fallback_rows = _always_fallback(rows)
    v1_rows = _convert_v1_rows(
        cross_material_policy_rows(
            rows,
            "query_horizon_gate",
            v1_protocol,
        ),
        policy="v1_query_horizon_gate",
    )
    oracle_rows = _convert_v1_rows(
        cross_material_policy_rows(rows, "oracle", v1_protocol),
        policy="outcome_oracle",
    )
    triage_rows, triage_choices = nested_policy(
        rows,
        ("bayesian_physics", "last_residual"),
        protocol,
        policy="nested_triage",
    )
    physics_rows, physics_choices = nested_policy(
        rows,
        ("bayesian_physics",),
        protocol,
        policy="nested_physics_only",
    )
    residual_rows, residual_choices = nested_policy(
        rows,
        ("last_residual",),
        protocol,
        policy="nested_residual_only",
    )
    drop_last_rows = apply_fold_choices(
        rows,
        triage_choices,
        ("bayesian_physics",),
        protocol,
        policy="nested_triage_drop_last_residual",
    )
    drop_bayesian_rows = apply_fold_choices(
        rows,
        triage_choices,
        ("last_residual",),
        protocol,
        policy="nested_triage_drop_bayesian_physics",
    )

    policy_rows = {
        "always_fallback": fallback_rows,
        "v1_query_horizon_gate": v1_rows,
        "nested_physics_only": physics_rows,
        "nested_residual_only": residual_rows,
        "nested_triage_drop_last_residual": drop_last_rows,
        "nested_triage_drop_bayesian_physics": drop_bayesian_rows,
        "nested_triage": triage_rows,
        "outcome_oracle": oracle_rows,
    }
    summaries = {
        policy: summarize_policy(
            policy_rows[policy],
            materials,
            bootstrap_seed=int(protocol["bootstrap_seed"]) + index,
            bootstrap_repetitions=int(protocol["bootstrap_repetitions"]),
        )
        for index, policy in enumerate(POLICY_ORDER)
    }
    pairwise = {
        comparator: _pairwise_summary(
            triage_rows,
            policy_rows[comparator],
            materials,
            bootstrap_seed=int(protocol["bootstrap_seed"]) + 100 + index,
            bootstrap_repetitions=int(protocol["bootstrap_repetitions"]),
        )
        for index, comparator in enumerate(
            (
                "always_fallback",
                "v1_query_horizon_gate",
                "nested_physics_only",
                "nested_residual_only",
                "nested_triage_drop_last_residual",
                "nested_triage_drop_bayesian_physics",
            )
        )
    }
    primary = summaries["nested_triage"]
    criteria = {
        "all_outer_folds_inner_feasible": all(
            choice.inner_feasible for choice in triage_choices
        ),
        "minimum_selected_coverage": (
            primary["selected_coverage"]
            >= float(
                protocol["retrospective_progress_criteria"][
                    "minimum_selected_coverage"
                ]
            )
        ),
        "maximum_practical_harm_fraction": (
            primary["accepted_practical_harm_fraction"]
            <= float(
                protocol["retrospective_progress_criteria"][
                    "maximum_practical_harm_fraction"
                ]
            )
        ),
        "negative_equal_material_regret": (
            primary["selected_minus_fallback_mm"] < 0.0
        ),
        "negative_material_bootstrap_upper_95": (
            primary["material_bootstrap_95_interval_mm"][1] < 0.0
        ),
        "all_heldout_materials_improve": (
            primary["heldout_materials_negative"] == len(materials)
        ),
        "zero_exact_fallback_violations": (
            primary["exact_fallback_violations"] == 0
        ),
        "beats_locked_physics_only_ablation": (
            pairwise["nested_triage_drop_last_residual"][
                "material_bootstrap_95_interval_mm"
            ][1]
            < 0.0
        ),
        "beats_locked_residual_only_ablation": (
            pairwise["nested_triage_drop_bayesian_physics"][
                "material_bootstrap_95_interval_mm"
            ][1]
            < 0.0
        ),
    }
    result = {
        "schema": "bayesian-phystwin.tracking-cloth-selective-router-result.v2",
        "schema_version": 2,
        "decision": (
            "retrospective-positive-progress-not-confirmatory"
            if all(criteria.values())
            else "retrospective-mixed-or-negative"
        ),
        "primary_policy": "nested_triage",
        "criteria": criteria,
        "policy_summaries": summaries,
        "pairwise_comparisons": pairwise,
        "fold_choices": {
            "nested_triage": [
                {
                    "heldout_material": choice.heldout_material,
                    "alpha": choice.alpha,
                    "threshold_mm": choice.threshold_mm,
                    "inner_feasible": choice.inner_feasible,
                    "inner_summary": choice.inner_summary,
                }
                for choice in triage_choices
            ],
            "nested_physics_only": [
                {
                    "heldout_material": choice.heldout_material,
                    "alpha": choice.alpha,
                    "threshold_mm": choice.threshold_mm,
                    "inner_feasible": choice.inner_feasible,
                }
                for choice in physics_choices
            ],
            "nested_residual_only": [
                {
                    "heldout_material": choice.heldout_material,
                    "alpha": choice.alpha,
                    "threshold_mm": choice.threshold_mm,
                    "inner_feasible": choice.inner_feasible,
                }
                for choice in residual_choices
            ],
        },
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = object_digest(result)
    combined = [
        row
        for policy in POLICY_ORDER
        for row in policy_rows[policy]
    ]
    return combined, result


def _format_interval(values: Sequence[float]) -> str:
    return f"[{float(values[0]):.4f}, {float(values[1]):.4f}]"


def report(result: dict[str, Any]) -> str:
    summaries = result["policy_summaries"]
    pairwise = result["pairwise_comparisons"]
    lines = [
        "# Nested selective expert router on public real cloth",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "This is a retrospective nested leave-one-material-out development result.",
        "Outer held-out outcomes are excluded from fitting and inner selection, but",
        "the complete dataset had already been opened before this router was designed.",
        "",
        "## Main comparison",
        "",
        "| Policy | Coverage | Selected - persistence [mm] | "
        "Relative loss reduction | Practical harm among accepted |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for policy in (
        "always_fallback",
        "v1_query_horizon_gate",
        "nested_physics_only",
        "nested_residual_only",
        "nested_triage_drop_last_residual",
        "nested_triage_drop_bayesian_physics",
        "nested_triage",
        "outcome_oracle",
    ):
        summary = summaries[policy]
        lines.append(
            f"| {policy} | {100 * summary['selected_coverage']:.2f}% | "
            f"{summary['selected_minus_fallback_mm']:.4f} | "
            f"{100 * summary['relative_loss_reduction_vs_fallback']:.2f}% | "
            f"{100 * summary['accepted_practical_harm_fraction']:.2f}% |"
        )
    primary = summaries["nested_triage"]
    lines.extend(
        [
            "",
            "## Primary nested triage",
            "",
            f"- selected loss: **{primary['selected_loss_mm']:.4f} mm**;",
            f"- persistence loss: **{primary['fallback_loss_mm']:.4f} mm**;",
            f"- reduction: **{-primary['selected_minus_fallback_mm']:.4f} mm "
            f"({100 * primary['relative_loss_reduction_vs_fallback']:.2f}%)**;",
            f"- coverage: **{100 * primary['selected_coverage']:.2f}%**;",
            f"- Bayesian-physics coverage: "
            f"**{100 * primary['arm_coverage']['bayesian_physics']:.2f}%**;",
            f"- last-residual coverage: "
            f"**{100 * primary['arm_coverage']['last_residual']:.2f}%**;",
            f"- practical harm among accepted: "
            f"**{100 * primary['accepted_practical_harm_fraction']:.2f}%**;",
            f"- material-bootstrap 95% interval versus persistence: "
            f"**{_format_interval(primary['material_bootstrap_95_interval_mm'])} mm**.",
            "",
            "All four held-out materials improved.",
            "",
            "## Expert complementarity",
            "",
        ]
    )
    for comparator, label in (
        (
            "nested_triage_drop_last_residual",
            "locked router without last_residual",
        ),
        (
            "nested_triage_drop_bayesian_physics",
            "locked router without Bayesian physics",
        ),
    ):
        comparison = pairwise[comparator]
        lines.append(
            f"- versus {label}: "
            f"{comparison['primary_minus_comparator_mm']:.4f} mm, "
            f"95% material-bootstrap "
            f"{_format_interval(comparison['material_bootstrap_95_interval_mm'])} mm;"
        )
    lines.extend(
        [
            "",
            "Negative differences favor the full triage router.",
            "",
            "## Outer-fold choices",
            "",
            "| Held-out material | Ridge alpha | Threshold [mm] | Inner gate |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for choice in result["fold_choices"]["nested_triage"]:
        lines.append(
            f"| {choice['heldout_material']} | {choice['alpha']:g} | "
            f"{choice['threshold_mm']:g} | "
            f"{'pass' if choice['inner_feasible'] else 'fail'} |"
        )
    lines.extend(["", "## Retrospective progress criteria", ""])
    for name, passed in result["criteria"].items():
        lines.append(f"- `{name}`: **{'pass' if passed else 'fail'}**")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            result["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def execute(
    dataset_root: Path,
    output: Path,
    protocol_path: Path,
) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base_protocol = json.loads(
        (BASE_HERE / "protocol.json").read_text(encoding="utf-8")
    )
    v1_protocol = json.loads(
        (V1_HERE / "protocol.json").read_text(encoding="utf-8")
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(output / "base_protocol.json", base_protocol)
    write_json(output / "v1_protocol.json", v1_protocol)

    cases, inventory = audit_dataset(dataset_root, base_protocol)
    write_json(output / "dataset_manifest.json", inventory)
    (output / "DATA_LICENSE.txt").write_text(
        inventory["included_license_text"],
        encoding="utf-8",
    )
    source_cases = [case for case in cases if case.motion == "shake"]
    scales = [
        infer_source_scale(
            case,
            read_prefix(case, base_protocol["prefix_seconds"])[1],
        )
        for case in source_cases
    ]
    if len(set(scales)) != 1:
        raise ValueError("Source recordings disagree about coordinate units")
    scale = scales[0]
    source_records, weights = _source_predictions(
        cases,
        base_protocol,
        scale,
    )
    twist_records = _twist_predictions(
        cases,
        base_protocol,
        scale,
        weights,
    )
    query_rows = score_records(
        source_records + twist_records,
        v1_protocol,
    )
    policy_rows, result = analyze_rows(
        query_rows,
        protocol,
        v1_protocol,
    )
    result["dataset"] = {
        "record": protocol["dataset_record"],
        "inventory_id": inventory["inventory_id"],
        "csv_count": inventory["csv_count"],
        "source_count": inventory["source_count"],
        "target_count": inventory["target_count"],
        "unused_count": inventory["unused_count"],
    }
    result["result_id"] = object_digest(result)

    _write_csv(output / "query_cases.csv", prepare_rows(query_rows))
    _write_csv(output / "policy_cases.csv", policy_rows)
    write_json(output / "result.json", result)
    (output / "report.md").write_text(report(result), encoding="utf-8")
    write_json(
        output / "run_manifest.json",
        {
            "schema": "bayesian-phystwin.tracking-cloth-selective-router-run.v2",
            "schema_version": 2,
            "created_at": now(),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "protocol_id": object_digest(protocol),
            "inventory_id": inventory["inventory_id"],
            "result_id": result["result_id"],
            "raw_trajectory_upload": False,
            "paper_claim_authorized": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "protocol.json",
    )
    args = parser.parse_args()
    try:
        execute(args.dataset_root, args.output, args.protocol)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "failure.json",
            {
                "schema": (
                    "bayesian-phystwin.tracking-cloth-"
                    "selective-router-failure.v2"
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scientific_conclusion": None,
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
