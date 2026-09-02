"""Nested material-level model and threshold selection for Tracking Cloth v2."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ridge import TARGET_ORDER, fit_ridge, predict_ridge, route_rows


@dataclass(frozen=True)
class FoldChoice:
    """Outer-fold tuning result."""

    heldout_material: str
    alpha: float
    threshold_mm: float
    inner_feasible: bool
    inner_summary: dict[str, Any]


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
            float(np.mean([bool(row["selected_practical_harm"]) for row in accepted]))
            if accepted
            else 0.0
        ),
        "accepted_strict_regression_fraction": (
            float(
                np.mean([bool(row["selected_strict_regression"]) for row in accepted])
            )
            if accepted
            else 0.0
        ),
        "exact_fallback_violations": int(
            np.sum([not bool(row["exact_fallback"]) for row in rows])
        ),
        "arm_counts": arm_counts,
        "arm_coverage": {arm: arm_counts[arm] / len(rows) for arm in TARGET_ORDER},
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
            candidates.append((feasible, summary, float(alpha), float(threshold)))
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

    choice_by_material = {choice.heldout_material: choice for choice in choices}
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
