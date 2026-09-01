"""Query scoring, source-frozen handoff policies, and confirmation summaries."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np

from bayesian_phystwin.selective_competence_bound_v1 import clopper_pearson_upper

from .data import InputView
from .model import PHYSICS_ARM

SELECTOR_ARMS = {
    "kinematic": ("persistence", "constant_velocity"),
    "matched_residual": ("persistence", "constant_velocity", "last_residual"),
    "physics_enabled": (
        "persistence",
        "constant_velocity",
        "last_residual",
        PHYSICS_ARM,
    ),
}
CENTRAL_PATCH = np.asarray([5, 6, 9, 10], dtype=int)


def object_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _horizon_index(inputs: InputView, horizon_seconds: float) -> int:
    target = inputs.times[inputs.cutoff] + horizon_seconds
    future = np.arange(inputs.cutoff + 1, len(inputs.times))
    if not len(future):
        raise ValueError("prediction input has no future samples")
    index = int(future[np.argmin(np.abs(inputs.times[future] - target))])
    if abs(inputs.times[index] - target) > 2.5 * np.median(np.diff(inputs.times)):
        raise ValueError("registered horizon is absent from prediction grid")
    return index


def query_loss_mm(
    prediction: np.ndarray,
    truth: np.ndarray,
    inputs: InputView,
    query: str,
    horizon_seconds: float,
) -> float:
    index = _horizon_index(inputs, horizon_seconds)
    observed = np.isfinite(truth[index]).all(axis=1)
    if query == "central_patch_centroid":
        observed &= np.isin(np.arange(20), CENTRAL_PATCH)
    if not np.any(observed):
        raise ValueError(f"no observed markers for {query}")
    predicted = prediction[index, observed]
    actual = truth[index, observed]
    if query == "cloth_shape":
        predicted = predicted - predicted.mean(axis=0)
        actual = actual - actual.mean(axis=0)
        loss_m = float(np.sqrt(np.mean(np.sum((predicted - actual) ** 2, axis=1))))
    elif query in {"cloth_centroid", "central_patch_centroid"}:
        loss_m = float(np.linalg.norm(predicted.mean(axis=0) - actual.mean(axis=0)))
    elif query == "shape_radius":
        predicted_radius = float(
            np.sqrt(np.mean(np.sum((predicted - predicted.mean(axis=0)) ** 2, axis=1)))
        )
        actual_radius = float(
            np.sqrt(np.mean(np.sum((actual - actual.mean(axis=0)) ** 2, axis=1)))
        )
        loss_m = abs(predicted_radius - actual_radius)
    else:
        raise ValueError(f"unknown query {query}")
    return 1000.0 * loss_m


def score_case(
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
    inputs: InputView,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    margin = (
        float(protocol["practical_harm_fraction_of_initial_diameter"])
        * 1000.0
        * inputs.initial_diameter_m
    )
    for query in protocol["queries"]:
        for horizon in protocol["horizons_seconds"]:
            losses = {
                arm: query_loss_mm(
                    prediction,
                    truth,
                    inputs,
                    query,
                    float(horizon),
                )
                for arm, prediction in predictions.items()
            }
            fallback = losses["persistence"]
            rows.append(
                {
                    "case_id": inputs.case.case_id,
                    "recording": inputs.case.path.name,
                    "material": inputs.case.material,
                    "interaction": inputs.case.interaction,
                    "repetition": inputs.case.repetition,
                    "query": query,
                    "horizon_seconds": float(horizon),
                    "initial_diameter_mm": 1000.0 * inputs.initial_diameter_m,
                    "practical_harm_margin_mm": margin,
                    "losses_mm": losses,
                    "practical_harm": {
                        arm: value > fallback + margin for arm, value in losses.items()
                    },
                }
            )
    return rows


def context_key(row: dict[str, Any]) -> tuple[str, str, float]:
    return (
        str(row["interaction"]),
        str(row["query"]),
        float(row["horizon_seconds"]),
    )


def context_string(key: tuple[str, str, float]) -> str:
    return f"{key[0]}|{key[1]}|{key[2]:.9g}"


def _best_arm(rows: list[dict[str, Any]], roster: tuple[str, ...]) -> str:
    return min(
        roster,
        key=lambda arm: (
            float(np.mean([row["losses_mm"][arm] for row in rows])),
            roster.index(arm),
        ),
    )


def fit_fold_policy(
    rep2_rows: list[dict[str, Any]],
    heldout_material: str,
    protocol: dict[str, Any],
) -> dict[str, dict[str, str]]:
    training = [row for row in rep2_rows if row["material"] != heldout_material]
    materials = sorted({row["material"] for row in training})
    if len(materials) != 3:
        raise ValueError("each policy fold must train on exactly three materials")
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in training:
        grouped[context_key(row)].append(row)
    expected_contexts = (
        len(protocol["interactions"])
        * len(protocol["queries"])
        * len(protocol["horizons_seconds"])
    )
    if len(grouped) != expected_contexts:
        raise ValueError("source policy context roster is incomplete")

    minimum_gain = float(protocol["primary_gate"]["minimum_relative_gain"])
    maximum_harm = float(
        protocol["primary_gate"]["maximum_training_practical_harm_fraction"]
    )
    minimum_physics_gain = float(
        protocol["primary_gate"]["minimum_source_incremental_relative_gain"]
    )
    output: dict[str, dict[str, str]] = {name: {} for name in SELECTOR_ARMS}
    for key, subset in sorted(grouped.items()):
        if len(subset) != 3:
            raise ValueError(
                "each source context must contain one row per training material"
            )
        fallback_mean = float(
            np.mean([row["losses_mm"]["persistence"] for row in subset])
        )
        matched_arm = _best_arm(subset, SELECTOR_ARMS["matched_residual"])
        matched_material_loss = {
            material: next(
                row["losses_mm"][matched_arm]
                for row in subset
                if row["material"] == material
            )
            for material in materials
        }
        for selector, roster in SELECTOR_ARMS.items():
            arm = _best_arm(subset, roster)
            mean_loss = float(np.mean([row["losses_mm"][arm] for row in subset]))
            material_regrets = [
                next(
                    row["losses_mm"][arm] - row["losses_mm"]["persistence"]
                    for row in subset
                    if row["material"] == material
                )
                for material in materials
            ]
            harm = float(
                np.mean([bool(row["practical_harm"][arm]) for row in subset])
            )
            accepted = bool(
                arm != "persistence"
                and mean_loss <= (1.0 - minimum_gain) * fallback_mean
                and max(material_regrets) <= 0.0
                and harm <= maximum_harm
            )
            if accepted and arm == PHYSICS_ARM:
                physics_mean = mean_loss
                matched_mean = float(
                    np.mean([row["losses_mm"][matched_arm] for row in subset])
                )
                incremental = [
                    next(
                        row["losses_mm"][PHYSICS_ARM]
                        for row in subset
                        if row["material"] == material
                    )
                    - matched_material_loss[material]
                    for material in materials
                ]
                accepted = bool(
                    physics_mean
                    <= (1.0 - minimum_physics_gain) * max(matched_mean, 1e-12)
                    and max(incremental) <= 0.0
                )
            output[selector][context_string(key)] = arm if accepted else "persistence"
    return output


def fit_cross_material_policies(
    rep2_rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    folds = {
        material: fit_fold_policy(rep2_rows, material, protocol)
        for material in protocol["materials"]
    }
    record = {
        "schema": "bayesian-phystwin.tracking-cloth-self-collision-policy.v1",
        "schema_version": 1,
        "outer_split": "leave-one-material-out",
        "fit_repetition": 1,
        "selection_repetition": 2,
        "confirmation_repetition": 3,
        "selector_arms": {key: list(value) for key, value in SELECTOR_ARMS.items()},
        "folds": folds,
        "target_outcomes_used": False,
    }
    record["policy_id"] = object_digest(record)
    return record


def apply_policy(
    rows: list[dict[str, Any]], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        fold = policy["folds"][row["material"]]
        key = context_string(context_key(row))
        fallback = float(row["losses_mm"]["persistence"])
        for selector in SELECTOR_ARMS:
            arm = fold[selector][key]
            selected = float(row["losses_mm"][arm])
            output.append(
                {
                    **{
                        name: value
                        for name, value in row.items()
                        if name not in {"losses_mm", "practical_harm"}
                    },
                    "selector": selector,
                    "selected_arm": arm,
                    "selected_loss_mm": selected,
                    "fallback_loss_mm": fallback,
                    "selected_minus_fallback_mm": selected - fallback,
                    "selected_practical_harm": bool(row["practical_harm"][arm]),
                    "physics_selected": arm == PHYSICS_ARM,
                    "exact_fallback": bool(
                        arm != "persistence" or selected == fallback
                    ),
                }
            )
    return output


def _material_bootstrap(
    material_values: np.ndarray,
    repetitions: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(material_values),
        size=(repetitions, len(material_values)),
    )
    distribution = material_values[indices].mean(axis=1)
    return np.quantile(distribution, [0.025, 0.975]).tolist()


def summarize_policy_rows(
    policy_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    seed_offset: int = 0,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for selector_index, selector in enumerate(SELECTOR_ARMS):
        subset = [row for row in policy_rows if row["selector"] == selector]
        materials = []
        for material in protocol["materials"]:
            material_rows = [row for row in subset if row["material"] == material]
            materials.append(
                {
                    "material": material,
                    "selected_loss_mm": float(
                        np.mean([row["selected_loss_mm"] for row in material_rows])
                    ),
                    "selected_minus_fallback_mm": float(
                        np.mean(
                            [row["selected_minus_fallback_mm"] for row in material_rows]
                        )
                    ),
                    "coverage": float(
                        np.mean(
                            [
                                row["selected_arm"] != "persistence"
                                for row in material_rows
                            ]
                        )
                    ),
                    "physics_coverage": float(
                        np.mean([row["physics_selected"] for row in material_rows])
                    ),
                }
            )
        material_regrets = np.asarray(
            [item["selected_minus_fallback_mm"] for item in materials], dtype=float
        )
        accepted = [row for row in subset if row["selected_arm"] != "persistence"]
        physics = [row for row in subset if row["physics_selected"]]
        harmful_physics = sum(bool(row["selected_practical_harm"]) for row in physics)
        summaries[selector] = {
            "selector": selector,
            "query_cases": len(subset),
            "selected_loss_mm": float(
                np.mean([row["selected_loss_mm"] for row in subset])
            ),
            "fallback_loss_mm": float(
                np.mean([row["fallback_loss_mm"] for row in subset])
            ),
            "selected_minus_fallback_mm": float(
                np.mean([row["selected_minus_fallback_mm"] for row in subset])
            ),
            "relative_gain_vs_fallback": float(
                -np.mean([row["selected_minus_fallback_mm"] for row in subset])
                / max(np.mean([row["fallback_loss_mm"] for row in subset]), 1e-12)
            ),
            "nonfallback_coverage": float(
                np.mean([row["selected_arm"] != "persistence" for row in subset])
            ),
            "physics_coverage": float(
                np.mean([row["physics_selected"] for row in subset])
            ),
            "selected_practical_harm_fraction_all_cases": float(
                np.mean([row["selected_practical_harm"] for row in subset])
            ),
            "selected_practical_harm_fraction_accepted": (
                float(np.mean([row["selected_practical_harm"] for row in accepted]))
                if accepted
                else 0.0
            ),
            "physics_practical_harm_fraction": (
                float(harmful_physics / len(physics)) if physics else 0.0
            ),
            "physics_harm_upper_95": clopper_pearson_upper(
                harmful_physics, len(physics), alpha=0.05
            ),
            "exact_fallback_violations": int(
                sum(not row["exact_fallback"] for row in subset)
            ),
            "materials_nonpositive": int(np.sum(material_regrets <= 0.0)),
            "materials_negative": int(np.sum(material_regrets < 0.0)),
            "material_bootstrap_95_interval_mm": _material_bootstrap(
                material_regrets,
                int(protocol["bootstrap_repetitions"]),
                int(protocol["bootstrap_seed"]) + seed_offset + selector_index,
            ),
            "material_results": materials,
        }
    return summaries


def incremental_summary(
    policy_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    seed_offset: int = 0,
) -> dict[str, Any]:
    residual = {
        (row["case_id"], row["query"], row["horizon_seconds"]): row
        for row in policy_rows
        if row["selector"] == "matched_residual"
    }
    physics = {
        (row["case_id"], row["query"], row["horizon_seconds"]): row
        for row in policy_rows
        if row["selector"] == "physics_enabled"
    }
    if residual.keys() != physics.keys():
        raise ValueError("matched selectors do not cover identical query cases")
    paired = []
    for key in sorted(residual):
        p_row, r_row = physics[key], residual[key]
        paired.append(
            {
                "material": p_row["material"],
                "difference_mm": p_row["selected_loss_mm"]
                - r_row["selected_loss_mm"],
                "physics_selected": p_row["physics_selected"],
            }
        )
    materials = []
    for material in protocol["materials"]:
        values = [row["difference_mm"] for row in paired if row["material"] == material]
        materials.append(
            {"material": material, "physics_minus_residual_mm": float(np.mean(values))}
        )
    material_values = np.asarray(
        [item["physics_minus_residual_mm"] for item in materials], dtype=float
    )
    residual_mean = float(
        np.mean(
            [
                row["selected_loss_mm"]
                for row in policy_rows
                if row["selector"] == "matched_residual"
            ]
        )
    )
    difference = float(np.mean([row["difference_mm"] for row in paired]))
    return {
        "physics_minus_residual_mm": difference,
        "incremental_relative_gain": -difference / max(residual_mean, 1e-12),
        "materials_nonpositive": int(np.sum(material_values <= 0.0)),
        "materials_negative": int(np.sum(material_values < 0.0)),
        "material_bootstrap_95_interval_mm": _material_bootstrap(
            material_values,
            int(protocol["bootstrap_repetitions"]),
            int(protocol["bootstrap_seed"]) + 100 + seed_offset,
        ),
        "material_results": materials,
        "improved_query_cases": int(
            np.sum([row["difference_mm"] < 0 for row in paired])
        ),
        "worsened_query_cases": int(
            np.sum([row["difference_mm"] > 0 for row in paired])
        ),
        "unchanged_query_cases": int(
            np.sum([row["difference_mm"] == 0 for row in paired])
        ),
    }


def source_gate(
    summaries: dict[str, Any],
    incremental: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    physics = summaries["physics_enabled"]
    residual = summaries["matched_residual"]
    gate = protocol["source_gate"]
    criteria = {
        "minimum_source_physics_coverage": physics["physics_coverage"]
        >= float(gate["minimum_physics_coverage"]),
        "minimum_source_incremental_relative_gain": incremental[
            "incremental_relative_gain"
        ]
        >= float(gate["minimum_incremental_relative_gain"]),
        "all_source_materials_nonpositive": incremental["materials_nonpositive"]
        == len(protocol["materials"]),
        "negative_source_material_bootstrap_upper_95": incremental[
            "material_bootstrap_95_interval_mm"
        ][1]
        < 0.0,
        "no_more_source_harm_than_residual": physics[
            "selected_practical_harm_fraction_all_cases"
        ]
        <= residual["selected_practical_harm_fraction_all_cases"],
        "maximum_source_physics_harm": physics["physics_practical_harm_fraction"]
        <= float(gate["maximum_physics_practical_harm_fraction"]),
        "zero_source_fallback_violations": physics["exact_fallback_violations"] == 0,
    }
    return {"pass": all(criteria.values()), "criteria": criteria}


def confirmation_gate(
    summaries: dict[str, Any],
    incremental: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    physics = summaries["physics_enabled"]
    residual = summaries["matched_residual"]
    gate = protocol["confirmation_gate"]
    criteria = {
        "minimum_physics_coverage": physics["physics_coverage"]
        >= float(gate["minimum_physics_coverage"]),
        "minimum_incremental_relative_gain": incremental["incremental_relative_gain"]
        >= float(gate["minimum_incremental_relative_gain"]),
        "negative_incremental_regret": incremental["physics_minus_residual_mm"] < 0.0,
        "all_materials_nonpositive": incremental["materials_nonpositive"]
        == len(protocol["materials"]),
        "negative_material_bootstrap_upper_95": incremental[
            "material_bootstrap_95_interval_mm"
        ][1]
        < 0.0,
        "no_more_harm_than_residual": physics[
            "selected_practical_harm_fraction_all_cases"
        ]
        <= residual["selected_practical_harm_fraction_all_cases"],
        "maximum_physics_harm": physics["physics_practical_harm_fraction"]
        <= float(gate["maximum_physics_practical_harm_fraction"]),
        "zero_exact_fallback_violations": physics["exact_fallback_violations"] == 0,
    }
    return {"pass": all(criteria.values()), "criteria": criteria}
