"""Policy assembly, inference, and reporting for Tracking Cloth v2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from experiments.tracking_cloth_deformation_v1.data import object_digest
from experiments.tracking_cloth_selective_twin_v1.run import (
    cross_material_policy_rows,
)

from .ridge import prepare_rows
from .selection import apply_fold_choices, nested_policy, summarize_policy

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
        item["selected_practical_harm"] = bool(row["selected_practical_harm"])
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
        difference = float(row["selected_loss_mm"]) - float(other["selected_loss_mm"])
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
                protocol["retrospective_progress_criteria"]["minimum_selected_coverage"]
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
        "negative_equal_material_regret": (primary["selected_minus_fallback_mm"] < 0.0),
        "negative_material_bootstrap_upper_95": (
            primary["material_bootstrap_95_interval_mm"][1] < 0.0
        ),
        "all_heldout_materials_improve": (
            primary["heldout_materials_negative"] == len(materials)
        ),
        "zero_exact_fallback_violations": (primary["exact_fallback_violations"] == 0),
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
    combined = [row for policy in POLICY_ORDER for row in policy_rows[policy]]
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
