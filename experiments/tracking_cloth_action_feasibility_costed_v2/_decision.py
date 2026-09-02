"""Cost-aware and support-robust decisions for Tracking Cloth V2.

V1 correctly included the registered probe cost in the contingent plan passed to
the exact certificate, but its empirical source-gain summary compared terminal
task losses only.  This module leaves V1 untouched and defines a separately
versioned objective in which a sensing decision is scored as

    terminal task loss + probe cost * source loss scale.

It additionally evaluates the exact bounded-support-miss envelope from
``support_robust_act_sense_fallback_certificate_v1`` directly on the complete
finite plan set produced by the parent act--sense--fallback certificate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from bayesian_phystwin.act_sense_fallback_certificate_v1 import (
    ActSenseFallbackCertificateV1,
    ContingentPlanV1,
    act_sense_fallback_certificate,
)

ATOL = 1e-12


@dataclass(frozen=True)
class RobustPlanDecision:
    """One material-conditional robust decision over a frozen plan roster."""

    parent: ActSenseFallbackCertificateV1
    support_miss_probability: float
    unknown_plan_loss_lower: np.ndarray
    unknown_plan_loss_upper: np.ndarray
    pairwise_worst_case_loss_gap: np.ndarray
    worst_case_regret: np.ndarray
    maximum_admissible_support_miss: np.ndarray
    minimax_plan_index: int
    output_plan_index: int
    output_mode: str
    used_fallback: bool

    @property
    def output_plan(self) -> ContingentPlanV1:
        return self.parent.plans[self.output_plan_index]

    @property
    def minimax_worst_case_regret(self) -> float:
        return float(self.worst_case_regret[self.minimax_plan_index])

    @property
    def selected_probe_index(self) -> int | None:
        if self.output_mode != "sense":
            return None
        return self.output_plan.probe_index

    def terminal_action(self, outcome_index: int | None = None) -> int:
        return self.output_plan.terminal_action(outcome_index)


def _fallback_action(losses: np.ndarray) -> int:
    mean_loss = np.mean(losses, axis=0)
    return int(np.flatnonzero(np.isclose(mean_loss, np.min(mean_loss)))[0])


def _validate_support_contract(
    support_miss_probability: float,
    action_lower: np.ndarray,
    action_upper: np.ndarray,
    action_count: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    epsilon = float(support_miss_probability)
    if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("support_miss_probability must lie in [0, 1]")
    lower = np.asarray(action_lower, dtype=np.float64)
    upper = np.asarray(action_upper, dtype=np.float64)
    if lower.shape != (action_count,) or upper.shape != (action_count,):
        raise ValueError("unknown action bounds must match the action roster")
    if (
        not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
    ):
        raise ValueError("unknown action bounds must be finite ordered intervals")
    return epsilon, lower, upper


def _plan_bounds(
    certificate: ActSenseFallbackCertificateV1,
    probe_costs: np.ndarray,
    action_lower: np.ndarray,
    action_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower: list[float] = []
    upper: list[float] = []
    for plan in certificate.plans:
        if plan.mode == "act":
            if plan.direct_action_index is None:
                raise RuntimeError("direct plan lost its action")
            action = plan.direct_action_index
            lower.append(float(action_lower[action]))
            upper.append(float(action_upper[action]))
            continue
        if plan.mode != "sense" or plan.probe_index is None:
            raise RuntimeError("unknown contingent plan mode")
        mapping = np.asarray(plan.terminal_action_by_outcome, dtype=np.int64)
        if mapping.size == 0:
            raise RuntimeError("sensing plan lost its terminal map")
        cost = float(probe_costs[plan.probe_index])
        lower.append(cost + float(np.min(action_lower[mapping])))
        upper.append(cost + float(np.max(action_upper[mapping])))
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _support_miss_budgets(
    represented_pairwise_gap: np.ndarray,
    unknown_plan_lower: np.ndarray,
    unknown_plan_upper: np.ndarray,
    regret_tolerance: float,
) -> np.ndarray:
    unknown_gap = unknown_plan_upper[:, None] - unknown_plan_lower[None, :]
    slope = np.maximum(0.0, unknown_gap - represented_pairwise_gap)
    pair_budget = np.ones_like(represented_pairwise_gap)
    already_over = represented_pairwise_gap > regret_tolerance + ATOL
    pair_budget[already_over] = 0.0
    increasing = (~already_over) & (slope > ATOL)
    pair_budget[increasing] = np.clip(
        (regret_tolerance - represented_pairwise_gap[increasing])
        / slope[increasing],
        0.0,
        1.0,
    )
    return np.min(pair_budget, axis=1)


def support_robust_decision(
    certificate: ActSenseFallbackCertificateV1,
    *,
    probe_costs: np.ndarray,
    support_miss_probability: float,
    unknown_action_loss_lower: np.ndarray,
    unknown_action_loss_upper: np.ndarray,
    regret_tolerance: float,
) -> RobustPlanDecision:
    """Apply the exact at-most-epsilon loss-box envelope to all complete plans."""

    action_count = certificate.direct_plan_count
    epsilon, action_lower, action_upper = _validate_support_contract(
        support_miss_probability,
        unknown_action_loss_lower,
        unknown_action_loss_upper,
        action_count,
    )
    costs = np.asarray(probe_costs, dtype=np.float64)
    if costs.shape != (certificate.probe_count,) or np.any(costs < 0.0):
        raise ValueError("probe_costs changed after parent plan enumeration")
    plan_lower, plan_upper = _plan_bounds(
        certificate,
        costs,
        action_lower,
        action_upper,
    )
    represented = np.asarray(
        certificate.plan_certificate.pairwise_worst_case_loss_gap,
        dtype=np.float64,
    )
    unknown_gap = plan_upper[:, None] - plan_lower[None, :]
    robust_pairwise = represented + epsilon * np.maximum(
        0.0,
        unknown_gap - represented,
    )
    np.fill_diagonal(robust_pairwise, 0.0)
    robust_regret = np.maximum(np.max(robust_pairwise, axis=1), 0.0)
    minimum = float(np.min(robust_regret))
    minimax = int(
        np.flatnonzero(np.isclose(robust_regret, minimum, rtol=0.0, atol=ATOL))[0]
    )
    budgets = _support_miss_budgets(
        represented,
        plan_lower,
        plan_upper,
        float(regret_tolerance),
    )
    if minimum > float(regret_tolerance) + ATOL:
        output = certificate.fallback_plan_index
        mode = "fallback"
        used_fallback = True
    else:
        output = minimax
        plan = certificate.plans[output]
        if plan.mode == "sense":
            mode = "sense"
            used_fallback = False
        elif plan.direct_action_index == certificate.fallback_action_index:
            mode = "fallback"
            used_fallback = True
        else:
            mode = "act"
            used_fallback = False
    return RobustPlanDecision(
        parent=certificate,
        support_miss_probability=epsilon,
        unknown_plan_loss_lower=plan_lower,
        unknown_plan_loss_upper=plan_upper,
        pairwise_worst_case_loss_gap=robust_pairwise,
        worst_case_regret=robust_regret,
        maximum_admissible_support_miss=budgets,
        minimax_plan_index=minimax,
        output_plan_index=output,
        output_mode=mode,
        used_fallback=used_fallback,
    )


def _resolve_actions(
    decision: RobustPlanDecision,
    probe_outcomes: np.ndarray,
    block_indices: np.ndarray,
) -> np.ndarray:
    if decision.output_mode in {"act", "fallback"}:
        return np.full(
            block_indices.size,
            decision.terminal_action(),
            dtype=np.int64,
        )
    probe_index = decision.selected_probe_index
    if probe_index is None:
        raise RuntimeError("sensing decision is missing its probe")
    return np.asarray(
        [
            decision.terminal_action(int(probe_outcomes[probe_index, index]))
            for index in block_indices
        ],
        dtype=np.int64,
    )


def _decision_record(
    decision: RobustPlanDecision,
    *,
    material: str,
    actions: list[str],
    informative_probe_indices: list[int],
    informative_probe_names: list[str],
    chosen: np.ndarray,
) -> dict[str, Any]:
    plan = decision.output_plan
    selected_probe = None
    selected_probe_global_index = None
    terminal_map = None
    if decision.output_mode == "sense":
        if plan.probe_index is None:
            raise RuntimeError("sensing plan has no local probe index")
        selected_probe = informative_probe_names[plan.probe_index]
        selected_probe_global_index = informative_probe_indices[plan.probe_index]
        terminal_map = [
            actions[int(index)] for index in plan.terminal_action_by_outcome
        ]
    return {
        "material": material,
        "mode": decision.output_mode,
        "selected_probe": selected_probe,
        "selected_probe_global_index": selected_probe_global_index,
        "terminal_action_by_probe_outcome": terminal_map,
        "direct_or_fallback_action": (
            actions[decision.terminal_action()]
            if decision.output_mode in {"act", "fallback"}
            else None
        ),
        "robust_worst_case_regret": decision.minimax_worst_case_regret,
        "output_plan_worst_case_regret": float(
            decision.worst_case_regret[decision.output_plan_index]
        ),
        "output_plan_support_miss_budget": float(
            decision.maximum_admissible_support_miss[decision.output_plan_index]
        ),
        "chosen_actions_by_source_repetition": [
            actions[int(value)] for value in chosen
        ],
    }


def decision_grid_v2(
    blocks: list[tuple[str, int]],
    losses: np.ndarray,
    probe_outcomes: np.ndarray,
    source_protocol: dict[str, Any],
    v2_protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a registered cost/tolerance/epsilon grid on source blocks."""

    materials = list(source_protocol["materials"])
    actions = list(source_protocol["interactions"])
    material_index = {material: index for index, material in enumerate(materials)}
    classes = np.asarray(
        [material_index[material] for material, _ in blocks],
        dtype=np.int64,
    )
    prior = np.full(len(blocks), 1.0 / len(blocks), dtype=np.float64)
    fallback = _fallback_action(losses)
    loss_scale = float(max(np.quantile(losses, 0.9), 1e-12))
    normalized_losses = losses / loss_scale
    informative_probe_indices = [
        index for index, row in enumerate(probe_outcomes) if np.unique(row).size >= 2
    ]
    informative_probe_outcomes = probe_outcomes[informative_probe_indices]
    informative_probe_names = [actions[index] for index in informative_probe_indices]
    contract = v2_protocol["support_robustness"]
    action_lower = np.asarray(
        contract["unknown_terminal_loss_lower_normalized"], dtype=np.float64
    )
    action_upper = np.asarray(
        contract["unknown_terminal_loss_upper_normalized"], dtype=np.float64
    )

    records: list[dict[str, Any]] = []
    selected_by_epsilon: dict[str, dict[str, Any]] = {}
    for epsilon in contract["support_miss_probability_grid"]:
        epsilon_records: list[dict[str, Any]] = []
        for probe_cost in source_protocol["probe_cost_grid"]:
            for tolerance in source_protocol["regret_tolerance_grid"]:
                outputs: list[dict[str, Any]] = []
                objective_losses: list[float] = []
                terminal_losses: list[float] = []
                oracle_losses: list[float] = []
                fallback_losses: list[float] = []
                for material in materials:
                    quotient = np.zeros(len(materials), dtype=np.float64)
                    quotient[material_index[material]] = 1.0
                    probe_costs = np.full(
                        len(informative_probe_indices),
                        float(probe_cost),
                        dtype=np.float64,
                    )
                    parent = act_sense_fallback_certificate(
                        prior,
                        quotient,
                        classes,
                        normalized_losses,
                        informative_probe_outcomes,
                        probe_costs,
                        fallback_action_index=fallback,
                        regret_tolerance=float(tolerance),
                        probe_names=informative_probe_names,
                        max_plan_count=int(source_protocol["max_plan_count"]),
                    )
                    decision = support_robust_decision(
                        parent,
                        probe_costs=probe_costs,
                        support_miss_probability=float(epsilon),
                        unknown_action_loss_lower=action_lower,
                        unknown_action_loss_upper=action_upper,
                        regret_tolerance=float(tolerance),
                    )
                    block_indices = np.flatnonzero(
                        classes == material_index[material]
                    )
                    chosen = _resolve_actions(
                        decision,
                        informative_probe_outcomes,
                        block_indices,
                    )
                    terminal = losses[block_indices, chosen]
                    objective = terminal.copy()
                    if decision.output_mode == "sense":
                        objective += float(probe_cost) * loss_scale
                    terminal_losses.extend(float(value) for value in terminal)
                    objective_losses.extend(float(value) for value in objective)
                    oracle_losses.extend(
                        float(value) for value in np.min(losses[block_indices], axis=1)
                    )
                    fallback_losses.extend(
                        float(value) for value in losses[block_indices, fallback]
                    )
                    outputs.append(
                        _decision_record(
                            decision,
                            material=material,
                            actions=actions,
                            informative_probe_indices=informative_probe_indices,
                            informative_probe_names=informative_probe_names,
                            chosen=chosen,
                        )
                    )
                modes = Counter(item["mode"] for item in outputs)
                mean_objective = float(np.mean(objective_losses))
                mean_terminal = float(np.mean(terminal_losses))
                mean_fallback = float(np.mean(fallback_losses))
                item = {
                    "support_miss_probability": float(epsilon),
                    "probe_cost": float(probe_cost),
                    "regret_tolerance": float(tolerance),
                    "mode_counts": {
                        mode: int(modes.get(mode, 0))
                        for mode in ("act", "sense", "fallback")
                    },
                    "mean_source_objective_loss": mean_objective,
                    "mean_source_terminal_loss": mean_terminal,
                    "mean_fallback_loss": mean_fallback,
                    "mean_oracle_loss": float(np.mean(oracle_losses)),
                    "relative_objective_gain_vs_fallback": float(
                        (mean_fallback - mean_objective) / max(mean_fallback, 1e-12)
                    ),
                    "relative_terminal_gain_vs_fallback": float(
                        (mean_fallback - mean_terminal) / max(mean_fallback, 1e-12)
                    ),
                    "outputs": outputs,
                }
                records.append(item)
                epsilon_records.append(item)
        candidates = [
            item
            for item in epsilon_records
            if item["mode_counts"]["sense"] > 0
            and item["relative_objective_gain_vs_fallback"]
            > float(v2_protocol["source_gate"]["minimum_relative_objective_gain"])
        ]
        if candidates:
            selected = min(
                candidates,
                key=lambda item: (
                    -item["relative_objective_gain_vs_fallback"],
                    item["mode_counts"]["fallback"],
                    item["probe_cost"],
                    item["regret_tolerance"],
                ),
            )
            selected_by_epsilon[str(float(epsilon))] = selected
        else:
            selected_by_epsilon[str(float(epsilon))] = {
                "status": "no-costed-source-setting-combines-sensing-and-gain",
                "support_miss_probability": float(epsilon),
            }
    primary = str(float(contract["primary_support_miss_probability"]))
    if primary not in selected_by_epsilon:
        raise ValueError("primary support-miss probability is absent from the grid")
    return records, {
        "fallback_action": actions[fallback],
        "fallback_action_index": fallback,
        "loss_scale": loss_scale,
        "informative_probe_indices": informative_probe_indices,
        "informative_probe_names": informative_probe_names,
        "selected_by_support_miss_probability": selected_by_epsilon,
        "primary_support_miss_probability": float(
            contract["primary_support_miss_probability"]
        ),
        "selected_primary_source_setting": selected_by_epsilon[primary],
        "objective_semantics": (
            "terminal-task-loss-plus-probe-cost-times-source-loss-scale"
        ),
    }


__all__ = [
    "RobustPlanDecision",
    "decision_grid_v2",
    "support_robust_decision",
]
