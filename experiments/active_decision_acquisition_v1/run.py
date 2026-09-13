#!/usr/bin/env python3
"""Run the registered controlled active decision-acquisition study."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.active_decision_acquisition_v1 import (
    ActiveDecisionPolicyV1,
    DeterministicDecisionProbeV1,
    conditioned_query_decision_certificate,
    minimum_cost_global_decision_identifying_probe_set,
    synthesize_minimax_active_decision_policy,
)

SCHEMA = "bayesian-phystwin.active-decision-acquisition-study"
SCHEMA_VERSION = 1
RESULT_SCHEMA = "bayesian-phystwin.active-decision-acquisition-result"


def _content_id(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("protocol must contain one JSON object")
    if value.get("schema") != SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported active-acquisition protocol")
    unsigned = dict(value)
    protocol_id = unsigned.pop("protocol_id", None)
    if protocol_id != _content_id(unsigned):
        raise ValueError("protocol identity mismatch")
    return value


def _problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[DeterministicDecisionProbeV1, ...],
    np.ndarray,
]:
    action_group_sizes = (16, 4, 4)
    action_groups: list[int] = []
    nuisance_codes: list[int] = []
    for action, count in enumerate(action_group_sizes):
        action_groups.extend([action] * count)
        nuisance_codes.extend(range(count))
    group = np.asarray(action_groups, dtype=np.int64)
    codes = np.asarray(nuisance_codes, dtype=np.int64)
    hypothesis_count = group.size
    prior = np.full(hypothesis_count, 1.0 / hypothesis_count)
    quotient = np.array([1.0])
    classes = np.zeros(hypothesis_count, dtype=np.int64)
    losses = np.ones((hypothesis_count, 3), dtype=np.float64)
    losses[np.arange(hypothesis_count), group] = 0.0

    probes = [
        DeterministicDecisionProbeV1(
            "decision-probe-0",
            (group != 0).astype(np.int64),
            1.0,
        ),
        DeterministicDecisionProbeV1(
            "decision-probe-1",
            (group == 2).astype(np.int64),
            1.0,
        ),
    ]
    for bit in range(4):
        probes.append(
            DeterministicDecisionProbeV1(
                f"nuisance-bit-{bit}",
                ((codes >> bit) & 1).astype(np.int64),
                1.0,
            )
        )
    return prior, quotient, classes, losses, tuple(probes), group


def _policy_node_map(policy: ActiveDecisionPolicyV1) -> dict[str, object]:
    return {node.state_id: node for node in policy.nodes}


def _policy_costs_by_hypothesis(
    policy: ActiveDecisionPolicyV1,
    probes: tuple[DeterministicDecisionProbeV1, ...],
) -> np.ndarray:
    nodes = _policy_node_map(policy)
    probe_by_id = {probe.probe_id: probe for probe in probes}
    costs = np.zeros(probes[0].hypothesis_count, dtype=np.float64)
    for hypothesis in range(costs.size):
        state_id = policy.root_state_id
        visited: set[str] = set()
        while True:
            if state_id in visited:
                raise RuntimeError("active policy contains a cycle")
            visited.add(state_id)
            node = nodes[state_id]
            if node.certified:
                break
            if node.selected_probe_id is None:
                costs[hypothesis] = math.inf
                break
            probe = probe_by_id[node.selected_probe_id]
            costs[hypothesis] += probe.cost
            outcome = int(probe.outcome_index[hypothesis])
            children = dict(node.outcome_children)
            if outcome not in children:
                raise RuntimeError("policy omits a feasible probe outcome")
            state_id = children[outcome]
    return costs


def _entropy(values: np.ndarray) -> float:
    counts = np.bincount(values)
    probabilities = counts[counts > 0] / values.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _greedy_entropy_costs(
    prior: np.ndarray,
    quotient: np.ndarray,
    classes: np.ndarray,
    losses: np.ndarray,
    probes: tuple[DeterministicDecisionProbeV1, ...],
) -> np.ndarray:
    costs = np.zeros(prior.size, dtype=np.float64)
    ordered = tuple(sorted(probes, key=lambda item: item.probe_id))
    for truth in range(prior.size):
        consistent = np.ones(prior.size, dtype=np.bool_)
        remaining = list(ordered)
        while True:
            certificate = conditioned_query_decision_certificate(
                prior,
                quotient,
                classes,
                losses,
                consistent_hypothesis_mask=consistent,
            )
            if certificate.has_tolerance_admissible_action:
                break
            candidates: list[tuple[float, str, DeterministicDecisionProbeV1]] = []
            for probe in remaining:
                outcomes = probe.outcome_index[consistent & (prior > 0.0)]
                candidates.append((-_entropy(outcomes), probe.probe_id, probe))
            if not candidates:
                costs[truth] = math.inf
                break
            _, _, selected = min(candidates)
            outcomes = selected.outcome_index[consistent & (prior > 0.0)]
            if np.unique(outcomes).size <= 1:
                costs[truth] = math.inf
                break
            costs[truth] += selected.cost
            consistent &= selected.outcome_index == selected.outcome_index[truth]
            remaining = [probe for probe in remaining if probe is not selected]
    return costs


def _fixed_order_costs(
    prior: np.ndarray,
    quotient: np.ndarray,
    classes: np.ndarray,
    losses: np.ndarray,
    order: tuple[DeterministicDecisionProbeV1, ...],
) -> np.ndarray:
    costs = np.zeros(prior.size, dtype=np.float64)
    for truth in range(prior.size):
        consistent = np.ones(prior.size, dtype=np.bool_)
        for probe in order:
            certificate = conditioned_query_decision_certificate(
                prior,
                quotient,
                classes,
                losses,
                consistent_hypothesis_mask=consistent,
            )
            if certificate.has_tolerance_admissible_action:
                break
            costs[truth] += probe.cost
            consistent &= probe.outcome_index == probe.outcome_index[truth]
        else:
            certificate = conditioned_query_decision_certificate(
                prior,
                quotient,
                classes,
                losses,
                consistent_hypothesis_mask=consistent,
            )
            if not certificate.has_tolerance_admissible_action:
                costs[truth] = math.inf
    return costs


def _minimum_full_state_probe_cost(
    probes: tuple[DeterministicDecisionProbeV1, ...],
) -> tuple[float, tuple[str, ...]]:
    best_cost = math.inf
    best: tuple[str, ...] = ()
    for size in range(len(probes) + 1):
        for selected in itertools.combinations(probes, size):
            signatures = (
                np.stack(
                    [probe.outcome_index for probe in selected],
                    axis=1,
                )
                if selected
                else np.zeros((probes[0].hypothesis_count, 0), dtype=int)
            )
            if np.unique(signatures, axis=0).shape[0] != signatures.shape[0]:
                continue
            cost = float(sum(probe.cost for probe in selected))
            ids = tuple(sorted(probe.probe_id for probe in selected))
            if cost < best_cost or (math.isclose(cost, best_cost) and ids < best):
                best_cost = cost
                best = ids
    return best_cost, best


def run(protocol: dict[str, Any]) -> dict[str, Any]:
    prior, quotient, classes, losses, probes, action_group = _problem()
    policy = synthesize_minimax_active_decision_policy(
        prior,
        quotient,
        classes,
        losses,
        probes,
        regret_tolerance=0.0,
    )
    policy_costs = _policy_costs_by_hypothesis(policy, probes)
    global_decision = minimum_cost_global_decision_identifying_probe_set(
        prior,
        classes,
        losses,
        probes,
    )
    full_state_cost, full_state_probes = _minimum_full_state_probe_cost(probes)
    entropy_costs = _greedy_entropy_costs(
        prior,
        quotient,
        classes,
        losses,
        probes,
    )

    fixed_expected: list[float] = []
    for order in itertools.permutations(probes):
        fixed_expected.append(
            float(prior @ _fixed_order_costs(prior, quotient, classes, losses, order))
        )

    radius_rows: list[dict[str, object]] = []
    for radius in protocol["robustness"]["simultaneous_loss_radius_sweep"]:
        radii = np.full_like(losses, float(radius))
        robust_policy = synthesize_minimax_active_decision_policy(
            prior,
            quotient,
            classes,
            losses,
            probes,
            loss_radius_by_hypothesis_action=radii,
        )
        radius_rows.append(
            {
                "radius": float(radius),
                "feasible": robust_policy.feasible,
                "worst_case_cost": (
                    robust_policy.root_worst_case_cost
                    if robust_policy.feasible
                    else None
                ),
            }
        )

    without_second = tuple(
        probe for probe in probes if probe.probe_id != "decision-probe-1"
    )
    impossible = synthesize_minimax_active_decision_policy(
        prior,
        quotient,
        classes,
        losses,
        without_second,
    )

    expected = protocol["required_checks"]
    checks = {
        "active_worst_case_cost": math.isclose(
            policy.root_worst_case_cost,
            expected["active_worst_case_cost"],
            abs_tol=1e-12,
        ),
        "active_uniform_expected_cost": math.isclose(
            float(prior @ policy_costs),
            expected["active_uniform_expected_cost"],
            abs_tol=1e-12,
        ),
        "global_decision_probe_cost": math.isclose(
            global_decision.total_cost,
            expected["global_decision_probe_cost"],
            abs_tol=1e-12,
        ),
        "full_state_probe_cost": math.isclose(
            full_state_cost,
            expected["full_state_probe_cost"],
            abs_tol=1e-12,
        ),
        "entropy_expected_cost": math.isclose(
            float(prior @ entropy_costs),
            expected["entropy_expected_cost"],
            abs_tol=1e-12,
        ),
        "loss_radius_0_49_feasible": next(
            row["feasible"] for row in radius_rows if row["radius"] == 0.49
        )
        is expected["loss_radius_0_49_feasible"],
        "loss_radius_0_51_feasible": next(
            row["feasible"] for row in radius_rows if row["radius"] == 0.51
        )
        is expected["loss_radius_0_51_feasible"],
        "unresolvable_control_feasible": (
            impossible.feasible is expected["unresolvable_control_feasible"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"registered active-acquisition checks failed: {checks}")

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "decision": "controlled-active-decision-acquisition-passed",
        "problem": {
            "hypothesis_count": int(prior.size),
            "action_count": int(losses.shape[1]),
            "action_group_counts": np.bincount(action_group).tolist(),
            "probe_count": len(probes),
        },
        "active_policy": {
            **policy.summary(),
            "uniform_expected_cost": float(prior @ policy_costs),
            "costs_by_action_group": [
                float(np.mean(policy_costs[action_group == group]))
                for group in range(losses.shape[1])
            ],
            "cost_reduction_vs_full_state": 1.0
            - policy.root_worst_case_cost / full_state_cost,
        },
        "global_decision_probe_set": {
            "selected_probe_ids": list(global_decision.selected_probe_ids),
            "total_cost": global_decision.total_cost,
            "conflict_pair_count": global_decision.conflict_pair_count,
        },
        "full_state_probe_set": {
            "selected_probe_ids": list(full_state_probes),
            "total_cost": full_state_cost,
        },
        "entropy_greedy": {
            "uniform_expected_cost": float(prior @ entropy_costs),
            "worst_case_cost": float(np.max(entropy_costs)),
        },
        "fixed_order_distribution": {
            "order_count": len(fixed_expected),
            "minimum_uniform_expected_cost": float(np.min(fixed_expected)),
            "median_uniform_expected_cost": float(np.median(fixed_expected)),
            "maximum_uniform_expected_cost": float(np.max(fixed_expected)),
        },
        "loss_radius_sweep": radius_rows,
        "unresolvable_control": {
            "removed_probe": "decision-probe-1",
            "feasible": impossible.feasible,
            "root_worst_case_cost": (
                impossible.root_worst_case_cost if impossible.feasible else None
            ),
        },
        "checks": checks,
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = _content_id(result)
    return result


def _summary(result: dict[str, Any]) -> str:
    active = result["active_policy"]
    entropy = result["entropy_greedy"]
    global_set = result["global_decision_probe_set"]
    full = result["full_state_probe_set"]
    lines = [
        "# Controlled active decision-acquisition result",
        "",
        f"Decision: `{result['decision']}`",
        f"Result ID: `{result['result_id']}`",
        "",
        "| Method | Worst-case probe cost | Uniform expected cost |",
        "| --- | ---: | ---: |",
        (
            "| Exact adaptive decision certificate | "
            f"{active['root_worst_case_cost']:.3f} | "
            f"{active['uniform_expected_cost']:.3f} |"
        ),
        (
            "| Greedy hypothesis entropy | "
            f"{entropy['worst_case_cost']:.3f} | "
            f"{entropy['uniform_expected_cost']:.3f} |"
        ),
        f"| Global decision-identifying set | {global_set['total_cost']:.3f} | n/a |",
        f"| Full-state-identifying set | {full['total_cost']:.3f} | n/a |",
        "",
        (
            "The adaptive policy spends one probe on the 16/24 hypothesis branch "
            "whose decision is immediately identified and a second probe only on "
            "the remaining branch. Complete state identification requires all six "
            "registered probes."
        ),
        "",
        (
            "Removing `decision-probe-1` leaves an observationally indistinguishable "
            "pair with opposing optimal actions; the exact policy reports the task "
            "as infeasible and therefore fails closed."
        ),
        "",
        "This is controlled finite-hypothesis evidence, not a real sensor result.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = _load_protocol(args.protocol)
    result = run(protocol)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(
        _summary(result),
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
