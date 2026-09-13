#!/usr/bin/env python3
"""Controlled robust/open-set extension of interventional cause attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin_experiments.robust_interventional_cause_attribution_v1 import (
    RobustAttributionPlanV1,
    RobustAttributionStatus,
    RobustCauseModelV1,
    RobustObservationDesignV1,
)

CAUSES = (
    "observation_bias",
    "physical_parameter",
    "physical_state",
    "realized_intervention",
    "source_local_discrepancy",
)
PHYSICAL_CAUSES = {"physical_parameter", "physical_state", "realized_intervention"}
ACTIONS = ("action-0", "action-1", "action-2")


def true_signatures() -> dict[str, np.ndarray]:
    return {
        "observation_bias": np.asarray([[1, 0, 0]] * 4, dtype=float),
        "physical_parameter": np.asarray(
            [[1, 0, 0], [0, 1, .2], [.2, .3, 1], [.5, 1, -.3]], dtype=float
        ),
        "physical_state": np.asarray(
            [[1, 0, 0], [1, .5, 0], [1, -.5, .5], [1, .2, -.6]], dtype=float
        ),
        "realized_intervention": np.asarray(
            [[1, 0, 0], [.5, -.5, 1], [-.5, 1, .1], [1, -1, .5]], dtype=float
        ),
        "source_local_discrepancy": np.asarray(
            [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float
        ),
    }


def _estimated_signatures(seed: int, bound: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for cause, values in true_signatures().items():
        perturbation = np.empty_like(values)
        for index in range(len(values)):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            perturbation[index] = 0.8 * bound * direction
        result[cause] = values + perturbation
    return result


def _projector_complement(design: np.ndarray) -> np.ndarray:
    left, singular, _ = np.linalg.svd(design, full_matrices=False)
    rank = int(np.count_nonzero(singular > 1e-10 * singular[0]))
    basis = left[:, :rank]
    return np.eye(design.shape[0]) - basis @ basis.T


def _fit(signature: np.ndarray, observation: np.ndarray) -> float:
    return float(np.dot(signature, observation) / np.dot(signature, signature))


def _closed_label(signatures: dict[str, np.ndarray], y: np.ndarray, order: tuple[int, ...]) -> tuple[str, float]:
    fits: dict[str, tuple[float, float]] = {}
    for cause in CAUSES:
        signature = signatures[cause][list(order)].reshape(-1)
        coefficient = _fit(signature, y)
        fits[cause] = (float(np.linalg.norm(y - signature * coefficient)), coefficient)
    label = min(CAUSES, key=lambda value: (fits[value][0], value))
    return label, fits[label][1]


def _bounded_noise(rng: np.random.Generator, radius: float) -> np.ndarray:
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    return direction * rng.uniform(0.0, radius)


def _plan(estimated: dict[str, np.ndarray], error_bound: float, noise_radius: float) -> RobustAttributionPlanV1:
    models = tuple(
        RobustCauseModelV1(
            cause_id=cause,
            intervention_ids=ACTIONS,
            response_blocks=tuple(estimated[cause][index, :, None] for index in range(3)),
            query_map=np.eye(1),
            signature_error_bounds=(error_bound,) * 3,
            coefficient_norm_bound=1.5,
            query_error_tolerance=0.2,
            minimum_effect_norm=0.2,
        )
        for cause in CAUSES
    )
    design = RobustObservationDesignV1(
        intervention_ids=ACTIONS,
        nuisance_blocks=(np.empty((3, 0)),) * 3,
        observation_noise_radii=(noise_radius,) * 3,
        nuisance_signature_error_bounds=(0.0,) * 3,
        nuisance_coefficient_norm_bound=0.0,
        intervention_costs=(1.0, 1.0, 1.0),
    )
    return RobustAttributionPlanV1(design, models, "why-is-the-twin-wrong-robust-v1")


def run(trials: int, seed: int) -> dict[str, object]:
    error_bound = 0.005
    noise_radius = 0.005
    estimated = _estimated_signatures(seed + 1, error_bound)
    plan = _plan(estimated, error_bound, noise_radius)
    truth = true_signatures()
    registered = np.column_stack([truth[cause][:3].reshape(-1) for cause in CAUSES])
    rng = np.random.default_rng(seed + 2)
    unknown = _projector_complement(registered) @ rng.normal(size=9)
    unknown /= np.linalg.norm(unknown)
    unknown = unknown.reshape(3, 3)

    counts = {
        name: {"correct": 0, "resolved": 0, "unknown_correct": 0, "unknown_total": 0, "false_physical": 0}
        for name in ("factual", "closed_interventional", "robust_open_set", "wrong_action")
    }
    held_squared = {name: [] for name in counts}
    bound_covered = 0
    bound_total = 0
    labels = (*CAUSES, "unregistered_cause")

    for index in range(trials):
        true_cause = labels[index % len(labels)]
        amplitude = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.6, 1.5))
        if true_cause == "unregistered_cause":
            source = unknown * amplitude
            held = np.asarray([.4, -.7, .2]) * amplitude
        else:
            source = truth[true_cause][:3] * amplitude
            held = truth[true_cause][3] * amplitude
        blocks = tuple(source[action] + _bounded_noise(rng, noise_radius) for action in range(3))
        y = np.concatenate(blocks)

        factual, factual_coefficient = _closed_label(estimated, y[:3], (0,))
        closed, closed_coefficient = _closed_label(estimated, y, (0, 1, 2))
        wrong, wrong_coefficient = _closed_label(estimated, y, (0, 2, 1))
        decision = plan.evaluate(blocks)
        present = [value for value in decision.cause_decisions if value.effect_present_certified]
        if decision.registered_family_falsified:
            robust = "unregistered_cause"
            robust_coefficient = 0.0
        elif len(present) == 1:
            robust = present[0].cause_id
            robust_coefficient = float(present[0].query_estimate[0])
        else:
            robust = "unresolved"
            robust_coefficient = 0.0

        methods = {
            "factual": (factual, factual_coefficient),
            "closed_interventional": (closed, closed_coefficient),
            "robust_open_set": (robust, robust_coefficient),
            "wrong_action": (wrong, wrong_coefficient),
        }
        for name, (label, coefficient) in methods.items():
            if label != "unresolved":
                counts[name]["resolved"] += 1
            counts[name]["correct"] += int(label == true_cause)
            if true_cause == "unregistered_cause":
                counts[name]["unknown_total"] += 1
                counts[name]["unknown_correct"] += int(label == "unregistered_cause")
                counts[name]["false_physical"] += int(label in PHYSICAL_CAUSES)
            else:
                prediction = estimated[label][3] * coefficient if label in CAUSES else np.zeros(3)
                held_squared[name].append(float(np.mean((prediction - held) ** 2)))

        if true_cause in CAUSES:
            cause_decision = decision.result_for(true_cause)
            if cause_decision.status is RobustAttributionStatus.ROBUSTLY_ATTRIBUTABLE:
                bound_total += 1
                bound_covered += int(
                    abs(float(cause_decision.query_estimate[0]) - amplitude)
                    <= cause_decision.query_error_bound + 1e-12
                )

    metrics: dict[str, object] = {}
    for name, values in counts.items():
        metrics[name] = {
            "resolved_coverage": values["resolved"] / trials,
            "overall_accuracy": values["correct"] / trials,
            "unknown_detection_recall": values["unknown_correct"] / max(1, values["unknown_total"]),
            "false_physical_promotion_on_unknown": values["false_physical"] / max(1, values["unknown_total"]),
            "registered_held_intervention_rmse": float(np.sqrt(np.mean(held_squared[name]))),
        }

    # Near-confounding counterexample: the true A signature lies inside the
    # registered uncertainty ball around B. A point label is therefore wrong,
    # while the robust error bound must refuse promotion.
    near_a = RobustCauseModelV1(
        "cause_a", ("u0", "u1"), (np.asarray([[1.0]]), np.asarray([[1.0]])),
        np.eye(1), (0.0, 0.02), 1.0, 0.1, 0.1
    )
    near_b = RobustCauseModelV1(
        "cause_b", ("u0", "u1"), (np.asarray([[1.0]]), np.asarray([[1.02]])),
        np.eye(1), (0.0, 0.02), 1.0, 0.1, 0.1
    )
    near_design = RobustObservationDesignV1(
        ("u0", "u1"), (np.empty((1, 0)), np.empty((1, 0))),
        (0.0, 0.0), (0.0, 0.0), 0.0, (1.0, 1.0)
    )
    near_plan = RobustAttributionPlanV1(near_design, (near_a, near_b), "near-confounding")
    near_decision = near_plan.evaluate((np.asarray([1.0]), np.asarray([1.02])))

    result: dict[str, object] = {
        "schema": "why-is-the-twin-wrong-robust-controlled-v1",
        "trials": trials,
        "seed": seed,
        "registered_causes": list(CAUSES),
        "unknown_trial_fraction": 1.0 / len(labels),
        "metrics": metrics,
        "robust_bound_coverage": bound_covered / max(1, bound_total),
        "robust_bound_evaluated_trials": bound_total,
        "plan_id": plan.plan_id,
        "cause_plans": [
            {
                "cause_id": value.cause_id,
                "minimum_robust_intervention_count": value.minimum_robust_intervention_count,
                "minimal_robust_intervention_sets": [list(item) for item in value.minimal_robust_intervention_sets],
                "minimum_robust_intervention_cost": value.minimum_robust_intervention_cost,
            }
            for value in plan.cause_plans
        ],
        "near_confounding_stress": {
            "closed_set_point_label": "cause_b",
            "robust_cause_a_status": near_decision.result_for("cause_a").status.value,
            "robust_cause_b_status": near_decision.result_for("cause_b").status.value,
            "family_falsified": near_decision.registered_family_falsified,
        },
        "criteria": {},
    }
    criteria = {
        "robust_overall_accuracy_at_least_0_99": metrics["robust_open_set"]["overall_accuracy"] >= .99,
        "unknown_detection_recall_at_least_0_99": metrics["robust_open_set"]["unknown_detection_recall"] >= .99,
        "closed_set_unknown_recall_zero": metrics["closed_interventional"]["unknown_detection_recall"] == 0.0,
        "robust_bound_coverage_one": result["robust_bound_coverage"] == 1.0,
        "wrong_action_accuracy_below_robust": metrics["wrong_action"]["overall_accuracy"] < metrics["robust_open_set"]["overall_accuracy"],
        "near_confounding_not_promoted": near_decision.result_for("cause_a").status is RobustAttributionStatus.IDENTIFIABLE_BUT_UNSTABLE,
    }
    result["criteria"] = criteria
    result["decision"] = "robust-open-set-attribution-supported" if all(criteria.values()) else "robust-open-set-attribution-not-supported"
    result["claim_boundary"] = (
        "Controlled finite-family mechanism evidence. It does not establish natural "
        "real-world cause labels, family completeness, unseen-object transfer, or safety."
    )
    result["result_id"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def write_report(result: dict[str, object], path: Path) -> None:
    metrics = result["metrics"]
    lines = [
        "# Robust/open-set interventional attribution", "",
        f"**Decision:** `{result['decision']}`", "",
        "| Method | Resolved | Overall accuracy | Unknown recall | False physical promotion | Held-action RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("factual", "closed_interventional", "robust_open_set", "wrong_action"):
        row = metrics[name]
        lines.append(
            f"| `{name}` | {100*row['resolved_coverage']:.2f}% | {100*row['overall_accuracy']:.2f}% | "
            f"{100*row['unknown_detection_recall']:.2f}% | {100*row['false_physical_promotion_on_unknown']:.2f}% | "
            f"{row['registered_held_intervention_rmse']:.4f} |"
        )
    lines += ["", f"Deterministic query-bound coverage: **{100*result['robust_bound_coverage']:.2f}%** over {result['robust_bound_evaluated_trials']} registered-cause trials.", "", "## Frozen criteria", ""]
    lines += [f"- `{key}`: **{'pass' if value else 'fail'}**" for key, value in result["criteria"].items()]
    lines += ["", "## Near-confounding stress", "", json.dumps(result["near_confounding_stress"], indent=2), "", "## Claim boundary", "", result["claim_boundary"]]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.trials, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_report(result, args.output / "REPORT.md")
    print(json.dumps({"decision": result["decision"], "result_id": result["result_id"]}, indent=2))
    return 0 if result["decision"] == "robust-open-set-attribution-supported" else 3


if __name__ == "__main__":
    raise SystemExit(main())
