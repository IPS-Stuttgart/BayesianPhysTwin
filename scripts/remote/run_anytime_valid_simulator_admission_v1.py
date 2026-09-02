#!/usr/bin/env python3
"""Run the frozen controlled anytime-valid simulator-admission evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin_experiments.anytime_valid_admission_v1 import (
    AnytimeAdmissionConfig,
    AnytimeSimulatorCorrectionGuard,
    BettingMixtureConfig,
    DeploymentState,
    geometric_alpha,
)

CONTRACT = "anytime-valid-simulator-admission-v1"
RESULT_CONTRACT = "anytime-valid-simulator-admission-result-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_protocol(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol must be a JSON object")
    if payload.get("schema_version") != 1 or payload.get("contract") != CONTRACT:
        raise ValueError("unsupported protocol identity")
    score = mapping(payload.get("registered_score"), label="score")
    if (
        score.get("name") != "symmetric-relative-paired-gain"
        or score.get("range") != [-1.0, 1.0]
        or score.get("raw_loss_claim_authorized") is not False
    ):
        raise ValueError("registered score changed")
    e_process = mapping(payload.get("e_process"), label="e-process")
    lambdas = [float(value) for value in cast(list[object], e_process["lambdas"])]
    if (
        e_process.get("family") != "fixed-mixture-betting"
        or lambdas != [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
        or float(cast(Any, e_process["global_promotion_alpha"])) != 0.05
        or float(cast(Any, e_process["global_revocation_alpha"])) != 0.05
        or int(cast(Any, e_process["minimum_promotion_observations"])) != 8
        or int(cast(Any, e_process["minimum_revocation_observations"])) != 5
    ):
        raise ValueError("e-process contract changed")
    evaluation = mapping(
        payload.get("controlled_evaluation"),
        label="controlled evaluation",
    )
    expected = {
        "seed": 20260902,
        "horizon": 500,
        "null_replications": 20000,
        "alternative_replications": 5000,
        "shift_replications": 5000,
    }
    for key, value in expected.items():
        if int(cast(Any, evaluation.get(key, -1))) != value:
            raise ValueError(f"controlled evaluation field changed: {key}")
    boundary = mapping(payload.get("information_boundary"), label="boundary")
    if (
        boundary.get("controlled_simulation_only") is not True
        or boundary.get("existing_dlo_target_outcomes_used") is not False
        or boundary.get("existing_deform360_outcomes_used") is not False
        or boundary.get("hyperparameter_selection_from_evaluation") is not False
        or boundary.get("post_result_threshold_changes") is not False
        or boundary.get("paper_claim_authorized") is not False
    ):
        raise ValueError("information boundary changed")
    return payload


def row_logsumexp(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    return maximum + np.log(np.sum(np.exp(values - maximum[:, None]), axis=1))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_crossings(first: np.ndarray, horizon: int) -> dict[str, object]:
    crossed = first >= 0
    times = first[crossed] + 1
    return {
        "replications": int(len(first)),
        "crossing_count": int(np.sum(crossed)),
        "crossing_probability": float(np.mean(crossed)),
        "wilson_95_interval": wilson_interval(int(np.sum(crossed)), len(first)),
        "median_first_crossing": (
            None if len(times) == 0 else float(np.median(times))
        ),
        "mean_first_crossing": None if len(times) == 0 else float(np.mean(times)),
        "horizon": horizon,
    }


def scenario_scores(
    *,
    name: str,
    time_index: int,
    replications: int,
    rng: np.random.Generator,
    previous_log_e: np.ndarray,
    positive_probability: float | None,
) -> np.ndarray:
    if name == "fair-rademacher":
        probability = 0.5
        magnitude: float | np.ndarray = 1.0
    elif name == "negative-drift-rademacher":
        probability = cast(float, positive_probability)
        magnitude = 1.0
    elif name == "predictable-heteroskedastic-null":
        probability = 0.5
        phase = 2.0 * math.pi * time_index / 37.0
        magnitude = 0.55 + 0.45 * math.sin(phase)
    elif name == "wealth-adaptive-null":
        probability = 0.5
        clipped = np.clip(previous_log_e, -20.0, 20.0)
        magnitude = 0.1 + 0.9 / (1.0 + np.exp(-clipped))
    elif name in {"moderate-positive-gain", "strong-positive-gain"}:
        probability = cast(float, positive_probability)
        magnitude = 1.0
    else:
        raise ValueError(f"unknown scenario: {name}")
    signs = np.where(rng.random(replications) < probability, 1.0, -1.0)
    return signs * magnitude


def simulate_monitoring(
    *,
    name: str,
    replications: int,
    horizon: int,
    rng: np.random.Generator,
    lambdas: np.ndarray,
    weights: np.ndarray,
    epoch_alpha: float,
    minimum_observations: int,
    positive_probability: float | None = None,
) -> dict[str, object]:
    component_logs = np.zeros((replications, len(lambdas)), dtype=np.float64)
    log_weights = np.log(weights)
    first_e = np.full(replications, -1, dtype=np.int64)
    first_naive = np.full(replications, -1, dtype=np.int64)
    first_spending = np.full(replications, -1, dtype=np.int64)
    cumulative = np.zeros(replications, dtype=np.float64)
    previous_log_e = np.zeros(replications, dtype=np.float64)
    final_log_e = previous_log_e

    for time_index in range(horizon):
        scores = scenario_scores(
            name=name,
            time_index=time_index,
            replications=replications,
            rng=rng,
            previous_log_e=previous_log_e,
            positive_probability=positive_probability,
        )
        component_logs += np.log1p(scores[:, None] * lambdas[None, :])
        final_log_e = row_logsumexp(component_logs + log_weights[None, :])
        previous_log_e = final_log_e
        count = time_index + 1
        eligible = count >= minimum_observations
        newly_e = (first_e < 0) & eligible & (final_log_e >= -math.log(epoch_alpha))
        first_e[newly_e] = time_index

        cumulative += scores
        mean = cumulative / count
        naive_threshold = math.sqrt(2.0 * math.log(1.0 / epoch_alpha) / count)
        newly_naive = (first_naive < 0) & eligible & (mean >= naive_threshold)
        first_naive[newly_naive] = time_index

        spending_alpha = epoch_alpha * 6.0 / (math.pi * math.pi * count * count)
        spending_threshold = math.sqrt(
            2.0 * math.log(1.0 / spending_alpha) / count
        )
        newly_spending = (
            (first_spending < 0) & eligible & (mean >= spending_threshold)
        )
        first_spending[newly_spending] = time_index

    fixed_horizon = final_log_e >= -math.log(epoch_alpha)
    return {
        "anytime_e_process": summarize_crossings(first_e, horizon),
        "naive_repeated_fixed_time_hoeffding": summarize_crossings(
            first_naive,
            horizon,
        ),
        "alpha_spending_hoeffding": summarize_crossings(first_spending, horizon),
        "fixed_horizon_e_process": {
            "replications": replications,
            "crossing_count": int(np.sum(fixed_horizon)),
            "crossing_probability": float(np.mean(fixed_horizon)),
            "wilson_95_interval": wilson_interval(
                int(np.sum(fixed_horizon)),
                replications,
            ),
            "horizon": horizon,
        },
    }


def simulate_abrupt_shift(
    *,
    replications: int,
    horizon: int,
    shift_index: int,
    positive_before: float,
    positive_after: float,
    rng: np.random.Generator,
    lambdas: np.ndarray,
    weights: np.ndarray,
    promotion_alpha: float,
    revocation_alpha: float,
    minimum_promotion: int,
    minimum_revocation: int,
) -> dict[str, object]:
    log_weights = np.log(weights)
    promotion_logs = np.zeros((replications, len(lambdas)), dtype=np.float64)
    revocation_logs = np.zeros_like(promotion_logs)
    promotion_count = np.zeros(replications, dtype=np.int64)
    revocation_count = np.zeros(replications, dtype=np.int64)
    active = np.zeros(replications, dtype=bool)
    ever_promoted = np.zeros(replications, dtype=bool)
    revoked = np.zeros(replications, dtype=bool)
    promotion_time = np.full(replications, -1, dtype=np.int64)
    revocation_time = np.full(replications, -1, dtype=np.int64)
    harmful_active_steps = np.zeros(replications, dtype=np.int64)

    for time_index in range(horizon):
        probability = positive_before if time_index < shift_index else positive_after
        scores = np.where(rng.random(replications) < probability, 1.0, -1.0)
        was_active = active.copy()
        if time_index >= shift_index:
            harmful_active_steps += was_active.astype(np.int64)

        promotable = ~ever_promoted
        if np.any(promotable):
            promotion_logs[promotable] += np.log1p(
                scores[promotable, None] * lambdas[None, :]
            )
            promotion_count[promotable] += 1
            log_e = row_logsumexp(
                promotion_logs[promotable] + log_weights[None, :]
            )
            indices = np.flatnonzero(promotable)
            crossed = (
                (promotion_count[promotable] >= minimum_promotion)
                & (log_e >= -math.log(promotion_alpha))
            )
            newly_promoted = indices[crossed]
            active[newly_promoted] = True
            ever_promoted[newly_promoted] = True
            promotion_time[newly_promoted] = time_index

        revocable = was_active & ~revoked
        if np.any(revocable):
            revocation_logs[revocable] += np.log1p(
                -scores[revocable, None] * lambdas[None, :]
            )
            revocation_count[revocable] += 1
            log_e = row_logsumexp(
                revocation_logs[revocable] + log_weights[None, :]
            )
            indices = np.flatnonzero(revocable)
            crossed = (
                (revocation_count[revocable] >= minimum_revocation)
                & (log_e >= -math.log(revocation_alpha))
            )
            newly_revoked = indices[crossed]
            active[newly_revoked] = False
            revoked[newly_revoked] = True
            revocation_time[newly_revoked] = time_index

    promoted_before_shift = ever_promoted & (promotion_time < shift_index)
    revoked_after_shift = promoted_before_shift & revoked & (
        revocation_time >= shift_index
    )
    delays = revocation_time[revoked_after_shift] - shift_index + 1
    unguarded_harmful_steps = horizon - shift_index
    return {
        "replications": replications,
        "promotion_probability": float(np.mean(ever_promoted)),
        "promotion_before_shift_probability": float(np.mean(promoted_before_shift)),
        "revocation_probability_given_pre_shift_promotion": (
            0.0
            if not np.any(promoted_before_shift)
            else float(np.mean(revoked_after_shift[promoted_before_shift]))
        ),
        "median_revocation_delay_after_shift": (
            None if len(delays) == 0 else float(np.median(delays))
        ),
        "mean_revocation_delay_after_shift": (
            None if len(delays) == 0 else float(np.mean(delays))
        ),
        "median_harmful_active_steps": float(np.median(harmful_active_steps)),
        "mean_harmful_active_steps": float(np.mean(harmful_active_steps)),
        "unguarded_harmful_active_steps_per_replication": unguarded_harmful_steps,
        "median_harmful_steps_fraction_of_unguarded": float(
            np.median(harmful_active_steps) / unguarded_harmful_steps
        ),
        "shift_index": shift_index,
        "horizon": horizon,
    }


def exercise_exact_identity(config: AnytimeAdmissionConfig) -> dict[str, object]:
    guard: AnytimeSimulatorCorrectionGuard[str] = AnytimeSimulatorCorrectionGuard(
        config
    )
    fallback = {"kind": "caller-owned-fallback", "payload": [1, 2, 3]}
    candidate = {"kind": "frozen-candidate", "payload": [4, 5, 6]}
    violations = 0
    if guard.select(fallback_belief=fallback, candidate_belief=candidate) is not fallback:
        violations += 1

    guard.register("delayed-first")
    guard.register("delayed-second")
    guard.resolve("delayed-second", candidate_loss=0.0, fallback_loss=1.0)
    guard.resolve("delayed-first", candidate_loss=0.0, fallback_loss=1.0)
    index = 0
    while guard.state is DeploymentState.FALLBACK and index < 300:
        guard.observe(
            f"promote-{index}",
            candidate_loss=0.0,
            fallback_loss=1.0,
        )
        index += 1
    if guard.state is not DeploymentState.CANDIDATE:
        raise RuntimeError("deterministic positive stream did not promote")
    if guard.select(fallback_belief=fallback, candidate_belief=candidate) is not candidate:
        violations += 1

    index = 0
    while guard.state is DeploymentState.CANDIDATE and index < 300:
        guard.observe(
            f"revoke-{index}",
            candidate_loss=1.0,
            fallback_loss=0.0,
        )
        index += 1
    if guard.state is not DeploymentState.FALLBACK:
        raise RuntimeError("deterministic negative stream did not revoke")
    selected = guard.select(fallback_belief=fallback, candidate_belief=candidate)
    if selected is not fallback:
        violations += 1
    if selected != {"kind": "caller-owned-fallback", "payload": [1, 2, 3]}:
        violations += 1
    return {
        "exact_fallback_identity_violations": violations,
        "promotion_observations": next(
            event.resolution_index + 1 for event in guard.events if event.promoted
        ),
        "revocation_observations_after_promotion": sum(
            event.state_before == DeploymentState.CANDIDATE.value
            for event in guard.events
        ),
        "delayed_outcomes_resolved_out_of_registration_order": True,
        "final_state": guard.state.value,
        "final_epoch": guard.epoch_index,
        "snapshot": guard.snapshot(),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_report(path: Path, result: Mapping[str, object]) -> None:
    nulls = mapping(result["null_scenarios"], label="null scenarios")
    alternatives = mapping(result["alternative_scenarios"], label="alternatives")
    shift = mapping(result["abrupt_shift"], label="shift")
    identity = mapping(result["identity_audit"], label="identity")
    lines = [
        "# Anytime-valid simulator-correction admission v1",
        "",
        f"- Decision: **{result['decision']}**",
        f"- Global ever-false-promotion budget: **{result['global_promotion_alpha']}**",
        f"- First-epoch alpha: **{result['first_epoch_promotion_alpha']}**",
        "- Registered score: symmetric relative paired gain in `[-1,1]`",
        "",
        "## Optional-stopping null audit",
        "",
        "| Scenario | E-process false promotion | Wilson 95% upper | Naive repeated |",
        "|---|---:|---:|---:|",
    ]
    for name, raw in nulls.items():
        scenario = mapping(raw, label=name)
        e_process = mapping(scenario["anytime_e_process"], label="e-process")
        naive = mapping(
            scenario["naive_repeated_fixed_time_hoeffding"],
            label="naive comparator",
        )
        interval = cast(list[float], e_process["wilson_95_interval"])
        lines.append(
            f"| `{name}` | {float(cast(Any, e_process['crossing_probability'])):.4f} "
            f"| {interval[1]:.4f} | "
            f"{float(cast(Any, naive['crossing_probability'])):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Positive-gain power",
            "",
            "| Scenario | Anytime power | Median promotion observation | Fixed-horizon power |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, raw in alternatives.items():
        scenario = mapping(raw, label=name)
        anytime = mapping(scenario["anytime_e_process"], label="anytime")
        fixed = mapping(scenario["fixed_horizon_e_process"], label="fixed")
        lines.append(
            f"| `{name}` | {float(cast(Any, anytime['crossing_probability'])):.4f} "
            f"| {anytime['median_first_crossing']} | "
            f"{float(cast(Any, fixed['crossing_probability'])):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Abrupt sign reversal",
            "",
            (
                "- Promotion before shift: "
                f"**{float(cast(Any, shift['promotion_before_shift_probability'])):.4f}**"
            ),
            (
                "- Revocation given pre-shift promotion: "
                f"**{float(cast(Any, shift['revocation_probability_given_pre_shift_promotion'])):.4f}**"
            ),
            (
                "- Median post-shift revocation delay: "
                f"**{shift['median_revocation_delay_after_shift']} observations**"
            ),
            (
                "- Median harmful active steps relative to unguarded: "
                f"**{float(cast(Any, shift['median_harmful_steps_fraction_of_unguarded'])):.4f}**"
            ),
            "",
            "## Exact fallback",
            "",
            (
                "- Identity violations: "
                f"**{identity['exact_fallback_identity_violations']}**"
            ),
            f"- Final state after deterministic reversal: **{identity['final_state']}**",
            "",
            "## Guarantee and boundary",
            "",
            str(result["guarantee"]),
            "",
            str(result["claim_boundary"]),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, result: Mapping[str, object]) -> None:
    rows: list[dict[str, object]] = []
    for group_name in ("null_scenarios", "alternative_scenarios"):
        group = mapping(result[group_name], label=group_name)
        for scenario_name, raw in group.items():
            scenario = mapping(raw, label=scenario_name)
            for method_name, method_raw in scenario.items():
                method = mapping(method_raw, label=method_name)
                rows.append(
                    {
                        "group": group_name,
                        "scenario": scenario_name,
                        "method": method_name,
                        "crossing_probability": method.get("crossing_probability"),
                        "median_first_crossing": method.get("median_first_crossing"),
                        "wilson_low": cast(list[object], method["wilson_95_interval"])[0],
                        "wilson_high": cast(list[object], method["wilson_95_interval"])[1],
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_protocol(protocol_path)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    e_process = mapping(protocol["e_process"], label="e-process")
    evaluation = mapping(protocol["controlled_evaluation"], label="evaluation")
    gate = mapping(protocol["mechanism_gate"], label="gate")
    lambdas = np.asarray(e_process["lambdas"], dtype=np.float64)
    weights = np.full(len(lambdas), 1.0 / len(lambdas), dtype=np.float64)
    global_promotion_alpha = float(cast(Any, e_process["global_promotion_alpha"]))
    global_revocation_alpha = float(cast(Any, e_process["global_revocation_alpha"]))
    first_promotion_alpha = geometric_alpha(global_promotion_alpha, 0)
    first_revocation_alpha = geometric_alpha(global_revocation_alpha, 0)
    minimum_promotion = int(cast(Any, e_process["minimum_promotion_observations"]))
    minimum_revocation = int(cast(Any, e_process["minimum_revocation_observations"]))
    horizon = int(cast(Any, evaluation["horizon"]))
    rng = np.random.default_rng(int(cast(Any, evaluation["seed"])))

    null_results: dict[str, object] = {}
    for raw in cast(list[object], evaluation["null_scenarios"]):
        scenario = mapping(raw, label="null scenario")
        name = str(scenario["name"])
        probability = scenario.get("positive_probability")
        null_results[name] = simulate_monitoring(
            name=name,
            replications=int(cast(Any, evaluation["null_replications"])),
            horizon=horizon,
            rng=rng,
            lambdas=lambdas,
            weights=weights,
            epoch_alpha=first_promotion_alpha,
            minimum_observations=minimum_promotion,
            positive_probability=(
                None if probability is None else float(cast(Any, probability))
            ),
        )

    alternative_results: dict[str, object] = {}
    for raw in cast(list[object], evaluation["alternative_scenarios"]):
        scenario = mapping(raw, label="alternative scenario")
        name = str(scenario["name"])
        alternative_results[name] = simulate_monitoring(
            name=name,
            replications=int(cast(Any, evaluation["alternative_replications"])),
            horizon=horizon,
            rng=rng,
            lambdas=lambdas,
            weights=weights,
            epoch_alpha=first_promotion_alpha,
            minimum_observations=minimum_promotion,
            positive_probability=float(cast(Any, scenario["positive_probability"])),
        )

    shift_contract = mapping(evaluation["abrupt_shift"], label="shift")
    shift_result = simulate_abrupt_shift(
        replications=int(cast(Any, evaluation["shift_replications"])),
        horizon=horizon,
        shift_index=int(cast(Any, shift_contract["shift_index"])),
        positive_before=float(
            cast(Any, shift_contract["positive_probability_before_shift"])
        ),
        positive_after=float(
            cast(Any, shift_contract["positive_probability_after_shift"])
        ),
        rng=rng,
        lambdas=lambdas,
        weights=weights,
        promotion_alpha=first_promotion_alpha,
        revocation_alpha=first_revocation_alpha,
        minimum_promotion=minimum_promotion,
        minimum_revocation=minimum_revocation,
    )

    config = AnytimeAdmissionConfig(
        candidate_id="frozen-correction-v1",
        fallback_id="caller-owned-physical-belief",
        total_promotion_alpha=global_promotion_alpha,
        total_revocation_alpha=global_revocation_alpha,
        minimum_promotion_observations=minimum_promotion,
        minimum_revocation_observations=minimum_revocation,
        betting=BettingMixtureConfig(lambdas=tuple(float(x) for x in lambdas)),
    )
    identity_result = exercise_exact_identity(config)

    maximum_null_upper = max(
        float(
            cast(
                list[object],
                mapping(
                    mapping(raw, label=name)["anytime_e_process"],
                    label="e-process",
                )["wilson_95_interval"],
            )[1]
        )
        for name, raw in null_results.items()
    )
    strong_power = float(
        cast(
            Any,
            mapping(
                mapping(
                    alternative_results["strong-positive-gain"],
                    label="strong alternative",
                )["anytime_e_process"],
                label="anytime",
            )["crossing_probability"],
        )
    )
    promotion_probability = float(
        cast(Any, shift_result["promotion_before_shift_probability"])
    )
    revocation_probability = float(
        cast(
            Any,
            shift_result["revocation_probability_given_pre_shift_promotion"],
        )
    )
    harmful_fraction = float(
        cast(Any, shift_result["median_harmful_steps_fraction_of_unguarded"])
    )
    identity_violations = int(
        cast(Any, identity_result["exact_fallback_identity_violations"])
    )
    checks = {
        "null_wilson_upper": maximum_null_upper
        <= float(cast(Any, gate["maximum_null_wilson_upper_bound"])),
        "strong_alternative_power": strong_power
        >= float(cast(Any, gate["minimum_strong_alternative_power"])),
        "shift_promotion_probability": promotion_probability
        >= float(cast(Any, gate["minimum_shift_promotion_probability"])),
        "shift_revocation_probability": revocation_probability
        >= float(
            cast(Any, gate["minimum_revocation_probability_given_promotion"])
        ),
        "harmful_active_steps_fraction": harmful_fraction
        <= float(
            cast(
                Any,
                gate["maximum_median_harmful_active_steps_fraction_of_unguarded"],
            )
        ),
        "exact_fallback_identity": identity_violations
        <= int(cast(Any, gate["maximum_exact_fallback_identity_violations"])),
    }
    supported = all(checks.values())
    result = {
        "schema_version": 1,
        "contract": RESULT_CONTRACT,
        "status": "complete",
        "decision": (
            "anytime-valid-admission-mechanism-supported"
            if supported
            else "anytime-valid-admission-mechanism-gate-failed"
        ),
        "global_promotion_alpha": global_promotion_alpha,
        "first_epoch_promotion_alpha": first_promotion_alpha,
        "global_revocation_alpha": global_revocation_alpha,
        "first_epoch_revocation_alpha": first_revocation_alpha,
        "null_scenarios": null_results,
        "alternative_scenarios": alternative_results,
        "abrupt_shift": shift_result,
        "identity_audit": identity_result,
        "mechanism_gate": {
            "passed": supported,
            "checks": checks,
            "maximum_null_wilson_upper_bound_observed": maximum_null_upper,
            "strong_alternative_power_observed": strong_power,
        },
        "guarantee": (
            "For each registered epoch, if the bounded paired gain X_t lies in "
            "[-1,1] and satisfies E[X_t|F_(t-1)]<=0, the fixed-mixture betting "
            "process is a nonnegative supermartingale and the probability of "
            "ever crossing 1/alpha_j is at most alpha_j. The geometric alpha "
            "schedule sums to the global promotion budget, so a union bound "
            "controls ever-false promotion across arbitrarily many restarts."
        ),
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": protocol["claim_boundary"],
        "protocol": protocol,
    }
    write_json(output_root / "result.json", result)
    write_report(output_root / "report.md", result)
    write_summary_csv(output_root / "monitoring-summary.csv", result)
    print(json.dumps(result["mechanism_gate"], indent=2, sort_keys=True))
    return 0 if supported else 2


if __name__ == "__main__":
    raise SystemExit(main())
