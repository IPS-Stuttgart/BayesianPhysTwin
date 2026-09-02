#!/usr/bin/env python3
"""Run the frozen controlled validation for anytime-valid simulator admission."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.anytime_valid_admission_v1 import (
    AnytimeAdmissionConfig,
    AnytimeAdmissionController,
    DeploymentState,
    epoch_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported protocol schema")
    if payload.get("contract") != "anytime-valid-simulator-admission-validation-v1":
        raise ValueError("unexpected protocol contract")
    if payload.get("status") != "frozen-before-controlled-execution":
        raise ValueError("protocol is not frozen")
    if payload.get("paper_claim_authorized") is not False:
        raise ValueError("controlled protocol cannot authorize a paper claim")
    validation = payload["controlled_validation"]
    if int(validation["replications"]) < 1000:
        raise ValueError("at least 1000 replications are required")
    if int(validation["horizon"]) < 100:
        raise ValueError("horizon is too short for sequential validation")
    return payload


def config_from_protocol(payload: dict[str, Any]) -> AnytimeAdmissionConfig:
    raw = payload["controller"]
    return AnytimeAdmissionConfig(
        alpha=float(raw["alpha"]),
        beta=float(raw["beta"]),
        loss_cap=float(raw["loss_cap"]),
        gain_margin=float(raw["gain_margin"]),
        harm_margin=float(raw["harm_margin"]),
        allow_reentry=bool(raw["allow_reentry"]),
        lambdas=tuple(float(value) for value in raw["lambdas"]),
    )


def gain_sequence(
    scenario: dict[str, Any],
    *,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    kind = scenario["kind"]
    if kind == "bernoulli_gain":
        positive = float(scenario["positive_gain"])
        negative = float(scenario["negative_gain"])
        probability = float(scenario["positive_probability"])
        return np.where(rng.random(horizon) < probability, positive, negative)
    if kind == "piecewise_bernoulli_gain":
        positive = float(scenario["positive_gain"])
        negative = float(scenario["negative_gain"])
        shift = int(scenario["shift_step"])
        probabilities = np.full(horizon, float(scenario["post_shift_positive_probability"]))
        probabilities[:shift] = float(scenario["pre_shift_positive_probability"])
        return np.where(rng.random(horizon) < probabilities, positive, negative)
    if kind == "adaptive_rademacher_gain":
        low = float(scenario["low_amplitude"])
        high = float(scenario["high_amplitude"])
        values = np.empty(horizon, dtype=np.float64)
        running_sum = 0.0
        for index in range(horizon):
            amplitude = high if running_sum > 0.0 else low
            sign = 1.0 if rng.random() < 0.5 else -1.0
            values[index] = sign * amplitude
            running_sum += values[index]
        return values
    raise ValueError(f"unsupported scenario kind: {kind}")


def running_z_admission(
    gains: np.ndarray,
    *,
    minimum_step: int,
    threshold: float,
) -> int | None:
    for count in range(minimum_step, len(gains) + 1):
        prefix = gains[:count]
        standard_deviation = float(np.std(prefix, ddof=1))
        if standard_deviation <= 0.0:
            continue
        z_value = float(np.mean(prefix)) / (standard_deviation / math.sqrt(count))
        if z_value >= threshold:
            return count
    return None


def fixed_horizon_z_rejects(gains: np.ndarray, *, threshold: float) -> bool:
    standard_deviation = float(np.std(gains, ddof=1))
    if standard_deviation <= 0.0:
        return False
    z_value = float(np.mean(gains)) / (standard_deviation / math.sqrt(len(gains)))
    return z_value >= threshold


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid binomial counts")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def optional_int_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "mean": None, "maximum": None}
    return {
        "count": len(values),
        "median": float(median(values)),
        "mean": float(np.mean(values)),
        "maximum": int(max(values)),
    }


def evaluate_scenario(
    scenario: dict[str, Any],
    *,
    config: AnytimeAdmissionConfig,
    replications: int,
    horizon: int,
    minimum_naive_step: int,
    naive_z: float,
    seed_sequence: np.random.SeedSequence,
) -> dict[str, Any]:
    child_seeds = seed_sequence.spawn(replications)
    admitted = 0
    revoked = 0
    admission_times: list[int] = []
    revocation_delays: list[int] = []
    naive_admitted = 0
    naive_times: list[int] = []
    fixed_horizon_rejected = 0
    exact_fallback_identity_violations = 0
    candidate_exposures = 0
    harmful_candidate_exposures = 0
    post_shift_candidate_exposures = 0
    post_shift_harmful_exposures = 0
    cumulative_deployed_regret: list[float] = []
    cumulative_unguarded_regret: list[float] = []
    final_e_values: list[float] = []
    maximum_e_values: list[float] = []
    shift_step = scenario.get("shift_step")

    fallback_token = object()
    candidate_token = object()

    for replication, child_seed in enumerate(child_seeds):
        rng = np.random.default_rng(child_seed)
        gains = gain_sequence(scenario, horizon=horizon, rng=rng)
        controller = AnytimeAdmissionController(
            config,
            candidate_id=f"{scenario['name']}-fixed-candidate",
        )
        admission_time: int | None = None
        revocation_time: int | None = None
        deployed_regret = 0.0
        unguarded_regret = float(np.sum(-gains))
        stream_maximum_e = 1.0

        for step, gain in enumerate(gains, start=1):
            state_before = controller.state
            selected = controller.select(
                fallback=fallback_token,
                candidate=candidate_token,
            )
            if state_before is DeploymentState.FALLBACK and selected is not fallback_token:
                exact_fallback_identity_violations += 1
            if state_before is DeploymentState.CANDIDATE:
                candidate_exposures += 1
                deployed_regret -= float(gain)
                if gain < 0.0:
                    harmful_candidate_exposures += 1
                if shift_step is not None and step > int(shift_step):
                    post_shift_candidate_exposures += 1
                    if gain < 0.0:
                        post_shift_harmful_exposures += 1

            fallback_loss = 0.5
            candidate_loss = fallback_loss - float(gain)
            record = controller.observe(
                candidate_loss=candidate_loss,
                fallback_loss=fallback_loss,
            )
            stream_maximum_e = max(stream_maximum_e, record.e_value)
            if record.event == "admit" and admission_time is None:
                admission_time = step
                admitted += 1
                admission_times.append(step)
            if record.event == "revoke" and revocation_time is None:
                revocation_time = step
                revoked += 1
                if shift_step is not None:
                    revocation_delays.append(max(0, step - int(shift_step)))
                break

        naive_time = running_z_admission(
            gains,
            minimum_step=minimum_naive_step,
            threshold=naive_z,
        )
        if naive_time is not None:
            naive_admitted += 1
            naive_times.append(naive_time)
        if fixed_horizon_z_rejects(gains, threshold=naive_z):
            fixed_horizon_rejected += 1

        cumulative_deployed_regret.append(deployed_regret)
        cumulative_unguarded_regret.append(unguarded_regret)
        final_e_values.append(float(record.e_value))
        maximum_e_values.append(stream_maximum_e)

    return {
        "name": scenario["name"],
        "kind": scenario["kind"],
        "replications": replications,
        "horizon": horizon,
        "anytime": {
            "ever_admitted_count": admitted,
            "ever_admitted_fraction": admitted / replications,
            "ever_admitted_wilson_95_interval": wilson_interval(admitted, replications),
            "admission_time": optional_int_summary(admission_times),
            "ever_revoked_count": revoked,
            "ever_revoked_fraction": revoked / replications,
            "revocation_delay_after_shift": optional_int_summary(revocation_delays),
            "mean_final_e_value": float(np.mean(final_e_values)),
            "median_maximum_e_value": float(np.median(maximum_e_values)),
        },
        "naive_repeated_z": {
            "ever_admitted_count": naive_admitted,
            "ever_admitted_fraction": naive_admitted / replications,
            "ever_admitted_wilson_95_interval": wilson_interval(
                naive_admitted,
                replications,
            ),
            "admission_time": optional_int_summary(naive_times),
        },
        "fixed_horizon_z": {
            "rejected_count": fixed_horizon_rejected,
            "rejected_fraction": fixed_horizon_rejected / replications,
            "wilson_95_interval": wilson_interval(
                fixed_horizon_rejected,
                replications,
            ),
        },
        "deployment": {
            "candidate_exposures": candidate_exposures,
            "harmful_candidate_exposures": harmful_candidate_exposures,
            "post_shift_candidate_exposures": post_shift_candidate_exposures,
            "post_shift_harmful_exposures": post_shift_harmful_exposures,
            "mean_cumulative_clipped_regret": float(
                np.mean(cumulative_deployed_regret)
            ),
            "mean_unguarded_cumulative_clipped_regret": float(
                np.mean(cumulative_unguarded_regret)
            ),
            "exact_fallback_identity_violations": exact_fallback_identity_violations,
        },
    }


def write_csv(path: Path, scenarios: list[dict[str, Any]]) -> None:
    rows = []
    for result in scenarios:
        rows.append(
            {
                "scenario": result["name"],
                "replications": result["replications"],
                "anytime_admission_fraction": result["anytime"][
                    "ever_admitted_fraction"
                ],
                "naive_repeated_z_admission_fraction": result[
                    "naive_repeated_z"
                ]["ever_admitted_fraction"],
                "fixed_horizon_z_rejection_fraction": result["fixed_horizon_z"][
                    "rejected_fraction"
                ],
                "anytime_revocation_fraction": result["anytime"][
                    "ever_revoked_fraction"
                ],
                "median_admission_time": result["anytime"]["admission_time"][
                    "median"
                ],
                "median_revocation_delay": result["anytime"][
                    "revocation_delay_after_shift"
                ]["median"],
                "mean_deployed_regret": result["deployment"][
                    "mean_cumulative_clipped_regret"
                ],
                "mean_unguarded_regret": result["deployment"][
                    "mean_unguarded_cumulative_clipped_regret"
                ],
                "exact_fallback_identity_violations": result["deployment"][
                    "exact_fallback_identity_violations"
                ],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Anytime-valid simulator admission v1",
        "",
        "## Controlled optional-stopping validation",
        "",
        "| Scenario | Anytime admit | Naive repeated z | Fixed-horizon z | Revoke | Median admit | Median revoke delay |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["scenarios"]:
        lines.append(
            "| {name} | {anytime:.3%} | {naive:.3%} | {fixed:.3%} | "
            "{revoke:.3%} | {admit_time} | {revoke_delay} |".format(
                name=result["name"],
                anytime=result["anytime"]["ever_admitted_fraction"],
                naive=result["naive_repeated_z"]["ever_admitted_fraction"],
                fixed=result["fixed_horizon_z"]["rejected_fraction"],
                revoke=result["anytime"]["ever_revoked_fraction"],
                admit_time=result["anytime"]["admission_time"]["median"],
                revoke_delay=result["anytime"]["revocation_delay_after_shift"][
                    "median"
                ],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Mechanism gate: **{payload['decision']}**",
            f"- Exact fallback identity violations: **{payload['exact_fallback_identity_violations']}**",
            f"- First-epoch alpha: **{payload['budget_audit']['first_epoch_alpha']:.6g}**",
            f"- Summed alpha over 100,000 epochs: **{payload['budget_audit']['alpha_sum_100000_epochs']:.6g}**",
            "",
            "## Guarantee boundary",
            "",
            payload["claim_boundary"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_protocol(protocol_path)
    config = config_from_protocol(protocol)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    validation = protocol["controlled_validation"]
    replications = int(validation["replications"])
    horizon = int(validation["horizon"])
    minimum_naive_step = int(validation["minimum_naive_monitoring_step"])
    naive_z = float(validation["naive_one_sided_z"])
    master = np.random.SeedSequence(int(validation["master_seed"]))
    scenario_seeds = master.spawn(len(validation["scenarios"]))

    results = [
        evaluate_scenario(
            scenario,
            config=config,
            replications=replications,
            horizon=horizon,
            minimum_naive_step=minimum_naive_step,
            naive_z=naive_z,
            seed_sequence=seed,
        )
        for scenario, seed in zip(validation["scenarios"], scenario_seeds, strict=True)
    ]
    by_name = {result["name"]: result for result in results}
    null_names = (
        "iid_zero_mean_null",
        "adaptive_amplitude_zero_mean_null",
    )
    identity_violations = sum(
        result["deployment"]["exact_fallback_identity_violations"]
        for result in results
    )
    gates = {
        "zero_fallback_identity_violations": identity_violations == 0,
        "anytime_no_more_frequent_than_naive_on_zero_mean_nulls": all(
            by_name[name]["anytime"]["ever_admitted_fraction"]
            <= by_name[name]["naive_repeated_z"]["ever_admitted_fraction"]
            for name in null_names
        ),
        "beneficial_small_has_power": by_name["beneficial_small"]["anytime"][
            "ever_admitted_count"
        ]
        > 0,
        "shift_admits": by_name["beneficial_large_then_harmful_shift"]["anytime"][
            "ever_admitted_count"
        ]
        > 0,
        "shift_revokes": by_name["beneficial_large_then_harmful_shift"]["anytime"][
            "ever_revoked_count"
        ]
        > 0,
    }
    alpha_sum = sum(
        epoch_budget(config.alpha, epoch, allow_reentry=True)
        for epoch in range(1, 100_001)
    )
    gates["summable_alpha_budget"] = alpha_sum <= config.alpha
    decision = "controlled-mechanism-pass" if all(gates.values()) else "controlled-mechanism-fail"

    payload = {
        "schema_version": 1,
        "contract": "anytime-valid-simulator-admission-controlled-result-v1",
        "status": "complete",
        "decision": decision,
        "protocol": str(protocol_path),
        "controller": protocol["controller"],
        "scenarios": results,
        "gates": gates,
        "exact_fallback_identity_violations": identity_violations,
        "budget_audit": {
            "first_epoch_alpha": epoch_budget(
                config.alpha,
                1,
                allow_reentry=True,
            ),
            "second_epoch_alpha": epoch_budget(
                config.alpha,
                2,
                allow_reentry=True,
            ),
            "alpha_sum_100000_epochs": alpha_sum,
            "total_alpha": config.alpha,
        },
        "claim_boundary": protocol["promotion_gate"]["claim_boundary"],
        "paper_claim_authorized": False,
    }
    (output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_root / "scenario-summary.csv", results)
    write_report(output_root / "report.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
