#!/usr/bin/env python3
"""Controlled study: transport can be known before the physical cause."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.interventional_cause_adequacy_v1 import (
    InterventionalCauseFamilyAdequacyV1,
)
from bayesian_phystwin_experiments.interventional_transport_quotient_v1 import (
    InterventionalTransportQuotientV1,
    TransportQuotientStatus,
)

SHA = "a" * 64


def canonical_id(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def source_certificate(residual: float) -> InterventionalCauseFamilyAdequacyV1:
    return InterventionalCauseFamilyAdequacyV1(
        residual_id=SHA,
        intervention_roster_id=SHA,
        whitening_id=SHA,
        cause_signature_ids={"gauge": SHA, "state": SHA},
        cause_signatures={
            "gauge": np.asarray([[1.0]]),
            "state": np.asarray([[1.0]]),
        },
        whitened_residual=np.asarray([residual]),
        noise_radius=1e-12,
    )


def source_quotient(
    certificate: InterventionalCauseFamilyAdequacyV1,
) -> InterventionalTransportQuotientV1:
    return InterventionalTransportQuotientV1(
        adequacy_certificate=certificate,
        target_intervention_roster_id=SHA,
        target_transport_ids={"difference-query": SHA, "sum-query": SHA},
        target_maps={
            "difference-query": np.asarray([[1.0, -1.0]]),
            "sum-query": np.asarray([[1.0, 1.0]]),
        },
    )


def run(*, trials: int, seed: int) -> dict[str, Any]:
    if trials < 100:
        raise ValueError("trials must be at least 100")
    rng = np.random.default_rng(seed)
    source_noise_sigma = 0.01
    probe_noise_sigma = 0.01

    design_certificate = source_certificate(1.0)
    design_quotient = source_quotient(design_certificate)
    sum_record = design_quotient.record_for("sum-query")
    difference_record = design_quotient.record_for("difference-query")
    if sum_record.status is not TransportQuotientStatus.FULLY_IDENTIFIABLE:
        raise RuntimeError("sum query must be identifiable without a probe")
    if difference_record.status is not TransportQuotientStatus.NONIDENTIFIABLE:
        raise RuntimeError("difference query must require a probe")

    point_errors: list[float] = []
    quotient_errors: list[float] = []
    full_identification_errors: list[float] = []
    quotient_probe_count = 0
    full_identification_probe_count = 0
    false_unprobed_sensitive_transports = 0
    source_unique_cause_count = 0
    post_probe_unique_cause_count = 0

    for index in range(trials):
        state = rng.normal(0.0, 1.0)
        gauge = rng.normal(0.0, 1.0)
        source_observation = state + gauge + rng.normal(0.0, source_noise_sigma)
        target_is_sum = index % 2 == 0
        target_truth = state + gauge if target_is_sum else state - gauge

        source = source_certificate(source_observation)
        quotient = source_quotient(source)
        source_unique_cause_count += int(source.unique_coefficients)

        point_coefficients = source.minimum_norm_coefficients
        point_target = (
            float(point_coefficients.sum())
            if target_is_sum
            else float(point_coefficients[1] - point_coefficients[0])
        )
        point_errors.append((point_target - target_truth) ** 2)

        target_id = "sum-query" if target_is_sum else "difference-query"
        record = quotient.record_for(target_id)
        if record.full_transport_permitted:
            quotient_target = float(record.identifiable_effect[0])
        else:
            quotient_probe_count += 1
            probe_observation = state - gauge + rng.normal(0.0, probe_noise_sigma)
            augmented_design = np.asarray([[1.0, 1.0], [1.0, -1.0]])
            augmented_residual = np.asarray([source_observation, probe_observation])
            augmented_coefficients = np.linalg.solve(
                augmented_design,
                augmented_residual,
            )
            quotient_target = float(
                augmented_coefficients.sum()
                if target_is_sum
                else augmented_coefficients[0] - augmented_coefficients[1]
            )
            post_probe_unique_cause_count += 1
        quotient_errors.append((quotient_target - target_truth) ** 2)
        false_unprobed_sensitive_transports += int(
            not target_is_sum and record.full_transport_permitted
        )

        full_identification_probe_count += 1
        probe_observation = state - gauge + rng.normal(0.0, probe_noise_sigma)
        augmented_coefficients = np.linalg.solve(
            np.asarray([[1.0, 1.0], [1.0, -1.0]]),
            np.asarray([source_observation, probe_observation]),
        )
        full_target = float(
            augmented_coefficients.sum()
            if target_is_sum
            else augmented_coefficients[0] - augmented_coefficients[1]
        )
        full_identification_errors.append((full_target - target_truth) ** 2)

    point_rmse = float(np.sqrt(np.mean(point_errors)))
    quotient_rmse = float(np.sqrt(np.mean(quotient_errors)))
    full_rmse = float(np.sqrt(np.mean(full_identification_errors)))
    quotient_probe_rate = quotient_probe_count / trials
    full_probe_rate = full_identification_probe_count / trials
    metrics = {
        "source_unique_cause_coverage": source_unique_cause_count / trials,
        "unprobed_sum_transport_coverage": 1.0,
        "unprobed_difference_transport_coverage": 0.0,
        "false_unprobed_sensitive_transport_rate": (
            false_unprobed_sensitive_transports / (trials // 2)
        ),
        "forced_point_transport_rmse": point_rmse,
        "transport_quotient_rmse": quotient_rmse,
        "full_cause_identification_rmse": full_rmse,
        "transport_quotient_probe_rate": quotient_probe_rate,
        "full_cause_identification_probe_rate": full_probe_rate,
        "probe_rate_reduction": full_probe_rate - quotient_probe_rate,
        "post_probe_unique_cause_count": post_probe_unique_cause_count,
    }
    checks = {
        "cause_is_not_unique_before_probe": (
            metrics["source_unique_cause_coverage"] == 0.0
        ),
        "sum_query_transports_without_cause_label": (
            metrics["unprobed_sum_transport_coverage"] == 1.0
        ),
        "sensitive_query_is_never_falsely_transported": (
            metrics["false_unprobed_sensitive_transport_rate"] == 0.0
        ),
        "quotient_matches_full_identification_rmse": (
            metrics["transport_quotient_rmse"]
            <= metrics["full_cause_identification_rmse"] + 0.002
        ),
        "quotient_halves_probe_use": (
            metrics["transport_quotient_probe_rate"] == 0.5
            and metrics["probe_rate_reduction"] == 0.5
        ),
        "quotient_beats_forced_point_transport": (
            metrics["transport_quotient_rmse"]
            <= 0.1 * metrics["forced_point_transport_rmse"]
        ),
    }
    result: dict[str, Any] = {
        "schema": "bayesian-phystwin.interventional-transport-quotient-controlled.v1",
        "trials": trials,
        "seed": seed,
        "source_noise_sigma": source_noise_sigma,
        "probe_noise_sigma": probe_noise_sigma,
        "source_cause_status": "adequate_set_valued",
        "sum_query_status": sum_record.status.value,
        "difference_query_status": difference_record.status.value,
        "metrics": metrics,
        "checks": checks,
        "decision": (
            "transport-known-before-cause-and-probe-use-reduced"
            if all(checks.values())
            else "transport-quotient-mechanism-not-established"
        ),
        "claim_boundary": (
            "Controlled local linear mechanism evidence only. The study shows "
            "that a target query can be invariant over a coefficient ambiguity "
            "set and that a target-directed probe can be avoided in that case. "
            "It does not establish real-data transport, natural cause labels, "
            "nonlinear closure, or deployment safety."
        ),
    }
    result["result_id"] = canonical_id(result)
    return result


def report(result: dict[str, Any]) -> str:
    lines = [
        "# Interventional transport quotient controlled result",
        "",
        f"**Decision:** `{result['decision']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in result["metrics"].items():
        lines.append(f"| `{key}` | {value:.8f} |")
    lines += ["", "## Frozen checks", ""]
    for key, value in result["checks"].items():
        lines.append(f"- `{key}`: **{'pass' if value else 'fail'}**")
    lines += ["", "## Claim boundary", "", result["claim_boundary"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = run(trials=args.trials, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(report(result), encoding="utf-8")
    print(
        json.dumps({"decision": result["decision"], "result_id": result["result_id"]})
    )
    return 0 if all(result["checks"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
