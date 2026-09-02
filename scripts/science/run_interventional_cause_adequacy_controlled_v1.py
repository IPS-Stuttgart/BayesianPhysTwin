#!/usr/bin/env python3
"""Controlled falsification for incomplete interventional cause families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.interventional_cause_adequacy_v1 import (
    CauseFamilyAdequacyStatus,
    InterventionalCauseFamilyAdequacyV1,
)

SHA = "a" * 64
CAUSES = (
    "observation_bias",
    "physical_parameter",
    "physical_state",
    "realized_intervention",
    "source_local_discrepancy",
)
UNKNOWN = "unregistered_hysteresis"


def canonical_id(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def signatures() -> tuple[dict[str, np.ndarray], np.ndarray]:
    factual = np.asarray([1.0, -0.5, 0.25])
    codes = {
        "observation_bias": ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        "physical_parameter": ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        "physical_state": ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        "realized_intervention": ([1.0, 1.0, 0.0], [0.0, 1.0, 1.0]),
        "source_local_discrepancy": ([1.0, 0.0, 1.0], [1.0, -1.0, 0.0]),
    }
    result: dict[str, np.ndarray] = {}
    for cause in CAUSES:
        action, contact = codes[cause]
        view = np.asarray(action) - 0.5 * np.asarray(contact)
        column = np.concatenate(
            [
                factual,
                np.asarray(action),
                np.asarray(contact),
                view,
            ]
        )
        result[cause] = (column / np.linalg.norm(column))[:, None]

    total = np.hstack([result[cause] for cause in CAUSES])
    raw_unknown = np.asarray(
        [0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, -1.0, 0.5, 0.0, 1.0, 1.0]
    )
    unknown = raw_unknown - total @ np.linalg.lstsq(total, raw_unknown, rcond=None)[0]
    unknown /= np.linalg.norm(unknown)
    return result, unknown


def classify(
    residual: np.ndarray,
    cause_signatures: dict[str, np.ndarray],
    *,
    noise_radius: float,
    permit_unknown: bool,
) -> tuple[str, InterventionalCauseFamilyAdequacyV1]:
    certificate = InterventionalCauseFamilyAdequacyV1(
        residual_id=SHA,
        intervention_roster_id=SHA,
        whitening_id=SHA,
        cause_signature_ids={cause: SHA for cause in CAUSES},
        cause_signatures=cause_signatures,
        whitened_residual=residual,
        noise_radius=noise_radius,
    )
    if (
        permit_unknown
        and certificate.status is CauseFamilyAdequacyStatus.UNMODELED_CAUSE
    ):
        return UNKNOWN, certificate
    coefficients = certificate.minimum_norm_coefficients
    selected = CAUSES[int(np.argmax(np.abs(coefficients)))]
    return selected, certificate


def run(*, trials: int, seed: int) -> dict[str, Any]:
    if trials < 100:
        raise ValueError("trials must be at least 100")
    rng = np.random.default_rng(seed)
    cause_signatures, unknown_signature = signatures()
    labels = (*CAUSES, UNKNOWN)
    sigma = 0.002
    noise_radius = 0.015

    factual_signatures = {
        cause: values[:3] for cause, values in cause_signatures.items()
    }
    broken_signatures = {
        cause: np.concatenate(
            [values[:3], values[6:9], values[3:6], values[9:12]]
        )[:, None]
        for cause, matrix in cause_signatures.items()
        for values in [matrix[:, 0]]
    }

    counts = {
        "factual_correct": 0,
        "forced_correct": 0,
        "adequacy_correct": 0,
        "broken_correct": 0,
        "registered_trials": 0,
        "unknown_trials": 0,
        "forced_unknown_physical": 0,
        "adequacy_unknown_physical": 0,
        "adequacy_registered_rejected": 0,
        "adequacy_unknown_detected": 0,
    }
    unexplained_registered: list[float] = []
    unexplained_unknown: list[float] = []

    for index in range(trials):
        truth = labels[index % len(labels)]
        amplitude = rng.uniform(0.8, 1.2) * rng.choice((-1.0, 1.0))
        signature = (
            unknown_signature[:, None]
            if truth == UNKNOWN
            else cause_signatures[truth]
        )
        residual = amplitude * signature[:, 0] + rng.normal(0.0, sigma, 12)

        factual_residual = residual[:3]
        factual_columns = np.hstack(
            [factual_signatures[cause] for cause in CAUSES]
        )
        factual_coefficients = np.linalg.lstsq(
            factual_columns,
            factual_residual,
            rcond=None,
        )[0]
        factual_label = CAUSES[int(np.argmax(np.abs(factual_coefficients)))]
        counts["factual_correct"] += int(factual_label == truth)

        forced_label, forced_certificate = classify(
            residual,
            cause_signatures,
            noise_radius=noise_radius,
            permit_unknown=False,
        )
        adequacy_label, adequacy_certificate = classify(
            residual,
            cause_signatures,
            noise_radius=noise_radius,
            permit_unknown=True,
        )
        broken_label, _ = classify(
            residual,
            broken_signatures,
            noise_radius=noise_radius,
            permit_unknown=True,
        )
        counts["forced_correct"] += int(forced_label == truth)
        counts["adequacy_correct"] += int(adequacy_label == truth)
        counts["broken_correct"] += int(broken_label == truth)

        if truth == UNKNOWN:
            counts["unknown_trials"] += 1
            counts["forced_unknown_physical"] += int(forced_label != UNKNOWN)
            counts["adequacy_unknown_physical"] += int(adequacy_label != UNKNOWN)
            counts["adequacy_unknown_detected"] += int(adequacy_label == UNKNOWN)
            unexplained_unknown.append(adequacy_certificate.unexplained_norm)
        else:
            counts["registered_trials"] += 1
            counts["adequacy_registered_rejected"] += int(
                adequacy_certificate.status
                is CauseFamilyAdequacyStatus.UNMODELED_CAUSE
            )
            unexplained_registered.append(adequacy_certificate.unexplained_norm)
        if forced_certificate.status is CauseFamilyAdequacyStatus.NO_DETECTABLE_ERROR:
            raise RuntimeError("controlled signal unexpectedly below the noise radius")

    metrics = {
        "factual_only_accuracy": counts["factual_correct"] / trials,
        "forced_registered_family_accuracy": counts["forced_correct"] / trials,
        "adequacy_gated_accuracy": counts["adequacy_correct"] / trials,
        "broken_relation_accuracy": counts["broken_correct"] / trials,
        "unknown_detection_recall": (
            counts["adequacy_unknown_detected"] / counts["unknown_trials"]
        ),
        "forced_unknown_false_physical_promotion": (
            counts["forced_unknown_physical"] / counts["unknown_trials"]
        ),
        "adequacy_unknown_false_physical_promotion": (
            counts["adequacy_unknown_physical"] / counts["unknown_trials"]
        ),
        "registered_false_unknown_rate": (
            counts["adequacy_registered_rejected"] / counts["registered_trials"]
        ),
        "mean_registered_unexplained_norm": float(np.mean(unexplained_registered)),
        "mean_unknown_unexplained_norm": float(np.mean(unexplained_unknown)),
    }
    checks = {
        "same_action_is_nonidentifying": metrics["factual_only_accuracy"] <= 0.20,
        "adequacy_gated_accuracy_at_least_99_percent": (
            metrics["adequacy_gated_accuracy"] >= 0.99
        ),
        "unknown_detection_at_least_99_percent": (
            metrics["unknown_detection_recall"] >= 0.99
        ),
        "forced_family_false_promotes_unknown": (
            metrics["forced_unknown_false_physical_promotion"] >= 0.99
        ),
        "adequacy_gate_controls_false_physical_promotion": (
            metrics["adequacy_unknown_false_physical_promotion"] <= 0.01
        ),
        "registered_causes_are_not_rejected": (
            metrics["registered_false_unknown_rate"] <= 0.01
        ),
        "broken_relation_loses_at_least_20_points": (
            metrics["broken_relation_accuracy"]
            <= metrics["adequacy_gated_accuracy"] - 0.20
        ),
    }
    result: dict[str, Any] = {
        "schema": "bayesian-phystwin.interventional-cause-adequacy-controlled.v1",
        "trials": trials,
        "seed": seed,
        "registered_causes": list(CAUSES),
        "unregistered_cause": UNKNOWN,
        "noise_sigma": sigma,
        "noise_radius": noise_radius,
        "metrics": metrics,
        "checks": checks,
        "decision": (
            "cause-family-adequacy-and-none-of-the-above-supported"
            if all(checks.values())
            else "cause-family-adequacy-not-supported"
        ),
        "claim_boundary": (
            "Controlled linear-Gaussian mechanism evidence only. This result does "
            "not establish that the registered natural cause family is complete, "
            "identify a real physical cause, validate real-data transfer, or "
            "authorize deployment."
        ),
    }
    result["result_id"] = canonical_id(result)
    return result


def report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    lines = [
        "# Interventional cause-family adequacy controlled result",
        "",
        f"**Decision:** `{result['decision']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        lines.append(f"| `{key}` | {value:.6f} |")
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
        json.dumps(
            {"decision": result["decision"], "result_id": result["result_id"]}
        )
    )
    return 0 if all(result["checks"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
