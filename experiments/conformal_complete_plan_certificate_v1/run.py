#!/usr/bin/env python3
"""Generate the conformal complete-plan act--sense--fallback result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.conformal_complete_plan_certificate_v1 import (
    CONFORMAL_COMPLETE_PLAN_CERTIFICATE_CLAIM_BOUNDARY,
    conformal_act_sense_fallback_decision,
    scaled_trajectory_conformal_plan_envelope,
    support_robust_plan_width_scales,
)
from bayesian_phystwin.support_robust_act_sense_fallback_certificate_v1 import (
    support_robust_act_sense_fallback_certificate,
)

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "result.json"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _certificate(*, resolved_left: bool):
    side = np.repeat(np.array([0, 1], dtype=np.int64), 2)
    losses = np.empty((4, 3), dtype=np.float64)
    losses[:, 0] = np.where(side == 0, 0.0, 4.0)
    losses[:, 1] = np.where(side == 1, 0.0, 4.0)
    losses[:, 2] = 1.5
    outcomes = np.vstack((side, side))
    if resolved_left:
        keep = side == 0
        losses = losses[keep]
        outcomes = outcomes[:, keep]
    count = losses.shape[0]
    return support_robust_act_sense_fallback_certificate(
        np.full(count, 1.0 / count),
        [1.0],
        np.zeros(count, dtype=np.int64),
        losses,
        outcomes,
        [0.0, 0.16],
        fallback_action_index=2,
        support_miss_probability_upper=0.05,
        unknown_terminal_loss_lower_by_action=[0.0, 0.0, 1.5],
        unknown_terminal_loss_upper_by_action=[0.8, 0.8, 1.5],
        unknown_probe_loss_lower=[0.0, 0.16],
        unknown_probe_loss_upper=[2.0, 0.16],
        regret_tolerance=0.25,
        probe_names=["quick_tug", "camera"],
    )


def evaluate(*, resolved_left: bool, calibration_score: float) -> dict[str, Any]:
    certificate = _certificate(resolved_left=resolved_left)
    scales = support_robust_plan_width_scales(
        certificate,
        minimum_scale=0.1,
    )
    registered = np.broadcast_to(
        certificate.support_robust_worst_case_regret,
        (4, 1, certificate.plan_count),
    ).copy()
    realized = registered + calibration_score * scales[None, None, :]
    envelope = scaled_trajectory_conformal_plan_envelope(
        realized,
        registered,
        scales,
        miscoverage=0.25,
    )
    decision = conformal_act_sense_fallback_decision(certificate, envelope)
    plan = decision.output_plan
    candidate = decision.candidate_plan_index
    return {
        "resolved_left": resolved_left,
        "support_miss_probability_upper": (
            certificate.support_miss_probability_upper
        ),
        "calibration_trajectory_count": envelope.trajectory_count,
        "miscoverage": envelope.miscoverage,
        "finite_sample_coverage_lower_bound": (
            envelope.finite_sample_coverage_lower_bound
        ),
        "calibration_score_quantile": envelope.score_quantile,
        "candidate_plan_index": candidate,
        "candidate_calibrated_regret_upper": (
            None
            if candidate is None
            else float(decision.calibrated_regret_upper_by_plan[candidate])
        ),
        "output_plan_index": decision.output_plan_index,
        "output_mode": decision.output_mode,
        "fallback_reason": decision.fallback_reason,
        "probe_name": plan.probe_name,
        "terminal_action_index": (
            decision.terminal_action()
            if decision.output_mode != "sense"
            else None
        ),
        "terminal_action_by_outcome": plan.terminal_action_by_outcome.tolist(),
        "registered_regret": float(
            decision.registered_worst_case_regret[decision.output_plan_index]
        ),
        "plan_scale": float(scales[decision.output_plan_index]),
        "conformal_inflation": float(
            envelope.inflation_by_plan[decision.output_plan_index]
        ),
        "calibrated_regret_upper": float(
            decision.calibrated_regret_upper_by_plan[decision.output_plan_index]
        ),
    }


def build() -> dict[str, Any]:
    rows = [
        evaluate(resolved_left=True, calibration_score=0.05),
        evaluate(resolved_left=False, calibration_score=0.00),
        evaluate(resolved_left=False, calibration_score=0.05),
        evaluate(resolved_left=False, calibration_score=0.10),
    ]
    expected = [
        ("act", None, 0),
        ("sense", "quick_tug", None),
        ("sense", "camera", None),
        ("fallback", None, 2),
    ]
    observed = [
        (row["output_mode"], row["probe_name"], row["terminal_action_index"])
        for row in rows
    ]
    if observed != expected:
        raise RuntimeError(f"calibrated phase diagram changed: {observed!r}")
    expected_upper = [0.08, 0.14, 0.24, 1.5]
    for row, expected_value in zip(rows, expected_upper, strict=True):
        if abs(row["calibrated_regret_upper"] - expected_value) > 1e-12:
            raise RuntimeError(
                "calibrated regret upper changed: "
                f"{row['calibrated_regret_upper']!r} != {expected_value!r}"
            )
    if abs(rows[3]["candidate_calibrated_regret_upper"] - 0.28) > 1e-12:
        raise RuntimeError("fallback candidate upper changed")

    unsigned = {
        "artifact_kind": "ConformalCompletePlanMechanismV1",
        "schema_version": 1,
        "actions": ["pull_left", "pull_right", "hold"],
        "probes": ["quick_tug", "camera"],
        "support_miss_probability_upper": 0.05,
        "regret_tolerance": 0.25,
        "plan_scale": (
            "registered unknown-plan loss width with a 0.1 positive floor"
        ),
        "calibration": {
            "unit": "complete trajectory",
            "trajectory_count": 4,
            "miscoverage": 0.25,
            "score": (
                "max over registered decisions and plans of positive excess "
                "regret divided by a fixed plan scale"
            ),
            "coverage": (
                "simultaneous over every candidate complete plan and registered "
                "decision on one exchangeable future trajectory"
            ),
        },
        "phase_diagram": rows,
        "theorem": {
            "trajectory_score": (
                "S_j=max_{d,p in C} max(0,R_jdp-B_jdp)/s_p"
            ),
            "calibrated_plan_upper": "B_new,dp + q_(1-alpha) * s_p",
            "selection_rule": (
                "execute the unique calibrated minimax complete plan only when "
                "its upper bound is within tolerance; otherwise reproduce fallback"
            ),
            "post_probe_rule": (
                "apply the terminal action map frozen inside the selected plan; "
                "do not re-optimize after observing the probe"
            ),
        },
        "interpretation": (
            "With a fixed 5% support-miss bound, low calibrated excess selects the "
            "cheap but fragile tug, moderate excess switches to the camera because "
            "its registered plan scale is smaller, and larger excess returns the "
            "exact hold fallback. When tether side is already decision-identified, "
            "the system acts directly."
        ),
        "claim_boundary": CONFORMAL_COMPLETE_PLAN_CERTIFICATE_CLAIM_BOUNDARY,
    }
    output = dict(unsigned)
    output["result_id"] = hashlib.sha256(canonical(unsigned)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        if RESULT.read_text(encoding="utf-8") != text:
            raise SystemExit("result.json is stale")
    else:
        RESULT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
