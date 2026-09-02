#!/usr/bin/env python3
"""Generate the support-robust act--sense--fallback mechanism result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.support_robust_act_sense_fallback_certificate_v1 import (
    SUPPORT_ROBUST_ACT_SENSE_FALLBACK_CLAIM_BOUNDARY,
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


def _problem(*, resolved_left: bool) -> tuple[object, ...]:
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
    return (
        np.full(count, 1.0 / count),
        np.array([1.0]),
        np.zeros(count, dtype=np.int64),
        losses,
        outcomes,
        [0.0, 0.16],
    )


def evaluate(*, resolved_left: bool, epsilon: float) -> dict[str, Any]:
    certificate = support_robust_act_sense_fallback_certificate(
        *_problem(resolved_left=resolved_left),
        fallback_action_index=2,
        support_miss_probability_upper=epsilon,
        unknown_terminal_loss_lower_by_action=[0.0, 0.0, 1.5],
        unknown_terminal_loss_upper_by_action=[0.8, 0.8, 1.5],
        unknown_probe_loss_lower=[0.0, 0.16],
        unknown_probe_loss_upper=[2.0, 0.16],
        regret_tolerance=0.25,
        probe_names=["quick_tug", "camera"],
    )
    plan = certificate.output_plan
    return {
        "resolved_left": resolved_left,
        "support_miss_probability_upper": epsilon,
        "output_mode": certificate.output_mode,
        "terminal_action_index": (
            certificate.terminal_action()
            if certificate.output_mode != "sense"
            else None
        ),
        "probe_name": plan.probe_name,
        "terminal_action_by_outcome": plan.terminal_action_by_outcome.tolist(),
        "minimax_worst_case_regret": certificate.minimax_worst_case_regret,
        "selected_support_miss_budget": certificate.selected_support_miss_budget,
        "plan_count": certificate.plan_count,
    }


def build() -> dict[str, Any]:
    rows = [
        evaluate(resolved_left=True, epsilon=0.10),
        evaluate(resolved_left=False, epsilon=0.00),
        evaluate(resolved_left=False, epsilon=0.10),
        evaluate(resolved_left=False, epsilon=0.20),
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
        raise RuntimeError(f"phase diagram changed: {observed!r}")
    if abs(rows[0]["minimax_worst_case_regret"] - 0.08) > 1e-12:
        raise RuntimeError("resolved action regret changed")
    if abs(rows[2]["minimax_worst_case_regret"] - 0.24) > 1e-12:
        raise RuntimeError("safer-probe regret changed")
    if abs(rows[3]["minimax_worst_case_regret"] - 0.32) > 1e-12:
        raise RuntimeError("fallback boundary changed")

    unsigned = {
        "artifact_kind": "SupportRobustActSenseFallbackMechanismV1",
        "schema_version": 1,
        "actions": ["pull_left", "pull_right", "hold"],
        "probes": ["quick_tug", "camera"],
        "regret_tolerance": 0.25,
        "phase_diagram": rows,
        "theorem": {
            "represented_gap": "Delta_0(p,b)",
            "unknown_box_gap": "M(p,b)=upper[p]-lower[b]",
            "at_most_epsilon_gap": (
                "Delta_0 + epsilon * max(0, M - Delta_0)"
            ),
            "ambiguity_class": (
                "at-most-epsilon mixture with an axis-aligned unknown plan-loss box"
            ),
        },
        "interpretation": (
            "The certificate acts when the terminal action is already identified, "
            "uses a cheap informative tug under represented support, switches to a "
            "costlier but support-robust camera under moderate misspecification, and "
            "returns the caller-owned hold fallback when no plan meets tolerance."
        ),
        "claim_boundary": SUPPORT_ROBUST_ACT_SENSE_FALLBACK_CLAIM_BOUNDARY,
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
