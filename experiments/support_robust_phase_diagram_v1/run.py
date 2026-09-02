"""Controlled exact phase diagram for support-robust active physical twins."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.act_sense_fallback_certificate_v1 import (
    act_sense_fallback_certificate,
)
from bayesian_phystwin.support_robust_phase_diagram_v1 import (
    SUPPORT_ROBUST_PHASE_DIAGRAM_CLAIM_BOUNDARY,
    support_robust_phase_diagram,
)

HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "result.json"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    side = np.asarray([0, 0, 1, 1], dtype=np.int64)
    prior = np.full(4, 0.25, dtype=np.float64)
    quotient = np.asarray([1.0], dtype=np.float64)
    classes = np.zeros(4, dtype=np.int64)
    terminal_losses = np.empty((4, 3), dtype=np.float64)
    terminal_losses[:, 0] = np.where(side == 0, 0.0, 4.0)
    terminal_losses[:, 1] = np.where(side == 1, 0.0, 4.0)
    terminal_losses[:, 2] = 1.5
    probe_outcomes = np.vstack((side, side))
    probe_costs = np.asarray([0.0, 0.1], dtype=np.float64)
    probe_names = ("quick_tug", "camera")
    return (
        prior,
        quotient,
        classes,
        terminal_losses,
        probe_outcomes,
        probe_costs,
        probe_names,
    )


def _plan_box() -> tuple[np.ndarray, np.ndarray, int, int]:
    (
        prior,
        quotient,
        classes,
        terminal_losses,
        probe_outcomes,
        probe_costs,
        probe_names,
    ) = _problem()
    certificate = act_sense_fallback_certificate(
        prior,
        quotient,
        classes,
        terminal_losses,
        probe_outcomes,
        probe_costs,
        fallback_action_index=2,
        regret_tolerance=0.25,
        probe_names=probe_names,
    )
    quick = next(
        index
        for index, plan in enumerate(certificate.plans)
        if plan.probe_name == "quick_tug"
        and tuple(plan.terminal_action_by_outcome.tolist()) == (0, 1)
    )
    camera = next(
        index
        for index, plan in enumerate(certificate.plans)
        if plan.probe_name == "camera"
        and tuple(plan.terminal_action_by_outcome.tolist()) == (0, 1)
    )
    lower = np.zeros(certificate.plan_count, dtype=np.float64)
    upper = np.full(certificate.plan_count, 4.0, dtype=np.float64)
    # The quick mechanical probe is exact on represented hypotheses but may be
    # strongly misleading off support. The camera pays a represented cost but
    # has the tighter declared unknown-physics plan-loss bound.
    upper[quick] = 3.0
    upper[camera] = 1.0
    return lower, upper, quick, camera


def _compress_output_regions(diagram) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for interval in diagram.interval_decisions:
        decision = interval.decision
        identity = (
            decision.output_plan_index,
            decision.output_mode,
            decision.selected_probe_name,
            decision.has_admissible_plan,
        )
        if regions and regions[-1]["identity"] == identity:
            regions[-1]["support_miss_right"] = interval.support_miss_right
            continue
        regions.append(
            {
                "identity": identity,
                "support_miss_left": interval.support_miss_left,
                "support_miss_right": interval.support_miss_right,
                "open_interval": True,
                "output_plan_index": decision.output_plan_index,
                "output_mode": decision.output_mode,
                "selected_probe_name": decision.selected_probe_name,
                "has_admissible_plan": decision.has_admissible_plan,
            }
        )
    for item in regions:
        item.pop("identity")
    return regions


def build_result() -> dict[str, Any]:
    (
        prior,
        quotient,
        classes,
        terminal_losses,
        probe_outcomes,
        probe_costs,
        probe_names,
    ) = _problem()
    lower, upper, quick, camera = _plan_box()
    diagram = support_robust_phase_diagram(
        prior,
        quotient,
        classes,
        terminal_losses,
        probe_outcomes,
        probe_costs,
        fallback_action_index=2,
        regret_tolerance=0.25,
        maximum_support_miss_probability=0.25,
        probe_names=probe_names,
        unknown_plan_loss_lower=lower,
        unknown_plan_loss_upper=upper,
    )
    probe_switch = 0.1 / 2.1
    fallback_switch = (0.25 - 0.1) / 0.9
    if diagram.decision_at(0.0).selected_probe_name != "quick_tug":
        raise RuntimeError("zero-miss phase no longer selects the quick tug")
    if diagram.decision_at(0.1).selected_probe_name != "camera":
        raise RuntimeError("moderate-miss phase no longer selects the camera")
    if diagram.decision_at(0.2).output_mode != "fallback":
        raise RuntimeError("large-miss phase no longer returns the fallback")
    if np.min(np.abs(diagram.breakpoints - probe_switch)) > 1e-10:
        raise RuntimeError("exact probe-switch breakpoint is missing")
    if np.min(np.abs(diagram.breakpoints - fallback_switch)) > 1e-10:
        raise RuntimeError("exact fallback breakpoint is missing")

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin.support-robust-phase-diagram-result.v1",
        "schema_version": 1,
        "study_id": "support-robust-act-sense-fallback-phase-diagram-v1",
        "terminal_actions": ["pull_left", "pull_right", "hold"],
        "probes": list(probe_names),
        "hypothesis_count": int(prior.size),
        "plan_count": diagram.plan_count,
        "regret_tolerance": diagram.regret_tolerance,
        "maximum_support_miss_probability": (diagram.maximum_support_miss_probability),
        "quick_tug_plan_index": quick,
        "camera_plan_index": camera,
        "quick_tug_maximum_admissible_support_miss": float(
            diagram.plan_maximum_admissible_support_miss[quick]
        ),
        "camera_maximum_admissible_support_miss": float(
            diagram.plan_maximum_admissible_support_miss[camera]
        ),
        "maximum_any_plan_admissible_support_miss": (
            diagram.maximum_any_plan_admissible_support_miss
        ),
        "exact_probe_switch_support_miss": probe_switch,
        "exact_fallback_switch_support_miss": fallback_switch,
        "selected_examples": {
            "epsilon_0": diagram.decision_at(0.0).summary(),
            "epsilon_0p1": diagram.decision_at(0.1).summary(),
            "epsilon_0p2": diagram.decision_at(0.2).summary(),
        },
        "all_breakpoints": diagram.breakpoints.tolist(),
        "point_decisions": [decision.summary() for decision in diagram.point_decisions],
        "open_interval_decisions": [
            interval.summary() for interval in diagram.interval_decisions
        ],
        "compressed_output_regions": _compress_output_regions(diagram),
        "interpretation": {
            "zero_miss": (
                "The represented support makes the zero-cost quick tug exactly "
                "decision-identifying."
            ),
            "moderate_miss": (
                "The camera overtakes the quick tug because its declared "
                "unknown-physics plan-loss box is tighter despite positive cost."
            ),
            "large_miss": (
                "Every learned direct or sensing plan exceeds the registered "
                "regret tolerance, so the caller-owned hold action is returned."
            ),
        },
        "claim_boundary": SUPPORT_ROBUST_PHASE_DIAGRAM_CLAIM_BOUNDARY,
    }
    unsigned = dict(result)
    result["result_id"] = _digest(unsigned)
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build_result()
    payload = _canonical(result)
    if args.write:
        RESULT_PATH.write_text(payload, encoding="utf-8")
    else:
        if not RESULT_PATH.exists():
            raise SystemExit(f"missing checked result: {RESULT_PATH}")
        if RESULT_PATH.read_text(encoding="utf-8") != payload:
            raise SystemExit("checked support-robust phase result is stale")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
