#!/usr/bin/env python3
"""Controlled act--sense--fallback mechanism experiment.

The benchmark separates state information from decision information.  A hidden
rope tether side determines whether pulling left or right is correct.  Friction
and texture remain physically latent nuisance variables.  A three-outcome
texture camera removes more state entropy than a two-outcome side tug, but only
the tug resolves the action ambiguity.

This is a controlled finite-hypothesis mechanism test.  It is not real robot
execution, provider validation, target-domain calibration, or a safety result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.act_sense_fallback_certificate_v1 import (
    ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY,
    ActSenseFallbackCertificateV1,
    act_sense_fallback_certificate,
)

ACTION_NAMES = ("pull_left", "pull_right", "hold")
PROBE_NAMES = ("tug_side", "camera_texture")
PROBE_COSTS = np.array([0.2, 0.05], dtype=np.float64)


def _canonical_id(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    side = np.repeat(np.array([0, 1], dtype=np.int64), 6)
    friction = np.tile(np.repeat(np.array([0, 1], dtype=np.int64), 3), 2)
    texture = np.tile(np.arange(3, dtype=np.int64), 4)

    losses = np.empty((12, 3), dtype=np.float64)
    losses[:, 0] = np.where(side == 0, 0.0, 4.0)
    losses[:, 1] = np.where(side == 1, 0.0, 4.0)
    losses[:, 2] = 1.5
    return side, friction, texture, losses


def _entropy_gain(outcomes: np.ndarray) -> float:
    counts = np.bincount(outcomes)
    probabilities = counts[counts > 0] / outcomes.size
    return float(-np.sum(probabilities * np.log(probabilities)))


def _best_probe_plan_regret(
    result: ActSenseFallbackCertificateV1,
    *,
    probe_name: str,
) -> float:
    candidates = [
        float(result.plan_certificate.worst_case_regret[index])
        for index, plan in enumerate(result.plans)
        if plan.probe_name == probe_name
    ]
    if not candidates:
        raise RuntimeError(f"probe has no enumerated plans: {probe_name}")
    return min(candidates)


def _case_record(
    *,
    name: str,
    mask: np.ndarray,
    tolerance: float,
    side: np.ndarray,
    texture: np.ndarray,
    losses: np.ndarray,
) -> tuple[dict[str, Any], ActSenseFallbackCertificateV1]:
    count = int(np.count_nonzero(mask))
    result = act_sense_fallback_certificate(
        np.full(count, 1.0 / count),
        [1.0],
        np.zeros(count, dtype=np.int64),
        losses[mask],
        np.vstack([side[mask], texture[mask]]),
        PROBE_COSTS,
        fallback_action_index=2,
        regret_tolerance=tolerance,
        probe_names=PROBE_NAMES,
    )
    record: dict[str, Any] = {
        "name": name,
        "hypothesis_count": count,
        "state_entropy_nats": math.log(count),
        "state_identified": count == 1,
        "regret_tolerance": tolerance,
        "minimax_worst_case_regret": (
            result.plan_certificate.minimax_worst_case_regret
        ),
        "output_mode": result.output_mode,
        "used_fallback": result.used_fallback,
        "output_plan_index": result.output_plan_index,
        "selected_probe": result.output_plan.probe_name,
        "terminal_action_if_direct": (
            ACTION_NAMES[result.terminal_action()]
            if result.output_mode != "sense"
            else None
        ),
        "terminal_action_by_outcome": (
            [
                ACTION_NAMES[result.terminal_action(outcome)]
                for outcome in range(result.output_plan.outcome_count)
            ]
            if result.output_mode == "sense"
            else []
        ),
        "plan_count": result.plan_count,
        "direct_plan_count": result.direct_plan_count,
        "sensing_plan_count": result.sensing_plan_count,
    }
    return record, result


def run_experiment() -> dict[str, Any]:
    side, friction, texture, losses = _problem()
    del friction  # It is a latent physical nuisance but action-independent here.

    act_record, act_result = _case_record(
        name="state_ambiguous_action_identified",
        mask=side == 0,
        tolerance=0.25,
        side=side,
        texture=texture,
        losses=losses,
    )
    sense_record, sense_result = _case_record(
        name="action_ambiguous_probe_resolves",
        mask=np.ones(len(side), dtype=bool),
        tolerance=0.25,
        side=side,
        texture=texture,
        losses=losses,
    )
    fallback_record, fallback_result = _case_record(
        name="action_ambiguous_probe_too_costly_for_tolerance",
        mask=np.ones(len(side), dtype=bool),
        tolerance=0.1,
        side=side,
        texture=texture,
        losses=losses,
    )

    tug_gain = _entropy_gain(side)
    camera_gain = _entropy_gain(texture)
    sense_record["probe_diagnostics"] = {
        "tug_side": {
            "outcome_count": 2,
            "state_entropy_gain_nats": tug_gain,
            "best_contingent_plan_worst_case_regret": _best_probe_plan_regret(
                sense_result,
                probe_name="tug_side",
            ),
        },
        "camera_texture": {
            "outcome_count": 3,
            "state_entropy_gain_nats": camera_gain,
            "best_contingent_plan_worst_case_regret": _best_probe_plan_regret(
                sense_result,
                probe_name="camera_texture",
            ),
        },
        "maximum_entropy_probe": "camera_texture",
        "decision_directed_probe": sense_result.output_plan.probe_name,
    }

    assert act_result.output_mode == "act"
    assert act_result.terminal_action() == 0
    assert act_result.plan_certificate.minimax_worst_case_regret == 0.0
    assert sense_result.output_mode == "sense"
    assert sense_result.output_plan.probe_name == "tug_side"
    assert sense_result.terminal_action(0) == 0
    assert sense_result.terminal_action(1) == 1
    assert math.isclose(
        sense_result.plan_certificate.minimax_worst_case_regret,
        0.2,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert camera_gain > tug_gain
    assert _best_probe_plan_regret(
        sense_result,
        probe_name="camera_texture",
    ) > _best_probe_plan_regret(
        sense_result,
        probe_name="tug_side",
    )
    assert fallback_result.output_mode == "fallback"
    assert fallback_result.terminal_action() == 2
    assert not fallback_result.has_admissible_plan

    result: dict[str, Any] = {
        "schema": "bayesian-phystwin.act-sense-fallback-controlled-result.v1",
        "schema_version": 1,
        "study_id": "occluded-rope-decision-information-v1",
        "hypotheses": {
            "count": 12,
            "latent_factors": {
                "tether_side": ["left", "right"],
                "friction": ["low", "high"],
                "texture": [0, 1, 2],
            },
            "visible_geometry": "identical within each registered case",
        },
        "terminal_actions": list(ACTION_NAMES),
        "probes": [
            {
                "name": "tug_side",
                "cost": float(PROBE_COSTS[0]),
                "outcomes": 2,
                "resolves": "tether_side",
            },
            {
                "name": "camera_texture",
                "cost": float(PROBE_COSTS[1]),
                "outcomes": 3,
                "resolves": "texture only",
            },
        ],
        "cases": [act_record, sense_record, fallback_record],
        "headline": {
            "state_ambiguous_but_act": True,
            "higher_entropy_probe_rejected": True,
            "decision_probe_selected": True,
            "exact_fallback_when_tolerance_fails": True,
        },
        "claim": (
            "A finite physical twin can certify a direct action when unresolved "
            "state is decision-irrelevant, choose a lower-entropy diagnostic probe "
            "when it resolves the action ambiguity, and return an exact caller-owned "
            "fallback when no direct or contingent plan meets the registered regret "
            "tolerance."
        ),
        "claim_boundary": ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY,
        "evidence_class": "controlled finite-hypothesis mechanism",
        "real_robot_execution": False,
        "target_domain_claim": False,
        "safety_claim": False,
    }
    result["result_id"] = _canonical_id(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("result.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    result = run_experiment()
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        if not args.output.is_file():
            raise SystemExit(f"missing checked result: {args.output}")
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"checked result is stale: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"result_id": result["result_id"], "status": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
