#!/usr/bin/env python3
"""Post-hoc observable tail/inlier gates for the frozen Deform360 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin import deform360_corruption_diagnostic as base
from bayesian_phystwin.phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    initialize_recursive_rbf_belief,
    update_recursive_rbf_belief,
)


GATE_RULES = ("p90", "inlier13")


def run_tail_filter(
    prior_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    center_ids: np.ndarray,
    update_observation_m: np.ndarray,
    update_available: np.ndarray,
    history_observation_m: np.ndarray,
    history_available: np.ndarray,
    *,
    gate_rule: str,
) -> dict[str, Any]:
    config = RecursiveRbfBeliefConfig(local_blend=0.25)
    belief = initialize_recursive_rbf_belief(
        center_ids,
        prior_m[0, center_ids],
        prior_m[0],
        config=config,
    )
    trajectory = prior_m.copy()
    history_values = base.frozen_history_dispersion(
        prior_m,
        history_observation_m,
        history_available,
        center_ids,
    )
    threshold, history_p95 = base.risk_threshold(history_values)
    records: list[dict[str, Any]] = []
    exact_fallback_count = 0

    for update_index, update in enumerate(base.UPDATE_FRAMES):
        stop = (
            base.UPDATE_FRAMES[update_index + 1] if update_index < 2 else len(prior_m)
        )
        available = update_available[update].copy()
        count = int(np.sum(available))
        residual = np.full((base.CENTER_COUNT, 3), np.nan, dtype=float)
        residual[available] = (
            update_observation_m[update, available]
            - prior_m[update, center_ids[available]]
        )
        radial = (
            np.empty(0, dtype=float)
            if count == 0
            else base.radial_residuals(residual[available])
        )
        median = None if count == 0 else float(np.median(radial))
        p90 = None if count == 0 else float(np.quantile(radial, 0.90))
        inlier_count = int(np.sum(radial <= threshold))
        if gate_rule == "p90":
            coherent = p90 is not None and p90 <= threshold
        elif gate_rule == "inlier13":
            coherent = inlier_count >= 13
        else:
            raise ValueError(gate_rule)
        accepted = count >= base.MINIMUM_UPDATE_CENTER_COUNT and coherent

        if accepted:
            belief, _ = update_recursive_rbf_belief(
                belief,
                update,
                prior_m[update, center_ids],
                residual,
                available,
                config=config,
            )
            for frame in range(update + 1, stop):
                correction = decode_recursive_rbf_belief(
                    belief,
                    prior_m[update],
                    forecast_frames=frame - update,
                    config=config,
                ).mean_m
                trajectory[frame] = (prior_m[frame] + correction).astype(
                    prior_m.dtype, copy=False
                )
        else:
            if not np.array_equal(
                trajectory[update + 1 : stop], prior_m[update + 1 : stop]
            ):
                raise AssertionError("tail-gate rejection violated exact fallback")
            exact_fallback_count += 1
        records.append(
            {
                "frame": update,
                "available_center_count": count,
                "median_radial_residual_m": median,
                "p90_radial_residual_m": p90,
                "inlier_count_under_frozen_threshold": inlier_count,
                "frozen_history_p95_dispersion_m": history_p95,
                "dispersion_threshold_m": threshold,
                "accepted": bool(accepted),
                "decision": "accepted" if accepted else "tail_or_support_exact_prior",
            }
        )

    score = base.score_hidden(trajectory, target_m, visibility, validity, center_ids)
    return {
        "scores": {
            "recursive_rbf_ungated": score,
            "recursive_rbf_risk_limited": score,
        },
        "updates": records,
        "rejected_exact_fallback_count": exact_fallback_count,
    }


def run(root: Path, *, include_full_stream: bool) -> dict[str, Any]:
    case_names = [
        f"{object_id}-ep{episode_id:04d}"
        for object_id, episodes in base.EXPECTED_SOURCE_EPISODES.items()
        for episode_id in episodes
    ]
    cases = [base.load_case(root / name) for name in case_names]
    baselines = {
        case["case"]: base.score_hidden(
            case["prior"],
            case["target"],
            case["visibility"],
            case["validity"],
            case["centers"],
        )
        for case in cases
    }
    groups = {case["case"]: case["object_id"] for case in cases}
    history_modes = (
        ("scheduled_only", "full_stream")
        if include_full_stream
        else ("scheduled_only",)
    )
    records: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    aggregates: dict[str, Any] = {}
    for history_mode in history_modes:
        records[history_mode] = {}
        aggregates[history_mode] = {}
        for gate_rule in GATE_RULES:
            records[history_mode][gate_rule] = {}
            aggregates[history_mode][gate_rule] = {}
            for condition in base.CONDITIONS:
                if history_mode == "full_stream" and condition == "clean":
                    continue
                condition_records: list[dict[str, Any]] = []
                seeds = (0,) if condition == "clean" else base.SEEDS
                for case in cases:
                    clean_obs, clean_available, _ = base.corrupt_center_stream(
                        case["target"],
                        case["visibility"],
                        case["validity"],
                        case["centers"],
                        condition="clean",
                        seed=0,
                        case_name=case["case"],
                    )
                    for seed in seeds:
                        corrupt_obs, corrupt_available, corruption = (
                            base.corrupt_center_stream(
                                case["target"],
                                case["visibility"],
                                case["validity"],
                                case["centers"],
                                condition=condition,
                                seed=seed,
                                case_name=case["case"],
                            )
                        )
                        result = run_tail_filter(
                            case["prior"],
                            case["target"],
                            case["visibility"],
                            case["validity"],
                            case["centers"],
                            corrupt_obs,
                            corrupt_available,
                            clean_obs
                            if history_mode == "scheduled_only"
                            else corrupt_obs,
                            clean_available
                            if history_mode == "scheduled_only"
                            else corrupt_available,
                            gate_rule=gate_rule,
                        )
                        condition_records.append(
                            {
                                "case": case["case"],
                                "object_id": case["object_id"],
                                "seed": seed,
                                "condition": condition,
                                "history_mode": history_mode,
                                "gate_rule": gate_rule,
                                "corruption": corruption,
                                **result,
                            }
                        )
                records[history_mode][gate_rule][condition] = condition_records
                aggregates[history_mode][gate_rule][condition] = base.aggregate_setting(
                    condition_records, baselines, groups
                )["recursive_rbf_risk_limited"]
                aggregates[history_mode][gate_rule][condition]["coverage"] = (
                    base.aggregate_setting(condition_records, baselines, groups)[
                        "coverage"
                    ]
                )
    return {
        "schema_version": 1,
        "protocol": {
            "gate_rules": {
                "p90": "q9 and current radial-residual p90 <= frozen threshold",
                "inlier13": "q9 and at least 13/16 current radial residuals <= frozen threshold",
            },
            "threshold": (
                "max(10 mm, 1.5 * p95 of per-frame median dispersion in [0,19))"
            ),
            "routing_inputs": "only physical prior plus measurements available by update",
            "shared_outcome_data_edited": False,
        },
        "physical_prior_by_case": baselines,
        "aggregates": aggregates,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-full-stream", action="store_true")
    args = parser.parse_args()
    result = run(args.root.resolve(), include_full_stream=args.include_full_stream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
