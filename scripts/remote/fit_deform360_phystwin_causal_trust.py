#!/usr/bin/env python3
"""Fit source-only causal trust from matched driven and zero-action Warp runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal4d_public.deform360_phystwin_trust import (
    fit_cardinality_normalized_source_causal_trust,
    fit_regime_gated_source_causal_trust,
    fit_source_causal_trust,
    load_official_phystwin_trust_episode,
    write_cardinality_normalized_source_causal_trust_artifact,
    write_regime_gated_source_causal_trust_artifact,
    write_source_causal_trust_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        action="append",
        nargs=5,
        required=True,
        metavar=("ID", "DATA", "DRIVEN_RESULT", "ZERO_RESULT", "SPLIT"),
        help="Repeat for each locked source action.",
    )
    parser.add_argument(
        "--normalize-action-response-by-controller-count",
        action="store_true",
        help=("Fit the post-hoc controller-cardinality-normalized source hypothesis."),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument(
        "--regime",
        action="append",
        nargs=2,
        metavar=("ID", "PREHENSILE_OR_NONPREHENSILE"),
        help=("Repeat for every source to enable the locked contact-regime policy."),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not np.isfinite(args.grid_step) or not 0.0 < args.grid_step <= 1.0:
        raise ValueError("grid step must be in (0, 1]")
    candidate_count = int(round(1.0 / args.grid_step))
    if not np.isclose(candidate_count * args.grid_step, 1.0):
        raise ValueError("grid step must divide the unit interval")
    grid = tuple(np.linspace(0.0, 1.0, candidate_count + 1))
    episodes = [
        load_official_phystwin_trust_episode(
            episode_id,
            data_path,
            driven_result,
            zero_result,
            split_path,
        )
        for episode_id, data_path, driven_result, zero_result, split_path in args.episode
    ]
    if args.regime and args.normalize_action_response_by_controller_count:
        raise ValueError(
            "controller normalization and the frozen v2 regime fit are distinct"
        )
    if args.regime:
        regime_pairs = [tuple(pair) for pair in args.regime]
        regimes = dict(regime_pairs)
        if len(regimes) != len(regime_pairs):
            raise ValueError("contact regime episode is repeated")
        episode_ids = {episode.episode_id for episode in episodes}
        if set(regimes) != episode_ids:
            raise ValueError("contact regimes must cover every source episode")
        result = fit_regime_gated_source_causal_trust(
            episodes,
            regimes,
            action_response_grid=grid,
            autonomous_drift_grid=grid,
        )
        write_regime_gated_source_causal_trust_artifact(args.output, result)
        selected_summary = {
            "policy": result["policy"],
            "prospective_source_gate": result["prospective_source_gate"],
            "pooled_source_tail": result["pooled_leave_one_action_out_tail"],
        }
    elif args.normalize_action_response_by_controller_count:
        result = fit_cardinality_normalized_source_causal_trust(
            episodes,
            action_response_grid=grid,
            autonomous_drift_grid=grid,
        )
        write_cardinality_normalized_source_causal_trust_artifact(args.output, result)
        selected_summary = {
            "selected_base_weights": result["selected_weights"],
            "effective_selected_action_response_by_episode": result[
                "effective_selected_action_response_by_episode"
            ],
            "pooled_source_tail": result["pooled_source_tail"],
        }
    else:
        result = fit_source_causal_trust(
            episodes,
            action_response_grid=grid,
            autonomous_drift_grid=grid,
        )
        write_source_causal_trust_artifact(args.output, result)
        selected_summary = {
            "selected_weights": result["selected_weights"],
            "pooled_source_tail": result["pooled_source_tail"],
        }
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "result_sha256": result["result_sha256"],
                "source_episode_count": len(episodes),
                **selected_summary,
                "leave_one_action_out": result["leave_one_action_out"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
