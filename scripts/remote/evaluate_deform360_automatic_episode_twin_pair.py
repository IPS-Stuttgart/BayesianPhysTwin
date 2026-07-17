#!/usr/bin/env python3
"""Evaluate a matched driven/zero-action automatic episode-twin pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from causal4d_public.deform360_phystwin_trust import (
    evaluate_cardinality_normalized_fixed_trust,
    load_official_phystwin_readout_trust_episode,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--target-final-data", type=Path, required=True)
    parser.add_argument("--simulator-final-data", type=Path, required=True)
    parser.add_argument("--readout-artifact", type=Path, required=True)
    parser.add_argument("--driven-result", type=Path, required=True)
    parser.add_argument("--zero-result", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--base-action-response", type=float, default=0.4)
    parser.add_argument("--autonomous-drift", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    episode = load_official_phystwin_readout_trust_episode(
        args.episode_id,
        args.target_final_data,
        args.simulator_final_data,
        args.readout_artifact,
        args.driven_result,
        args.zero_result,
        args.split_json,
    )
    metrics = evaluate_cardinality_normalized_fixed_trust(
        episode,
        base_action_response=args.base_action_response,
        autonomous_drift=args.autonomous_drift,
    )
    result = {
        "schema_version": 1,
        "artifact_kind": "Deform360AutomaticEpisodeTwinMatchedPairEvaluation",
        "episode_id": args.episode_id,
        "fixed_trust": {
            "base_action_response": args.base_action_response,
            "autonomous_drift": args.autonomous_drift,
            "normalization": "controller_count",
        },
        "metrics": metrics,
        "input_sha256": {
            "target_final_data": _sha256_file(args.target_final_data),
            "simulator_final_data": _sha256_file(args.simulator_final_data),
            "readout_artifact": _sha256_file(args.readout_artifact),
            "driven_result": _sha256_file(args.driven_result),
            "zero_result": _sha256_file(args.zero_result),
            "split_json": _sha256_file(args.split_json),
        },
        "information_boundary": {
            "object_observation_frames_used_at_prediction_time": [0],
            "post_initial_object_observation_used": False,
            "source_outcomes_scored": True,
            "calibration_outcome_read": False,
            "target_outcome_read": False,
        },
        "claim_boundary": (
            "source-only benchmark control; no independent transfer or SOTA claim"
        ),
    }
    result["result_sha256"] = _result_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
