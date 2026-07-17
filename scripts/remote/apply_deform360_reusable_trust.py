#!/usr/bin/env python3
"""Apply the frozen reusable-twin trust gate to one prediction-only episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform360_independent_source import sha256_file
from causal4d_public.deform360_reusable_trust import (
    build_deform360_trust_features,
    load_reusable_twin_trust_candidate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-artifact", type=Path, required=True)
    parser.add_argument("--reference-prediction", type=Path, required=True)
    parser.add_argument("--robot", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--expected-prediction-sha256")
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _result_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = _parse_args()
    reference_sha256 = sha256_file(args.reference_prediction)
    if args.expected_prediction_sha256 is not None:
        _require(
            reference_sha256 == args.expected_prediction_sha256,
            "reference prediction checksum differs from the lock",
        )
    model = load_reusable_twin_trust_candidate(args.trust_artifact)
    with np.load(args.reference_prediction, allow_pickle=False) as stored:
        _require("prediction_m" in stored, "reference archive lacks prediction_m")
        _require("persistence_m" in stored, "reference archive lacks persistence_m")
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    with np.load(args.robot, allow_pickle=False) as stored:
        _require("actions" in stored and "openings" in stored, "robot archive incomplete")
        actions = np.asarray(stored["actions"], dtype=np.float64)
        openings = np.asarray(stored["openings"], dtype=np.float64)
    reference_prediction = np.asarray(arrays["prediction_m"], dtype=np.float64)
    persistence = np.asarray(arrays["persistence_m"], dtype=np.float64)
    _require(
        reference_prediction.shape == persistence.shape,
        "reference prediction and persistence shapes differ",
    )
    response = (
        reference_prediction - persistence
    ) / model.reference_response_alpha
    features = build_deform360_trust_features(
        actions,
        openings,
        response,
        persistence,
    )
    decision = model.decide(features)
    if decision.alpha == 0.0:
        trusted_prediction = persistence.copy()
        _require(
            np.array_equal(trusted_prediction, persistence),
            "zero trust does not reproduce persistence exactly",
        )
    else:
        trusted_prediction = persistence + decision.alpha * response
    _require(np.all(np.isfinite(trusted_prediction)), "trusted prediction is not finite")

    arrays["reference_prediction_m"] = np.asarray(arrays["prediction_m"])
    arrays["prediction_m"] = trusted_prediction
    arrays["trust_alpha"] = np.asarray(decision.alpha, dtype=np.float64)
    arrays["trust_closure_accepted"] = np.asarray(
        decision.closure_accepted, dtype=np.bool_
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTwinTrustedPrediction",
        "object_id": args.object_id,
        "episode_id": args.episode_id,
        "decision": {
            "alpha": decision.alpha,
            "raw_alpha": decision.raw_alpha,
            "closure_accepted": decision.closure_accepted,
            "closure_value": decision.closure_value,
        },
        "model": {
            "result_sha256": model.result_sha256,
            "reference_response_alpha": model.reference_response_alpha,
        },
        "input_sha256": {
            "trust_artifact": sha256_file(args.trust_artifact),
            "reference_prediction": reference_sha256,
            "robot": sha256_file(args.robot),
        },
        "output_sha256": sha256_file(args.output_npz),
        "features": {name: float(features[name]) for name in model.feature_names},
        "information_boundary": {
            "known_future_robot_action_used": True,
            "frame_zero_object_geometry_used": True,
            "predicted_simulator_response_used": True,
            "post_initial_object_observation_used": False,
            "tactile_used": False,
            "symbolic_action_label_used": False,
            "object_outcome_used": False,
        },
        "claim_boundary": (
            "prediction-only reusable-twin trust arm; official multi-episode claims "
            "require the exact Deform360 split, evaluator, and a fresh lock"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
