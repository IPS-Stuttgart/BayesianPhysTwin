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
from causal4d_public.deform360_reusable_physics import (
    validate_reusable_physics_response,
    validate_reusable_physics_selection,
)
from causal4d_public.deform360_reusable_trust import (
    build_deform360_trust_features,
    load_reusable_twin_trust_candidate,
)
from causal4d_public.deform360_reusable_trust_protocol import (
    authorize_reusable_trust_episode,
    load_reusable_trust_protocol,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-artifact", type=Path, required=True)
    parser.add_argument("--reference-prediction", type=Path, required=True)
    parser.add_argument(
        "--application-prediction",
        type=Path,
        help=(
            "Physical response to scale after trust is inferred from the fixed "
            "reference response. Defaults to --reference-prediction."
        ),
    )
    parser.add_argument("--robot", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--expected-prediction-sha256")
    parser.add_argument("--expected-application-prediction-sha256")
    parser.add_argument("--fresh-parent-lock", type=Path)
    parser.add_argument("--physics-addendum", type=Path)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--reference-response-json", type=Path)
    parser.add_argument("--application-response-json", type=Path)
    parser.add_argument("--physics-selection", type=Path)
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
    fresh_locks = (
        args.fresh_parent_lock,
        args.physics_addendum,
        args.execution_lock,
    )
    _require(
        not any(value is not None for value in fresh_locks)
        or all(value is not None for value in fresh_locks),
        "fresh parent, physics, and execution locks must be supplied together",
    )
    authorization = None
    fresh_evidence = None
    if args.fresh_parent_lock is not None:
        protocol = load_reusable_trust_protocol(
            args.fresh_parent_lock, args.physics_addendum, args.execution_lock
        )
        authorization = authorize_reusable_trust_episode(
            protocol,
            object_id=args.object_id,
            episode_id=args.episode_id,
            operation="held-prediction",
        )
        _require(
            args.reference_response_json is not None
            and args.application_response_json is not None
            and args.physics_selection is not None,
            "fresh prediction requires response and physical-selection artifacts",
        )
        selection_payload = json.loads(
            args.physics_selection.read_text(encoding="utf-8")
        )
        selection = validate_reusable_physics_selection(
            selection_payload, protocol=protocol
        )
        _require(
            selection["object_id"] == args.object_id,
            "physical selection belongs to another object",
        )
        reference_payload = json.loads(
            args.reference_response_json.read_text(encoding="utf-8")
        )
        application_payload = json.loads(
            args.application_response_json.read_text(encoding="utf-8")
        )
        reference_response = validate_reusable_physics_response(
            reference_payload, protocol=protocol, verify_archive=True
        )
        application_response = validate_reusable_physics_response(
            application_payload, protocol=protocol, verify_archive=True
        )
        _require(
            reference_response["episode_key"] == authorization["episode_key"]
            and application_response["episode_key"] == authorization["episode_key"],
            "physical response belongs to another held episode",
        )
        expected_reference = {
            name: float(protocol["addendum"]["reference_trust_response"][name])
            for name in ("init_spring_y", "drag_damping", "dashpot_damping")
        }
        _require(
            reference_response["physical_parameters"] == expected_reference,
            "trust response is not the frozen reference tuple",
        )
        _require(
            application_response["physical_parameters"]
            == selection["selected_physical_parameters"],
            "application response is not the selected physical tuple",
        )
        _require(
            Path(reference_payload["prediction_archive"]["path"]).resolve()
            == args.reference_prediction.resolve()
            and Path(application_payload["prediction_archive"]["path"]).resolve()
            == (args.application_prediction or args.reference_prediction).resolve(),
            "response metadata and prediction archives differ",
        )
        fresh_evidence = {
            "physical_selection": {
                "path": str(args.physics_selection.resolve()),
                "file_sha256": sha256_file(args.physics_selection),
                "result_sha256": selection["result_sha256"],
                "selected_candidate_index": selection["selected_candidate_index"],
                "selected_physical_parameters": selection[
                    "selected_physical_parameters"
                ],
            },
            "physical_responses": {
                "reference_result_sha256": reference_response["result_sha256"],
                "application_result_sha256": application_response["result_sha256"],
            },
        }
    reference_sha256 = sha256_file(args.reference_prediction)
    if args.expected_prediction_sha256 is not None:
        _require(
            reference_sha256 == args.expected_prediction_sha256,
            "reference prediction checksum differs from the lock",
        )
    application_path = args.application_prediction or args.reference_prediction
    application_sha256 = sha256_file(application_path)
    if args.expected_application_prediction_sha256 is not None:
        _require(
            application_sha256 == args.expected_application_prediction_sha256,
            "application prediction checksum differs from the lock",
        )
    model = load_reusable_twin_trust_candidate(args.trust_artifact)
    with np.load(args.reference_prediction, allow_pickle=False) as stored:
        _require("prediction_m" in stored, "reference archive lacks prediction_m")
        _require("persistence_m" in stored, "reference archive lacks persistence_m")
        reference_arrays = {name: np.asarray(stored[name]) for name in stored.files}
    with np.load(application_path, allow_pickle=False) as stored:
        _require("prediction_m" in stored, "application archive lacks prediction_m")
        _require("persistence_m" in stored, "application archive lacks persistence_m")
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    with np.load(args.robot, allow_pickle=False) as stored:
        _require("actions" in stored and "openings" in stored, "robot archive incomplete")
        actions = np.asarray(stored["actions"], dtype=np.float64)
        openings = np.asarray(stored["openings"], dtype=np.float64)
    reference_prediction = np.asarray(
        reference_arrays["prediction_m"], dtype=np.float64
    )
    reference_persistence = np.asarray(
        reference_arrays["persistence_m"], dtype=np.float64
    )
    application_prediction = np.asarray(arrays["prediction_m"], dtype=np.float64)
    persistence = np.asarray(arrays["persistence_m"], dtype=np.float64)
    _require(
        reference_prediction.shape == reference_persistence.shape,
        "reference prediction and persistence shapes differ",
    )
    _require(
        application_prediction.shape == persistence.shape,
        "application prediction and persistence shapes differ",
    )
    _require(
        np.array_equal(reference_persistence, persistence),
        "reference and application persistence differ",
    )
    reference_response = (
        reference_prediction - reference_persistence
    ) / model.reference_response_alpha
    application_response = (
        application_prediction - persistence
    ) / model.reference_response_alpha
    features = build_deform360_trust_features(
        actions,
        openings,
        reference_response,
        reference_persistence,
    )
    decision = model.decide(features)
    if decision.alpha == 0.0:
        trusted_prediction = persistence.copy()
        _require(
            np.array_equal(trusted_prediction, persistence),
            "zero trust does not reproduce persistence exactly",
        )
    else:
        trusted_prediction = persistence + decision.alpha * application_response
    _require(np.all(np.isfinite(trusted_prediction)), "trusted prediction is not finite")

    arrays["reference_prediction_m"] = np.asarray(
        reference_arrays["prediction_m"]
    )
    arrays["application_prediction_m"] = np.asarray(arrays["prediction_m"])
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
            "application_prediction": application_sha256,
            "robot": sha256_file(args.robot),
        },
        "output": {
            "path": str(args.output_npz.resolve()),
            "sha256": sha256_file(args.output_npz),
        },
        "output_sha256": sha256_file(args.output_npz),
        "features": {name: float(features[name]) for name in model.feature_names},
        "information_boundary": {
            "known_future_robot_action_used": True,
            "frame_zero_object_geometry_used": True,
            "predicted_simulator_response_used": True,
            "trust_inferred_from_fixed_reference_response": True,
            "candidate_physics_cannot_change_trust": True,
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
    if authorization is not None:
        payload["prospective_authorization"] = authorization
        payload.update(fresh_evidence)
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
