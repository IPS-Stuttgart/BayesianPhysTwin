from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from causal4d_public.deform360_reusable_trust import (
    build_deform360_trust_features,
    load_reusable_twin_trust_candidate,
)


def _candidate_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "artifact_kind": "Deform360ReusableTwinTrustCandidate",
        "schema_version": 1,
        "policy": "test",
        "closure_feature": "mean_minimum_gripper_closure",
        "closure_rule": {"mode": "threshold", "threshold": 0.5},
        "reference_response_alpha": 0.9,
        "maximum_alpha": 1.2,
        "feature_names": ["mean_minimum_gripper_closure"],
        "ridge": 0.1,
        "coefficients": [0.4, 0.0],
        "feature_mean": [0.0],
        "feature_scale": [1.0],
        "closure_search": [],
        "ridge_search": [],
        "fit_episode_keys": [],
        "information_boundary": {},
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["result_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def test_trust_candidate_applies_exact_closure_fallback(tmp_path) -> None:
    payload = _candidate_payload()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    model = load_reusable_twin_trust_candidate(path)

    rejected = model.decide({"mean_minimum_gripper_closure": 0.49})
    accepted = model.decide({"mean_minimum_gripper_closure": 0.5})

    assert rejected.alpha == 0.0
    assert not rejected.closure_accepted
    assert accepted.alpha == pytest.approx(0.4)
    assert accepted.closure_accepted


def test_trust_candidate_loads_from_diagnosis(tmp_path) -> None:
    candidate = _candidate_payload()
    path = tmp_path / "diagnosis.json"
    path.write_text(
        json.dumps(
            {
                "artifact_kind": "Deform360SameObjectTrustDiagnosis",
                "full_source_candidate": candidate,
            }
        ),
        encoding="utf-8",
    )

    model = load_reusable_twin_trust_candidate(path)

    assert model.result_sha256 == candidate["result_sha256"]


def test_trust_candidate_rejects_tampering(tmp_path) -> None:
    payload = _candidate_payload()
    payload["coefficients"] = [0.8, 0.0]
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_reusable_twin_trust_candidate(path)


@pytest.mark.parametrize("gripper_count", [1, 2])
def test_build_trust_features_supports_robot_action_layouts(
    gripper_count: int,
) -> None:
    frames = 81
    points = 8
    centres = np.zeros((frames, gripper_count, 3), dtype=np.float64)
    centres[..., 0] = np.linspace(0.0, 0.1, frames)[:, None]
    centres[..., 2] = np.linspace(0.0, 0.05, frames)[:, None]
    actions = np.repeat(centres[:, :, None, :], 5, axis=2)
    openings = np.linspace(0.05, 0.02, frames)[:, None]
    openings = np.repeat(openings, gripper_count, axis=1)
    if gripper_count == 1:
        actions = actions[:, 0]
        openings = openings[:, 0]
    persistence = np.zeros((76, points, 3), dtype=np.float64)
    persistence[0, :, 0] = np.linspace(0.0, 0.2, points)
    response = np.repeat(persistence[0:1], 76, axis=0)
    response[:, :, 2] += np.linspace(0.0, 0.01, 76)[:, None]

    features = build_deform360_trust_features(
        actions,
        openings,
        response,
        persistence,
    )

    assert features["gripper_count"] == gripper_count
    assert features["bimanual"] == (gripper_count == 2)
    assert features["response_mean_displacement_m"] > 0.0
    assert all(np.isfinite(value) for value in features.values())
