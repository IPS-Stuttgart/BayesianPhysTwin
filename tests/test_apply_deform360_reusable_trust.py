from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/remote/apply_deform360_reusable_trust.py"


def _candidate(path: Path, *, threshold: float) -> None:
    payload = {
        "artifact_kind": "Deform360ReusableTwinTrustCandidate",
        "schema_version": 1,
        "policy": "test",
        "closure_feature": "mean_minimum_gripper_closure",
        "closure_rule": {"mode": "threshold", "threshold": threshold},
        "reference_response_alpha": 0.9,
        "maximum_alpha": 1.2,
        "feature_names": ["mean_minimum_gripper_closure"],
        "ridge": 0.1,
        "coefficients": [0.45, 0.0],
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
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_apply(tmp_path: Path, *, threshold: float) -> tuple[np.ndarray, dict]:
    frames = 76
    points = 8
    persistence = np.zeros((frames, points, 3), dtype=np.float64)
    persistence[:, :, 0] = np.linspace(0.0, 0.2, points)[None]
    reference = persistence.copy()
    reference[:, :, 2] += 0.9 * np.linspace(0.0, 0.01, frames)[:, None]
    application = persistence.copy()
    application[:, :, 2] += 0.9 * np.linspace(0.0, 0.02, frames)[:, None]
    reference_path = tmp_path / "reference.npz"
    application_path = tmp_path / "application.npz"
    np.savez_compressed(
        reference_path, prediction_m=reference, persistence_m=persistence
    )
    np.savez_compressed(
        application_path, prediction_m=application, persistence_m=persistence
    )

    robot_frames = 81
    centres = np.zeros((robot_frames, 2, 3), dtype=np.float64)
    centres[..., 2] = np.linspace(0.0, 0.05, robot_frames)[:, None]
    actions = np.repeat(centres[:, :, None], 5, axis=2)
    openings = np.repeat(
        np.linspace(0.05, 0.01, robot_frames)[:, None], 2, axis=1
    )
    robot_path = tmp_path / "robot.npz"
    np.savez_compressed(robot_path, actions=actions, openings=openings)
    candidate_path = tmp_path / "candidate.json"
    _candidate(candidate_path, threshold=threshold)
    output_npz = tmp_path / "trusted.npz"
    output_json = tmp_path / "trusted.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trust-artifact",
            str(candidate_path),
            "--reference-prediction",
            str(reference_path),
            "--application-prediction",
            str(application_path),
            "--robot",
            str(robot_path),
            "--output-npz",
            str(output_npz),
            "--output-json",
            str(output_json),
            "--object-id",
            "003-cable",
            "--episode-id",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with np.load(output_npz, allow_pickle=False) as stored:
        prediction = np.asarray(stored["prediction_m"])
    return prediction, json.loads(output_json.read_text(encoding="utf-8"))


def test_apply_trust_infers_from_reference_and_scales_application(tmp_path) -> None:
    prediction, payload = _run_apply(tmp_path, threshold=0.0)
    z = 0.45 * np.linspace(0.0, 0.02, 76)
    expected = np.repeat(z[:, None], prediction.shape[1], axis=1)

    assert prediction[:, :, 2] == pytest.approx(expected)
    assert payload["decision"]["alpha"] == pytest.approx(0.45)
    assert (
        payload["input_sha256"]["reference_prediction"]
        != payload["input_sha256"]["application_prediction"]
    )
    assert payload["information_boundary"][
        "candidate_physics_cannot_change_trust"
    ]


def test_apply_trust_rejection_is_exact_persistence_with_other_physics(
    tmp_path,
) -> None:
    prediction, payload = _run_apply(tmp_path, threshold=1.1)
    persistence = np.zeros_like(prediction)
    persistence[:, :, 0] = np.linspace(0.0, 0.2, prediction.shape[1])[None]

    assert np.array_equal(prediction, persistence)
    assert payload["decision"]["alpha"] == 0.0
