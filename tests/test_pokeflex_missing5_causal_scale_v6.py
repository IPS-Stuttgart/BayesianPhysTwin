from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.pokeflex_missing5_causal_scale_v6 import (
    FEATURE_NAMES,
    V5_EFFECTIVE_SCALES,
    V6_CANDIDATE_SCALES,
    build_causal_scale_model,
    causal_scale_feature,
    causal_scale_vertices,
    extract_source_frame_rows,
    select_causal_scale,
    validate_causal_scale_model,
)
from bayesian_phystwin.pokeflex_missing5_execution_v5 import (
    canonical_payload_sha256,
    file_sha256,
)
from bayesian_phystwin.pokeflex_missing5_scale import SOURCE_TAKES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "configs" / "sota" / "pokeflex_missing5_causal_scale_v6.json"
RESULT_PATH = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_missing5_causal_scale_v6"
    / "source_result.json"
)
MODEL_FILE_SHA256 = "130fd65b84723b05a8a7b6926b2c82bea2f356d3ff35c613fb7d6bf4f51373fa"
RESULT_FILE_SHA256 = "733dd4376ea7fcb62d67528479cf70c6d040f71b3ddf29e2dfb6e00950d645d1"


def _artifact(object_name: str, take_id: str) -> dict[str, Any]:
    maximum_frame = 40
    updates: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    take_number = int(take_id.rpartition("_T")[2])
    for target_frame in range(6, 36):
        phase = target_frame / maximum_frame
        update = {
            "target_frame": target_frame,
            "accepted": True,
            "action_supported": True,
            "rms_update_m": 0.001 + 0.00001 * target_frame + 0.000001 * take_number,
            "prior_motion_rms_m": 0.002 + 0.00002 * target_frame,
            "correction_to_prior_motion_ratio": 0.5 + phase,
            "correction_prior_motion_cosine": -0.2 + 0.4 * phase,
        }
        updates.append(update)
        target: dict[str, Any] = {"target_frame": target_frame}
        if object_name in V6_CANDIDATE_SCALES:
            baseline_scale = V5_EFFECTIVE_SCALES[object_name]
            candidate_scale = V6_CANDIDATE_SCALES[object_name]
            baseline = 5.0 + 0.01 * take_number
            gain = 0.02 if phase >= 0.50 else -0.02
            target[
                "checkpoint_action_local_state_relative_0.4_"
                f"residual_scale_{baseline_scale:g}"
            ] = baseline
            target[
                "checkpoint_action_local_state_relative_0.4_"
                f"residual_scale_{candidate_scale:g}"
            ] = baseline - gain
        targets.append(target)
    return {
        "artifact_kind": "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "future_observation_used": False,
        "official_target_outcome_used": False,
        "held_v8_accessed": False,
        "take": {"id": take_id, "maximum_frame": maximum_frame},
        "updates": updates,
        "targets": targets,
    }


def _model() -> dict[str, Any]:
    payloads = [
        _artifact(object_name, take_id)
        for object_name, take_ids in SOURCE_TAKES.items()
        for take_id in take_ids
    ]
    hashes = {
        take_id: f"{index + 1:064x}"
        for index, take_id in enumerate(
            sorted(take for take_ids in SOURCE_TAKES.values() for take in take_ids)
        )
    }
    return build_causal_scale_model(
        payloads,
        source_artifact_file_sha256s=hashes,
        parent_bindings={"parent": "1" * 64},
        source_gate={"passed": True},
    )


def _update(target_frame: int, maximum_frame: int = 40) -> dict[str, object]:
    phase = target_frame / maximum_frame
    return {
        "accepted": True,
        "rms_update_m": 0.001 + 0.00001 * target_frame,
        "prior_motion_rms_m": 0.002 + 0.00002 * target_frame,
        "correction_to_prior_motion_ratio": 0.5 + phase,
        "correction_prior_motion_cosine": -0.2 + 0.4 * phase,
    }


def test_hidden_source_outcome_does_not_change_causal_features() -> None:
    first = _artifact("3dPrintedCylinder", "3dPrintedCylinder_T1")
    second = deepcopy(first)
    second["targets"][0][
        "checkpoint_action_local_state_relative_0.4_residual_scale_0.375"
    ] = 1000.0

    first_rows = extract_source_frame_rows(first)
    second_rows = extract_source_frame_rows(second)

    np.testing.assert_array_equal(first_rows[0]["features"], second_rows[0]["features"])
    assert first_rows[0]["candidate_gain_mm"] != second_rows[0]["candidate_gain_mm"]
    assert len(first_rows[0]["features"]) == len(FEATURE_NAMES)


def test_state_residual_is_not_a_prior_scale_feature() -> None:
    first = _update(30)
    second = {**first, "state_residual_m": 1000.0}

    np.testing.assert_array_equal(
        causal_scale_feature(first, target_frame=30, maximum_frame=40),
        causal_scale_feature(second, target_frame=30, maximum_frame=40),
    )


def test_model_admits_supported_region_and_falls_back_elsewhere() -> None:
    model = _model()
    validation = validate_causal_scale_model(model)
    assert validation["passed"]

    admitted = select_causal_scale(
        model,
        object_name="3dPrintedCylinder",
        update=_update(34),
        target_frame=34,
        maximum_frame=40,
        supported=True,
    )
    assert admitted.admitted
    assert admitted.selected_scale == 0.375

    rejected = select_causal_scale(
        model,
        object_name="3dPrintedCylinder",
        update=_update(8),
        target_frame=8,
        maximum_frame=40,
        supported=True,
    )
    assert not rejected.admitted
    assert rejected.selected_scale == 0.25

    nonpromoted = select_causal_scale(
        model,
        object_name="Sponge",
        update=_update(34),
        target_frame=34,
        maximum_frame=40,
        supported=True,
    )
    assert not nonpromoted.admitted
    assert nonpromoted.selected_scale == 0.125


def test_invalid_and_unsupported_updates_fail_closed() -> None:
    model = _model()
    invalid = select_causal_scale(
        model,
        object_name="3dPrintedHeart",
        update={**_update(34), "rms_update_m": float("nan")},
        target_frame=34,
        maximum_frame=40,
        supported=True,
    )
    unsupported = select_causal_scale(
        model,
        object_name="3dPrintedHeart",
        update=_update(34),
        target_frame=34,
        maximum_frame=40,
        supported=False,
    )

    assert not invalid.admitted
    assert invalid.selected_scale == 0.1875
    assert invalid.reason == "invalid-causal-feature-v5-exact-fallback"
    assert not unsupported.admitted
    assert unsupported.selected_scale == 0.1875


def test_rejected_and_unsupported_vertex_paths_are_bit_exact() -> None:
    model = _model()
    target_prior = np.arange(18, dtype=np.float64).reshape(6, 3) / 100.0
    correction = np.full_like(target_prior, 0.003)
    v5 = target_prior + 0.25 * correction
    rejected = select_causal_scale(
        model,
        object_name="3dPrintedCylinder",
        update=_update(8),
        target_frame=8,
        maximum_frame=40,
        supported=True,
    )

    fallback = causal_scale_vertices(
        target_prior,
        correction,
        v5,
        rejected,
        supported=True,
    )
    unsupported = causal_scale_vertices(
        target_prior,
        correction,
        v5,
        rejected,
        supported=False,
    )

    assert fallback.tobytes() == v5.tobytes()
    assert unsupported.tobytes() == target_prior.tobytes()


def test_model_tampering_is_rejected() -> None:
    model = _model()
    model["policy"]["gain_margin_mm"] = 0.0

    with pytest.raises(ValueError, match="model changed"):
        validate_causal_scale_model(model)


def test_frozen_source_artifacts_validate_and_bind_closed_boundaries() -> None:
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    validation = validate_causal_scale_model(model)
    assert validation["model_sha256"] == (
        "4b6835f6ab57787be007855141081a3a10cea30eba736d47863b68fa7acf6ffa"
    )
    assert file_sha256(MODEL_PATH) == MODEL_FILE_SHA256
    assert file_sha256(RESULT_PATH) == RESULT_FILE_SHA256
    assert result["result_sha256"] == canonical_payload_sha256(result, "result_sha256")
    assert result["result_sha256"] == (
        "e7f17d5bd9045a3634e4f07b32ae9217ea8432ddd24ed387ab1c3512dd18483f"
    )
    assert result["model_sha256"] == model["model_sha256"]
    assert result["model_file_sha256"] == MODEL_FILE_SHA256
    assert result["source_gate"]["passed"] is True
    assert result["official_target_outcomes_used"] is False
    assert result["held_v8_accessed"] is False
