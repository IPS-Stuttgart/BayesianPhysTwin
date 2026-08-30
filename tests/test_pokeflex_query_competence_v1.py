from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.pokeflex_query_competence_v1 import (
    FEATURE_NAMES,
    PARENT_PUBLIC78_PROTOCOL_SHA256,
    PRIMARY_FEATURES,
    FrozenPhysicalRiskModelV1,
    PokeFlexFrameV1,
    consume_stage_attempt_v1,
    deterministic_split_v1,
    evaluate_policy_v1,
    file_sha256,
    fit_risk_model_v1,
    load_protocol_v1,
    load_take_artifact_v1,
    run_source_stage_v1,
    run_validation_stage_v1,
    score_frames_v1,
    validate_source_result_v1,
)

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "protocols/pokeflex_query_competence_retrospective_v1.json"
)
PROTOCOL_FILE_SHA256 = (
    "70f3f9bba296582e59e89776225e12e8bd517958a3a0845242eb9fe739574906"
)
PROTOCOL_SHA256 = "853ba1017e8781ddf324b9400df5f4393f2fca390f192e2757ce67a81c3e2354"
IMPLEMENTATION_COMMIT = "4524852c0235fc736c12ebaa82118cf81e5f19bf"


def _update(target_frame: int, *, harmful: bool) -> dict[str, object]:
    rms_update_m = 0.040 if harmful else 0.0008
    return {
        "source_frame": target_frame - 1,
        "target_frame": target_frame,
        "accepted": True,
        "reason": "accepted",
        "action_supported": True,
        "rms_update_m": rms_update_m,
        "maximum_update_m": 2.0 * rms_update_m,
        "associated_points": 120,
        "camera_biases_m": [[0.0001, 0.0, 0.0], [0.0, -0.0001, 0.0]],
        "prior_motion_rms_m": 0.002,
        "correction_to_prior_motion_ratio": rms_update_m / 0.002,
        "correction_prior_motion_cosine": 0.8,
        "previous_correction_cosine": 0.7,
        "force_y": 4.0,
        "force_y_delta": 0.2,
        "effective_information_mass": 50.0,
        "median_robust_weight": 0.9,
        "downweighted_fraction": 0.1,
        "assignment_variance_m2_mean": 1e-7,
        "condition_number": 12.0,
    }


def _artifact(take_id: str, *, safe_error: float = 0.9) -> dict[str, object]:
    targets = []
    updates = []
    candidate_key = "checkpoint_action_local_state_relative_0.4_residual_scale_0.125"
    for index in range(10):
        target_frame = 6 + index
        harmful = index >= 5
        updates.append(_update(target_frame, harmful=harmful))
        targets.append(
            {
                "target_frame": target_frame,
                "released_checkpoint_CD_UL1_mm": 1.0,
                candidate_key: 1.2 if harmful else safe_error,
            }
        )
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke",
        "future_observation_used": False,
        "retrospective_prediction_role": (
            "previously exposed public action; fixed all18 scale; "
            "never prospective evidence"
        ),
        "public_transfer_protocol_sha256": PARENT_PUBLIC78_PROTOCOL_SHA256,
        "candidate_effective_scale": 0.125,
        "take": {"id": take_id},
        "targets": targets,
        "updates": updates,
    }


def _write_artifact(
    root: Path, take_id: str, payload: dict[str, object]
) -> dict[str, object]:
    path = root / f"{take_id}.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def test_deterministic_split_keeps_every_object_in_every_stage() -> None:
    take_ids = [
        f"Object{index:02d}_T{take}" for index in range(18) for take in range(1, 5)
    ]
    split = deterministic_split_v1(take_ids)
    assert tuple(len(split[name]) for name in split) == (18, 18, 36)
    for name, roster in split.items():
        assert len({take_id.rpartition("_T")[0] for take_id in roster}) == 18, name
    assert split == deterministic_split_v1(reversed(take_ids))


def test_frozen_protocol_identity_and_rosters() -> None:
    assert file_sha256(PROTOCOL_PATH) == PROTOCOL_FILE_SHA256
    protocol = load_protocol_v1(PROTOCOL_PATH)
    assert protocol["protocol_sha256"] == PROTOCOL_SHA256
    assert protocol["implementation"]["git_commit"] == IMPLEMENTATION_COMMIT
    split = protocol["split"]
    assert tuple(len(split[name]) for name in split) == (18, 18, 42)
    assert len(set().union(*(set(roster) for roster in split.values()))) == 78
    for roster in split.values():
        assert len({take_id.rpartition("_T")[0] for take_id in roster}) == 18


def test_target_changes_cannot_change_preoutcome_features(tmp_path: Path) -> None:
    first = _artifact("FoamDice_T2", safe_error=0.9)
    second = _artifact("FoamDice_T2", safe_error=0.1)
    first_row = _write_artifact(tmp_path, "FoamDice_T2", first)
    first_frames = load_take_artifact_v1(
        tmp_path / first_row["filename"],
        take_id="FoamDice_T2",
        expected_sha256=str(first_row["sha256"]),
        expected_bytes=int(first_row["bytes"]),
    )
    second_row = _write_artifact(tmp_path, "FoamDice_T2", second)
    second_frames = load_take_artifact_v1(
        tmp_path / second_row["filename"],
        take_id="FoamDice_T2",
        expected_sha256=str(second_row["sha256"]),
        expected_bytes=int(second_row["bytes"]),
    )
    assert not np.array_equal(
        [frame.candidate_error_mm for frame in first_frames],
        [frame.candidate_error_mm for frame in second_frames],
    )
    for first_frame, second_frame in zip(first_frames, second_frames, strict=True):
        np.testing.assert_array_equal(
            first_frame.feature_vector, second_frame.feature_vector
        )


def test_artifact_loader_rejects_future_observation_use(tmp_path: Path) -> None:
    payload = _artifact("FoamDice_T2")
    payload["future_observation_used"] = True
    row = _write_artifact(tmp_path, "FoamDice_T2", payload)
    with pytest.raises(ValueError, match="future observations"):
        load_take_artifact_v1(
            tmp_path / row["filename"],
            take_id="FoamDice_T2",
            expected_sha256=str(row["sha256"]),
            expected_bytes=int(row["bytes"]),
        )


def test_artifact_loader_enforces_direct_causal_prefix_and_booleans(
    tmp_path: Path,
) -> None:
    payload = _artifact("FoamDice_T2")
    payload["updates"][0]["source_frame"] = payload["updates"][0]["target_frame"]
    row = _write_artifact(tmp_path, "FoamDice_T2", payload)
    with pytest.raises(ValueError, match="one-frame causal prefix"):
        load_take_artifact_v1(
            tmp_path / row["filename"],
            take_id="FoamDice_T2",
            expected_sha256=str(row["sha256"]),
            expected_bytes=int(row["bytes"]),
        )

    payload = _artifact("FoamDice_T2")
    payload["updates"][0]["accepted"] = "true"
    row = _write_artifact(tmp_path, "FoamDice_T2", payload)
    with pytest.raises(ValueError, match="non-boolean PokeFlex accepted"):
        load_take_artifact_v1(
            tmp_path / row["filename"],
            take_id="FoamDice_T2",
            expected_sha256=str(row["sha256"]),
            expected_bytes=int(row["bytes"]),
        )


def test_attempt_ledger_is_consumed_before_stage_execution(tmp_path: Path) -> None:
    protocol = {
        "protocol_sha256": "1" * 64,
        "execution": {
            "root": str(tmp_path),
            "source_attempt_filename": "source-attempt.json",
            "source_result_filename": "source-result.json",
            "validation_attempt_filename": "validation-attempt.json",
            "validation_result_filename": "validation-result.json",
        },
    }
    output = tmp_path / "source-result.json"
    attempt = consume_stage_attempt_v1(
        protocol,
        protocol_file_sha256="2" * 64,
        stage="source",
        output=output,
    )
    assert json.loads(attempt.read_text(encoding="utf-8"))["stage"] == "source"
    with pytest.raises(ValueError, match="attempt already consumed"):
        consume_stage_attempt_v1(
            protocol,
            protocol_file_sha256="2" * 64,
            stage="source",
            output=output,
        )


def _frame(
    object_index: int,
    frame_index: int,
    *,
    harmful: bool,
) -> PokeFlexFrameV1:
    features = np.zeros(len(FEATURE_NAMES), dtype=float)
    features[FEATURE_NAMES.index("candidate_disagreement_mm")] = 5.0 if harmful else 0.1
    features[FEATURE_NAMES.index("candidate_motion_ratio_log1p")] = (
        2.0 if harmful else 0.05
    )
    features[FEATURE_NAMES.index("update_accepted")] = 1.0
    features[FEATURE_NAMES.index("action_supported")] = 1.0
    return PokeFlexFrameV1(
        take_id=f"Object{object_index:02d}_T1",
        object_name=f"Object{object_index:02d}",
        target_frame=frame_index + 1,
        feature_vector=features,
        candidate_available=True,
        fallback_error_mm=1.0,
        candidate_error_mm=1.2 if harmful else 0.9,
    )


def test_object_balanced_risk_model_separates_harm() -> None:
    frames = tuple(
        _frame(object_index, frame_index, harmful=frame_index >= 5)
        for object_index in range(18)
        for frame_index in range(10)
    )
    model = fit_risk_model_v1(
        frames,
        model_name="model_disagreement_only",
        selected_feature_names=PRIMARY_FEATURES,
    )
    scores = score_frames_v1(model, frames)
    assert float(np.max(scores[:5])) < float(np.min(scores[5:10]))
    evaluation = evaluate_policy_v1(
        frames,
        scores,
        0.5,
        bootstrap_seed=7,
        bootstrap_replicates=200,
    )
    assert evaluation["accepted_object_count"] == 18
    assert evaluation["harmful_accepted_count"] == 0
    assert evaluation["gate_passed"] is True
    assert evaluation["gate_checks"]["exact_fallback_identity"] is True

    record = model.to_record()
    restored = FrozenPhysicalRiskModelV1.from_record(record)
    assert restored.artifact_id == model.artifact_id
    record["unexpected"] = True
    with pytest.raises(ValueError, match="record fields changed"):
        FrozenPhysicalRiskModelV1.from_record(record)


def _synthetic_protocol_and_artifacts(
    root: Path,
) -> tuple[dict[str, object], dict[str, tuple[str, ...]]]:
    rosters = {
        "risk_train": tuple(f"Object{index:02d}_T1" for index in range(18)),
        "threshold_select": tuple(f"Object{index:02d}_T2" for index in range(18)),
        "validation": tuple(
            take_id
            for index in range(18)
            for take_id in (f"Object{index:02d}_T3", f"Object{index:02d}_T4")
        ),
    }
    inventory: dict[str, object] = {}
    for take_id in rosters["risk_train"] + rosters["threshold_select"]:
        inventory[take_id] = _write_artifact(root, take_id, _artifact(take_id))
    for take_id in rosters["validation"]:
        inventory[take_id] = {
            "filename": f"{take_id}.json",
            "bytes": 1,
            "sha256": "0" * 64,
        }
    protocol: dict[str, object] = {
        "protocol_sha256": "1" * 64,
        "artifact_inventory": inventory,
        "split": rosters,
    }
    return protocol, rosters


def test_source_stage_never_opens_validation_artifacts(tmp_path: Path) -> None:
    protocol, _ = _synthetic_protocol_and_artifacts(tmp_path)
    source_result = run_source_stage_v1(
        protocol,
        tmp_path,
        bootstrap_replicates=200,
    )
    assert source_result["validation_take_count_opened"] == 0
    assert source_result["validation_authorized"] is True
    validate_source_result_v1(source_result, protocol)


def test_validation_uses_exact_source_result_and_frozen_roster(tmp_path: Path) -> None:
    protocol, rosters = _synthetic_protocol_and_artifacts(tmp_path)
    source_result = run_source_stage_v1(
        protocol,
        tmp_path,
        bootstrap_replicates=200,
    )
    for take_id in rosters["validation"]:
        protocol["artifact_inventory"][take_id] = _write_artifact(
            tmp_path, take_id, _artifact(take_id)
        )
    result = run_validation_stage_v1(
        protocol,
        source_result,
        tmp_path,
        bootstrap_replicates=200,
    )
    assert result["validation_take_count"] == 36
    assert result["retrospective_data"] is True
    assert result["prospective_confirmation"] is False
    assert result["primary_gate_passed"] is True

    tampered = dict(source_result)
    tampered["validation_authorized"] = False
    with pytest.raises(ValueError, match="identity changed"):
        run_validation_stage_v1(protocol, tampered, tmp_path, bootstrap_replicates=200)
