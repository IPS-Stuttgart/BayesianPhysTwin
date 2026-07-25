from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_selective_virtual_sensing_artifacts import (
    BACKBONE_ARCHIVE_FILENAME,
    BACKBONE_SEAL_FILENAME,
    QUALITY_FAILURE_FILENAME,
    build_selective_backbone_seal,
    build_selective_prediction_cohort_seal,
    build_selective_raw_camera_measurement_case,
    build_selective_virtual_sensing_prediction_case,
    record_selective_quality_failure,
    selective_case_records,
    validate_selective_backbone_seal,
    validate_selective_prediction_cohort_seal,
    validate_selective_quality_failure,
)
from bayesian_phystwin.deform360_selective_virtual_sensing_protocol import (
    PROTOCOL_ID,
)


PROTOCOL = (
    Path(__file__).parents[1]
    / "configs"
    / "sota"
    / "deform360_selective_virtual_sensing_v1.json"
)


def _backbone(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source = tmp_path / "frame-zero.json"
    stage = tmp_path / "stage.json"
    source.write_text('{"frame_zero_only":true}\n', encoding="utf-8")
    stage.write_text('{"future_object_read":false}\n', encoding="utf-8")
    rng = np.random.default_rng(4)
    points = rng.uniform(-0.2, 0.2, size=(24, 3)).astype(np.float32)
    case = tmp_path / "005-thread-ep0005"
    seal = build_selective_backbone_seal(
        PROTOCOL,
        case,
        object_id="005-thread",
        episode_id=5,
        frame_zero_points_m=points,
        frame_zero_reconstruction_manifest=source,
        prediction_stage_manifest=stage,
    )
    return case, seal


def _measurement(case: Path, output: Path, seal: dict[str, object]) -> None:
    from bayesian_phystwin.deform360_online_belief_evaluation import _sha256

    with np.load(case / BACKBONE_ARCHIVE_FILENAME, allow_pickle=False) as stored:
        persistence = np.asarray(stored["persistence_m"])
    centers = np.arange(16, dtype=np.int64)
    measurement = np.full_like(persistence, np.nan)
    visible = np.zeros(persistence.shape[:2], dtype=bool)
    valid = visible.copy()
    for frame in (19, 38, 57):
        measurement[frame, centers] = persistence[frame, centers] + np.array(
            [0.01, 0.0, 0.0], dtype=np.float32
        )
        visible[frame, centers] = True
        valid[frame, centers] = True
    output.mkdir()
    archive = output / "measurement.npz"
    np.savez_compressed(
        archive,
        measurement_m=measurement,
        measurement_visibility=visible,
        measurement_validity=valid,
        center_ids=centers,
        selected_cameras=np.asarray([f"camera-{index}" for index in range(8)]),
        update_frames=np.asarray((19, 38, 57), dtype=np.int64),
    )
    manifest = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalRawCameraMeasurement",
        "protocol_id": PROTOCOL_ID,
        "case": seal["case"],
        "object_id": seal["object_id"],
        "episode_id": seal["episode_id"],
        "episode_key": seal["episode_key"],
        "inputs": {
            "prediction_seal": {
                "sha256": _sha256(case / BACKBONE_SEAL_FILENAME)
            }
        },
        "output": {"measurement_archive_sha256": _sha256(archive)},
        "information_boundary": {
            "target_data_read": False,
            "outcome_manifest_read": False,
            "future_reconstruction_after_frame_zero_read": False,
        },
    }
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    import hashlib

    manifest["result_sha256"] = hashlib.sha256(encoded).hexdigest()
    (output / "measurement_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prediction_for_record(
    tmp_path: Path,
    record: dict[str, object],
    prediction_root: Path,
) -> None:
    case_name = str(record["case"])
    sources = tmp_path / "sources"
    sources.mkdir(exist_ok=True)
    source = sources / f"{case_name}-frame-zero.json"
    stage = sources / f"{case_name}-stage.json"
    source.write_text('{"frame_zero_only":true}\n', encoding="utf-8")
    stage.write_text('{"future_object_read":false}\n', encoding="utf-8")
    seed = int.from_bytes(case_name.encode("utf-8"), "little") % (2**32)
    points = np.random.default_rng(seed).uniform(
        -0.2, 0.2, size=(24, 3)
    ).astype(np.float32)
    case = tmp_path / "backbones" / case_name
    seal = build_selective_backbone_seal(
        PROTOCOL,
        case,
        object_id=str(record["object_id"]),
        episode_id=int(record["episode_id"]),
        frame_zero_points_m=points,
        frame_zero_reconstruction_manifest=source,
        prediction_stage_manifest=stage,
    )
    measurement = tmp_path / "measurements" / case_name
    measurement.parent.mkdir(exist_ok=True)
    _measurement(case, measurement, seal)
    build_selective_virtual_sensing_prediction_case(
        PROTOCOL, case, measurement, prediction_root / case_name
    )


def test_case_records_are_exactly_twenty_four_locked_episodes() -> None:
    records = selective_case_records(PROTOCOL)

    assert len(records) == 24
    assert records[0] == {
        "case": "005-thread-ep0005",
        "object_id": "005-thread",
        "episode_id": 5,
        "episode_key": "005-thread/5",
        "stratum": "filament",
    }


def test_backbone_seal_contains_only_frame_zero_persistence(tmp_path: Path) -> None:
    case, seal = _backbone(tmp_path)

    validate_selective_backbone_seal(seal, protocol_path=PROTOCOL, case_dir=case)
    with np.load(case / BACKBONE_ARCHIVE_FILENAME, allow_pickle=False) as stored:
        prediction = np.asarray(stored["prediction_m"])
        persistence = np.asarray(stored["persistence_m"])
        points = np.asarray(stored["frame_zero_points_m"])
    np.testing.assert_array_equal(prediction, persistence)
    np.testing.assert_array_equal(persistence, np.repeat(points[None], 76, axis=0))
    assert seal["information_boundary"]["future_object_track_read"] is False


def test_backbone_tampering_is_rejected(tmp_path: Path) -> None:
    case, seal = _backbone(tmp_path)
    seal["material_point_count"] = 25

    with pytest.raises(ValueError, match="content checksum changed"):
        validate_selective_backbone_seal(
            seal, protocol_path=PROTOCOL, case_dir=case
        )


def test_prediction_artifact_is_built_before_any_target(tmp_path: Path) -> None:
    case, backbone_seal = _backbone(tmp_path)
    measurement = tmp_path / "measurement"
    _measurement(case, measurement, backbone_seal)
    output = tmp_path / "prediction"

    seal = build_selective_virtual_sensing_prediction_case(
        PROTOCOL, case, measurement, output
    )

    assert seal["information_boundary"]["target_data_read"] is False
    assert seal["information_boundary"]["future_dense_reconstruction_read"] is False
    with np.load(
        output / "virtual_sensing_prediction.npz", allow_pickle=False
    ) as stored:
        prediction = np.asarray(stored["prediction_m"])
        persistence = np.asarray(stored["persistence_m"])
    assert np.any(prediction[20:] != persistence[20:])


def test_public_builders_do_not_accept_target_or_outcome_inputs() -> None:
    for function in (
        build_selective_backbone_seal,
        build_selective_prediction_cohort_seal,
        build_selective_raw_camera_measurement_case,
        build_selective_virtual_sensing_prediction_case,
        record_selective_quality_failure,
    ):
        parameters = inspect.signature(function).parameters
        assert "target" not in parameters
        assert "outcome" not in parameters


def test_quality_failure_is_target_free_and_tamper_evident(tmp_path: Path) -> None:
    failure_dir = tmp_path / "005-thread-ep0005"
    failure = record_selective_quality_failure(
        PROTOCOL,
        failure_dir,
        object_id="005-thread",
        episode_id=5,
        stage="frame-zero-reconstruction",
        error_type="RuntimeError",
        error_message="fixture reconstruction failure",
    )

    validate_selective_quality_failure(failure, protocol_path=PROTOCOL)
    assert failure["information_boundary"]["target_data_read"] is False
    assert (failure_dir / QUALITY_FAILURE_FILENAME).is_file()
    failure["error_message"] = "changed"
    with pytest.raises(ValueError, match="content checksum changed"):
        validate_selective_quality_failure(failure, protocol_path=PROTOCOL)


def test_prediction_cohort_seal_enforces_locked_evaluability_gate(
    tmp_path: Path,
) -> None:
    records = selective_case_records(PROTOCOL)
    chosen_objects: dict[str, list[str]] = {}
    successful_cases: set[str] = set()
    for record in records:
        stratum = str(record["stratum"])
        object_id = str(record["object_id"])
        selected = chosen_objects.setdefault(stratum, [])
        if object_id not in selected and len(selected) < 3:
            selected.append(object_id)
            successful_cases.add(str(record["case"]))

    prediction_root = tmp_path / "predictions"
    failure_root = tmp_path / "failures"
    for record in records:
        case_name = str(record["case"])
        if case_name in successful_cases:
            _prediction_for_record(tmp_path, dict(record), prediction_root)
        else:
            record_selective_quality_failure(
                PROTOCOL,
                failure_root / case_name,
                object_id=str(record["object_id"]),
                episode_id=int(record["episode_id"]),
                stage="prediction-prefix-staging",
                error_type="FixtureFailure",
                error_message="predefined target-free fixture failure",
            )

    output = tmp_path / "prediction-cohort-seal.json"
    seal = build_selective_prediction_cohort_seal(
        PROTOCOL, prediction_root, failure_root, output
    )

    validate_selective_prediction_cohort_seal(
        seal,
        protocol_path=PROTOCOL,
        require_eligible=True,
        prediction_root=prediction_root,
        failure_root=failure_root,
    )
    assert seal["prediction_count"] == 9
    assert seal["quality_failure_count"] == 15
    assert seal["evaluable_object_count"] == 9
    assert seal["evaluable_object_count_by_stratum"] == {
        "filament": 3,
        "sheet": 3,
        "volumetric": 3,
    }
    assert seal["eligible_to_open_targets"] is True


def test_rehashed_false_cohort_eligibility_is_rejected(tmp_path: Path) -> None:
    failure_root = tmp_path / "failures"
    for record in selective_case_records(PROTOCOL):
        record_selective_quality_failure(
            PROTOCOL,
            failure_root / str(record["case"]),
            object_id=str(record["object_id"]),
            episode_id=int(record["episode_id"]),
            stage="prediction-prefix-staging",
            error_type="FixtureFailure",
            error_message="target-free fixture failure",
        )
    seal = build_selective_prediction_cohort_seal(
        PROTOCOL,
        tmp_path / "predictions",
        failure_root,
        tmp_path / "cohort.json",
    )
    seal["eligible_to_open_targets"] = True
    unsigned = dict(seal)
    unsigned.pop("result_sha256")
    import hashlib

    seal["result_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="evaluability calculation changed"):
        validate_selective_prediction_cohort_seal(
            seal, protocol_path=PROTOCOL
        )
