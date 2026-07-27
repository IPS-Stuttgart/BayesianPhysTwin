import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_dynamic_tapnextpp_artifacts import (
    PREDICTION_ARCHIVE_FILENAME,
    PREDICTION_SEAL_FILENAME,
    TECHNICAL_FAILURE_FILENAME,
    authorize_source_scoring,
    build_prediction_seal,
    build_source_admission,
    build_source_barrier,
    deform360_case_hash,
    record_technical_failure,
    validate_prediction_seal,
    validate_source_admission,
)
from bayesian_phystwin.deform360_object_exclusion import (
    file_sha256,
    load_object_exclusion_manifest,
)
from bayesian_phystwin.observation_belief import save_observation_belief
from bayesian_phystwin.tapnextpp_dynamic_multiview import (
    build_dynamic_tapnextpp_observation_belief,
    fuse_dynamic_tapnextpp_multiview,
)
from tests.test_tapnextpp_dynamic_multiview import _synthetic_input

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _protocol(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "deform360-dynamic-tapnextpp-provider-v1",
            }
        ),
        encoding="utf-8",
    )


def _camera_records(count: int = 8) -> list[dict[str, object]]:
    return [
        {
            "camera_name": f"camera-{index}",
            "rgb_frame_count": 76,
            "depth_frame_count": 76,
            "mask_frame_count": 76,
            "calibration_valid": True,
            "frame_zero_projected_support_count": 32,
        }
        for index in range(count)
    ]


def _admission(
    path: Path,
    *,
    bimanual: str = "yes",
    physical_node_count: int = 128,
    camera_count: int = 8,
) -> dict[str, object]:
    return build_source_admission(
        path,
        object_id="fresh-object",
        episode_id=2,
        category="cloth",
        bimanual=bimanual,
        episode_frame_count=76,
        robot_frame_count=76,
        physical_node_count=physical_node_count,
        camera_records=_camera_records(camera_count),
        source_sha256={
            "metadata": "a" * 64,
            "robot": "b" * 64,
            "physical_geometry": "c" * 64,
        },
    )


def test_admission_rejects_manifest_enum_typo(tmp_path: Path) -> None:
    admission = _admission(
        tmp_path / "admission.json",
        bimanual="yess",
    )
    assert admission["admitted"] is False
    assert "invalid-bimanual-enum" in admission["rejection_reasons"]


def test_protocol_uses_amended_hash_only_exclusion() -> None:
    protocol_path = (
        REPOSITORY_ROOT
        / "configs"
        / "sota"
        / "deform360_dynamic_tapnextpp_provider_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    boundary = protocol["fresh_object_boundary"]
    exclusion_path = REPOSITORY_ROOT / boundary["exclusion_artifact"]
    exclusion = load_object_exclusion_manifest(exclusion_path)

    assert len(exclusion["object_hashes"]) == 94
    assert exclusion["exclusion_sha256"] == boundary["exclusion_sha256"]
    assert file_sha256(exclusion_path) == boundary["exclusion_file_sha256"]
    assert protocol["amendments"][0]["cohort_selected_before_amendment"] is False


def test_admission_rejects_54_nodes_before_backend_runtime(tmp_path: Path) -> None:
    admission = _admission(
        tmp_path / "admission.json",
        physical_node_count=54,
    )
    assert admission["admitted"] is False
    assert "physical-backend-node-count" in admission["rejection_reasons"]


def test_admission_rejects_insufficient_camera_panel(tmp_path: Path) -> None:
    admission = _admission(
        tmp_path / "admission.json",
        camera_count=7,
    )
    assert admission["admitted"] is False
    assert "insufficient-eligible-camera-panel" in admission["rejection_reasons"]


def test_admission_is_hash_only_and_tamper_evident(tmp_path: Path) -> None:
    path = tmp_path / "admission.json"
    admission = _admission(path)
    assert admission["admitted"] is True
    encoded = path.read_text(encoding="utf-8")
    assert "fresh-object" not in encoded
    assert "episode_id" not in encoded
    validate_source_admission(path)
    admission["category"] = "changed"
    with pytest.raises(ValueError, match="checksum changed"):
        validate_source_admission(admission)


def _prediction_inputs(tmp_path: Path) -> dict[str, Path | str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol = tmp_path / "protocol.json"
    _protocol(protocol)
    admission_path = tmp_path / "admission.json"
    admission = _admission(admission_path)
    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps(
            {
                "protocol_id": "deform360-dynamic-tapnextpp-provider-v1",
                "case_hash": admission["case_hash"],
            }
        ),
        encoding="utf-8",
    )
    fused = fuse_dynamic_tapnextpp_multiview(**_synthetic_input(3))
    belief = build_dynamic_tapnextpp_observation_belief(
        fused,
        case_id=str(admission["case_hash"]),
        frame_ids=np.asarray([0, 1, 2, 3]),
        entity_ids=np.asarray([7]),
        entity_birth_frames=np.asarray([0]),
        entity_update_frames=np.asarray([3]),
        camera_names=("camera-0", "camera-1", "camera-2"),
        query_schedule_sha256="d" * 64,
    )
    belief_path = tmp_path / "belief.npz"
    save_observation_belief(belief_path, belief)
    frame_zero = np.column_stack(
        (
            np.linspace(0.0, 0.1, 12),
            np.zeros(12),
            np.ones(12),
        )
    )
    baseline = np.repeat(frame_zero[None], 76, axis=0)
    candidate = baseline.copy()
    candidate[:, :, 0] += np.linspace(0.0, 0.01, 76)[:, None]
    archive = tmp_path / "prediction.npz"
    np.savez_compressed(
        archive,
        baseline_prediction_m=baseline,
        candidate_prediction_m=candidate,
        persistence_prediction_m=baseline,
        measurement_entity_ids=np.asarray([7, 8]),
        hidden_entity_ids=np.asarray([0, 1, 2, 3]),
        update_frames=np.asarray([19, 38, 57]),
    )
    return {
        "protocol_path": protocol,
        "admission_path": admission_path,
        "query_schedule_path": schedule,
        "observation_belief_path": belief_path,
        "prediction_archive_path": archive,
        "code_revision": "abcdef123456",
        "environment_sha256": "e" * 64,
    }


def test_prediction_seal_binds_disjoint_prediction_before_scoring(
    tmp_path: Path,
) -> None:
    inputs = _prediction_inputs(tmp_path)
    output = tmp_path / "sealed"
    seal = build_prediction_seal(output, **inputs)
    validate_prediction_seal(
        output / PREDICTION_SEAL_FILENAME,
        protocol_path=inputs["protocol_path"],
        admission_path=inputs["admission_path"],
        query_schedule_path=inputs["query_schedule_path"],
        observation_belief_path=inputs["observation_belief_path"],
        prediction_dir=output,
    )
    assert seal["information_boundary"]["future_identity_read"] is False
    assert seal["prediction_archive"]["hidden_identity_count"] == 4
    archive = output / PREDICTION_ARCHIVE_FILENAME
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive checksum changed"):
        validate_prediction_seal(
            seal,
            protocol_path=inputs["protocol_path"],
            admission_path=inputs["admission_path"],
            query_schedule_path=inputs["query_schedule_path"],
            observation_belief_path=inputs["observation_belief_path"],
            prediction_dir=output,
        )


def test_prediction_archive_rejects_identity_leakage(tmp_path: Path) -> None:
    inputs = _prediction_inputs(tmp_path)
    archive = Path(inputs["prediction_archive_path"])
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    arrays["hidden_entity_ids"] = np.asarray([2, 7])
    np.savez_compressed(archive, **arrays)
    with pytest.raises(ValueError, match="overlap"):
        build_prediction_seal(tmp_path / "sealed", **inputs)


def test_barrier_counts_predictions_failures_and_missing_separately(
    tmp_path: Path,
) -> None:
    first_inputs = _prediction_inputs(tmp_path / "first")
    first_output = tmp_path / "first-seal"
    first_output.parent.mkdir(parents=True, exist_ok=True)
    build_prediction_seal(first_output, **first_inputs)

    second_root = tmp_path / "second"
    second_root.mkdir()
    protocol = second_root / "protocol.json"
    _protocol(protocol)
    admission_path = second_root / "admission.json"
    second_admission = build_source_admission(
        admission_path,
        object_id="second-object",
        episode_id=1,
        category="rope",
        bimanual="no",
        episode_frame_count=76,
        robot_frame_count=76,
        physical_node_count=128,
        camera_records=_camera_records(),
        source_sha256={
            "metadata": "1" * 64,
            "robot": "2" * 64,
            "physical_geometry": "3" * 64,
        },
    )
    failure_dir = second_root / "failure"
    record_technical_failure(
        failure_dir,
        protocol_path=protocol,
        admission_path=admission_path,
        stage="tapnextpp-runtime",
        reason_code="runtime-failure",
        code_revision="abcdef123456",
    )
    first_admission = validate_source_admission(
        first_inputs["admission_path"]
    )
    third_hash = deform360_case_hash("third-object", 0)
    barrier_path = tmp_path / "barrier.json"
    barrier = build_source_barrier(
        barrier_path,
        expected_case_hashes=[
            first_admission["case_hash"],
            second_admission["case_hash"],
            third_hash,
        ],
        prediction_seals=[first_output / PREDICTION_SEAL_FILENAME],
        technical_failures=[
            failure_dir / TECHNICAL_FAILURE_FILENAME
        ],
    )
    assert barrier["counts"] == {
        "expected": 3,
        "ordinary_predictions": 1,
        "retained_technical_failures": 1,
        "missing": 1,
    }
    with pytest.raises(ValueError, match="incomplete"):
        authorize_source_scoring(barrier_path)

    complete = build_source_barrier(
        barrier_path,
        expected_case_hashes=[
            first_admission["case_hash"],
            second_admission["case_hash"],
        ],
        prediction_seals=[first_output / PREDICTION_SEAL_FILENAME],
        technical_failures=[
            failure_dir / TECHNICAL_FAILURE_FILENAME
        ],
    )
    assert complete["complete"] is True
    authorize_source_scoring(barrier_path)


def test_public_artifact_builders_accept_no_scoring_input() -> None:
    for function in (
        build_source_admission,
        build_prediction_seal,
        record_technical_failure,
        build_source_barrier,
    ):
        parameters = inspect.signature(function).parameters
        assert "target" not in parameters
        assert "outcome" not in parameters
