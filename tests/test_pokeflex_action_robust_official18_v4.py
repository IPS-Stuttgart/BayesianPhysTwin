import json
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import pytest

from bayesian_phystwin.pokeflex_action_robust_official18_v4 import (
    COMPLETION_ARTIFACT_KIND,
    EXPECTED_PROTOCOL_SHA256,
    PREDICTION_BARRIER_ARTIFACT_KIND,
    SOURCE_MANIFEST_ARTIFACT_KIND,
    build_author_source_manifest,
    completion_sha256,
    evaluate_official18_v4,
    load_archived_public13_result,
    load_official18_v4_protocol,
    prediction_barrier_sha256,
    protocol_sha256,
    source_manifest_sha256,
    validate_author_source_manifest,
    validate_official18_v4_protocol,
    validate_prediction_barrier,
    validate_prospective_completion,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "sota" / "pokeflex_action_robust_official18_v4.json"
PUBLIC13_PATH = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_action_robust_all18_v4_public13_retrospective"
    / "result.json"
)
MISSING_TAKES = (
    "Pillow_T8",
    "3dPrintedCylinder_T7",
    "3dPrintedHeart_T14",
    "Sponge_T10",
    "3dPrintedPizza_T13",
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_completion(
    protocol: dict[str, object],
    public13: dict[str, object],
    source_manifest: dict[str, object],
    prediction_barrier: dict[str, object],
) -> dict[str, object]:
    public_frames = [frame for row in public13["objects"] for frame in row["frames"]]
    public_baseline_sum = sum(frame["baseline_CD_UL1_mm"] for frame in public_frames)
    prospective_frame_count = 5 * 100
    baseline = (
        6.498 * (len(public_frames) + prospective_frame_count) - public_baseline_sum
    ) / prospective_frame_count
    objects = []
    for take_id in MISSING_TAKES:
        frames = [
            {
                "baseline_CD_UL1_mm": baseline,
                "global_CD_UL1_mm": baseline - 0.05,
                "target_frame": frame,
                "update_supported": True,
                "v4_all18_CD_UL1_mm": baseline - 0.10,
            }
            for frame in range(6, 106)
        ]
        objects.append(
            {
                "baseline_mean_CD_UL1_mm": baseline,
                "frames": frames,
                "global_mean_CD_UL1_mm": baseline - 0.05,
                "object_name": take_id.rpartition("_T")[0],
                "scored_frame_count": len(frames),
                "supported_frame_count": len(frames),
                "take_id": take_id,
                "v4_all18_mean_CD_UL1_mm": baseline - 0.10,
            }
        )
    completion = {
        "artifact_kind": COMPLETION_ARTIFACT_KIND,
        "completion_sha256": "",
        "future_observation_used_for_prediction": False,
        "held_v8_accessed": False,
        "objects": objects,
        "parameter_selection_from_this_cohort": False,
        "prediction_barrier_passed": True,
        "prediction_barrier_sha256": prediction_barrier["prediction_barrier_sha256"],
        "protocol_sha256": protocol["protocol_sha256"],
        "replacement_used": False,
        "schema_version": 1,
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "target_mesh_access_before_barrier": False,
    }
    completion["completion_sha256"] = completion_sha256(completion)
    return completion


def _synthetic_source_manifest(protocol: dict[str, object]) -> dict[str, object]:
    manifest = {
        "artifact_kind": SOURCE_MANIFEST_ARTIFACT_KIND,
        "created_before_prediction": True,
        "held_v8_accessed": False,
        "member_payload_decoded": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "schema_version": 1,
        "source_manifest_sha256": "",
        "source_root_embedded": False,
        "takes": [
            {
                "camera_panel_sufficient": True,
                "episode_length": 120,
                "evaluator_compatible": True,
                "member_manifest_sha256": f"{index + 10:064x}",
                "mesh_frame_count": 120,
                "mesh_frames_contiguous_from_one": True,
                "official_take_identity_verified": True,
                "required_streams_present": True,
                "source_payload_bytes": 1000 + index,
                "source_payload_name": f"{take_id}.zip",
                "source_payload_sha256": f"{index + 20:064x}",
                "take_id": take_id,
            }
            for index, take_id in enumerate(MISSING_TAKES)
        ],
        "target_geometry_decoded": False,
        "target_metric_computed": False,
    }
    manifest["source_manifest_sha256"] = source_manifest_sha256(manifest)
    return manifest


def _write_source_zip(
    root: Path,
    take_id: str,
    *,
    omit_prefix: str | None = None,
) -> None:
    object_name = take_id.rpartition("_T")[0]
    path = root / object_name / f"{take_id}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    member_names = [f"{take_id}/robot_data.json"]
    member_names.extend(
        f"{take_id}/meshes/mesh-f{frame:05d}.obj" for frame in range(1, 8)
    )
    member_names.append(f"{take_id}/mesh_confidence/00001.npy")
    member_names.extend(
        f"{take_id}/{sensor}/{camera}/{modality}/00001.bin"
        for sensor, modalities in (
            ("kinect", ("color", "depth")),
            ("realsense", ("color", "depth")),
            ("volucam", ("color",)),
        )
        for camera in (0, 1)
        for modality in modalities
    )
    with ZipFile(path, "w") as archive:
        for name in member_names:
            if omit_prefix is None or not name.startswith(f"{take_id}/{omit_prefix}"):
                archive.writestr(name, f"fixture:{name}".encode())


def _synthetic_prediction_barrier(
    protocol: dict[str, object],
    source_manifest: dict[str, object],
) -> dict[str, object]:
    revision = "1" * 40
    barrier = {
        "artifact_kind": PREDICTION_BARRIER_ARTIFACT_KIND,
        "held_v8_accessed": False,
        "implementation_checkout_clean": True,
        "implementation_revision": revision,
        "passed": True,
        "prediction_barrier_sha256": "",
        "predictions": [
            {
                "future_observation_used": False,
                "implementation_revision": revision,
                "prediction_file_sha256": f"{index + 30:064x}",
                "seal_sha256": f"{index + 40:064x}",
                "status": "prediction_success",
                "take_id": take_id,
            }
            for index, take_id in enumerate(MISSING_TAKES)
        ],
        "protocol_sha256": protocol["protocol_sha256"],
        "schema_version": 1,
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "target_mesh_accessed": False,
    }
    barrier["prediction_barrier_sha256"] = prediction_barrier_sha256(barrier)
    return barrier


def test_official18_v4_protocol_is_exact_and_source_bound() -> None:
    protocol = load_official18_v4_protocol(PROTOCOL_PATH)

    assert protocol_sha256(protocol) == EXPECTED_PROTOCOL_SHA256
    assert protocol["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256
    assert protocol["target_cohort"]["prospective_take_ids"] == list(MISSING_TAKES)
    assert protocol["target_cohort"]["replacement_allowed"] is False
    assert protocol["method"]["target_outcome_adaptation"] == "forbidden"
    assert protocol["custody"]["required_prospective_prediction_seal_count"] == 5


def test_protocol_rejects_self_consistent_post_lock_mutation() -> None:
    protocol = _load(PROTOCOL_PATH)
    protocol["gates"]["baseline_reproduction"]["absolute_tolerance_mm"] = 0.01
    protocol["protocol_sha256"] = protocol_sha256(protocol)

    with pytest.raises(ValueError, match="registered protocol changed"):
        validate_official18_v4_protocol(protocol)


def test_archived_public13_component_is_byte_bound() -> None:
    protocol = load_official18_v4_protocol(PROTOCOL_PATH)
    result = load_archived_public13_result(PUBLIC13_PATH, protocol)

    assert len(result["objects"]) == 13
    assert result["parameter_selection_from_this_cohort"] is False
    assert result["future_or_missing_official_takes_accessed"] is False
    assert result["held_v8_accessed"] is False


def test_author_source_builder_uses_opaque_zip_inventory(tmp_path: Path) -> None:
    protocol = load_official18_v4_protocol(PROTOCOL_PATH)
    for take_id in MISSING_TAKES:
        _write_source_zip(tmp_path, take_id)

    manifest = build_author_source_manifest(tmp_path, protocol)
    validation = validate_author_source_manifest(manifest, protocol)

    assert validation["take_count"] == 5
    assert manifest["member_payload_decoded"] is False
    assert manifest["target_geometry_decoded"] is False
    assert manifest["source_root_embedded"] is False
    assert all(row["mesh_frame_count"] == 7 for row in manifest["takes"])
    assert all(row["episode_length"] == 7 for row in manifest["takes"])
    assert all("/" not in row["source_payload_name"] for row in manifest["takes"])


def test_author_source_builder_rejects_incomplete_camera_panel(
    tmp_path: Path,
) -> None:
    protocol = load_official18_v4_protocol(PROTOCOL_PATH)
    for take_id in MISSING_TAKES:
        omitted = "realsense/1/depth/" if take_id == MISSING_TAKES[0] else None
        _write_source_zip(tmp_path, take_id, omit_prefix=omitted)

    with pytest.raises(ValueError, match="required stream is missing"):
        build_author_source_manifest(tmp_path, protocol)


def test_prospective_completion_enforces_custody_and_exact_cohort() -> None:
    protocol = load_official18_v4_protocol(PROTOCOL_PATH)
    public13 = load_archived_public13_result(PUBLIC13_PATH, protocol)
    source_manifest = _synthetic_source_manifest(protocol)
    prediction_barrier = _synthetic_prediction_barrier(protocol, source_manifest)
    completion = _synthetic_completion(
        protocol,
        public13,
        source_manifest,
        prediction_barrier,
    )

    source_validation = validate_author_source_manifest(source_manifest, protocol)
    barrier_validation = validate_prediction_barrier(
        prediction_barrier,
        protocol,
        source_manifest,
    )
    validation = validate_prospective_completion(
        completion,
        protocol,
        source_manifest,
        prediction_barrier,
    )

    assert source_validation["take_count"] == 5
    assert barrier_validation["prediction_count"] == 5
    assert validation["prospective_take_count"] == 5
    assert validation["completion_sha256"] == completion["completion_sha256"]

    decoded = deepcopy(source_manifest)
    decoded["target_geometry_decoded"] = True
    decoded["source_manifest_sha256"] = source_manifest_sha256(decoded)
    with pytest.raises(ValueError, match="target geometry was decoded"):
        validate_author_source_manifest(decoded, protocol)

    future_barrier = deepcopy(prediction_barrier)
    future_barrier["predictions"][0]["future_observation_used"] = True
    future_barrier["prediction_barrier_sha256"] = prediction_barrier_sha256(
        future_barrier
    )
    with pytest.raises(ValueError, match="future observation leaked"):
        validate_prediction_barrier(future_barrier, protocol, source_manifest)

    leaked = deepcopy(completion)
    leaked["future_observation_used_for_prediction"] = True
    leaked["completion_sha256"] = completion_sha256(leaked)
    with pytest.raises(ValueError, match="future observation leaked"):
        validate_prospective_completion(
            leaked,
            protocol,
            source_manifest,
            prediction_barrier,
        )

    replaced = deepcopy(completion)
    replaced["replacement_used"] = True
    replaced["completion_sha256"] = completion_sha256(replaced)
    with pytest.raises(ValueError, match="replacement was used"):
        validate_prospective_completion(
            replaced,
            protocol,
            source_manifest,
            prediction_barrier,
        )


def test_official18_combiner_separates_direct_and_prospective_claims() -> None:
    protocol = load_official18_v4_protocol(PROTOCOL_PATH)
    public13 = load_archived_public13_result(PUBLIC13_PATH, protocol)
    source_manifest = _synthetic_source_manifest(protocol)
    prediction_barrier = _synthetic_prediction_barrier(protocol, source_manifest)
    completion = _synthetic_completion(
        protocol,
        public13,
        source_manifest,
        prediction_barrier,
    )

    result = evaluate_official18_v4(
        PUBLIC13_PATH,
        source_manifest,
        prediction_barrier,
        completion,
        protocol,
    )

    assert result["official_take_count"] == 18
    assert result["prospective_take_count"] == 5
    assert result["baseline_frame_balanced_CD_UL1_mm"] == pytest.approx(6.498)
    assert result["v4_frame_balanced_CD_UL1_mm"] < 6.498
    assert result["baseline_reproduction_passed"] is True
    assert result["direct_full18_benchmark_gate_passed"] is True
    assert result["prospective_missing5_transfer_gate_passed"] is True
    assert result["published_full_split_improvement_authorized"] is True
    assert result["fully_prospective_official18_claim_authorized"] is False


def test_prospective_regression_prevents_full_split_authorization() -> None:
    protocol = load_official18_v4_protocol(PROTOCOL_PATH)
    public13 = load_archived_public13_result(PUBLIC13_PATH, protocol)
    source_manifest = _synthetic_source_manifest(protocol)
    prediction_barrier = _synthetic_prediction_barrier(protocol, source_manifest)
    completion = _synthetic_completion(
        protocol,
        public13,
        source_manifest,
        prediction_barrier,
    )
    for row in completion["objects"]:
        for frame in row["frames"]:
            frame["v4_all18_CD_UL1_mm"] = frame["baseline_CD_UL1_mm"] + 0.10
        row["v4_all18_mean_CD_UL1_mm"] = row["baseline_mean_CD_UL1_mm"] + 0.10
    completion["completion_sha256"] = completion_sha256(completion)

    result = evaluate_official18_v4(
        PUBLIC13_PATH,
        source_manifest,
        prediction_barrier,
        completion,
        protocol,
    )

    assert result["prospective_missing5_transfer_gate_passed"] is False
    assert result["published_full_split_improvement_authorized"] is False
