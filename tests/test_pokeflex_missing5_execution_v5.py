import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from bayesian_phystwin.pokeflex_action_robust_official18_v4 import (
    SOURCE_MANIFEST_ARTIFACT_KIND,
    build_author_source_manifest,
    load_archived_public13_result,
    load_official18_v4_protocol,
    source_manifest_sha256,
)
from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    CHECKPOINT_SHA256,
    UPSTREAM_COMMIT,
    cd_ul1_mm,
    surface_sample,
)
from bayesian_phystwin.pokeflex_missing5_execution_v5 import (
    INPUT_STAGE_KIND,
    PREDICTION_SEAL_KIND,
    RESULT_KIND,
    TARGET_TAKE_IDS,
    build_prediction_barrier,
    evaluate_result,
    file_sha256,
    input_stage_sha256,
    load_execution_protocol,
    prediction_seal_sha256,
    result_sha256,
    score_one_prediction,
    validate_input_stage,
    validate_prediction_seal,
    validate_result,
    verify_implementation_files,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "configs" / "sota" / "pokeflex_action_robust_official18_v4.json"
COMPLETION_PATH = (
    ROOT / "configs" / "sota" / "pokeflex_missing5_scale_completion_v5.json"
)
EXECUTION_PATH = ROOT / "configs" / "sota" / "pokeflex_missing5_execution_v5.json"
PUBLIC13_PATH = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_action_robust_all18_v4_public13_retrospective"
    / "result.json"
)
RUNNER_PATH = ROOT / "scripts" / "held" / "run_pokeflex_missing5_v5.py"
CURRENT_MAIN_PARITY_PATH = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_missing5_execution_v5"
    / "current_main_public_parity_3dPrintedCylinder_T1.json"
)
CURRENT_MAIN_PARITY_FILE_SHA256 = (
    "83f7a312d739e6974fcc861159c9797a56f7b22a7ae72fb75581a662a5e6ae46"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _protocols() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    parent = load_official18_v4_protocol(PARENT_PATH)
    completion = _load(COMPLETION_PATH)
    execution = load_execution_protocol(EXECUTION_PATH, completion, parent)
    return execution, completion, parent


def _source_manifest(parent: dict[str, object]) -> dict[str, object]:
    manifest = {
        "artifact_kind": SOURCE_MANIFEST_ARTIFACT_KIND,
        "created_before_prediction": True,
        "held_v8_accessed": False,
        "member_payload_decoded": False,
        "protocol_sha256": parent["protocol_sha256"],
        "schema_version": 1,
        "source_manifest_sha256": "",
        "source_root_embedded": False,
        "takes": [
            {
                "camera_panel_sufficient": True,
                "episode_length": 7,
                "evaluator_compatible": True,
                "member_manifest_sha256": f"{index + 10:064x}",
                "mesh_frame_count": 7,
                "mesh_frames_contiguous_from_one": True,
                "official_take_identity_verified": True,
                "required_streams_present": True,
                "source_payload_bytes": 1000 + index,
                "source_payload_name": f"{take_id}.zip",
                "source_payload_sha256": f"{index + 20:064x}",
                "take_id": take_id,
            }
            for index, take_id in enumerate(TARGET_TAKE_IDS)
        ],
        "target_geometry_decoded": False,
        "target_metric_computed": False,
    }
    manifest["source_manifest_sha256"] = source_manifest_sha256(manifest)
    return manifest


def _input_stage(
    take_id: str,
    execution: dict[str, object],
    source_manifest: dict[str, object],
) -> dict[str, object]:
    source = next(row for row in source_manifest["takes"] if row["take_id"] == take_id)
    template = f"{take_id}/meshes/mesh-f00001.obj"
    members = {f"{take_id}/robot_data.json", template}
    members.update(
        f"{take_id}/kinect/{camera}/camera_parameters.json" for camera in (0, 1)
    )
    members.update(
        f"{take_id}/kinect/{camera}/depth/{frame:05d}.png"
        for camera in (0, 1)
        for frame in range(1, 7)
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": INPUT_STAGE_KIND,
        "execution_protocol_sha256": execution["execution_protocol_sha256"],
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "take_id": take_id,
        "object_name": take_id.rpartition("_T")[0],
        "source_archive_name": source["source_payload_name"],
        "source_archive_sha256": source["source_payload_sha256"],
        "source_member_manifest_sha256": source["member_manifest_sha256"],
        "frame_limit": 7,
        "template_frame": 1,
        "authorized_template_member": template,
        "authorized_template_mesh_decoded": True,
        "authorized_template_mesh_decode_count": 1,
        "future_target_mesh_member_decoded_count": 0,
        "target_metric_computed": False,
        "inputs": [
            {
                "archive_member": member,
                "staged_relative_path": member,
                "sha256": f"{index + 100:064x}",
                "byte_count": index + 1,
            }
            for index, member in enumerate(sorted(members))
        ],
        "implementation_revision": "1" * 40,
        "implementation_clean": True,
        "held_v8_accessed": False,
        "input_stage_sha256": "",
    }
    payload["input_stage_sha256"] = input_stage_sha256(payload)
    return payload


def _prediction_seal(
    root: Path,
    take_id: str,
    execution: dict[str, object],
    completion: dict[str, object],
    parent: dict[str, object],
    source_manifest: dict[str, object],
):
    output = root / take_id
    output.mkdir(parents=True)
    stage = _input_stage(take_id, execution, source_manifest)
    stage_path = output / "input_stage_manifest.json"
    stage_path.write_text(json.dumps(stage, sort_keys=True), encoding="utf-8")
    baseline = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 1.0, 0.0], [0.1, 0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    supported = np.asarray([False, True], dtype=np.bool_)
    arrays = {
        "baseline_vertices_m": baseline,
        "global_vertices_m": baseline.copy(),
        "v4_vertices_m": baseline.copy(),
        "v5_vertices_m": baseline.copy(),
        "faces": np.asarray(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64
        ),
        "target_frames": np.asarray([6, 7], dtype=np.int64),
        "source_frames": np.asarray([5, 6], dtype=np.int64),
        "history_start_frames": np.asarray([1, 2], dtype=np.int64),
        "history_end_frames": np.asarray([5, 6], dtype=np.int64),
        "update_supported": supported,
        "update_accepted": supported.copy(),
        "action_supported": supported.copy(),
        "robot_history_supported": np.asarray([True, True], dtype=np.bool_),
        "correction_rms_m": np.asarray([0.0, 0.001], dtype=np.float64),
    }
    arrays["global_vertices_m"][1, :, 0] += 0.01
    arrays["v4_vertices_m"][1, :, 0] += 0.02
    arrays["v5_vertices_m"][1, :, 0] += 0.015
    npz_path = output / "prediction.npz"
    np.savez_compressed(npz_path, **arrays)
    method = execution["method"]
    seal = {
        "schema_version": 1,
        "artifact_kind": PREDICTION_SEAL_KIND,
        "execution_protocol_sha256": execution["execution_protocol_sha256"],
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "take_id": take_id,
        "object_name": take_id.rpartition("_T")[0],
        "implementation_revision": "1" * 40,
        "implementation_clean": True,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "input_stage_manifest": stage_path.name,
        "input_stage_manifest_file_sha256": file_sha256(stage_path),
        "input_stage_sha256": stage["input_stage_sha256"],
        "prediction_npz": npz_path.name,
        "prediction_npz_sha256": file_sha256(npz_path),
        "predicted_frame_count": 2,
        "supported_frame_count": 1,
        "global_fallback_mismatch_count": 0,
        "v4_fallback_mismatch_count": 0,
        "v5_fallback_mismatch_count": 0,
        "global_effective_scale": method["global_effective_scale"],
        "v4_effective_scale": method["v4_effective_scales"][take_id],
        "v5_effective_scale": method["v5_effective_scales"][take_id],
        "template_frame": 1,
        "authorized_template_mesh_read": True,
        "authorized_template_mesh_read_count": 1,
        "future_target_mesh_read": False,
        "future_target_mesh_read_count": 0,
        "future_observation_used": False,
        "target_metric_computed": False,
        "held_v8_accessed": False,
        "seal_sha256": "",
    }
    seal["seal_sha256"] = prediction_seal_sha256(seal)
    seal_path = output / "seal.json"
    seal_path.write_text(json.dumps(seal, sort_keys=True), encoding="utf-8")
    archive = validate_prediction_seal(
        seal_path,
        execution,
        completion,
        parent,
        source_manifest,
    )
    return archive, arrays


def _write_source_zip(path: Path, take_id: str) -> None:
    robot = [
        {
            "frame": frame,
            "forces": [0.0, 0.0 if frame == 1 else 4.0, 0.0],
        }
        for frame in range(1, 8)
    ]
    triangle = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr(f"{take_id}/robot_data.json", json.dumps(robot))
        for frame in range(1, 8):
            archive.writestr(
                f"{take_id}/meshes/mesh-f{frame:05d}.obj",
                triangle,
            )
        archive.writestr(f"{take_id}/mesh_confidence/00001.npy", b"fixture")
        for camera in (0, 1):
            archive.writestr(
                f"{take_id}/kinect/{camera}/camera_parameters.json",
                "{}",
            )
            archive.writestr(f"{take_id}/kinect/{camera}/color/00001.bin", b"c")
            for frame in range(1, 7):
                archive.writestr(
                    f"{take_id}/kinect/{camera}/depth/{frame:05d}.png",
                    b"d",
                )
            archive.writestr(
                f"{take_id}/realsense/{camera}/color/00001.bin",
                b"c",
            )
            archive.writestr(
                f"{take_id}/realsense/{camera}/depth/00001.bin",
                b"d",
            )
            archive.writestr(
                f"{take_id}/volucam/{camera}/color/00001.bin",
                b"c",
            )


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "pokeflex_missing5_v5_runner_test",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_execution_lock_binds_template_boundary_and_implementation() -> None:
    execution, _, _ = _protocols()

    assert (
        execution["prediction_input_boundary"][
            "authorized_template_mesh_count_per_take"
        ]
        == 1
    )
    assert (
        execution["prediction_input_boundary"][
            "future_target_mesh_member_decoding_before_barrier"
        ]
        == "forbidden"
    )
    assert execution["custody"]["required_prediction_count"] == 5
    verify_implementation_files(execution, ROOT)


def test_input_stage_allows_one_template_and_rejects_an_extra_mesh() -> None:
    execution, completion, parent = _protocols()
    source_manifest = _source_manifest(parent)
    stage = _input_stage(TARGET_TAKE_IDS[0], execution, source_manifest)

    validation = validate_input_stage(
        stage,
        execution,
        completion,
        parent,
        source_manifest,
    )
    assert validation["input_count"] == 16

    changed = deepcopy(stage)
    take_id = TARGET_TAKE_IDS[0]
    changed["inputs"].append(
        {
            "archive_member": f"{take_id}/meshes/mesh-f00006.obj",
            "staged_relative_path": f"{take_id}/meshes/mesh-f00006.obj",
            "sha256": "f" * 64,
            "byte_count": 1,
        }
    )
    changed["input_stage_sha256"] = input_stage_sha256(changed)
    with pytest.raises(ValueError, match="inventory|more than one mesh"):
        validate_input_stage(
            changed,
            execution,
            completion,
            parent,
            source_manifest,
        )


def test_stage_command_physically_excludes_future_meshes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution, _, parent = _protocols()
    source_root = tmp_path / "source"
    for take_id in TARGET_TAKE_IDS:
        _write_source_zip(
            source_root / take_id.rpartition("_T")[0] / f"{take_id}.zip",
            take_id,
        )
    source_manifest = build_author_source_manifest(source_root, parent)
    source_manifest_path = tmp_path / "source_manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "_git_clean", lambda _: True)
    monkeypatch.setattr(runner, "_git_revision", lambda _: "1" * 40)
    take_id = TARGET_TAKE_IDS[0]
    stage_dir = tmp_path / "stage"

    runner._stage(
        source_root / take_id.rpartition("_T")[0] / f"{take_id}.zip",
        stage_dir,
        source_manifest_path,
        EXECUTION_PATH,
        COMPLETION_PATH,
        PARENT_PATH,
    )

    staged_meshes = sorted(stage_dir.rglob("*.obj"))
    assert [path.name for path in staged_meshes] == ["mesh-f00001.obj"]
    assert not (stage_dir / take_id / "meshes" / "mesh-f00006.obj").exists()
    stage = _load(stage_dir / "stage.json")
    assert stage["execution_protocol_sha256"] == execution["execution_protocol_sha256"]
    assert stage["future_target_mesh_member_decoded_count"] == 0


def test_prediction_archive_enforces_exact_fallback_and_five_case_barrier(
    tmp_path: Path,
) -> None:
    execution, completion, parent = _protocols()
    source_manifest = _source_manifest(parent)
    archives = [
        _prediction_seal(
            tmp_path,
            take_id,
            execution,
            completion,
            parent,
            source_manifest,
        )[0]
        for take_id in TARGET_TAKE_IDS
    ]

    barrier = build_prediction_barrier(
        [archive.seal_path for archive in archives],
        execution,
        completion,
        parent,
        source_manifest,
    )
    assert barrier["scoring_authorized"] is True
    assert barrier["future_target_mesh_accessed"] is False
    assert barrier["prediction_count"] == 5

    with pytest.raises(ValueError, match="incomplete"):
        build_prediction_barrier(
            [archive.seal_path for archive in archives[:-1]],
            execution,
            completion,
            parent,
            source_manifest,
        )

    first = archives[0]
    with np.load(first.npz_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    arrays["v5_vertices_m"][0, 0, 0] += 1e-6
    np.savez_compressed(first.npz_path, **arrays)
    seal = _load(first.seal_path)
    seal["prediction_npz_sha256"] = file_sha256(first.npz_path)
    seal["seal_sha256"] = prediction_seal_sha256(seal)
    first.seal_path.write_text(json.dumps(seal, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="exact fallback"):
        validate_prediction_seal(
            first.seal_path,
            execution,
            completion,
            parent,
            source_manifest,
        )


def test_split_scorer_matches_registered_surface_metric(tmp_path: Path) -> None:
    execution, completion, parent = _protocols()
    source_manifest = _source_manifest(parent)
    archive, arrays = _prediction_seal(
        tmp_path,
        TARGET_TAKE_IDS[0],
        execution,
        completion,
        parent,
        source_manifest,
    )
    target_vertices = arrays["baseline_vertices_m"][1] + np.asarray([0.005, 0.0, 0.0])
    target_faces = arrays["faces"]

    row = score_one_prediction(
        archive,
        [7],
        lambda _: (target_vertices, target_faces),
        execution,
    )
    count = execution["evaluation"]["surface_sample_count"]
    seed = execution["evaluation"]["surface_sample_seed"] + 7
    expected = cd_ul1_mm(
        surface_sample(arrays["baseline_vertices_m"][1], target_faces, count, seed),
        surface_sample(target_vertices, target_faces, count, seed),
    )
    assert row["frames"][0]["baseline_CD_UL1_mm"] == expected


def test_current_main_public_parity_record_is_complete() -> None:
    execution, _, _ = _protocols()
    assert file_sha256(CURRENT_MAIN_PARITY_PATH) == CURRENT_MAIN_PARITY_FILE_SHA256
    parity = _load(CURRENT_MAIN_PARITY_PATH)

    assert parity["artifact_kind"] == (
        "PokeFlexMissingFiveV5CurrentMainPublicExecutionParity"
    )
    assert (
        parity["current_execution_protocol_sha256"]
        == execution["execution_protocol_sha256"]
    )
    assert parity["all_passed"] is True
    assert parity["take_id"] == "3dPrintedCylinder_T1"
    assert parity["target_cohort_accessed"] is False
    assert parity["held_v8_accessed"] is False

    expected_arrays = {
        "action_supported",
        "baseline_vertices_m",
        "correction_rms_m",
        "faces",
        "global_vertices_m",
        "history_end_frames",
        "history_start_frames",
        "robot_history_supported",
        "source_frames",
        "target_frames",
        "update_accepted",
        "update_supported",
        "v5_vertices_m",
    }
    array_checks = parity["array_checks"]
    assert set(array_checks["byte_identical"]) == expected_arrays
    assert array_checks["count"] == len(expected_arrays)
    assert array_checks["maximum_absolute_difference"] == 0.0

    score_checks = parity["score_checks"]
    assert set(score_checks) == {
        "baseline_CD_UL1_mm",
        "global_CD_UL1_mm",
        "v4_CD_UL1_mm",
        "v5_CD_UL1_mm",
    }
    for check in score_checks.values():
        assert check == {
            "all_exactly_equal": True,
            "frame_count": 97,
            "maximum_absolute_difference_mm": 0.0,
        }


def test_result_evaluation_treats_public13_v5_as_unchanged_v4() -> None:
    execution, completion, parent = _protocols()
    public13 = load_archived_public13_result(PUBLIC13_PATH, parent)
    objects = []
    for index, take_id in enumerate(TARGET_TAKE_IDS):
        v4 = 4.0 + index
        v5 = v4 - 0.1
        objects.append(
            {
                "take_id": take_id,
                "object_name": take_id.rpartition("_T")[0],
                "v4_mean_CD_UL1_mm": v4,
                "v5_mean_CD_UL1_mm": v5,
                "frames": [
                    {
                        "target_frame": 6,
                        "v4_CD_UL1_mm": v4,
                        "v5_CD_UL1_mm": v5,
                    }
                ],
            }
        )

    aggregate = evaluate_result(objects, public13, execution, completion)

    assert aggregate["prospective_v5_vs_v4_win_count"] == 5
    assert aggregate["official_take_count"] == 18
    assert (
        aggregate["official_scored_frame_count"]
        == sum(len(row["frames"]) for row in public13["objects"]) + 5
    )
    assert (
        aggregate["official18_v5_frame_balanced_CD_UL1_mm"]
        < aggregate["official18_v4_frame_balanced_CD_UL1_mm"]
    )


def test_result_validator_recomputes_gates_and_rejects_resigned_mutation(
    tmp_path: Path,
) -> None:
    execution, completion, parent = _protocols()
    source_manifest = _source_manifest(parent)
    archives = [
        _prediction_seal(
            tmp_path,
            take_id,
            execution,
            completion,
            parent,
            source_manifest,
        )[0]
        for take_id in TARGET_TAKE_IDS
    ]
    barrier = build_prediction_barrier(
        [archive.seal_path for archive in archives],
        execution,
        completion,
        parent,
        source_manifest,
    )
    public13 = load_archived_public13_result(PUBLIC13_PATH, parent)
    objects = []
    for index, take_id in enumerate(TARGET_TAKE_IDS):
        baseline = 5.0 + index
        global_score = baseline - 0.05
        v4 = baseline - 0.10
        v5 = baseline - 0.20
        objects.append(
            {
                "take_id": take_id,
                "object_name": take_id.rpartition("_T")[0],
                "scored_frame_count": 1,
                "supported_frame_count": 1,
                "baseline_mean_CD_UL1_mm": baseline,
                "global_mean_CD_UL1_mm": global_score,
                "v4_mean_CD_UL1_mm": v4,
                "v5_mean_CD_UL1_mm": v5,
                "frames": [
                    {
                        "target_frame": 6,
                        "update_supported": True,
                        "baseline_CD_UL1_mm": baseline,
                        "global_CD_UL1_mm": global_score,
                        "v4_CD_UL1_mm": v4,
                        "v5_CD_UL1_mm": v5,
                    }
                ],
                "target_meshes": [
                    {
                        "target_frame": 6,
                        "archive_member": f"{take_id}/meshes/mesh-f00006.obj",
                        "sha256": f"{index + 200:064x}",
                        "byte_count": 1,
                    }
                ],
            }
        )
    result = {
        "schema_version": 1,
        "artifact_kind": RESULT_KIND,
        "execution_protocol_sha256": execution["execution_protocol_sha256"],
        "completion_protocol_sha256": completion["protocol_sha256"],
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "prediction_barrier_sha256": barrier["prediction_barrier_sha256"],
        "prediction_barrier_passed": True,
        "target_mesh_access_before_barrier": False,
        "target_meshes_opened_after_complete_barrier": True,
        "future_observation_used_for_prediction": False,
        "parameter_selection_from_this_cohort": False,
        "replacement_used": False,
        "target_adaptation_used": False,
        "objects": objects,
        "aggregate": evaluate_result(objects, public13, execution, completion),
        "held_v8_accessed": False,
        "result_sha256": "",
    }
    result["result_sha256"] = result_sha256(result)
    validation = validate_result(
        result,
        public13,
        execution,
        completion,
        barrier,
        source_manifest,
    )
    assert validation["passed"] is True

    changed = deepcopy(result)
    changed["aggregate"]["prospective_v5_vs_v4_win_count"] = 0
    changed["result_sha256"] = result_sha256(changed)
    with pytest.raises(ValueError, match="aggregate"):
        validate_result(
            changed,
            public13,
            execution,
            completion,
            barrier,
            source_manifest,
        )
