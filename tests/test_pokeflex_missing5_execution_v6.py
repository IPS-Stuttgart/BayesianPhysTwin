from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bayesian_phystwin.pokeflex_action_robust_official18_v4 import (
    load_archived_public13_result,
    load_official18_v4_protocol,
)
from bayesian_phystwin.pokeflex_missing5_execution_v5 import (
    TARGET_TAKE_IDS,
    file_sha256,
)
from bayesian_phystwin.pokeflex_missing5_execution_v5 import (
    load_execution_protocol as load_v5_execution_protocol,
)
from bayesian_phystwin.pokeflex_missing5_execution_v6 import (
    IMPLEMENTATION_FILE_PATHS,
    PREDICTION_SEAL_KIND,
    RESULT_KIND,
    apply_causal_scale_sequence,
    build_execution_protocol,
    build_prediction_barrier,
    evaluate_result,
    execution_protocol_sha256,
    load_execution_protocol,
    prediction_seal_sha256,
    result_sha256,
    score_one_prediction,
    validate_execution_protocol,
    validate_prediction_barrier,
    validate_prediction_seal,
    validate_result,
    verify_implementation_files,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "configs" / "sota" / "pokeflex_action_robust_official18_v4.json"
COMPLETION_PATH = (
    ROOT / "configs" / "sota" / "pokeflex_missing5_scale_completion_v5.json"
)
V5_EXECUTION_PATH = ROOT / "configs" / "sota" / "pokeflex_missing5_execution_v5.json"
EXECUTION_PATH = ROOT / "configs" / "sota" / "pokeflex_missing5_execution_v6.json"
MODEL_PATH = ROOT / "configs" / "sota" / "pokeflex_missing5_causal_scale_v6.json"
SOURCE_RESULT_PATH = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_missing5_causal_scale_v6"
    / "source_result.json"
)
PUBLIC13_PATH = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_action_robust_all18_v4_public13_retrospective"
    / "result.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    parent = load_official18_v4_protocol(PARENT_PATH)
    completion = _load(COMPLETION_PATH)
    v5_execution = load_v5_execution_protocol(
        V5_EXECUTION_PATH,
        completion,
        parent,
    )
    return (
        v5_execution,
        completion,
        parent,
        _load(MODEL_PATH),
        _load(SOURCE_RESULT_PATH),
    )


def _execution() -> dict[str, Any]:
    v5_execution, completion, parent, model, source_result = _inputs()
    return build_execution_protocol(
        v5_execution,
        completion,
        parent,
        model,
        source_result,
        locked_at_utc="2026-08-10T00:00:00Z",
        model_file_sha256=file_sha256(MODEL_PATH),
        source_result_file_sha256=file_sha256(SOURCE_RESULT_PATH),
        implementation_file_sha256s={
            path: f"{index + 1:064x}"
            for index, path in enumerate(IMPLEMENTATION_FILE_PATHS)
        },
    )


def _registered_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    v5_execution, completion, parent, model, source_result = _inputs()
    execution, _, _ = load_execution_protocol(
        EXECUTION_PATH,
        v5_execution,
        completion,
        parent,
        MODEL_PATH,
        SOURCE_RESULT_PATH,
    )
    return execution, v5_execution, completion, parent, model, source_result


def _v5_test_helpers() -> Any:
    path = ROOT / "tests" / "test_pokeflex_missing5_execution_v5.py"
    spec = importlib.util.spec_from_file_location("pokeflex_v5_test_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v6_prediction_seal(
    root: Path,
    parent_archive: Any,
    model: dict[str, Any],
    execution: dict[str, Any],
    source_manifest: dict[str, Any],
) -> Path:
    output = root / parent_archive.take_id
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = [
        {
            "target_frame": 6,
            "accepted": False,
            "rms_update_m": 0.0,
            "prior_motion_rms_m": 0.0,
            "correction_to_prior_motion_ratio": 0.0,
            "correction_prior_motion_cosine": None,
        },
        {
            "target_frame": 7,
            "accepted": True,
            "rms_update_m": 1e-12,
            "prior_motion_rms_m": 1e-12,
            "correction_to_prior_motion_ratio": 1e-12,
            "correction_prior_motion_cosine": 0.0,
        },
    ]
    arrays, decisions = apply_causal_scale_sequence(
        model,
        object_name=parent_archive.take_id.rpartition("_T")[0],
        baseline_vertices_m=parent_archive.baseline_vertices_m,
        v5_vertices_m=parent_archive.v5_vertices_m,
        target_frames=parent_archive.target_frames,
        update_supported=parent_archive.update_supported,
        update_rows=rows,
    )
    npz_path = output / "prediction.npz"
    np.savez_compressed(
        npz_path,
        v6_vertices_m=arrays["v6_vertices_m"],
        target_frames=arrays["target_frames"],
        selected_scale=arrays["selected_scale"],
        candidate_admitted=arrays["candidate_admitted"],
        predicted_lower_gain_mm=arrays["predicted_lower_gain_mm"],
        minimum_source_distance=arrays["minimum_source_distance"],
        support_radius=arrays["support_radius"],
    )
    parent_seal = json.loads(parent_archive.seal_path.read_text(encoding="utf-8"))
    seal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PREDICTION_SEAL_KIND,
        "execution_protocol_sha256": execution["execution_protocol_sha256"],
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "take_id": parent_archive.take_id,
        "object_name": parent_archive.take_id.rpartition("_T")[0],
        "implementation_revision": "1" * 40,
        "implementation_clean": True,
        "parent_v5_execution_protocol_sha256": execution[
            "parent_v5_execution_protocol_sha256"
        ],
        "parent_v5_seal_sha256": parent_seal["seal_sha256"],
        "parent_v5_seal_file_sha256": file_sha256(parent_archive.seal_path),
        "parent_v5_prediction_npz_sha256": file_sha256(parent_archive.npz_path),
        "causal_scale_model_sha256": execution["causal_scale_model_sha256"],
        "causal_scale_model_file_sha256": file_sha256(MODEL_PATH),
        "source_result_sha256": execution["source_result_sha256"],
        "source_result_file_sha256": file_sha256(SOURCE_RESULT_PATH),
        "input_stage_sha256": parent_seal["input_stage_sha256"],
        "prediction_npz": npz_path.name,
        "prediction_npz_sha256": file_sha256(npz_path),
        "predicted_frame_count": 2,
        "candidate_admission_count": int(np.sum(arrays["candidate_admitted"])),
        "unsupported_fallback_mismatch_count": 0,
        "rejected_fallback_mismatch_count": 0,
        "future_observation_used": False,
        "future_target_mesh_read": False,
        "future_target_mesh_read_count": 0,
        "target_metric_computed": False,
        "decisions": decisions,
        "held_v8_accessed": False,
        "seal_sha256": "",
    }
    seal["seal_sha256"] = prediction_seal_sha256(seal)
    seal_path = output / "seal.json"
    seal_path.write_text(json.dumps(seal, sort_keys=True), encoding="utf-8")
    return seal_path


def test_execution_lock_binds_parent_model_and_exact_fallbacks() -> None:
    v5_execution, completion, parent, model, source_result = _inputs()
    execution = _execution()

    validation = validate_execution_protocol(
        execution,
        v5_execution,
        completion,
        parent,
        model,
        source_result,
        bind_registered_digest=False,
    )

    assert validation["passed"]
    assert (
        execution["parent_v5_execution_protocol_sha256"]
        == (v5_execution["execution_protocol_sha256"])
    )
    assert execution["causal_scale_model_sha256"] == model["model_sha256"]
    assert execution["source_result_sha256"] == source_result["result_sha256"]
    assert execution["method"]["rejected_frame_action"] == (
        "byte-identical V5 prediction"
    )
    assert execution["method"]["unsupported_frame_action"] == (
        "byte-identical released checkpoint"
    )
    assert execution["official_target_outcomes_used_to_build_protocol"] is False


def test_registered_execution_lock_and_implementation_files_validate() -> None:
    v5_execution, completion, parent, _, _ = _inputs()
    execution, model, source_result = load_execution_protocol(
        EXECUTION_PATH,
        v5_execution,
        completion,
        parent,
        MODEL_PATH,
        SOURCE_RESULT_PATH,
    )

    verify_implementation_files(execution, ROOT)
    assert execution["execution_protocol_sha256"] == (
        "e875495295acc8de4b7da70cdcaff9947838b8e92f1785dbbe32f9fc0c67e78b"
    )
    assert model["model_sha256"] == execution["causal_scale_model_sha256"]
    assert source_result["result_sha256"] == execution["source_result_sha256"]


def test_v6_seal_reproduces_decisions_and_rejects_resigned_mutation(
    tmp_path: Path,
) -> None:
    execution, v5_execution, completion, parent, model, source_result = (
        _registered_inputs()
    )
    helpers = _v5_test_helpers()
    source_manifest = helpers._source_manifest(parent)
    parent_archive, _ = helpers._prediction_seal(
        tmp_path / "v5",
        "3dPrintedCylinder_T7",
        v5_execution,
        completion,
        parent,
        source_manifest,
    )
    seal_path = _v6_prediction_seal(
        tmp_path / "v6",
        parent_archive,
        model,
        execution,
        source_manifest,
    )

    archive = validate_prediction_seal(
        seal_path,
        parent_archive.seal_path,
        execution,
        v5_execution,
        completion,
        parent,
        source_manifest,
        model,
        source_result,
    )
    assert archive.v6_vertices_m[0].tobytes() == (
        parent_archive.baseline_vertices_m[0].tobytes()
    )
    assert archive.v6_vertices_m[1].tobytes() == (
        parent_archive.v5_vertices_m[1].tobytes()
    )
    loaded_frames: list[int] = []

    def mesh_loader(frame: int) -> tuple[np.ndarray, np.ndarray]:
        loaded_frames.append(frame)
        index = int(np.flatnonzero(parent_archive.target_frames == frame)[0])
        return parent_archive.baseline_vertices_m[index], parent_archive.faces

    score_one_prediction(
        archive,
        [6, 7],
        mesh_loader,
        execution,
        v5_execution,
    )
    assert loaded_frames == [6, 7]

    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["decisions"][1]["v6_candidate_admitted"] = True
    seal["seal_sha256"] = prediction_seal_sha256(seal)
    seal_path.write_text(json.dumps(seal, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="decision record changed"):
        validate_prediction_seal(
            seal_path,
            parent_archive.seal_path,
            execution,
            v5_execution,
            completion,
            parent,
            source_manifest,
            model,
            source_result,
        )


def test_sequence_preserves_checkpoint_and_v5_fallback_bytes() -> None:
    model = _load(MODEL_PATH)
    baseline = np.arange(36, dtype=np.float64).reshape(2, 6, 3) / 100.0
    v5 = baseline.copy()
    v5[1] += 0.001
    frames = np.asarray([6, 7], dtype=np.int64)
    supported = np.asarray([False, True], dtype=np.bool_)
    rows: list[dict[str, Any]] = [
        {
            "target_frame": 6,
            "accepted": False,
            "rms_update_m": 0.0,
            "prior_motion_rms_m": 0.0,
            "correction_to_prior_motion_ratio": 0.0,
            "correction_prior_motion_cosine": None,
        },
        {
            "target_frame": 7,
            "accepted": True,
            "rms_update_m": 1e-12,
            "prior_motion_rms_m": 1e-12,
            "correction_to_prior_motion_ratio": 1e-12,
            "correction_prior_motion_cosine": 0.0,
        },
    ]

    arrays, diagnostics = apply_causal_scale_sequence(
        model,
        object_name="3dPrintedCylinder",
        baseline_vertices_m=baseline,
        v5_vertices_m=v5,
        target_frames=frames,
        update_supported=supported,
        update_rows=rows,
    )

    assert arrays["v6_vertices_m"][0].tobytes() == baseline[0].tobytes()
    assert not bool(arrays["candidate_admitted"][1])
    assert arrays["v6_vertices_m"][1].tobytes() == v5[1].tobytes()
    assert diagnostics[1]["v6_decision_reason"].endswith("v5-exact-fallback")


def _prospective_objects(*, regress: bool = False) -> list[dict[str, Any]]:
    rows = []
    for take_id in TARGET_TAKE_IDS:
        object_name = take_id.rpartition("_T")[0]
        v5 = 5.0
        if object_name == "3dPrintedCylinder":
            v6 = 5.1 if regress else 4.8
        elif object_name == "3dPrintedHeart":
            v6 = 4.9
        else:
            v6 = v5
        rows.append(
            {
                "take_id": take_id,
                "object_name": object_name,
                "v5_mean_CD_UL1_mm": v5,
                "v6_mean_CD_UL1_mm": v6,
                "frames": [
                    {
                        "target_frame": 6,
                        "v5_CD_UL1_mm": v5,
                        "v6_CD_UL1_mm": v6,
                    }
                ],
            }
        )
    return rows


def test_result_gate_rewards_joint_transfer_and_rejects_object_regression() -> None:
    _, _, parent, _, _ = _inputs()
    public13 = load_archived_public13_result(PUBLIC13_PATH, parent)
    execution = _execution()

    passing = evaluate_result(_prospective_objects(), public13, execution)
    failing = evaluate_result(
        _prospective_objects(regress=True),
        public13,
        execution,
    )

    assert passing["prospective_v6_vs_v5_win_count"] == 2
    assert passing["prospective_v6_vs_v5_tie_count"] == 3
    assert passing["prospective_v6_vs_v5_gate_passed"]
    assert failing["prospective_v6_vs_v5_regression_count"] == 1
    assert not failing["prospective_v6_vs_v5_gate_passed"]


def test_barrier_and_result_validate_end_to_end(tmp_path: Path) -> None:
    execution, v5_execution, completion, parent, model, source_result = (
        _registered_inputs()
    )
    helpers = _v5_test_helpers()
    source_manifest = helpers._source_manifest(parent)
    parent_archives = []
    parent_seal_paths = []
    v6_seal_paths = []
    for take_id in TARGET_TAKE_IDS:
        parent_archive, _ = helpers._prediction_seal(
            tmp_path / "v5",
            take_id,
            v5_execution,
            completion,
            parent,
            source_manifest,
        )
        parent_archives.append(parent_archive)
        parent_seal_paths.append(parent_archive.seal_path)
        v6_seal_paths.append(
            _v6_prediction_seal(
                tmp_path / "v6",
                parent_archive,
                model,
                execution,
                source_manifest,
            )
        )

    barrier = build_prediction_barrier(
        v6_seal_paths,
        parent_seal_paths,
        execution,
        v5_execution,
        completion,
        parent,
        source_manifest,
        model,
        source_result,
    )
    barrier_validation = validate_prediction_barrier(
        barrier,
        execution,
        source_manifest,
    )
    assert barrier_validation["passed"]

    objects = []
    for seal_path, parent_archive in zip(v6_seal_paths, parent_archives, strict=True):
        archive = validate_prediction_seal(
            seal_path,
            parent_archive.seal_path,
            execution,
            v5_execution,
            completion,
            parent,
            source_manifest,
            model,
            source_result,
        )
        target_meshes: list[dict[str, Any]] = []

        def mesh_loader(
            frame: int,
            *,
            bound_archive: Any = archive,
            records: list[dict[str, Any]] = target_meshes,
        ) -> tuple[np.ndarray, np.ndarray]:
            index = int(np.flatnonzero(bound_archive.target_frames == frame)[0])
            member = f"{bound_archive.take_id}/meshes/mesh-f{frame:05d}.obj"
            records.append(
                {
                    "target_frame": frame,
                    "archive_member": member,
                    "sha256": f"{frame:064x}",
                    "byte_count": 100 + frame,
                }
            )
            return (
                bound_archive.parent_v5.baseline_vertices_m[index]
                + np.asarray([0.005, 0.0, 0.0]),
                bound_archive.parent_v5.faces,
            )

        row = score_one_prediction(
            archive,
            [6, 7],
            mesh_loader,
            execution,
            v5_execution,
        )
        row["target_meshes"] = target_meshes
        objects.append(row)

    public13 = load_archived_public13_result(PUBLIC13_PATH, parent)
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RESULT_KIND,
        "execution_protocol_sha256": execution["execution_protocol_sha256"],
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
        "aggregate": evaluate_result(objects, public13, execution),
        "held_v8_accessed": False,
        "result_sha256": "",
    }
    result["result_sha256"] = result_sha256(result)

    validation = validate_result(
        result,
        public13,
        execution,
        barrier,
        source_manifest,
    )
    assert validation["passed"]
    assert validation["result_sha256"] == result["result_sha256"]


def test_execution_resigned_tampering_is_rejected() -> None:
    v5_execution, completion, parent, model, source_result = _inputs()
    execution = _execution()
    changed = deepcopy(execution)
    changed["method"]["rejected_frame_action"] = "use candidate"
    changed["execution_protocol_sha256"] = execution_protocol_sha256(changed)

    try:
        validate_execution_protocol(
            changed,
            v5_execution,
            completion,
            parent,
            model,
            source_result,
            bind_registered_digest=False,
        )
    except ValueError as error:
        assert "fallback changed" in str(error)
    else:
        raise AssertionError("resigned fallback mutation was accepted")
