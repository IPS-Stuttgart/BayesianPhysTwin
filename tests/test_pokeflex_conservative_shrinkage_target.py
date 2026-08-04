import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    CHECKPOINT_SHA256,
    FRESH12_EXCLUSION_AUDIT_SHA256,
    FRESH12_PUBLIC_TARGET_TAKE_IDS,
    FRESH12_PUBLIC_ZIP_SHA256,
    INSTANCE_FRESH12_PUBLIC_TARGET_TAKE_IDS,
    INSTANCE_FRESH12_PUBLIC_ZIP_SHA256,
    OFFICIAL13_PUBLIC_DEVELOPMENT_OVERLAP_TAKE_IDS,
    OFFICIAL13_PUBLIC_PROSPECTIVE_TAKE_IDS,
    OFFICIAL13_PUBLIC_TARGET_TAKE_IDS,
    OFFICIAL18_DEVELOPMENT_OVERLAP_TAKE_IDS,
    OFFICIAL18_MISSING_PUBLIC_TAKE_IDS,
    OFFICIAL18_PROSPECTIVE_TAKE_IDS,
    OFFICIAL18_TARGET_TAKE_IDS,
    SELECTED_ARM,
    SOURCE_RESULT_SHA256,
    TARGET_OBJECTS,
    TARGET_PROTOCOL_FRESH12_PUBLIC_V1,
    TARGET_PROTOCOL_INSTANCE_FRESH12_V2,
    TARGET_PROTOCOL_OFFICIAL13_PUBLIC_V1,
    TARGET_PROTOCOL_OFFICIAL18_V1,
    TARGET_PROTOCOL_V2,
    TARGET_TAKE_IDS,
    UPSTREAM_COMMIT,
    action_field_history_is_supported,
    build_prediction_barrier,
    canonical_payload_sha256,
    evaluate_target_metrics,
    file_sha256,
    load_pokeflex_shrinkage_target_protocol,
    prediction_seal_sha256,
    protocol_requires_robot_history,
    score_one_prediction,
    target_protocol_sha256,
    validate_pokeflex_shrinkage_target_protocol,
    validate_prediction_seal,
)
from bayesian_phystwin.pokeflex_instance_shrinkage import (
    BASE_EFFECTIVE_SCALE,
    INSTANCE_SCALE_CALIBRATION_FILE_SHA256,
    INSTANCE_SCALE_CALIBRATION_SHA256,
)

ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = (
    ROOT / "configs" / "sota" / "pokeflex_conservative_shrinkage_target_v1.json"
)
PROTOCOL_V2_PATH = (
    ROOT / "configs" / "sota" / "pokeflex_conservative_shrinkage_target_v2.json"
)
OFFICIAL18_PROTOCOL_PATH = (
    ROOT / "configs" / "sota" / "pokeflex_conservative_shrinkage_official18_v1.json"
)
OFFICIAL13_PUBLIC_PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "sota"
    / "pokeflex_conservative_shrinkage_official13_public_v1.json"
)
FRESH12_PUBLIC_PROTOCOL_PATH = (
    ROOT / "configs" / "sota" / "pokeflex_conservative_shrinkage_fresh12_public_v1.json"
)
INSTANCE_FRESH12_PROTOCOL_PATH = (
    ROOT / "configs" / "sota" / "pokeflex_instance_shrinkage_fresh12_v2.json"
)
FRESH12_AUDIT_PATH = (
    ROOT / "configs" / "sota" / "pokeflex_fresh12_exclusion_audit_v1.json"
)
RUNNER_PATH = (
    ROOT / "scripts" / "held" / "run_pokeflex_conservative_shrinkage_target.py"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _protocol_v2() -> dict[str, object]:
    return json.loads(PROTOCOL_V2_PATH.read_text(encoding="utf-8"))


def _official18_protocol() -> dict[str, object]:
    return json.loads(OFFICIAL18_PROTOCOL_PATH.read_text(encoding="utf-8"))


def _official13_public_protocol() -> dict[str, object]:
    return json.loads(OFFICIAL13_PUBLIC_PROTOCOL_PATH.read_text(encoding="utf-8"))


def _fresh12_public_protocol() -> dict[str, object]:
    return json.loads(FRESH12_PUBLIC_PROTOCOL_PATH.read_text(encoding="utf-8"))


def _instance_fresh12_protocol() -> dict[str, object]:
    return json.loads(INSTANCE_FRESH12_PROTOCOL_PATH.read_text(encoding="utf-8"))


def _write_prediction(
    root: Path,
    take_id: str,
    *,
    revision: str = "1" * 40,
    corrupt_fallback: bool = False,
    protocol: dict[str, object] | None = None,
    robot_history_supported: np.ndarray | None = None,
) -> Path:
    protocol = _protocol() if protocol is None else protocol
    case_root = root / take_id
    case_root.mkdir(parents=True)
    tetrahedron = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
        ],
        dtype=np.float64,
    )
    baseline = np.repeat(tetrahedron[None, :, :], 2, axis=0)
    candidate = baseline.copy()
    candidate[1, :, 0] = 0.001
    supported = np.asarray([False, True], dtype=np.bool_)
    if corrupt_fallback:
        candidate[0, 0, 0] = 0.002
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    npz_path = case_root / "prediction.npz"
    arrays = dict(
        baseline_vertices_m=baseline,
        candidate_vertices_m=candidate,
        faces=faces,
        target_frames=np.asarray([6, 7], dtype=np.int64),
        source_frames=np.asarray([5, 6], dtype=np.int64),
        history_start_frames=np.asarray([1, 2], dtype=np.int64),
        history_end_frames=np.asarray([5, 6], dtype=np.int64),
        update_supported=supported,
        update_accepted=np.asarray([False, True], dtype=np.bool_),
        action_supported=np.asarray([False, True], dtype=np.bool_),
        correction_rms_m=np.asarray([0.0, 0.001], dtype=np.float64),
    )
    if protocol_requires_robot_history(str(protocol["protocol_id"])):
        if robot_history_supported is None:
            robot_history_supported = np.asarray([False, True], dtype=np.bool_)
        arrays["robot_history_supported"] = robot_history_supported
    if protocol["protocol_id"] == TARGET_PROTOCOL_INSTANCE_FRESH12_V2:
        global_candidate = baseline.copy()
        global_candidate[1, :, 0] = 0.0005
        arrays["global_candidate_vertices_m"] = global_candidate
    np.savez_compressed(npz_path, **arrays)
    object_name, _, _ = take_id.rpartition("_T")
    seal = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexConservativeShrinkagePredictionSeal",
        "protocol_sha256": protocol["protocol_sha256"],
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "selected_arm": SELECTED_ARM,
        "take_id": take_id,
        "object_name": object_name,
        "implementation_revision": revision,
        "implementation_clean": True,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "prediction_npz": npz_path.name,
        "prediction_npz_sha256": file_sha256(npz_path),
        "predicted_frame_count": 2,
        "fallback_mismatch_count": 0,
        "future_mesh_read": False,
        "future_mesh_read_count": 0,
    }
    if protocol_requires_robot_history(str(protocol["protocol_id"])):
        seal["missing_robot_history_frame_count"] = int(
            np.sum(~np.asarray(robot_history_supported, dtype=np.bool_))
        )
    if protocol["protocol_id"] in {
        TARGET_PROTOCOL_FRESH12_PUBLIC_V1,
        TARGET_PROTOCOL_INSTANCE_FRESH12_V2,
    }:
        archive_sha256 = (
            FRESH12_PUBLIC_ZIP_SHA256
            if protocol["protocol_id"] == TARGET_PROTOCOL_FRESH12_PUBLIC_V1
            else INSTANCE_FRESH12_PUBLIC_ZIP_SHA256
        )
        seal["source_archive_name"] = f"{take_id}.zip"
        seal["source_archive_sha256"] = archive_sha256[take_id]
        seal["source_stage_manifest_name"] = "source_stage_manifest.json"
        seal["source_stage_manifest_sha256"] = "2" * 64
        seal["source_stage_manifest_file_sha256"] = "3" * 64
    if protocol["protocol_id"] == TARGET_PROTOCOL_INSTANCE_FRESH12_V2:
        multiplier = float(
            protocol["method"]["instance_scale_calibration"]["multipliers"][object_name]
        )
        seal["correction_multiplier"] = multiplier
        seal["effective_scale"] = BASE_EFFECTIVE_SCALE * multiplier
        seal["instance_scale_calibration_sha256"] = INSTANCE_SCALE_CALIBRATION_SHA256
        seal["instance_scale_calibration_file_sha256"] = (
            INSTANCE_SCALE_CALIBRATION_FILE_SHA256
        )
        seal["global_fallback_mismatch_count"] = 0
    seal["seal_sha256"] = prediction_seal_sha256(seal)
    seal_path = case_root / "seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    return seal_path


def test_target_protocol_matches_canonical_lock() -> None:
    loaded = load_pokeflex_shrinkage_target_protocol(PROTOCOL_PATH)

    assert loaded["protocol_sha256"] == target_protocol_sha256(loaded)
    assert loaded["source_gate"]["result_sha256"] == SOURCE_RESULT_SHA256
    assert tuple(loaded["target_cohort"]["objects"]) == TARGET_OBJECTS


def test_v2_target_protocol_locks_preoutcome_missing_pose_fallback() -> None:
    loaded = load_pokeflex_shrinkage_target_protocol(PROTOCOL_V2_PATH)

    assert loaded["protocol_sha256"] == target_protocol_sha256(loaded)
    assert loaded["protocol_id"] == TARGET_PROTOCOL_V2
    assert loaded["preoutcome_amendment"]["target_mesh_outcome_opened"] is False


def test_official18_protocol_locks_exact_split_and_overlap_boundary() -> None:
    loaded = load_pokeflex_shrinkage_target_protocol(OFFICIAL18_PROTOCOL_PATH)

    assert loaded["protocol_sha256"] == target_protocol_sha256(loaded)
    assert loaded["protocol_id"] == TARGET_PROTOCOL_OFFICIAL18_V1
    assert tuple(loaded["target_cohort"]["take_ids"]) == OFFICIAL18_TARGET_TAKE_IDS
    assert (
        tuple(loaded["target_cohort"]["prospective_take_ids"])
        == OFFICIAL18_PROSPECTIVE_TAKE_IDS
    )
    assert (
        tuple(loaded["target_cohort"]["development_overlap_take_ids"])
        == OFFICIAL18_DEVELOPMENT_OVERLAP_TAKE_IDS
    )


def test_official13_public_protocol_locks_public_subset_and_claim_boundary() -> None:
    loaded = load_pokeflex_shrinkage_target_protocol(OFFICIAL13_PUBLIC_PROTOCOL_PATH)

    assert loaded["protocol_sha256"] == target_protocol_sha256(loaded)
    assert loaded["protocol_id"] == TARGET_PROTOCOL_OFFICIAL13_PUBLIC_V1
    assert (
        tuple(loaded["target_cohort"]["take_ids"]) == OFFICIAL13_PUBLIC_TARGET_TAKE_IDS
    )
    assert (
        tuple(loaded["target_cohort"]["prospective_take_ids"])
        == OFFICIAL13_PUBLIC_PROSPECTIVE_TAKE_IDS
    )
    assert (
        tuple(loaded["target_cohort"]["development_overlap_take_ids"])
        == OFFICIAL13_PUBLIC_DEVELOPMENT_OVERLAP_TAKE_IDS
    )
    assert (
        tuple(loaded["target_cohort"]["missing_official_take_ids"])
        == OFFICIAL18_MISSING_PUBLIC_TAKE_IDS
    )
    assert (
        loaded["gates"]["direct_metric_reference"]["published_aggregate_is_gating"]
        is False
    )


def test_fresh12_public_protocol_locks_all_prospective_archives() -> None:
    loaded = load_pokeflex_shrinkage_target_protocol(FRESH12_PUBLIC_PROTOCOL_PATH)

    assert loaded["protocol_sha256"] == target_protocol_sha256(loaded)
    assert loaded["protocol_id"] == TARGET_PROTOCOL_FRESH12_PUBLIC_V1
    assert tuple(loaded["target_cohort"]["take_ids"]) == FRESH12_PUBLIC_TARGET_TAKE_IDS
    assert (
        tuple(loaded["target_cohort"]["prospective_take_ids"])
        == FRESH12_PUBLIC_TARGET_TAKE_IDS
    )
    assert loaded["target_cohort"]["development_overlap_take_ids"] == []
    assert loaded["preoutcome_storage_amendment"]["target_metric_computed"] is False
    assert loaded["freshness_audit"]["selected_zip_sha256"] == FRESH12_PUBLIC_ZIP_SHA256


def test_fresh12_exclusion_audit_is_canonical_and_disjoint() -> None:
    audit = json.loads(FRESH12_AUDIT_PATH.read_text(encoding="utf-8"))
    excluded = tuple(audit["prior_exposure_audit"]["take_ids"])
    selected = tuple(audit["selection"]["take_ids"])

    assert audit["audit_sha256"] == FRESH12_EXCLUSION_AUDIT_SHA256
    assert audit["audit_sha256"] == canonical_payload_sha256(
        audit,
        digest_field="audit_sha256",
    )
    assert len(excluded) == 84
    assert len(selected) == 12
    assert set(excluded).isdisjoint(selected)
    assert selected == FRESH12_PUBLIC_TARGET_TAKE_IDS
    assert audit["selection"]["zip_sha256"] == FRESH12_PUBLIC_ZIP_SHA256


def test_action_field_history_rejects_missing_end_effector_pose() -> None:
    transform = np.eye(4).tolist()
    complete = {
        frame: {"T_WT": transform, "T_WE": transform, "forces": [0.0, 4.0, 0.0]}
        for frame in range(2, 6)
    }

    assert action_field_history_is_supported(complete, 5) is True
    del complete[4]["T_WE"]
    assert action_field_history_is_supported(complete, 5) is False


def test_v2_prediction_requires_robot_support_and_exact_fallback(
    tmp_path: Path,
) -> None:
    protocol = _protocol_v2()
    seal = _write_prediction(tmp_path, TARGET_TAKE_IDS[0], protocol=protocol)
    archive = validate_prediction_seal(seal, protocol)

    assert archive.update_supported.tolist() == [False, True]

    invalid = _write_prediction(
        tmp_path / "invalid",
        TARGET_TAKE_IDS[0],
        protocol=protocol,
        robot_history_supported=np.asarray([False, False], dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="incomplete robot history"):
        validate_prediction_seal(invalid, protocol)


def test_target_protocol_rejects_resigned_target_replacement() -> None:
    payload = copy.deepcopy(_protocol())
    payload["target_cohort"]["objects"][-1] = "Replacement"
    payload["protocol_sha256"] = target_protocol_sha256(payload)

    with pytest.raises(ValueError, match="target object cohort changed"):
        validate_pokeflex_shrinkage_target_protocol(payload)


def test_prediction_seal_enforces_exact_fallback(tmp_path: Path) -> None:
    seal = _write_prediction(
        tmp_path,
        TARGET_TAKE_IDS[0],
        corrupt_fallback=True,
    )

    with pytest.raises(ValueError, match="exact fallback"):
        validate_prediction_seal(seal, _protocol())


def test_barrier_requires_all_targets_and_one_revision(tmp_path: Path) -> None:
    paths = [_write_prediction(tmp_path, take_id) for take_id in TARGET_TAKE_IDS]
    barrier = build_prediction_barrier(paths, _protocol())

    assert barrier["prediction_count"] == 8
    assert barrier["scoring_authorized"] is True
    with pytest.raises(ValueError, match="incomplete"):
        build_prediction_barrier(paths[:-1], _protocol())

    replacement_root = tmp_path / "different"
    paths[-1] = _write_prediction(
        replacement_root,
        TARGET_TAKE_IDS[-1],
        revision="2" * 40,
    )
    with pytest.raises(ValueError, match="revisions differ"):
        build_prediction_barrier(paths, _protocol())


def test_official18_barrier_requires_all_exact_takes(tmp_path: Path) -> None:
    protocol = _official18_protocol()
    paths = [
        _write_prediction(tmp_path, take_id, protocol=protocol)
        for take_id in OFFICIAL18_TARGET_TAKE_IDS
    ]

    barrier = build_prediction_barrier(paths, protocol)

    assert barrier["prediction_count"] == 18
    assert tuple(barrier["target_take_ids"]) == OFFICIAL18_TARGET_TAKE_IDS
    with pytest.raises(ValueError, match="incomplete"):
        build_prediction_barrier(paths[:-1], protocol)


def test_official13_public_barrier_requires_all_public_takes(tmp_path: Path) -> None:
    protocol = _official13_public_protocol()
    paths = [
        _write_prediction(tmp_path, take_id, protocol=protocol)
        for take_id in OFFICIAL13_PUBLIC_TARGET_TAKE_IDS
    ]

    barrier = build_prediction_barrier(paths, protocol)

    assert barrier["prediction_count"] == 13
    assert tuple(barrier["target_take_ids"]) == OFFICIAL13_PUBLIC_TARGET_TAKE_IDS
    with pytest.raises(ValueError, match="incomplete"):
        build_prediction_barrier(paths[:-1], protocol)


def test_fresh12_barrier_requires_all_registered_archives(tmp_path: Path) -> None:
    protocol = _fresh12_public_protocol()
    paths = [
        _write_prediction(tmp_path, take_id, protocol=protocol)
        for take_id in FRESH12_PUBLIC_TARGET_TAKE_IDS
    ]

    barrier = build_prediction_barrier(paths, protocol)

    assert barrier["prediction_count"] == 12
    assert tuple(barrier["target_take_ids"]) == FRESH12_PUBLIC_TARGET_TAKE_IDS
    with pytest.raises(ValueError, match="incomplete"):
        build_prediction_barrier(paths[:-1], protocol)


def test_fresh12_seal_rejects_wrong_source_archive(tmp_path: Path) -> None:
    protocol = _fresh12_public_protocol()
    seal_path = _write_prediction(
        tmp_path,
        FRESH12_PUBLIC_TARGET_TAKE_IDS[0],
        protocol=protocol,
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["source_archive_sha256"] = "0" * 64
    seal["seal_sha256"] = prediction_seal_sha256(seal)
    seal_path.write_text(json.dumps(seal), encoding="utf-8")

    with pytest.raises(ValueError, match="source archive changed"):
        validate_prediction_seal(seal_path, protocol)


def test_instance_seal_and_scorer_preserve_three_arms(tmp_path: Path) -> None:
    protocol = _instance_fresh12_protocol()
    seal_path = _write_prediction(
        tmp_path,
        INSTANCE_FRESH12_PUBLIC_TARGET_TAKE_IDS[0],
        protocol=protocol,
    )
    archive = validate_prediction_seal(seal_path, protocol)
    assert archive.global_candidate_vertices_m is not None
    assert np.array_equal(
        archive.global_candidate_vertices_m[~archive.update_supported],
        archive.baseline_vertices_m[~archive.update_supported],
    )

    tetrahedron = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
        dtype=np.float64,
    )
    result = score_one_prediction(
        archive,
        [7],
        lambda _: (tetrahedron + np.asarray([0.002, 0.0, 0.0]), archive.faces),
        protocol,
        jaccard=lambda *_: 0.9,
    )

    assert "global_candidate_mean_CD_UL1_mm" in result
    assert "global_candidate_CD_UL1_mm" in result["frames"][0]


def test_prediction_function_has_no_target_mesh_loader() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    prediction = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_predict"
    )
    called_names = {
        node.func.id
        for node in ast.walk(prediction)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "_load_mesh" not in called_names
    assert "score_one_prediction" not in called_names
    assert "_load_official_template" in called_names


def test_one_object_scorer_uses_only_registered_active_frames(tmp_path: Path) -> None:
    archive = validate_prediction_seal(
        _write_prediction(tmp_path, TARGET_TAKE_IDS[0]),
        _protocol(),
    )
    tetrahedron = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
        dtype=np.float64,
    )
    loaded_frames = []

    def mesh_loader(frame: int) -> tuple[np.ndarray, np.ndarray]:
        loaded_frames.append(frame)
        return tetrahedron + np.asarray([0.002, 0.0, 0.0]), archive.faces

    result = score_one_prediction(
        archive,
        [7],
        mesh_loader,
        _protocol(),
        jaccard=lambda *_: 0.9,
    )

    assert loaded_frames == [7]
    assert result["scored_frame_count"] == 1
    assert result["candidate_jaccard_valid_count"] == 1


def test_target_aggregation_applies_direct_and_paired_gates() -> None:
    rows = [
        {
            "object_name": object_name,
            "baseline_mean_CD_UL1_mm": 6.2,
            "candidate_mean_CD_UL1_mm": 6.0,
            "candidate_jaccard_valid_count": 10,
            "candidate_mean_jaccard_valid": 0.83,
            "scored_frame_count": 10,
        }
        for object_name in TARGET_OBJECTS
    ]

    result = evaluate_target_metrics(rows, _protocol())

    assert result["direct_metric_reference_passed"] is True
    assert result["paired_transfer_passed"] is True
    assert result["all_target_gates_passed"] is True
    assert result["object_win_count"] == 8


def test_official18_aggregation_applies_reproduction_and_prospective_gates() -> None:
    rows = []
    for take_id in OFFICIAL18_TARGET_TAKE_IDS:
        object_name, _, _ = take_id.rpartition("_T")
        rows.append(
            {
                "object_name": object_name,
                "take_id": take_id,
                "baseline_mean_CD_UL1_mm": 6.498,
                "candidate_mean_CD_UL1_mm": 6.0,
                "candidate_jaccard_valid_count": 0,
                "candidate_mean_jaccard_valid": None,
                "scored_frame_count": 1,
                "frames": [
                    {
                        "baseline_CD_UL1_mm": 6.498,
                        "candidate_CD_UL1_mm": 6.0,
                        "candidate_jaccard": None,
                    }
                ],
            }
        )

    result = evaluate_target_metrics(rows, _official18_protocol())

    assert result["baseline_reproduction_passed"] is True
    assert result["candidate_below_published_reference_passed"] is True
    assert result["paired_transfer_passed"] is True
    assert result["all_target_gates_passed"] is True
    assert result["prospective_take_count"] == 15
    assert result["development_overlap_take_count"] == 3
    assert result["jaccard_is_gating"] is False


def test_official13_public_aggregation_gates_only_prospective_pairing() -> None:
    rows = []
    for take_id in OFFICIAL13_PUBLIC_TARGET_TAKE_IDS:
        object_name, _, _ = take_id.rpartition("_T")
        candidate = (
            7.0 if take_id in OFFICIAL13_PUBLIC_DEVELOPMENT_OVERLAP_TAKE_IDS else 6.0
        )
        rows.append(
            {
                "object_name": object_name,
                "take_id": take_id,
                "baseline_mean_CD_UL1_mm": 6.498,
                "candidate_mean_CD_UL1_mm": candidate,
                "candidate_jaccard_valid_count": 0,
                "candidate_mean_jaccard_valid": None,
                "scored_frame_count": 1,
                "frames": [
                    {
                        "baseline_CD_UL1_mm": 6.498,
                        "candidate_CD_UL1_mm": candidate,
                        "candidate_jaccard": None,
                    }
                ],
            }
        )

    result = evaluate_target_metrics(rows, _official13_public_protocol())

    assert result["public_official_subset_take_count"] == 13
    assert result["prospective_take_count"] == 10
    assert result["development_overlap_take_count"] == 3
    assert result["prospective_object_win_count"] == 10
    assert result["paired_transfer_passed"] is True
    assert result["all_target_gates_passed"] is True
    assert result["published_reference_is_contextual_only"] is True
    assert result["published_direct_comparison_authorized"] is False


def test_fresh12_aggregation_gates_all_twelve_against_checkpoint() -> None:
    rows = []
    for take_id in FRESH12_PUBLIC_TARGET_TAKE_IDS:
        object_name, _, _ = take_id.rpartition("_T")
        rows.append(
            {
                "object_name": object_name,
                "take_id": take_id,
                "baseline_mean_CD_UL1_mm": 6.5,
                "candidate_mean_CD_UL1_mm": 6.4,
                "candidate_jaccard_valid_count": 0,
                "candidate_mean_jaccard_valid": None,
                "scored_frame_count": 1,
                "frames": [
                    {
                        "baseline_CD_UL1_mm": 6.5,
                        "candidate_CD_UL1_mm": 6.4,
                        "candidate_jaccard": None,
                    }
                ],
            }
        )

    result = evaluate_target_metrics(rows, _fresh12_public_protocol())

    assert result["fresh_public_take_count"] == 12
    assert result["development_overlap_take_count"] == 0
    assert result["fresh12_object_win_count"] == 12
    assert result["paired_transfer_passed"] is True
    assert result["all_target_gates_passed"] is True
    assert result["published_reference_is_contextual_only"] is True
    assert result["published_direct_comparison_authorized"] is False
