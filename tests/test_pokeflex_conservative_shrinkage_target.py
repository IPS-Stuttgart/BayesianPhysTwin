import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.pokeflex_conservative_shrinkage_target import (
    CHECKPOINT_SHA256,
    SELECTED_ARM,
    SOURCE_RESULT_SHA256,
    TARGET_OBJECTS,
    TARGET_TAKE_IDS,
    UPSTREAM_COMMIT,
    build_prediction_barrier,
    evaluate_target_metrics,
    file_sha256,
    load_pokeflex_shrinkage_target_protocol,
    prediction_seal_sha256,
    score_one_prediction,
    target_protocol_sha256,
    validate_pokeflex_shrinkage_target_protocol,
    validate_prediction_seal,
)

ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = (
    ROOT / "configs" / "sota" / "pokeflex_conservative_shrinkage_target_v1.json"
)
RUNNER_PATH = (
    ROOT / "scripts" / "held" / "run_pokeflex_conservative_shrinkage_target.py"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _write_prediction(
    root: Path,
    take_id: str,
    *,
    revision: str = "1" * 40,
    corrupt_fallback: bool = False,
) -> Path:
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
    np.savez_compressed(
        npz_path,
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
    object_name, _, _ = take_id.rpartition("_T")
    seal = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexConservativeShrinkagePredictionSeal",
        "protocol_sha256": _protocol()["protocol_sha256"],
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
    seal["seal_sha256"] = prediction_seal_sha256(seal)
    seal_path = case_root / "seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    return seal_path


def test_target_protocol_matches_canonical_lock() -> None:
    loaded = load_pokeflex_shrinkage_target_protocol(PROTOCOL_PATH)

    assert loaded["protocol_sha256"] == target_protocol_sha256(loaded)
    assert loaded["source_gate"]["result_sha256"] == SOURCE_RESULT_SHA256
    assert tuple(loaded["target_cohort"]["objects"]) == TARGET_OBJECTS


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
