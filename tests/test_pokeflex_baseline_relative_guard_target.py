from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.pokeflex_baseline_relative_guard_target import (
    CERTIFICATE_SHA256,
    CHECKPOINT_SHA256,
    MINIMUM_SUPPORTED_OBJECT_COUNT,
    MINIMUM_WIN_COUNT,
    PROTOCOL_FILE_SHA256,
    TARGET_TAKE_IDS,
    UPSTREAM_COMMIT,
    PredictionArchive,
    build_prediction_barrier,
    evaluate_target_metrics,
    file_sha256,
    load_protocol,
    protocol_sha256,
    score_one_prediction,
    seal_sha256,
    validate_prediction_barrier,
    validate_prediction_seal,
    validate_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "configs"
    / "sota"
    / "pokeflex_baseline_relative_guard_public_paired_v2.json"
)


def _protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL)


def _rows(
    improvement_by_take: dict[str, float], supported_by_take: dict[str, int]
) -> list[dict[str, object]]:
    rows = []
    for take_id in TARGET_TAKE_IDS:
        baseline = 5.0
        improvement = improvement_by_take.get(take_id, 0.0)
        rows.append(
            {
                "take_id": take_id,
                "object_name": take_id.rpartition("_T")[0],
                "scored_frame_count": 2,
                "supported_frame_count": supported_by_take.get(take_id, 0),
                "baseline_mean_CD_UL1_mm": baseline,
                "candidate_mean_CD_UL1_mm": baseline * (1.0 - improvement),
                "candidate_jaccard_valid_count": 0,
                "candidate_mean_jaccard_valid": None,
            }
        )
    return rows


def _prediction_seal(
    root: Path, take_id: str, *, accepted: bool = False
) -> Path:
    take_root = root / take_id
    take_root.mkdir(parents=True)
    baseline = np.zeros((2, 4, 3), dtype=np.float64)
    candidate = baseline.copy()
    if accepted:
        candidate[0] += 0.001
    accepted_rows = np.asarray([accepted, False], dtype=np.bool_)
    npz_path = take_root / "prediction.npz"
    np.savez_compressed(
        npz_path,
        baseline_vertices_m=baseline,
        candidate_vertices_m=candidate,
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        target_frames=np.asarray([6, 7], dtype=np.int64),
        source_frames=np.asarray([5, 6], dtype=np.int64),
        history_start_frames=np.asarray([1, 2], dtype=np.int64),
        history_end_frames=np.asarray([5, 6], dtype=np.int64),
        raw_update_supported=accepted_rows,
        guard_in_source_support=accepted_rows,
        guard_accepted=accepted_rows,
        update_accepted=accepted_rows,
        action_supported=accepted_rows,
        robot_history_supported=accepted_rows,
        association_count=np.asarray([3, 0], dtype=np.int64),
        raw_correction_rms_m=np.asarray([0.001, 0.0]),
        correction_field_rms_m=np.asarray([0.001, 0.0]),
        guard_predicted_regret_mm=np.asarray(
            [-0.2 if accepted else np.nan, np.nan]
        ),
        guard_upper_regret_mm=np.asarray(
            [-0.1 if accepted else np.nan, np.nan]
        ),
    )
    seal = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexBaselineRelativeGuardPredictionSeal",
        "protocol_sha256": _protocol()["protocol_sha256"],
        "certificate_sha256": CERTIFICATE_SHA256,
        "take_id": take_id,
        "object_name": take_id.rpartition("_T")[0],
        "implementation_revision": "a" * 40,
        "implementation_clean": True,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "prediction_npz": npz_path.name,
        "prediction_npz_sha256": file_sha256(npz_path),
        "predicted_frame_count": 2,
        "guard_accepted_frame_count": int(accepted),
        "fallback_mismatch_count": 0,
        "future_mesh_read": False,
        "future_mesh_read_count": 0,
    }
    seal["seal_sha256"] = seal_sha256(seal)
    seal_path = take_root / "seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    return seal_path


def test_protocol_locks_fresh_twelve_object_cohort_and_certificate() -> None:
    protocol = _protocol()

    assert file_sha256(PROTOCOL) == PROTOCOL_FILE_SHA256
    assert validate_protocol(protocol)["passed"] is True
    assert tuple(protocol["target_cohort"]["take_ids"]) == TARGET_TAKE_IDS
    assert protocol["target_cohort"]["replacement_allowed"] is False
    assert protocol["development_guard"]["claim_status"].startswith("post-open")
    assert len(protocol["development_guard"]["certificate"]["coefficients"]) == 7
    assert MINIMUM_WIN_COUNT == 10
    assert MINIMUM_SUPPORTED_OBJECT_COUNT == 10


def test_protocol_rejects_a_changed_certificate() -> None:
    protocol = _protocol()
    changed = copy.deepcopy(protocol)
    changed["development_guard"]["certificate"]["coefficients"][0] += 0.1
    changed["protocol_sha256"] = protocol_sha256(changed)

    with pytest.raises(ValueError, match="guard certificate changed"):
        validate_protocol(changed)


def test_transfer_gate_requires_breadth_bootstrap_and_no_losses() -> None:
    improvements = {take_id: 0.02 for take_id in TARGET_TAKE_IDS[:10]}
    support = {take_id: 1 for take_id in TARGET_TAKE_IDS[:10]}
    passed = evaluate_target_metrics(_rows(improvements, support), _protocol())

    assert passed["object_win_count"] == 10
    assert passed["object_tie_count"] == 2
    assert passed["object_loss_count"] == 0
    assert passed["supported_object_count"] == 10
    assert passed["paired_transfer_passed"] is True

    harmful = dict(improvements)
    harmful[TARGET_TAKE_IDS[-1]] = -0.001
    failed = evaluate_target_metrics(_rows(harmful, support), _protocol())
    assert failed["object_loss_count"] == 1
    assert failed["paired_transfer_passed"] is False


def test_target_aggregation_is_object_balanced() -> None:
    improvements = {take_id: 0.01 for take_id in TARGET_TAKE_IDS}
    support = dict.fromkeys(TARGET_TAKE_IDS, 1)
    result = evaluate_target_metrics(_rows(improvements, support), _protocol())

    assert result["baseline_object_balanced_CD_UL1_mm"] == 5.0
    assert result["candidate_object_balanced_CD_UL1_mm"] == pytest.approx(4.95)
    assert result["object_balanced_relative_CD_UL1_improvement"] == pytest.approx(
        0.01
    )
    assert np.isfinite(
        result["bootstrap_upper_candidate_minus_baseline_CD_UL1_mm"]
    )


def test_score_records_each_active_frame_once(tmp_path: Path) -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    archive = PredictionArchive(
        take_id=TARGET_TAKE_IDS[0],
        seal_path=tmp_path / "seal.json",
        npz_path=tmp_path / "prediction.npz",
        implementation_revision="a" * 40,
        baseline_vertices_m=np.stack((vertices, vertices)),
        candidate_vertices_m=np.stack((vertices, vertices)),
        faces=faces,
        target_frames=np.asarray([6, 7], dtype=np.int64),
        update_supported=np.asarray([True, False]),
    )

    result = score_one_prediction(
        archive,
        (6, 7),
        lambda _frame: (vertices + 0.001, faces),
        _protocol(),
        jaccard=lambda *_args: 1.0,
    )

    assert result["scored_frame_count"] == 2
    assert result["supported_frame_count"] == 1
    assert [row["target_frame"] for row in result["frames"]] == [6, 7]


def test_seal_and_complete_barrier_preserve_exact_fallback(tmp_path: Path) -> None:
    seal_paths = [
        _prediction_seal(tmp_path, take_id, accepted=index == 0)
        for index, take_id in enumerate(TARGET_TAKE_IDS)
    ]

    first = validate_prediction_seal(seal_paths[0], _protocol())
    assert first.update_supported.tolist() == [True, False]
    assert np.array_equal(
        first.candidate_vertices_m[1].view(np.uint64),
        first.baseline_vertices_m[1].view(np.uint64),
    )
    barrier = build_prediction_barrier(seal_paths, _protocol())
    assert validate_prediction_barrier(barrier, _protocol())["passed"] is True
    assert barrier["prediction_count"] == len(TARGET_TAKE_IDS)


def test_seal_rejects_guard_acceptance_without_source_support(
    tmp_path: Path,
) -> None:
    seal_path = _prediction_seal(tmp_path, TARGET_TAKE_IDS[0], accepted=True)
    with np.load(seal_path.parent / "prediction.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["guard_in_source_support"] = np.asarray([False, False])
    np.savez_compressed(seal_path.parent / "prediction.npz", **arrays)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["prediction_npz_sha256"] = file_sha256(seal_path.parent / "prediction.npz")
    seal["seal_sha256"] = seal_sha256(seal)
    seal_path.write_text(json.dumps(seal), encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-support"):
        validate_prediction_seal(seal_path, _protocol())
