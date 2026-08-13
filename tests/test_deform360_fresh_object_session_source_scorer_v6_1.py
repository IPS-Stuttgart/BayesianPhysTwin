from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_fresh_object_session_candidate_v6_1 as candidate
import bayesian_phystwin.deform360_fresh_object_session_source_scorer_v6_1 as scorer
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_fresh_object_session_source_v6 import (
    B0,
    B1,
    D1_NATIVE,
    VT1_OBSERVED,
    VT1_SANDWICH,
    VT1_WORKING,
)
from bayesian_phystwin.deform360_joint_sparse_endpoint_v5 import (
    Deform360ReservedViewGeometryV5,
)

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / (
    "protocols/amendments/"
    "deform360_official_hub_fresh_object_session_v6_1_source_scoring.json"
)
PROCESSOR = ROOT / (
    "scripts/remote/process_deform360_fresh_object_session_source_endpoint_v6_1.py"
)


def _processor_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "v61_source_endpoint_processor", PROCESSOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _trajectory(*, nodes: int = 4) -> np.ndarray:
    result = np.zeros((76, nodes, 3), dtype=np.float64)
    result[..., 0] = np.linspace(-0.15, 0.15, nodes)[None]
    result[..., 2] = 1.0
    return result


def _candidate_arrays(*, d1_shift_m: float) -> candidate.Deform360V61CandidateArrays:
    physical = _trajectory()
    baseline = candidate.build_deform360_v61_technical_fallback_arrays(
        physical_prediction_m=physical,
        b0_trajectory_m=physical,
        b1_trajectory_m=physical + np.array([0.005, 0.0, 0.0]),
    )
    arrays = {
        name: np.array(value, copy=True) for name, value in baseline.arrays.items()
    }
    arrays[f"trajectory__{D1_NATIVE}"][58:76, :, 0] += d1_shift_m
    return candidate.Deform360V61CandidateArrays(arrays=arrays, risk_score=0.5)


def _publish_candidate(
    root: Path, *, d1_shift_m: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    seal = candidate.publish_deform360_v61_candidate_artifact(
        _candidate_arrays(d1_shift_m=d1_shift_m),
        root,
        candidate_revision=scorer.CANDIDATE_REVISION,
        outer_held_out_object_id="outer-object",
        object_id="target-object",
        episode_id=3,
        stratum="sheet",
        fit_object_ids=tuple(f"fit-{index}" for index in range(8)),
        source_artifacts={"prefix/input.json": _digest("prefix")},
    )
    prediction = {
        "outer_held_out_object_id": "outer-object",
        "object_id": "target-object",
        "episode_id": 3,
        "variants": seal["variant_artifacts"],
    }
    return seal, prediction


def _view(
    camera_id: str,
    *,
    supported_local_frames: set[int],
) -> Deform360ReservedViewGeometryV5:
    height = width = 64
    masks = np.zeros((18, height, width), dtype=np.bool_)
    for local_index in supported_local_frames:
        masks[local_index, 16:48, 8:56] = True
    intrinsics = np.array(
        [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return Deform360ReservedViewGeometryV5(
        object_id="target-object",
        episode_id=3,
        camera_id=camera_id,
        frame_indices=np.arange(58, 76, dtype=np.int64),
        depth_m=np.ones((18, height, width), dtype=np.float32),
        object_mask=masks,
        intrinsics=intrinsics,
        camera_to_world=np.eye(4, dtype=np.float64),
        source_artifact_ids={f"endpoint/{camera_id}.npz": _digest(camera_id)},
    )


def _partial_views() -> tuple[Deform360ReservedViewGeometryV5, ...]:
    return (
        _view("reserved-a", supported_local_frames=set(range(9))),
        _view("reserved-b", supported_local_frames=set(range(9, 18))),
    )


def test_source_scoring_amendment_is_content_addressed_and_target_closed() -> None:
    amendment = scorer.load_deform360_v61_source_scoring_amendment(AMENDMENT)

    assert amendment["amendment_id"] == scorer.SOURCE_SCORING_AMENDMENT_ID
    assert amendment["candidate_barrier"]["prediction_record_count"] == 100
    assert amendment["candidate_barrier"]["technical_failure_record_count"] == 0
    assert amendment["endpoint_carrier"]["partial_mask_frames_allowed"] is True
    assert amendment["endpoint_carrier"]["camera_substitution_allowed"] is False
    assert amendment["information_boundary"]["source_suffix_opened"] is False
    assert amendment["information_boundary"]["target_outcomes_opened"] is False
    assert amendment["information_boundary"]["human_approval_required"] is False
    runtime = amendment["source_processing"]["runtime"]
    assert runtime["gsplat_distribution_version"] == "1.4.0+pt24cu121"
    assert runtime["gsplat_wheel_sha256"] == (
        "2efb8b8f4ad3275db05707fa6f9cf110482e7fd269c78a4cc7dc5b08cfc957ff"
    )
    assert runtime["gsplat_extension_sha256"] == (
        "e0b664c9d6f355e611bdfa720103b86b399ded3dcc5ecfaf59eaade992f1359b"
    )
    assert runtime["jit_compilation_used"] is False
    assert runtime["nvcc_required"] is False


def test_partial_reserved_view_support_scores_one_common_roster(
    tmp_path: Path,
) -> None:
    _seal, prediction = _publish_candidate(tmp_path / "candidate", d1_shift_m=0.2)

    scores, support = scorer.score_deform360_v61_candidate_artifact(
        prediction=prediction,
        candidate_directory=tmp_path / "candidate",
        reserved_views=_partial_views(),
    )

    assert support["admitted_cell_count"] == 18
    assert support["admitted_cell_count_by_reserved_view"] == {
        "reserved-a": 9,
        "reserved-b": 9,
    }
    assert support["minimum_admitted_views_per_frame"] == 1
    assert support["common_query_count"] == scores[B0]["query_count"]
    assert {scores[item]["query_count"] for item in (B0, B1, D1_NATIVE)} == {
        support["common_query_count"]
    }
    assert scores[B0]["mean_raw_radius"] == pytest.approx(0.01)
    assert scores[B0]["mean_log_determinant"] == pytest.approx(3.0 * np.log(1e-4))


def test_challenger_motion_cannot_change_query_admission(tmp_path: Path) -> None:
    _, near_prediction = _publish_candidate(tmp_path / "near", d1_shift_m=0.0)
    _, far_prediction = _publish_candidate(tmp_path / "far", d1_shift_m=0.5)

    near_scores, near_support = scorer.score_deform360_v61_candidate_artifact(
        prediction=near_prediction,
        candidate_directory=tmp_path / "near",
        reserved_views=_partial_views(),
    )
    far_scores, far_support = scorer.score_deform360_v61_candidate_artifact(
        prediction=far_prediction,
        candidate_directory=tmp_path / "far",
        reserved_views=_partial_views(),
    )

    assert (
        near_support["common_query_roster_id"] == far_support["common_query_roster_id"]
    )
    assert near_scores[B0] == far_scores[B0]
    assert near_scores[D1_NATIVE]["query_count"] == far_scores[D1_NATIVE]["query_count"]
    assert far_scores[D1_NATIVE]["point_loss"] > near_scores[D1_NATIVE]["point_loss"]
    assert (
        far_scores[D1_NATIVE]["maximum_raw_mahalanobis_norm"]
        > near_scores[D1_NATIVE]["maximum_raw_mahalanobis_norm"]
    )


def test_unavailable_public_tactile_variants_are_exact_b0_scores(
    tmp_path: Path,
) -> None:
    _, prediction = _publish_candidate(tmp_path / "candidate", d1_shift_m=0.1)
    scores, _support = scorer.score_deform360_v61_candidate_artifact(
        prediction=prediction,
        candidate_directory=tmp_path / "candidate",
        reserved_views=_partial_views(),
    )
    fallback = {
        key: value
        for key, value in scores[B0].items()
        if key not in {"available", "prediction_artifact_id"}
    }

    for variant_id in (VT1_WORKING, VT1_OBSERVED, VT1_SANDWICH):
        assert scores[variant_id]["available"] is False
        assert scores[variant_id]["prediction_artifact_id"] is None
        assert {
            key: value
            for key, value in scores[variant_id].items()
            if key not in {"available", "prediction_artifact_id"}
        } == fallback


def test_time_spanning_support_failure_is_terminal(tmp_path: Path) -> None:
    _, prediction = _publish_candidate(tmp_path / "candidate", d1_shift_m=0.0)
    insufficient = (
        _view("reserved-a", supported_local_frames=set(range(8))),
        _view("reserved-b", supported_local_frames=set(range(8, 18))),
    )

    with pytest.raises(
        scorer.SourceEndpointSupportError,
        match="time-spanning support contract",
    ):
        scorer.score_deform360_v61_candidate_artifact(
            prediction=prediction,
            candidate_directory=tmp_path / "candidate",
            reserved_views=insufficient,
        )


def test_processor_retains_empty_mask_frames_and_requires_two_view_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = _processor_module()
    masks = np.ones((81, 4, 5), dtype=np.uint8)
    masks[17] = 0
    written: dict[str, np.ndarray] = {}

    class _Dataset:
        def __enter__(self) -> _Dataset:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def create_dataset(
            self, name: str, *, data: np.ndarray, **_kwargs: object
        ) -> None:
            written[name] = np.asarray(data)

    fake_h5py = types.SimpleNamespace(File=lambda *_args, **_kwargs: _Dataset())
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)
    support = processor._write_partial_masks(tmp_path / "masks.h5", masks)

    np.testing.assert_array_equal(written["data"], masks)
    assert support.shape == (81,)
    assert support.dtype == np.bool_
    assert support[17] == np.bool_(False)
    assert int(np.sum(support)) == 80

    second = np.ones(81, dtype=np.bool_)
    with pytest.raises(ValueError, match="fewer than two"):
        processor._validate_frame_support({"a": support, "b": second})
    third = np.ones(81, dtype=np.bool_)
    counts = processor._validate_frame_support({"a": support, "b": second, "c": third})
    assert counts[17] == 2
    assert counts[0] == 3


def test_processor_never_uses_modern_fps_mode_or_camera_substitution() -> None:
    source = PROCESSOR.read_text(encoding="utf-8")

    assert '"-vsync"' in source
    assert '"-fps_mode"' not in source
    assert "fixed_panel | set(reserved)" in source
    assert "camera_substitution" not in source
    assert 'selector_package_root = selector_root / "src"' in source
    assert 'parser.add_argument("--gsplat-wheel"' in source
    assert 'importlib.metadata.version("gsplat")' in source
    assert "build.ninja" not in source


def test_terminal_failure_receipt_is_not_a_scored_negative() -> None:
    authorization = {
        "schema": scorer.SOURCE_SUFFIX_AUTHORIZATION_SCHEMA,
        "schema_version": 1,
        "source_scoring_amendment_id": scorer.SOURCE_SCORING_AMENDMENT_ID,
        "scorer_revision": "1" * 40,
        "runner_name": "workstation2",
        "workflow_run_id": 123,
        "workflow_run_attempt": 1,
        "candidate_revision": scorer.CANDIDATE_REVISION,
        "candidate_workflow_run_id": scorer.CANDIDATE_WORKFLOW_RUN_ID,
        "candidate_workflow_run_attempt": scorer.CANDIDATE_WORKFLOW_RUN_ATTEMPT,
        "candidate_execution_receipt_id": scorer.CANDIDATE_EXECUTION_RECEIPT_ID,
        "candidate_execution_receipt_file_sha256": (
            scorer.CANDIDATE_EXECUTION_RECEIPT_FILE_SHA256
        ),
        "candidate_panel_receipt_id": scorer.CANDIDATE_PANEL_RECEIPT_ID,
        "candidate_panel_receipt_file_sha256": (
            scorer.CANDIDATE_PANEL_RECEIPT_FILE_SHA256
        ),
        "raw_prediction_batch_id": scorer.RAW_PREDICTION_BATCH_ID,
        "raw_prediction_batch_file_sha256": (scorer.RAW_PREDICTION_BATCH_FILE_SHA256),
        "upstream_source_plan_id": scorer.UPSTREAM_SOURCE_PLAN_ID,
        "upstream_source_plan_file_sha256": (scorer.UPSTREAM_SOURCE_PLAN_FILE_SHA256),
        "prediction_record_count": 100,
        "technical_failure_record_count": 0,
        "development_suffix_access_authorized": True,
        "confirmation_payloads_opened": False,
        "human_approval_required": False,
        "information_boundary": scorer._AUTHORIZATION_BOUNDARY,  # noqa: SLF001
    }
    authorization["authorization_id"] = content_id(authorization)

    receipt = scorer.build_deform360_v61_source_scoring_technical_failure_receipt(
        scorer_revision="1" * 40,
        runner_name="workstation2",
        workflow_run_id=123,
        workflow_run_attempt=1,
        authorization=authorization,
        terminal_stage="endpoint-processing",
        exit_code=2,
        source_suffix_opened=True,
        retained_artifact_file_sha256={"terminal-failures/a.json": "2" * 64},
    )

    assert receipt["status"] == "source-scoring-technical-failure-retained"
    assert receipt["source_gate_evaluated"] is False
    assert receipt["source_gate_passed"] is None
    assert receipt["source_continuation_authorized"] is False
    assert receipt["independent_confirmation_authorized"] is False
    assert receipt["information_boundary"]["source_suffix_opened"] is True


def test_success_receipt_preserves_authorized_workflow_lineage() -> None:
    authorization = {
        "schema": scorer.SOURCE_SUFFIX_AUTHORIZATION_SCHEMA,
        "schema_version": 1,
        "source_scoring_amendment_id": scorer.SOURCE_SCORING_AMENDMENT_ID,
        "scorer_revision": "1" * 40,
        "runner_name": "workstation2",
        "workflow_run_id": 456,
        "workflow_run_attempt": 1,
        "candidate_revision": scorer.CANDIDATE_REVISION,
        "candidate_workflow_run_id": scorer.CANDIDATE_WORKFLOW_RUN_ID,
        "candidate_workflow_run_attempt": scorer.CANDIDATE_WORKFLOW_RUN_ATTEMPT,
        "candidate_execution_receipt_id": scorer.CANDIDATE_EXECUTION_RECEIPT_ID,
        "candidate_execution_receipt_file_sha256": (
            scorer.CANDIDATE_EXECUTION_RECEIPT_FILE_SHA256
        ),
        "candidate_panel_receipt_id": scorer.CANDIDATE_PANEL_RECEIPT_ID,
        "candidate_panel_receipt_file_sha256": (
            scorer.CANDIDATE_PANEL_RECEIPT_FILE_SHA256
        ),
        "raw_prediction_batch_id": scorer.RAW_PREDICTION_BATCH_ID,
        "raw_prediction_batch_file_sha256": scorer.RAW_PREDICTION_BATCH_FILE_SHA256,
        "upstream_source_plan_id": scorer.UPSTREAM_SOURCE_PLAN_ID,
        "upstream_source_plan_file_sha256": scorer.UPSTREAM_SOURCE_PLAN_FILE_SHA256,
        "prediction_record_count": 100,
        "technical_failure_record_count": 0,
        "development_suffix_access_authorized": True,
        "confirmation_payloads_opened": False,
        "human_approval_required": False,
        "information_boundary": scorer._AUTHORIZATION_BOUNDARY,  # noqa: SLF001
    }
    authorization["authorization_id"] = content_id(authorization)
    endpoint = {"manifest_id": "2" * 64}
    result = {
        "source_gate_passed": False,
        "source_continuation_authorized": False,
        "claim_authorized": False,
        "evidence_id": "3" * 64,
        "result_id": "4" * 64,
    }

    receipt = scorer.build_deform360_v61_source_scoring_receipt(
        scorer_revision="1" * 40,
        authorization=authorization,
        endpoint_manifest=endpoint,
        result=result,
        outcome_count=100,
        artifacts={"source-result": "5" * 64},
    )

    assert receipt["runner_name"] == "workstation2"
    assert receipt["workflow_run_id"] == 456
    assert receipt["workflow_run_attempt"] == 1
    assert receipt["source_gate_passed"] is False
    scorer.validate_deform360_v61_source_scoring_receipt(receipt)
