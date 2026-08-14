from __future__ import annotations

import hashlib
import importlib.util
import json
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
    result: np.ndarray = np.zeros((76, nodes, 3), dtype=np.float64)
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
    masks: np.ndarray = np.zeros((18, height, width), dtype=np.bool_)
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


def _authorization(
    *, scorer_revision: str = "1" * 40, workflow_run_id: int = 123
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema": scorer.SOURCE_SUFFIX_AUTHORIZATION_SCHEMA,
        "schema_version": 1,
        "source_scoring_amendment_id": scorer.SOURCE_SCORING_AMENDMENT_ID,
        "scorer_revision": scorer_revision,
        "runner_name": "workstation2",
        "workflow_run_id": workflow_run_id,
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
        "information_boundary": dict(scorer._AUTHORIZATION_BOUNDARY),  # noqa: SLF001
    }
    return {**identity, "authorization_id": content_id(identity)}


def _source_plan(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    objects = [
        {
            "object_id": f"object-{index:02d}",
            "episode_id": index,
            "stratum": "sheet" if index < 5 else "volumetric",
            "all_camera_ids": ["camera-a", "camera-b", "camera-c"],
            "reserved_endpoint_camera_ids": ["camera-a", "camera-b"],
            "raw_prefix_range_half_open": [0, 58],
        }
        for index in range(10)
    ]
    identity: dict[str, Any] = {
        "schema": "bayesian-phystwin.deform360-joint-sparse-source-prediction-plan",
        "schema_version": 6,
        "implementation_revision": scorer.UPSTREAM_REVISION,
        "execution_lock_id": scorer.EXECUTION_LOCK_ID,
        "objects": objects,
    }
    plan = {**identity, "plan_id": content_id(identity)}
    monkeypatch.setattr(scorer, "UPSTREAM_SOURCE_PLAN_ID", plan["plan_id"])
    return plan


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _endpoint_objects(plan: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for row in plan["objects"]:
        views: list[dict[str, Any]] = []
        for camera_id in row["reserved_endpoint_camera_ids"]:
            relative = Path(row["object_id"]) / f"{camera_id}.npz"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                camera_to_world=np.eye(4, dtype=np.float64),
                depth_m=np.ones((18, 4, 5), dtype=np.float32),
                frame_indices=np.arange(58, 76, dtype=np.int64),
                intrinsics=np.array(
                    [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ),
                object_mask=np.ones((18, 4, 5), dtype=np.bool_),
                raw_frame_indices=np.arange(58, 76, dtype=np.int64),
            )
            views.append(
                {
                    "camera_id": camera_id,
                    "endpoint_archive": {
                        "path": relative.as_posix(),
                        "sha256": scorer._sha256_file(path),  # noqa: SLF001
                    },
                }
            )
        objects.append(
            {
                "object_id": row["object_id"],
                "episode_id": row["episode_id"],
                "stratum": row["stratum"],
                "all_camera_ids": row["all_camera_ids"],
                "raw_endpoint_range_half_open": [58, 76],
                "reserved_views": views,
            }
        )
    return objects


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


def test_score_rejects_non_boolean_masks_and_falsey_non_config(
    tmp_path: Path,
) -> None:
    _, prediction = _publish_candidate(tmp_path / "candidate", d1_shift_m=0.0)
    views = list(_partial_views())
    object.__setattr__(
        views[0],
        "object_mask",
        np.asarray(views[0].object_mask, dtype=np.float64),
    )

    with pytest.raises(ValueError, match="mask must contain only booleans"):
        scorer.score_deform360_v61_candidate_artifact(
            prediction=prediction,
            candidate_directory=tmp_path / "candidate",
            reserved_views=views,
        )

    with pytest.raises(
        TypeError,
        match="config must be a Deform360V61SourceScoreConfig",
    ):
        scorer.score_deform360_v61_candidate_artifact(
            prediction=prediction,
            candidate_directory=tmp_path / "candidate",
            reserved_views=_partial_views(),
            config=0,
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
    masks: np.ndarray = np.ones((81, 4, 5), dtype=np.uint8)
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

    second: np.ndarray = np.ones(81, dtype=np.bool_)
    with pytest.raises(ValueError, match="fewer than two"):
        processor._validate_frame_support({"a": support, "b": second})
    third: np.ndarray = np.ones(81, dtype=np.bool_)
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


def test_processor_accepts_the_pinned_tagged_gsplat_build() -> None:
    processor = _processor_module()

    assert processor._gsplat_versions_match(  # noqa: SLF001
        module_version="1.4.0+pt24cu121",
        distribution_version="1.4.0+pt24cu121",
        expected_base_version="1.4.0",
        expected_distribution_version="1.4.0+pt24cu121",
    )
    assert not processor._gsplat_versions_match(  # noqa: SLF001
        module_version="1.4.0",
        distribution_version="1.4.0+pt24cu121",
        expected_base_version="1.4.0",
        expected_distribution_version="1.4.0+pt24cu121",
    )
    assert not processor._gsplat_versions_match(  # noqa: SLF001
        module_version="1.4.0+pt24cu122",
        distribution_version="1.4.0+pt24cu122",
        expected_base_version="1.4.0",
        expected_distribution_version="1.4.0+pt24cu121",
    )


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


def test_contract_helpers_fail_closed_and_json_publication_is_idempotent(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        scorer._mapping([], name="value")  # noqa: SLF001
    with pytest.raises(ValueError, match="JSON array"):
        scorer._sequence("value", name="value")  # noqa: SLF001
    with pytest.raises(ValueError, match="JSON array"):
        scorer._sequence(1, name="value")  # noqa: SLF001

    output = tmp_path / "receipt.json"
    assert scorer._publish_or_validate_json(  # noqa: SLF001
        {"status": "sealed"}, output, label="receipt"
    ) == {"status": "sealed"}
    assert scorer._publish_or_validate_json(  # noqa: SLF001
        {"status": "sealed"}, output, label="receipt"
    ) == {"status": "sealed"}
    with pytest.raises(ValueError, match="existing receipt differs"):
        scorer._publish_or_validate_json(  # noqa: SLF001
            {"status": "changed"}, output, label="receipt"
        )


def test_source_plan_and_endpoint_manifest_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _source_plan(monkeypatch)
    authorization = _authorization()
    scorer.validate_deform360_v61_source_plan(plan)

    root = tmp_path / "endpoint"
    objects = _endpoint_objects(plan, root)
    manifest = scorer.build_deform360_v61_source_endpoint_manifest(
        authorization=authorization,
        source_plan=plan,
        processor_revision="2" * 40,
        objects=objects,
    )
    assert (
        scorer.validate_deform360_v61_source_endpoint_manifest(
            manifest,
            authorization=authorization,
            source_plan=plan,
        )
        == manifest
    )

    views, sources = scorer._endpoint_views_by_object(  # noqa: SLF001
        endpoint_manifest=manifest,
        endpoint_root=root,
    )
    assert set(views) == {row["object_id"] for row in plan["objects"]}
    assert all(len(item) == 2 for item in views.values())
    assert all(len(item) == 2 for item in sources.values())
    assert views["object-00"][0].frame_indices.tolist() == list(range(58, 76))

    changed = dict(manifest)
    changed["information_boundary"] = {}
    with pytest.raises(ValueError, match="endpoint manifest contract changed"):
        scorer.validate_deform360_v61_source_endpoint_manifest(
            changed,
            authorization=authorization,
            source_plan=plan,
        )


def test_endpoint_archive_loader_rejects_member_and_frame_changes(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.npz"
    base = {
        "camera_to_world": np.eye(4, dtype=np.float64),
        "depth_m": np.ones((18, 4, 5), dtype=np.float32),
        "frame_indices": np.arange(58, 76, dtype=np.int64),
        "intrinsics": np.array(
            [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "object_mask": np.ones((18, 4, 5), dtype=np.bool_),
        "raw_frame_indices": np.arange(58, 76, dtype=np.int64),
    }
    np.savez(good, **base)
    view = scorer.load_deform360_v61_source_endpoint_view(
        good,
        object_id="object",
        episode_id=1,
        camera_id="camera-a",
        raw_endpoint_range_half_open=(58, 76),
        source_artifact_ids={"endpoint.npz": scorer._sha256_file(good)},  # noqa: SLF001
    )
    assert view.object_mask.dtype == np.bool_

    wrong_members = tmp_path / "wrong-members.npz"
    np.savez(
        wrong_members, **{key: value for key, value in base.items() if key != "depth_m"}
    )
    with pytest.raises(ValueError, match="cannot load source endpoint archive"):
        scorer.load_deform360_v61_source_endpoint_view(
            wrong_members,
            object_id="object",
            episode_id=1,
            camera_id="camera-a",
            raw_endpoint_range_half_open=(58, 76),
            source_artifact_ids={"endpoint.npz": _digest("wrong")},
        )

    wrong_frames = tmp_path / "wrong-frames.npz"
    np.savez(wrong_frames, **{**base, "raw_frame_indices": np.arange(59, 77)})
    with pytest.raises(ValueError, match="raw frame roster changed"):
        scorer.load_deform360_v61_source_endpoint_view(
            wrong_frames,
            object_id="object",
            episode_id=1,
            camera_id="camera-a",
            raw_endpoint_range_half_open=(58, 76),
            source_artifact_ids={"endpoint.npz": _digest("wrong")},
        )


def test_geometry_edge_paths_are_deterministic() -> None:
    valid: np.ndarray = np.ones((8, 8), dtype=np.bool_)
    first = scorer._target_pixel_indices(  # noqa: SLF001
        valid,
        object_id="object",
        camera_id="camera",
        frame=58,
        maximum=8,
    )
    second = scorer._target_pixel_indices(  # noqa: SLF001
        valid,
        object_id="object",
        camera_id="camera",
        frame=58,
        maximum=8,
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert len(first[0]) == 8

    points = np.array([[100.0, 100.0, 1.0]], dtype=np.float64)
    visible = scorer._visible_node_indices(  # noqa: SLF001
        points,
        depth=np.ones((4, 4), dtype=np.float64),
        valid_target=np.ones((4, 4), dtype=np.bool_),
        intrinsics=np.eye(3, dtype=np.float64),
        camera_to_world=np.eye(4, dtype=np.float64),
        config=scorer.Deform360V61SourceScoreConfig(),
    )
    assert visible.size == 0

    candidate = _trajectory()
    outside = tuple(
        _view(camera, supported_local_frames=set(range(18)))
        for camera in ("reserved-a", "reserved-b")
    )
    candidate[:, :, 0] = 100.0
    with pytest.raises(scorer.SourceEndpointSupportError):
        scorer._prepare_score_cells(  # noqa: SLF001
            object_id="target-object",
            episode_id=3,
            b0_trajectory_m=candidate,
            reserved_views=outside,
            config=scorer.Deform360V61SourceScoreConfig(),
        )


def test_failure_retention_hashes_files_and_excludes_its_own_receipt(
    tmp_path: Path,
) -> None:
    authorization_path = tmp_path / "authorization.json"
    _write_json(authorization_path, _authorization())
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "runtime.log").write_text("failed", encoding="utf-8")
    output = artifact_root / "technical-failure.json"

    receipt = scorer.retain_deform360_v61_source_scoring_failure(
        scorer_revision="1" * 40,
        runner_name="workstation2",
        workflow_run_id=123,
        workflow_run_attempt=1,
        authorization_path=authorization_path,
        terminal_stage="endpoint-processing",
        exit_code=2,
        source_suffix_opened=True,
        artifact_root=artifact_root,
        output_path=output,
    )
    assert set(receipt["retained_artifacts"]) == {"runtime.log"}
    assert output.is_file()

    repeated = scorer.retain_deform360_v61_source_scoring_failure(
        scorer_revision="1" * 40,
        runner_name="workstation2",
        workflow_run_id=123,
        workflow_run_attempt=1,
        authorization_path=authorization_path,
        terminal_stage="endpoint-processing",
        exit_code=2,
        source_suffix_opened=True,
        artifact_root=artifact_root,
        output_path=output,
    )
    assert repeated == receipt


def test_source_suffix_authorization_replays_every_frozen_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    panel_path = candidate_root / scorer.CANDIDATE_RECEIPT_FILENAME
    batch_path = candidate_root / scorer.CANDIDATE_BATCH_FILENAME
    execution_path = tmp_path / "candidate-execution.json"
    source_plan_path = tmp_path / "source-plan.json"
    output = tmp_path / "authorization.json"
    _write_json(panel_path, {"sealed": True})
    _write_json(batch_path, {"sealed": True})
    _write_json(
        execution_path,
        {
            "receipt_id": scorer.CANDIDATE_EXECUTION_RECEIPT_ID,
            "candidate_panel_receipt_id": scorer.CANDIDATE_PANEL_RECEIPT_ID,
            "workflow_run_id": scorer.CANDIDATE_WORKFLOW_RUN_ID,
            "workflow_run_attempt": scorer.CANDIDATE_WORKFLOW_RUN_ATTEMPT,
        },
    )
    plan = _source_plan(monkeypatch)
    _write_json(source_plan_path, plan)

    monkeypatch.setattr(
        scorer,
        "CANDIDATE_PANEL_RECEIPT_FILE_SHA256",
        scorer._sha256_file(panel_path),  # noqa: SLF001
    )
    monkeypatch.setattr(
        scorer,
        "RAW_PREDICTION_BATCH_FILE_SHA256",
        scorer._sha256_file(batch_path),  # noqa: SLF001
    )
    monkeypatch.setattr(
        scorer,
        "CANDIDATE_EXECUTION_RECEIPT_FILE_SHA256",
        scorer._sha256_file(execution_path),  # noqa: SLF001
    )
    monkeypatch.setattr(
        scorer,
        "UPSTREAM_SOURCE_PLAN_FILE_SHA256",
        scorer._sha256_file(source_plan_path),  # noqa: SLF001
    )
    monkeypatch.setattr(
        scorer,
        "load_deform360_v61_source_scoring_amendment",
        lambda _path: {},
    )
    monkeypatch.setattr(
        scorer,
        "load_deform360_joint_sparse_source_execution_lock_v5",
        lambda _path: {"execution_lock_id": scorer.EXECUTION_LOCK_ID},
    )
    monkeypatch.setattr(
        scorer,
        "validate_deform360_v61_candidate_panel",
        lambda **_kwargs: {
            "receipt_id": scorer.CANDIDATE_PANEL_RECEIPT_ID,
            "candidate_revision": scorer.CANDIDATE_REVISION,
            "raw_prediction_batch_id": scorer.RAW_PREDICTION_BATCH_ID,
            "prediction_record_count": 100,
            "technical_failure_record_count": 0,
        },
    )
    monkeypatch.setattr(
        scorer,
        "validate_deform360_v61_candidate_execution_receipt",
        lambda value: value,
    )

    authorization = scorer.build_deform360_v61_source_suffix_authorization(
        source_scoring_amendment_path=AMENDMENT,
        execution_lock_path=tmp_path / "lock.json",
        candidate_root=candidate_root,
        candidate_execution_receipt_path=execution_path,
        upstream_source_plan_path=source_plan_path,
        scorer_revision="3" * 40,
        runner_name="workstation2",
        workflow_run_id=789,
        workflow_run_attempt=1,
    )
    assert authorization["upstream_source_plan_id"] == plan["plan_id"]
    assert authorization["development_suffix_access_authorized"] is True
    assert authorization["confirmation_payloads_opened"] is False
    assert (
        scorer.publish_deform360_v61_source_suffix_authorization(authorization, output)
        == authorization
    )
    assert (
        scorer.publish_deform360_v61_source_suffix_authorization(authorization, output)
        == authorization
    )


def test_candidate_path_and_support_report_are_content_addressed(
    tmp_path: Path,
) -> None:
    directory = scorer._candidate_directory(  # noqa: SLF001
        tmp_path,
        ordered_ids=("outer", "target"),
        outer_id="outer",
        target_id="target",
    )
    assert directory == (tmp_path / "candidate-artifacts" / "00-outer" / "01-target")
    report = scorer._support_report(  # noqa: SLF001
        prediction={
            "prediction_record_id": _digest("prediction"),
            "outer_held_out_object_id": "outer",
            "object_id": "target",
            "episode_id": 3,
        },
        support={"common_query_count": 4},
    )
    assert report["reserved_views_used_for_prediction"] is False
    assert report["report_id"] == content_id(
        {key: value for key, value in report.items() if key != "report_id"}
    )


def test_source_score_publication_is_complete_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_ids = tuple(f"object-{index:02d}" for index in range(10))
    cohort = {object_id: {"object_id": object_id} for object_id in object_ids}
    records = [
        {
            "prediction_record_id": _digest(f"prediction-{outer}-{target}"),
            "outer_held_out_object_id": outer,
            "object_id": target,
            "episode_id": object_ids.index(target),
        }
        for outer in object_ids
        for target in object_ids
    ]
    batch = {
        "prediction_batch_id": scorer.RAW_PREDICTION_BATCH_ID,
        "records": records,
    }
    authorization = _authorization(scorer_revision="4" * 40, workflow_run_id=456)
    endpoint_manifest = {"manifest_id": _digest("endpoint-manifest")}
    result = {
        "source_gate_passed": False,
        "source_continuation_authorized": False,
        "claim_authorized": False,
        "evidence_id": _digest("evidence"),
        "result_id": _digest("result"),
    }

    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    batch_path = candidate_root / scorer.CANDIDATE_BATCH_FILENAME
    panel_path = candidate_root / scorer.CANDIDATE_RECEIPT_FILENAME
    execution_path = tmp_path / "execution.json"
    plan_path = tmp_path / "plan.json"
    authorization_path = tmp_path / "authorization.json"
    endpoint_manifest_path = tmp_path / "endpoint-manifest.json"
    candidate_directory = tmp_path / "candidate-cell"
    candidate_directory.mkdir()
    candidate_seal = candidate_directory / scorer.CANDIDATE_SEAL_FILENAME
    for path, value in (
        (batch_path, {}),
        (panel_path, {}),
        (execution_path, {}),
        (plan_path, {}),
        (authorization_path, authorization),
        (endpoint_manifest_path, endpoint_manifest),
        (candidate_seal, {}),
    ):
        _write_json(path, value)

    monkeypatch.setattr(
        scorer, "load_deform360_v61_source_scoring_amendment", lambda _path: {}
    )
    monkeypatch.setattr(
        scorer,
        "load_deform360_joint_sparse_source_execution_lock_v5",
        lambda _path: {},
    )
    monkeypatch.setattr(scorer, "_cohort", lambda _lock: cohort)
    monkeypatch.setattr(
        scorer,
        "validate_deform360_v61_candidate_panel",
        lambda **_kwargs: {"receipt_id": scorer.CANDIDATE_PANEL_RECEIPT_ID},
    )
    monkeypatch.setattr(
        scorer,
        "validate_deform360_v6_raw_nested_batch",
        lambda _value, **_kwargs: batch,
    )
    monkeypatch.setattr(
        scorer,
        "build_deform360_v61_source_suffix_authorization",
        lambda **_kwargs: authorization,
    )
    monkeypatch.setattr(
        scorer,
        "_load_exact_source_plan",
        lambda _path: {"plan_id": scorer.UPSTREAM_SOURCE_PLAN_ID},
    )
    monkeypatch.setattr(
        scorer,
        "validate_deform360_v61_source_endpoint_manifest",
        lambda _value, **_kwargs: endpoint_manifest,
    )
    monkeypatch.setattr(
        scorer,
        "_endpoint_views_by_object",
        lambda **_kwargs: (
            {object_id: () for object_id in object_ids},
            {
                object_id: {f"endpoint/{object_id}.npz": _digest(object_id)}
                for object_id in object_ids
            },
        ),
    )
    monkeypatch.setattr(
        scorer, "_candidate_directory", lambda *_args, **_kwargs: candidate_directory
    )
    monkeypatch.setattr(
        scorer,
        "score_deform360_v61_candidate_artifact",
        lambda **_kwargs: ({B0: {"point_loss": 1.0}}, {"common_query_count": 4}),
    )
    monkeypatch.setattr(
        scorer,
        "build_deform360_v6_raw_nested_outcome",
        lambda **kwargs: {
            "outcome_id": _digest(str(kwargs["prediction_record_id"])),
            "prediction_record_id": kwargs["prediction_record_id"],
        },
    )
    evidence = {"evidence_id": result["evidence_id"], "outcome_count": 100}
    monkeypatch.setattr(
        scorer,
        "assemble_deform360_v6_nested_evidence",
        lambda **_kwargs: evidence,
    )
    monkeypatch.setattr(
        scorer,
        "publish_deform360_v6_nested_evidence",
        lambda value, path, **_kwargs: scorer._publish_or_validate_json(  # noqa: SLF001
            value, path, label="source evidence"
        ),
    )
    monkeypatch.setattr(
        scorer,
        "evaluate_deform360_v6_nested_source_gate",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        scorer,
        "publish_deform360_v6_nested_result",
        lambda value, path, **_kwargs: scorer._publish_or_validate_json(  # noqa: SLF001
            value, path, label="source result"
        ),
    )

    kwargs = {
        "source_scoring_amendment_path": AMENDMENT,
        "execution_lock_path": tmp_path / "lock.json",
        "candidate_root": candidate_root,
        "candidate_execution_receipt_path": execution_path,
        "upstream_source_plan_path": plan_path,
        "authorization_path": authorization_path,
        "endpoint_manifest_path": endpoint_manifest_path,
        "endpoint_root": tmp_path,
        "output_root": tmp_path / "scores",
        "scorer_revision": "4" * 40,
        "runner_name": "workstation2",
        "workflow_run_id": 456,
        "workflow_run_attempt": 1,
    }
    receipt = scorer.publish_deform360_v61_source_scores(**kwargs)
    assert receipt["status"] == "source-reference-retained"
    assert receipt["outcome_count"] == 100
    assert len(list((tmp_path / "scores" / "source-outcomes").glob("*.json"))) == 100
    assert scorer.publish_deform360_v61_source_scores(**kwargs) == receipt
