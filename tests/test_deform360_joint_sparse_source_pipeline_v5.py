from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

import bayesian_phystwin.deform360_joint_sparse_source_runner_v5 as source_runner
from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_joint_sparse_endpoint_v5 import (
    select_reserved_endpoint_views_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    evaluate_deform360_joint_sparse_source_gate_v5,
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_scoring_v5 import (
    build_deform360_joint_sparse_source_endpoint_plan_v5,
    publish_deform360_joint_sparse_source_scores_v5,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / (
    "protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json"
)
REVISION = "1" * 40


def _sha256(path: Path) -> str:
    return source_runner._sha256_file(path)


def _physical_archive(path: Path) -> None:
    rows = np.linspace(-0.02, 0.02, 16, dtype=np.float32)
    xy = np.stack(np.meshgrid(rows[:8], rows, indexing="xy"), axis=-1).reshape(-1, 2)
    frame_zero = np.column_stack((xy, np.ones(128, dtype=np.float32)))
    trajectory = np.repeat(frame_zero[None], 76, axis=0)
    np.savez(
        path,
        prediction_m=trajectory,
        persistence_m=trajectory,
        driven_readout_m=trajectory,
        zero_action_readout_m=trajectory,
        action_support=np.zeros(128, dtype=np.float32),
        frame_zero_points_m=frame_zero,
    )


def _source_objects(
    lock: Mapping[str, Any],
    *,
    physical_sha256: str,
    invalid_sha256: str,
) -> list[dict[str, Any]]:
    rows = cast(
        Sequence[Mapping[str, Any]],
        cast(Mapping[str, Any], lock["cohort"])["development_objects"],
    )
    result = []
    for index, row in enumerate(rows):
        object_id = cast(str, row["object_id"])
        cameras = tuple(f"camera-{camera}" for camera in range(6))
        reserved = select_reserved_endpoint_views_v5(object_id, cameras, count=2)
        likelihood = tuple(camera for camera in cameras if camera not in reserved)[:2]
        result.append(
            {
                "object_id": object_id,
                "episode_id": row["episode_id"],
                "stratum": row["stratum"],
                "raw_prefix_range_half_open": [100 + index, 158 + index],
                "all_camera_ids": list(cameras),
                "reserved_endpoint_camera_ids": list(reserved),
                "physical": {
                    "path": "physical.npz",
                    "sha256": physical_sha256,
                    "physical_mode": "warp_twin",
                },
                "visual_windows": [
                    {
                        "camera_id": camera,
                        "decoded_uniform": {
                            "path": "invalid-provider.npz",
                            "sha256": invalid_sha256,
                        },
                        "metric_prefix": {
                            "path": "invalid-provider.npz",
                            "sha256": invalid_sha256,
                        },
                    }
                    for camera in likelihood
                ],
                "contact_prefix": {
                    "path": "contact",
                    "manifest_file_sha256": "2" * 64,
                    "materialization_id": "3" * 64,
                },
            }
        )
    return result


def _endpoint_archive(path: Path, *, raw_start: int) -> None:
    np.savez(
        path,
        frame_indices=np.arange(58, 76, dtype=np.int64),
        raw_frame_indices=np.arange(raw_start, raw_start + 18, dtype=np.int64),
        depth_m=np.ones((18, 8, 8), dtype=np.float32),
        object_mask=np.ones((18, 8, 8), dtype=np.bool_),
        intrinsics=np.asarray(
            [[100.0, 0.0, 3.5], [0.0, 100.0, 3.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        camera_to_world=np.eye(4, dtype=np.float64),
    )


def test_technical_source_dry_run_seals_before_scoring(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(LOCK_PATH)
    input_root = tmp_path / "prefix-inputs"
    input_root.mkdir()
    _physical_archive(input_root / "physical.npz")
    (input_root / "invalid-provider.npz").write_bytes(b"not-an-npz")
    (input_root / "contact").mkdir()
    source_plan = source_runner.build_deform360_joint_sparse_source_prediction_plan_v5(
        lock=lock,
        implementation_revision=REVISION,
        objects=_source_objects(
            lock,
            physical_sha256=_sha256(input_root / "physical.npz"),
            invalid_sha256=_sha256(input_root / "invalid-provider.npz"),
        ),
    )
    source_plan_path = tmp_path / "source-plan.json"
    write_atomic_json(source_plan, source_plan_path, overwrite=False)
    monkeypatch.setattr(
        source_runner,
        "_verified_contact_directory",
        lambda root, record: root / cast(str, record["path"]),
    )
    prediction_root = tmp_path / "predictions"
    prediction_receipt = (
        source_runner.publish_deform360_joint_sparse_source_prediction_panel_v5(
            execution_lock_path=LOCK_PATH,
            source_plan_path=source_plan_path,
            input_root=input_root,
            output_root=prediction_root,
        )
    )
    assert prediction_receipt["prediction_record_count"] == 100
    assert len(list((prediction_root / "source-seals").glob("*.json"))) == 100
    assert not list(prediction_root.rglob("*confirmation*"))

    batch = load_strict_json_object(
        prediction_root / "source-prediction-batch.json",
        label="source prediction batch",
    )
    endpoint_root = tmp_path / "endpoint-inputs"
    endpoint_root.mkdir()
    endpoint_objects = []
    for row in cast(Sequence[Mapping[str, Any]], source_plan["objects"]):
        object_id = cast(str, row["object_id"])
        raw_start = cast(Sequence[int], row["raw_prefix_range_half_open"])[1]
        views = []
        for camera_id in cast(Sequence[str], row["reserved_endpoint_camera_ids"]):
            relative = Path(object_id) / f"{camera_id}.npz"
            path = endpoint_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _endpoint_archive(path, raw_start=raw_start)
            views.append(
                {
                    "camera_id": camera_id,
                    "endpoint_archive": {
                        "path": relative.as_posix(),
                        "sha256": _sha256(path),
                    },
                }
            )
        endpoint_objects.append(
            {
                "object_id": object_id,
                "episode_id": row["episode_id"],
                "stratum": row["stratum"],
                "all_camera_ids": row["all_camera_ids"],
                "raw_endpoint_range_half_open": [raw_start, raw_start + 18],
                "reserved_views": views,
            }
        )
    endpoint_plan = build_deform360_joint_sparse_source_endpoint_plan_v5(
        lock=lock,
        source_prediction_plan=source_plan,
        prediction_batch=batch,
        source_prediction_receipt=prediction_receipt,
        objects=endpoint_objects,
    )
    endpoint_plan_path = tmp_path / "endpoint-plan.json"
    write_atomic_json(endpoint_plan, endpoint_plan_path, overwrite=False)

    scoring_root = tmp_path / "scores"
    scoring_receipt = publish_deform360_joint_sparse_source_scores_v5(
        execution_lock_path=LOCK_PATH,
        source_prediction_plan_path=source_plan_path,
        source_prediction_root=prediction_root,
        endpoint_plan_path=endpoint_plan_path,
        endpoint_input_root=endpoint_root,
        output_root=scoring_root,
    )
    assert scoring_receipt["endpoint_report_count"] == 100
    assert scoring_receipt["outcome_count"] == 100
    assert (scoring_root / "source-suffix-opening-authorization.json").is_file()
    assert len(list((scoring_root / "endpoint-reports").glob("*.json"))) == 100
    assert not list(scoring_root.rglob("*confirmation*"))
    evidence = load_strict_json_object(
        scoring_root / "source-evidence.json",
        label="source evidence",
    )
    gate = evaluate_deform360_joint_sparse_source_gate_v5(evidence, lock)
    assert gate["gate_passed"] is False
    assert gate["confirmation_access_authorized"] is False
