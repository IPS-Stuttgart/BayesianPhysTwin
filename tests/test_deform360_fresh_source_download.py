from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesian_phystwin.deform360_fresh_source_download import (
    DATASET_REVISION,
    build_fresh_download_manifest,
    download_fresh_episode_sources_from_index,
    download_fresh_source_queue,
    download_fresh_source_queue_by_object,
    fresh_source_download_plan,
    select_episode_camera_source_files,
    validate_fresh_download_root,
)


def _canonical_sha256(payload: dict[str, object], *, digest_key: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _queue(tmp_path: Path, *, count: int = 12) -> Path:
    candidates = [
        {
            "queue_rank": index + 1,
            "object_id": f"{index + 1:03d}-source-object",
            "episode_id": 0,
            "category": ("filament", "sheet", "volumetric")[index % 3],
        }
        for index in range(count)
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360FreshSourceStagingQueue",
        "status": "source_only_locked_before_payload",
        "source_lock": {"implementation_commit": "1" * 40},
        "candidates": candidates,
        "information_boundary": {
            "episode_media_read_before_queue_lock": False,
            "processed_geometry_read_before_queue_lock": False,
            "future_object_positions_deserialized": False,
            "outcome_or_metric_read": False,
            "held_v8_target_query_score_barrier_or_outcome_access": False,
        },
    }
    payload["queue_sha256"] = _canonical_sha256(payload, digest_key="queue_sha256")
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_metadata(path: Path, *, bimanual: str = "no") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "object": path.parent.name,
                "sequences": {"0": {"bimanual": bimanual}},
            }
        ),
        encoding="utf-8",
    )


def test_download_plan_is_exact_and_tamper_evident(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    plan = fresh_source_download_plan(queue)

    assert plan.revision == DATASET_REVISION
    assert len(plan.object_ids) == 12
    assert plan.allow_patterns[0] == "raw/001-source-object/*"
    assert plan.allow_patterns[-1] == "raw/012-source-object/*"

    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["candidates"][0]["episode_id"] = 1
    queue.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum changed"):
        fresh_source_download_plan(queue)


def test_download_root_rejects_unqueued_objects(tmp_path: Path) -> None:
    plan = fresh_source_download_plan(_queue(tmp_path))
    output = tmp_path / "download"
    (output / "raw" / "999-unlocked").mkdir(parents=True)

    with pytest.raises(ValueError, match="unlocked objects"):
        validate_fresh_download_root(output, plan=plan, require_complete=False)


def test_queue_download_inventories_without_deserializing_payload(
    tmp_path: Path,
) -> None:
    queue = _queue(tmp_path)
    output = tmp_path / "download"
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        root = Path(str(kwargs["local_dir"]))
        for pattern in kwargs["allow_patterns"]:
            object_id = str(pattern).split("/")[1]
            object_root = root / "raw" / object_id
            _write_metadata(object_root / "metadata.json")
            (object_root / "episode_0000").mkdir()
            (object_root / "episode_0000" / "future.pkl").write_bytes(
                b"not a valid pickle"
            )
        return str(root)

    manifest = download_fresh_source_queue(
        queue,
        output,
        max_workers=3,
        snapshot_download=snapshot_download,
    )

    assert len(calls) == 1
    assert calls[0]["revision"] == DATASET_REVISION
    assert calls[0]["max_workers"] == 3
    assert manifest["object_count"] == 12
    assert manifest["information_boundary"]["episode_payload_deserialized"] is False
    assert manifest["information_boundary"]["target_metrics_opened"] is False
    assert manifest["manifest_sha256"] == _canonical_sha256(
        manifest, digest_key="manifest_sha256"
    )


def test_manifest_rejects_invalid_metadata_enum(tmp_path: Path) -> None:
    plan = fresh_source_download_plan(_queue(tmp_path))
    output = tmp_path / "download"
    for object_id in plan.object_ids:
        _write_metadata(
            output / "raw" / object_id / "metadata.json",
            bimanual="yess" if object_id == plan.object_ids[-1] else "no",
        )

    with pytest.raises(ValueError, match="metadata enum domain changed"):
        build_fresh_download_manifest(output, plan=plan)


def test_by_object_download_never_lists_the_global_tree(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    plan = fresh_source_download_plan(queue)
    output = tmp_path / "download"
    listed_paths: list[str] = []

    def list_repo_tree(**kwargs: object) -> list[SimpleNamespace]:
        path = str(kwargs["path_in_repo"])
        listed_paths.append(path)
        return [
            SimpleNamespace(path=f"{path}/metadata.json", blob_id="metadata"),
            SimpleNamespace(path=f"{path}/episode_0000/data.bin", blob_id="data"),
        ]

    def hub_download(**kwargs: object) -> str:
        path = output / str(kwargs["filename"])
        if path.name == "metadata.json":
            _write_metadata(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"source bytes")
        return str(path)

    manifest = download_fresh_source_queue_by_object(
        queue,
        output,
        max_workers=2,
        object_delay_seconds=0.0,
        list_repo_tree=list_repo_tree,
        hub_download=hub_download,
    )

    assert listed_paths == [f"raw/{object_id}" for object_id in plan.object_ids]
    assert manifest["object_count"] == len(plan.object_ids)


def test_episode_source_selection_keeps_exact_camera_pairs_only() -> None:
    prefix = "raw/001-source-object"
    paths = [
        f"{prefix}/metadata.json",
        f"{prefix}/calibration_refined/intrinsics.npy",
        f"{prefix}/calibration_refined/extrinsics.npy",
        f"{prefix}/brics-odroid-001_cam0/camera_100.mp4",
        f"{prefix}/brics-odroid-001_cam0/camera_100.txt",
        f"{prefix}/brics-odroid-001_cam0/camera_200.mp4",
        f"{prefix}/brics-odroid-001_cam0/camera_200.txt",
        f"{prefix}/brics-odroid_tactilel_left/tactile_100.npy",
        f"{prefix}/brics-odroid_tactilel_left/tactile_100.txt",
    ]

    selected = select_episode_camera_source_files(
        paths,
        object_id="001-source-object",
        episode_id=0,
    )

    assert f"{prefix}/brics-odroid-001_cam0/camera_100.mp4" in selected
    assert f"{prefix}/brics-odroid-001_cam0/camera_100.txt" in selected
    assert not any("camera_200" in path for path in selected)
    assert not any("tactile" in path for path in selected)


def test_indexed_episode_download_is_camera_only(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    plan = fresh_source_download_plan(queue)
    output = tmp_path / "download"

    def list_object_files(object_id: str) -> list[str]:
        prefix = f"raw/{object_id}"
        return [
            f"{prefix}/metadata.json",
            f"{prefix}/calibration_refined/intrinsics.npy",
            f"{prefix}/calibration_refined/extrinsics.npy",
            f"{prefix}/brics-odroid-001_cam0/camera_100.mp4",
            f"{prefix}/brics-odroid-001_cam0/camera_100.txt",
            f"{prefix}/brics-odroid_tactilel_left/tactile_100.npy",
            f"{prefix}/brics-odroid_tactilel_left/tactile_100.txt",
        ]

    def hub_download(**kwargs: object) -> str:
        path = output / str(kwargs["filename"])
        if path.name == "metadata.json":
            _write_metadata(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"source bytes")
        return str(path)

    manifest = download_fresh_episode_sources_from_index(
        queue,
        output,
        max_workers=2,
        object_delay_seconds=0.0,
        list_object_files=list_object_files,
        hub_download=hub_download,
    )

    assert manifest["object_count"] == len(plan.object_ids)
    assert manifest["download_scope"] == "queued_episode_camera_source_only"
    assert manifest["tactile_included"] is False
    assert not list(output.rglob("*tactile*"))
