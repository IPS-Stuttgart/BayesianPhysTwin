from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from bayesian_phystwin import deform360_joint_sparse_public_endpoint_v5_2 as module
from bayesian_phystwin._portable_contracts import load_strict_json_object

ROOT = Path(__file__).resolve().parents[1]
PROCESSING_LOCK_PATH = ROOT / (
    "protocols/locks/deform360_joint_sparse_public_endpoint_processing_v5_2.json"
)
PROCESSING_RUNNER_PATH = ROOT / (
    "scripts/remote/process_deform360_joint_sparse_public_endpoint_v5_2.py"
)


def _episode(tmp_path: Path, *, camera: str = "camera-0") -> Path:
    episode = tmp_path / "processed" / "object" / "episode_0000"
    camera_dir = episode / camera
    camera_dir.mkdir(parents=True)
    depth = np.arange(81 * 4 * 5, dtype=np.uint16).reshape(81, 4, 5)
    mask = np.ones_like(depth, dtype=np.uint8)
    with h5py.File(camera_dir / "rendered_depth.h5", "w") as stream:
        stream.create_dataset("data", data=depth)
    with h5py.File(camera_dir / "mask_refined.h5", "w") as stream:
        stream.create_dataset("data", data=mask)
    np.save(episode / "undistorted_intrinsics.npy", {camera: np.eye(3)})
    np.save(episode / "extrinsics.npy", {camera: np.eye(4)})
    return episode


def test_materializes_exact_metric_endpoint_archive(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    output = tmp_path / "endpoint.npz"
    record = module.materialize_public_endpoint_archive_v5_2(
        object_id="object",
        camera_id="camera-0",
        raw_endpoint_range_half_open=(144, 162),
        episode_directory=episode,
        output_path=output,
    )

    assert record["archive_sha256"]
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "camera_to_world",
            "depth_m",
            "frame_indices",
            "intrinsics",
            "object_mask",
            "raw_frame_indices",
        }
        np.testing.assert_array_equal(archive["frame_indices"], np.arange(58, 76))
        np.testing.assert_array_equal(archive["raw_frame_indices"], np.arange(144, 162))
        assert archive["depth_m"].shape == (18, 4, 5)
        assert archive["depth_m"].dtype == np.float32
        assert archive["object_mask"].dtype == np.bool_
        np.testing.assert_array_equal(archive["intrinsics"], np.eye(3))
        np.testing.assert_array_equal(archive["camera_to_world"], np.eye(4))


def test_refuses_nonmillimetre_depth(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    depth_path = episode / "camera-0" / "rendered_depth.h5"
    with h5py.File(depth_path, "w") as stream:
        stream.create_dataset("data", data=np.ones((81, 4, 5), dtype=np.float32))

    with pytest.raises(ValueError, match="uint16 millimetres"):
        module.materialize_public_endpoint_archive_v5_2(
            object_id="object",
            camera_id="camera-0",
            raw_endpoint_range_half_open=(144, 162),
            episode_directory=episode,
            output_path=tmp_path / "endpoint.npz",
        )


def test_refuses_archive_overwrite(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    output = tmp_path / "endpoint.npz"
    output.write_bytes(b"existing")

    with pytest.raises(ValueError, match="already exists"):
        module.materialize_public_endpoint_archive_v5_2(
            object_id="object",
            camera_id="camera-0",
            raw_endpoint_range_half_open=(144, 162),
            episode_directory=episode,
            output_path=output,
        )


def test_refuses_symlinked_source_file(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    depth_path = episode / "camera-0" / "rendered_depth.h5"
    real_depth_path = episode / "camera-0" / "real-rendered-depth.h5"
    depth_path.rename(real_depth_path)
    depth_path.symlink_to(real_depth_path)

    with pytest.raises(ValueError, match="path must not contain symlinks"):
        module.materialize_public_endpoint_archive_v5_2(
            object_id="object",
            camera_id="camera-0",
            raw_endpoint_range_half_open=(144, 162),
            episode_directory=episode,
            output_path=tmp_path / "endpoint.npz",
        )


def test_processing_lock_binds_public_sealed_panel_without_human_approval() -> None:
    processing_lock = load_strict_json_object(
        PROCESSING_LOCK_PATH, label="public endpoint processing lock"
    )
    execution_lock = {
        "execution_lock_id": processing_lock["execution_lock_id"],
        "public_measurements": {
            "dataset_repository": processing_lock["dataset_repository"],
            "dataset_revision": processing_lock["dataset_revision"],
        },
    }

    validated = module.validate_public_endpoint_processing_lock_v5_2(
        processing_lock,
        execution_lock=execution_lock,
        source_plan={"plan_id": processing_lock["source_prediction_plan_id"]},
        prediction_batch={
            "prediction_batch_id": processing_lock["prediction_batch_id"]
        },
        prediction_receipt={
            "receipt_id": processing_lock["source_prediction_receipt_id"]
        },
    )

    assert validated == processing_lock
    assert validated["information_boundary"]["human_approval_required"] is False
    assert validated["information_boundary"]["new_measurements_required"] is False
    runtime = validated["processing"]["runtime"]
    assert runtime["ffmpeg_sha256"] == (
        "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
    )
    assert validated["processing"]["windowing"]["video_materialization"] == (
        "exact-ffmpeg-libx264-crf12-30hz-legacy-vsync-cfr"
    )


def test_processing_runner_uses_only_the_locked_legacy_sync_option() -> None:
    source = PROCESSING_RUNNER_PATH.read_text(encoding="utf-8")

    assert '"-vsync"' in source
    assert '"-fps_mode"' not in source


def test_materializes_complete_public_endpoint_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "source-plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    prediction_root = tmp_path / "prediction"
    prediction_root.mkdir()
    (prediction_root / "source-prediction-batch.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (prediction_root / "source-prediction-receipt.json").write_text(
        "{}\n", encoding="utf-8"
    )
    processing_lock_path = tmp_path / "processing-lock.json"
    processing_lock_path.write_text("{}\n", encoding="utf-8")
    processed_root = tmp_path / "processed"
    processed_root.mkdir()

    execution_lock = {
        "execution_lock_id": "execution-lock",
        "public_measurements": {
            "dataset_repository": "dataset/repository",
            "dataset_revision": "1" * 40,
            "endpoint_geometry_derivation_repository": "geometry/repository",
            "endpoint_geometry_derivation_revision": "2" * 40,
        },
    }
    plan = {
        "plan_id": "source-plan",
        "objects": [
            {
                "object_id": "object",
                "episode_id": "episode_0000",
                "stratum": "cloth",
                "all_camera_ids": ["camera-0", "camera-1", "camera-2"],
                "raw_prefix_range_half_open": [63, 144],
            }
        ],
    }
    prediction_batch = {"prediction_batch_id": "prediction-batch"}
    prediction_receipt = {"receipt_id": "prediction-receipt"}
    processing_lock = {"lock_id": "processing-lock"}

    monkeypatch.setattr(
        module,
        "load_deform360_joint_sparse_source_execution_lock_v5",
        lambda _path: execution_lock,
    )
    monkeypatch.setattr(
        module,
        "validate_deform360_joint_sparse_source_prediction_plan_v5_2",
        lambda _value, *, lock: plan,
    )
    monkeypatch.setattr(
        module,
        "validate_deform360_joint_sparse_source_prediction_batch_v5",
        lambda _value, lock: prediction_batch,
    )
    monkeypatch.setattr(
        module,
        "validate_deform360_joint_sparse_source_prediction_receipt_v5_2",
        lambda _value, **_kwargs: prediction_receipt,
    )
    monkeypatch.setattr(
        module,
        "validate_public_endpoint_processing_lock_v5_2",
        lambda _value, **_kwargs: processing_lock,
    )
    monkeypatch.setattr(
        module,
        "select_reserved_endpoint_views_v5",
        lambda _object_id, cameras, *, count: tuple(cameras[:count]),
    )

    materialized: list[tuple[str, str]] = []

    def fake_materialize(**kwargs: object) -> dict[str, object]:
        object_id = str(kwargs["object_id"])
        camera_id = str(kwargs["camera_id"])
        destination = Path(str(kwargs["output_path"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"{object_id}:{camera_id}".encode())
        materialized.append((object_id, camera_id))
        return {
            "object_id": object_id,
            "camera_id": camera_id,
            "archive_sha256": camera_id.rjust(64, "0"),
            "source_files_sha256": {},
        }

    monkeypatch.setattr(
        module,
        "materialize_public_endpoint_archive_v5_2",
        fake_materialize,
    )

    objects_output = tmp_path / "objects.json"
    manifest_output = tmp_path / "manifest.json"
    manifest = module.materialize_public_endpoint_inputs_v5_2(
        execution_lock_path=tmp_path / "execution-lock.json",
        source_prediction_plan_path=plan_path,
        source_prediction_root=prediction_root,
        processing_lock_path=processing_lock_path,
        processed_root=processed_root,
        output_root=tmp_path / "endpoint-output",
        objects_output_path=objects_output,
        manifest_output_path=manifest_output,
    )

    assert materialized == [
        ("object", "camera-0"),
        ("object", "camera-1"),
    ]
    objects = load_strict_json_object(
        objects_output, label="materialized public endpoint objects"
    )
    assert [view["camera_id"] for view in objects["objects"][0]["reserved_views"]] == [
        "camera-0",
        "camera-1",
    ]
    assert manifest["execution_lock_id"] == "execution-lock"
    assert manifest["source_prediction_plan_id"] == "source-plan"
    assert manifest["source_prediction_receipt_id"] == "prediction-receipt"
    assert manifest["prediction_batch_id"] == "prediction-batch"
    assert manifest["processing_lock_id"] == "processing-lock"
    assert manifest["manifest_id"]
    assert (
        load_strict_json_object(manifest_output, label="public endpoint manifest")
        == manifest
    )
