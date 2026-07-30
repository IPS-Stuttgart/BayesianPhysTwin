from __future__ import annotations

from pathlib import Path

import pytest

from bayesian_phystwin.cloth_sim2real_protocol import (
    EXPECTED_CLOTHS,
    EXPECTED_TASKS,
    ClothSim2RealProtocolConfig,
    build_cloth_sim2real_dataset_manifest,
)


def _write_dataset(root: Path, config: ClothSim2RealProtocolConfig) -> Path:
    dataset = root / "Benchmarking_cloth"
    for cloth_id in EXPECTED_CLOTHS:
        for repeat_index in range(3):
            for task in EXPECTED_TASKS:
                frame_count = (
                    config.minimum_dynamic_frames
                    if task == "dynamic"
                    else config.minimum_quasi_static_frames
                )
                cloud = dataset / f"{cloth_id}_{repeat_index}" / task / "cloud"
                cloud.mkdir(parents=True)
                for frame in range(frame_count):
                    (cloud / f"{frame:05d}.ply").write_bytes(b"ply\n")
    return dataset


def test_manifest_locks_repeat_splits_without_reading_coordinates(
    tmp_path: Path,
) -> None:
    config = ClothSim2RealProtocolConfig(
        minimum_dynamic_frames=12,
        minimum_quasi_static_frames=16,
        minimum_prefix_fit_frames=2,
        minimum_prefix_validation_frames=1,
    )
    dataset = _write_dataset(tmp_path, config)

    manifest = build_cloth_sim2real_dataset_manifest(
        dataset,
        archive_sha256="1" * 64,
        config=config,
    )

    assert len(manifest.cases) == 18
    assert sum(case.split == "source" for case in manifest.cases) == 6
    assert sum(case.split == "calibration" for case in manifest.cases) == 6
    assert sum(case.split == "target" for case in manifest.cases) == 6
    assert manifest.descriptor()["information_boundary"]["point_coordinates_read"] is False
    assert manifest.artifact_sha256 != "0" * 64


def test_manifest_rejects_a_missing_target_case(tmp_path: Path) -> None:
    config = ClothSim2RealProtocolConfig(
        minimum_dynamic_frames=12,
        minimum_quasi_static_frames=16,
        minimum_prefix_fit_frames=2,
        minimum_prefix_validation_frames=1,
    )
    dataset = _write_dataset(tmp_path, config)
    missing = dataset / "linen_rag_2" / "dynamic"
    for path in (missing / "cloud").iterdir():
        path.unlink()
    (missing / "cloud").rmdir()
    missing.rmdir()

    with pytest.raises(ValueError, match="missing point-cloud directory"):
        build_cloth_sim2real_dataset_manifest(
            dataset,
            archive_sha256="2" * 64,
            config=config,
        )


def test_manifest_rejects_noncontiguous_frames(tmp_path: Path) -> None:
    config = ClothSim2RealProtocolConfig(
        minimum_dynamic_frames=12,
        minimum_quasi_static_frames=16,
        minimum_prefix_fit_frames=2,
        minimum_prefix_validation_frames=1,
    )
    dataset = _write_dataset(tmp_path, config)
    cloud = dataset / "cotton_rag_0" / "dynamic" / "cloud"
    (cloud / "00011.ply").rename(cloud / "00012.ply")

    with pytest.raises(ValueError, match="not contiguous"):
        build_cloth_sim2real_dataset_manifest(
            dataset,
            archive_sha256="3" * 64,
            config=config,
        )


def test_manifest_digest_binds_frame_count(tmp_path: Path) -> None:
    config = ClothSim2RealProtocolConfig(
        minimum_dynamic_frames=12,
        minimum_quasi_static_frames=16,
        minimum_prefix_fit_frames=2,
        minimum_prefix_validation_frames=1,
    )
    dataset = _write_dataset(tmp_path, config)
    first = build_cloth_sim2real_dataset_manifest(
        dataset,
        archive_sha256="4" * 64,
        config=config,
    )
    cloud = dataset / "chequered_rag_0" / "dynamic" / "cloud"
    (cloud / "00012.ply").write_bytes(b"ply\n")
    second = build_cloth_sim2real_dataset_manifest(
        dataset,
        archive_sha256="4" * 64,
        config=config,
    )

    assert first.artifact_sha256 != second.artifact_sha256
