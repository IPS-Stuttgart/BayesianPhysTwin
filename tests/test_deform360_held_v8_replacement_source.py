from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.deform360_held_v8_replacement_source as source


CAMERAS = (
    "brics-odroid-001_cam0",
    "brics-odroid-002_cam0",
    "brics-odroid-004_cam0",
    "brics-odroid-006_cam0",
    "brics-odroid-007_cam0",
    "brics-odroid-007_cam1",
    "brics-odroid-008_cam0",
    "brics-odroid-008_cam1",
    "brics-odroid-010_cam0",
    "brics-odroid-010_cam1",
    "brics-odroid-011_cam0",
    "brics-odroid-012_cam0",
    "brics-odroid-012_cam1",
    "brics-odroid-013_cam0",
    "brics-odroid-014_cam0",
    "brics-odroid-014_cam1",
    "brics-odroid-015_cam0",
    "brics-odroid-015_cam1",
    "brics-odroid-016_cam0",
    "brics-odroid-017_cam0",
    "brics-odroid-017_cam1",
    "brics-odroid-018_cam0",
    "brics-odroid-018_cam1",
    "brics-odroid-019_cam0",
    "brics-odroid-019_cam1",
    "brics-odroid-021_cam0",
    "brics-odroid-021_cam1",
    "brics-odroid-022_cam0",
    "brics-odroid-022_cam1",
    "brics-odroid-023_cam0",
    "brics-odroid-024_cam0",
    "brics-odroid-024_cam1",
    "brics-odroid-025_cam0",
    "brics-odroid-025_cam1",
    "brics-odroid-027_cam0",
    "brics-odroid-027_cam1",
    "brics-odroid-028_cam0",
)


def _metadata_bytes() -> bytes:
    return json.dumps(
        {
            "object": source.REPLACEMENT_OBJECT_ID,
            "sequences": {
                "3": {
                    "action": "drag center",
                    "bimanual": "no",
                    "nonprehensile": "no",
                }
            },
        },
        sort_keys=True,
    ).encode()


def _bytes_for_path(path: str) -> bytes:
    if path.endswith("metadata.json"):
        return _metadata_bytes()
    return f"fixture:{path}".encode()


def _inventory_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for camera_index, camera in enumerate(CAMERAS):
        for episode in range(10):
            if camera == source.REFERENCE_CAMERA:
                stem = f"{camera}_{1_769_121_824_929_786 + episode}"
            else:
                stem = (
                    f"{camera}_{1_700_000_000_000_000 + camera_index * 100 + episode}"
                )
            for suffix in (".mp4", ".txt"):
                path = f"{source.REMOTE_OBJECT_ROOT}/{camera}/{stem}{suffix}"
                payload = _bytes_for_path(path)
                entry: dict[str, Any] = {
                    "type": "file",
                    "path": path,
                    "size": len(payload),
                    "oid": hashlib.sha1(payload).hexdigest(),  # noqa: S324 - Git blob fixture
                }
                if suffix == ".mp4":
                    entry["lfs"] = {"oid": hashlib.sha256(payload).hexdigest()}
                entries.append(entry)
    for relative in (*source._CALIBRATION_RELATIVE_PATHS, "metadata.json"):
        path = f"{source.REMOTE_OBJECT_ROOT}/{relative}"
        payload = _bytes_for_path(path)
        entry = {
            "type": "file",
            "path": path,
            "size": len(payload),
            "oid": hashlib.sha1(payload).hexdigest(),  # noqa: S324 - Git blob fixture
        }
        if path.endswith(".npy"):
            entry["lfs"] = {"oid": hashlib.sha256(payload).hexdigest()}
        entries.append(entry)
    entries.extend(
        [
            {
                "type": "file",
                "path": f"{source.REMOTE_OBJECT_ROOT}/brics-odroid_tactile0/audio.wav",
                "size": 1,
                "oid": "a" * 40,
            },
            {
                "type": "directory",
                "path": f"{source.REMOTE_OBJECT_ROOT}/brics-odroid_tactile0",
                "size": 0,
                "oid": "b" * 40,
            },
        ]
    )
    return entries


def _fixture_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, Any]], source.SelectedInventory]:
    entries = _inventory_entries()
    preliminary = source.derive_selected_inventory(
        entries,
        expected_records_sha256=source._canonical_sha256(
            sorted(
                (
                    source._inventory_file_record(item)
                    for item in entries
                    if item.get("type") == "file"
                    and (
                        item["path"].endswith("metadata.json")
                        or item["path"].endswith("dist.npy")
                        or item["path"].endswith("extrinsics.npy")
                        or item["path"].endswith("intrinsics.npy")
                        or any(
                            item["path"].endswith(f"/{stem}{suffix}")
                            for camera, stem in _selected_stems(entries).items()
                            for suffix in (".mp4", ".txt")
                        )
                    )
                ),
                key=lambda record: record["path"],
            )
        ),
        expected_total_size_bytes=_selected_total_size(entries),
    )
    monkeypatch.setattr(source, "INVENTORY_RECORDS_SHA256", preliminary.records_sha256)
    monkeypatch.setattr(
        source, "EXPECTED_DOWNLOAD_SIZE_BYTES", preliminary.total_size_bytes
    )
    content_records = [
        {
            "path": path,
            "size_bytes": len(_bytes_for_path(path)),
            "sha256": hashlib.sha256(_bytes_for_path(path)).hexdigest(),
        }
        for path in preliminary.allow_patterns
    ]
    monkeypatch.setattr(
        source,
        "DOWNLOADED_CONTENT_RECORDS_SHA256",
        source._canonical_sha256(content_records),
    )
    return entries, preliminary


def _selected_stems(entries: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for camera in CAMERAS:
        mp4 = sorted(
            item["path"]
            for item in entries
            if item.get("type") == "file"
            and item["path"].startswith(f"{source.REMOTE_OBJECT_ROOT}/{camera}/")
            and item["path"].endswith(".mp4")
        )
        result[camera] = Path(mp4[3]).stem
    return result


def _selected_total_size(entries: list[dict[str, Any]]) -> int:
    stems = _selected_stems(entries)
    selected = []
    for item in entries:
        path = item.get("path", "")
        if any(
            path.endswith(f"/{stem}{suffix}")
            for stem in stems.values()
            for suffix in (".mp4", ".txt")
        ) or path in {
            f"{source.REMOTE_OBJECT_ROOT}/{relative}"
            for relative in (*source._CALIBRATION_RELATIVE_PATHS, "metadata.json")
        }:
            selected.append(item)
    return sum(item["size"] for item in selected)


def _write_download(local_dir: str, allow_patterns: list[str]) -> str:
    root = Path(local_dir)
    for relative in allow_patterns:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_bytes_for_path(relative))
    return str(root)


def _write_valid_robot(path: Path, *, frame_count: int) -> None:
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    transforms[:, 0, 3] = np.arange(frame_count, dtype=np.float64) * 0.001
    openings = np.full(frame_count, 0.02, dtype=np.float64)
    actions = np.zeros((frame_count, 5, 3), dtype=np.float64)
    actions[:, 0, :] = transforms[:, :3, 3]
    actions[:, 1:4, :] = transforms[:, :3, :3]
    actions[:, 4, 0] = openings
    np.savez(
        path,
        format_version=np.asarray(1, dtype=np.uint16),
        actions=actions,
        T_worlds=transforms,
        openings=openings,
        bimanual=np.asarray(False, dtype=np.bool_),
    )


def _fake_command_runner(
    commands: list[tuple[str, ...]],
):
    def run(command: tuple[str, ...], *, cwd: Path, env: dict[str, str]) -> object:
        commands.append(tuple(command))
        assert env["PYTHONPATH"] == str(cwd)
        assert env["PYTHONNOUSERSITE"] == "1"
        if command[2] == source._UNDISTORT_SCRIPT:
            output = Path(command[4]) / "episode_0000"
            downloaded_cameras = json.loads(command[5])
            aligned_cameras = [
                camera
                for camera in downloaded_cameras
                if camera not in source.UNCALIBRATED_CAMERAS
            ]
            output.mkdir(parents=True)
            (output / "alignment.json").write_text(
                json.dumps(
                    {
                        "episode_index": 0,
                        "cameras": aligned_cameras,
                        "frame_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            (output / "undistorted_intrinsics.npy").write_bytes(b"intrinsics")
            (output / "extrinsics.npy").write_bytes(b"extrinsics")
            for camera in aligned_cameras:
                camera_dir = output / camera
                camera_dir.mkdir()
                for name in source._CAMERA_OUTPUT_FILENAMES:
                    payload = b"{}" if name.endswith(".json") else name.encode()
                    (camera_dir / name).write_bytes(payload)
        elif command[2] == source._ROBOT_SCRIPT:
            episode = Path(command[3]) / "episode_0003"
            cameras = json.loads(command[4])
            robot = episode / "robot"
            robot.mkdir()
            _write_valid_robot(robot / "robot.npz", frame_count=2)
            (robot / "robot.meta.json").write_text(
                json.dumps(
                    {
                        "parameters": {
                            "seed": 0,
                            "bimanual": False,
                            "cameras": cameras,
                        },
                        "outputs": {"bimanual": False, "num_frames": 2},
                    }
                ),
                encoding="utf-8",
            )
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("unexpected processing command")
        return object()

    return run


def _paths(tmp_path: Path) -> source.ReplacementSourcePaths:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    code = tmp_path / "code"
    code.mkdir()
    python = tmp_path / "python"
    python.write_bytes(b"fixture executable")
    python.chmod(0o500)
    return source.ReplacementSourcePaths(
        download_root=tmp_path / "fresh-download",
        aligned_root=tmp_path / "fresh-aligned",
        inventory_manifest=manifests / "inventory.json",
        content_manifest=manifests / "content.json",
        aligned_source_manifest=manifests / "aligned-source.json",
        processing_code_root=code,
        python_executable=python,
    )


def test_frozen_public_inventory_constants_match_actual_preflight() -> None:
    assert source.INVENTORY_RECORDS_SHA256 == (
        "9a3c4755cc635bdd1702d6739f540401c532dae3c394f85416da54e81161a839"
    )
    assert source.DOWNLOADED_CONTENT_RECORDS_SHA256 == (
        "874c31f58f9f6679e6d625621e37a9d2591c708c551c0fd592b101c770650df2"
    )
    assert source.EXPECTED_DOWNLOAD_SIZE_BYTES == 95_165_257
    assert source.HF_DATASET_REVISION == ("7fea8e20231a47641d1d2bc8791920ec4e62ec5e")
    assert source.PROCESSING_CODE_REVISION == (
        "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
    )


def test_inventory_derivation_selects_only_pair_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries, expected = _fixture_selected(monkeypatch)
    selected = source.derive_selected_inventory(entries)

    assert selected == expected
    assert selected.camera_names == CAMERAS
    assert len(selected.allow_patterns) == 78
    assert selected.allow_patterns == tuple(sorted(selected.allow_patterns))
    assert selected.selected_stems[source.REFERENCE_CAMERA] == (
        source.REFERENCE_SELECTED_STEM
    )
    assert not any(
        "tactile" in path or path.endswith(".wav") for path in selected.allow_patterns
    )
    for camera in CAMERAS:
        stem = _selected_stems(entries)[camera]
        assert (
            f"{source.REMOTE_OBJECT_ROOT}/{camera}/{stem}.mp4"
            in selected.allow_patterns
        )
        assert (
            f"{source.REMOTE_OBJECT_ROOT}/{camera}/{stem}.txt"
            in selected.allow_patterns
        )


def test_inventory_derivation_rejects_missing_or_mismatched_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries, _selected = _fixture_selected(monkeypatch)
    missing = [
        item
        for item in entries
        if item.get("path")
        != f"{source.REMOTE_OBJECT_ROOT}/{CAMERAS[1]}/{_selected_stems(entries)[CAMERAS[1]]}.txt"
    ]
    with pytest.raises(ValueError, match="ten MP4/TXT pairs"):
        source.derive_selected_inventory(missing)

    mismatch = [dict(item) for item in entries]
    target = next(
        item
        for item in mismatch
        if item.get("path", "").startswith(f"{source.REMOTE_OBJECT_ROOT}/{CAMERAS[2]}/")
        and item.get("path", "").endswith(".txt")
    )
    target["path"] = target["path"].replace(".txt", "-different.txt")
    with pytest.raises(ValueError, match="not exact pairs"):
        source.derive_selected_inventory(mismatch)


def test_acquire_align_and_revalidate_exact_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries, selected = _fixture_selected(monkeypatch)
    paths = _paths(tmp_path)
    permit = object()
    permit_record = {
        "kind": "test-source-permit",
        "case_name": source.REPLACEMENT_CASE_NAME,
        "lock_artifact_sha256": "1" * 64,
    }
    permit_calls: list[tuple[object, str, str]] = []
    inventory_calls: list[dict[str, Any]] = []
    download_calls: list[dict[str, Any]] = []
    commands: list[tuple[str, ...]] = []

    def consume(value: object, *, case_name: str, operation: str) -> dict[str, Any]:
        permit_calls.append((value, case_name, operation))
        return permit_record

    def inventory(**kwargs: Any) -> list[dict[str, Any]]:
        inventory_calls.append(kwargs)
        return entries

    def download(**kwargs: Any) -> str:
        download_calls.append(kwargs)
        return _write_download(kwargs["local_dir"], kwargs["allow_patterns"])

    manifest = source.acquire_and_align_replacement_source(
        paths,
        source_permit=permit,
        consume_source_permit=consume,
        expected_source_permit=permit_record,
        inventory_provider=inventory,
        snapshot_downloader=download,
        command_runner=_fake_command_runner(commands),
        revision_reader=lambda _root: source.PROCESSING_CODE_REVISION,
    )

    assert permit_calls == [
        (permit, source.REPLACEMENT_CASE_NAME, source.SOURCE_OPERATION)
    ]
    assert inventory_calls == [
        {
            "repo_id": source.HF_REPO_ID,
            "repo_type": "dataset",
            "revision": source.HF_DATASET_REVISION,
            "path_in_repo": source.REMOTE_OBJECT_ROOT,
            "recursive": True,
            "expand": True,
        }
    ]
    assert len(download_calls) == 1
    assert set(download_calls[0]) == {
        "repo_id",
        "repo_type",
        "revision",
        "local_dir",
        "allow_patterns",
    }
    assert download_calls[0]["allow_patterns"] == list(selected.allow_patterns)
    assert "ignore_patterns" not in download_calls[0]
    assert len(commands) == 2
    assert json.loads(commands[0][5]) == list(CAMERAS)
    assert json.loads(commands[1][4]) == [
        camera for camera in CAMERAS if camera not in source.UNCALIBRATED_CAMERAS
    ]
    assert (paths.aligned_root / source.REPLACEMENT_OBJECT_ID / "episode_0003").is_dir()
    assert not (
        paths.aligned_root / source.REPLACEMENT_OBJECT_ID / "episode_0000"
    ).exists()
    for path in (
        paths.inventory_manifest,
        paths.content_manifest,
        paths.aligned_source_manifest,
    ):
        assert stat.S_IMODE(os.lstat(path).st_mode) == 0o400

    artifact = source.validate_aligned_source_manifest(
        manifest,
        expected_source_permit=permit_record,
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    assert artifact["aligned_episode_dir"].endswith(
        f"/{source.REPLACEMENT_OBJECT_ID}/episode_0003"
    )
    assert artifact["semantics"] == source.SEMANTICS
    assert artifact["camera_census"]["downloaded_camera_count"] == 37
    assert artifact["camera_census"]["aligned_camera_count"] == 32
    assert artifact["camera_census"]["uncalibrated_skipped_cameras"] == list(
        source.UNCALIBRATED_CAMERAS
    )
    assert artifact["no_tactile_audio_or_other_episode_downloaded"] is True


def test_source_permit_is_consumed_before_remote_or_filesystem_access(
    tmp_path: Path,
) -> None:
    touched = False

    def inventory(**_kwargs: Any) -> list[object]:
        nonlocal touched
        touched = True
        return []

    def reject(_permit: object, *, case_name: str, operation: str) -> dict[str, Any]:
        assert case_name == source.REPLACEMENT_CASE_NAME
        assert operation == source.SOURCE_OPERATION
        raise RuntimeError("permit rejected")

    with pytest.raises(RuntimeError, match="permit rejected"):
        source.acquire_and_align_replacement_source(
            source.ReplacementSourcePaths(
                download_root=tmp_path / "does-not-exist" / "download",
                aligned_root=tmp_path / "does-not-exist" / "aligned",
                inventory_manifest=tmp_path / "does-not-exist" / "inventory.json",
                content_manifest=tmp_path / "does-not-exist" / "content.json",
                aligned_source_manifest=tmp_path / "does-not-exist" / "aligned.json",
                processing_code_root=tmp_path / "does-not-exist" / "code",
                python_executable=tmp_path / "does-not-exist" / "python",
            ),
            source_permit=object(),
            consume_source_permit=reject,
            inventory_provider=inventory,
        )
    assert touched is False


def test_pinned_virtualenv_python_symlink_is_accepted(tmp_path: Path) -> None:
    target = tmp_path / "python-target"
    target.write_bytes(b"executable fixture")
    target.chmod(0o500)
    lexical = tmp_path / "python"
    lexical.symlink_to(target)

    assert source._validated_executable(lexical, label="test Python") == lexical


def test_validator_rejects_mode_change_and_aligned_byte_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries, _selected = _fixture_selected(monkeypatch)
    paths = _paths(tmp_path)
    permit_record = {"kind": "test", "case_name": source.REPLACEMENT_CASE_NAME}
    manifest = source.acquire_and_align_replacement_source(
        paths,
        source_permit=object(),
        consume_source_permit=lambda _permit, **_kwargs: permit_record,
        expected_source_permit=permit_record,
        inventory_provider=lambda **_kwargs: entries,
        snapshot_downloader=lambda **kwargs: _write_download(
            kwargs["local_dir"], kwargs["allow_patterns"]
        ),
        command_runner=_fake_command_runner([]),
        revision_reader=lambda _root: source.PROCESSING_CODE_REVISION,
    )

    manifest.chmod(0o600)
    with pytest.raises(ValueError, match="mode must be 0o400"):
        source.validate_aligned_source_manifest(manifest)
    manifest.chmod(0o400)

    unexpected = paths.download_root / "raw" / "unexpected-empty-source"
    unexpected.mkdir()
    with pytest.raises(ValueError, match="unexpected directory"):
        source.validate_aligned_source_manifest(manifest)
    unexpected.rmdir()

    video = (
        paths.aligned_root
        / source.REPLACEMENT_OBJECT_ID
        / "episode_0003"
        / next(
            camera for camera in CAMERAS if camera not in source.UNCALIBRATED_CAMERAS
        )
        / "undistorted.mp4"
    )
    video.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="aligned source bytes changed"):
        source.validate_aligned_source_manifest(manifest)


def test_download_scan_rejects_symlink_even_with_exact_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _entries, selected = _fixture_selected(monkeypatch)
    root = tmp_path / "download"
    _write_download(str(root), list(selected.allow_patterns))
    target = root / selected.allow_patterns[0]
    target.unlink()
    target.symlink_to(root / selected.allow_patterns[1])

    with pytest.raises(ValueError, match="contains a symlink"):
        source._scan_exact_download(root, selected)
