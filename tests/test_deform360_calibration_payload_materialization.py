from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "materialize_deform360_calibration_payloads.py"


def _module() -> Any:
    specification = importlib.util.spec_from_file_location(
        "deform360_calibration_payload_materialization",
        SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _file(module: Any, path: str, *, size: int = 1) -> Any:
    return module.HubFile(path=path, size=size)


def _unit() -> SimpleNamespace:
    return SimpleNamespace(
        object_id="001-object",
        episode_id=1,
        stratum="sheet",
        metadata_path="raw/001-object/metadata.json",
        metadata_sha256="a" * 64,
    )


def _complete_files(module: Any) -> list[Any]:
    prefix = "raw/001-object"
    files = [
        _file(module, f"{prefix}/metadata.json"),
        _file(module, f"{prefix}/calibration_refined/intrinsics.npy"),
        _file(module, f"{prefix}/calibration_refined/extrinsics.npy"),
        _file(module, f"{prefix}/calibration_refined/dist.npy"),
    ]
    for camera in ("brics-odroid-001_cam0", "brics-odroid-002_cam0"):
        for episode in range(2):
            stem = f"capture_{episode:02d}"
            files.extend(
                (
                    _file(module, f"{prefix}/{camera}/{stem}.mp4", size=100),
                    _file(module, f"{prefix}/{camera}/{stem}.txt", size=10),
                )
            )
    for sensor in ("brics-odroid_tactilel_left", "brics-odroid_tactiler_right"):
        files.append(_file(module, f"{prefix}/{sensor}/median_shared.npy"))
        for episode in range(2):
            stem = f"touch_{episode:02d}"
            files.extend(
                (
                    _file(module, f"{prefix}/{sensor}/{stem}.npy", size=100),
                    _file(module, f"{prefix}/{sensor}/{stem}.txt", size=10),
                )
            )
    return files


def test_plan_selects_only_locked_episode_and_excludes_video_downloads() -> None:
    module = _module()
    plan = module.build_unit_plan(_unit(), _complete_files(module))

    assert plan.status == "ready"
    assert len(plan.camera_recordings) == 2
    assert len(plan.tactile_recordings) == 2
    assert all("_01." in media.path for _, media, _ in plan.camera_recordings)
    camera_downloads = [path for path in plan.materialization_paths if "cam" in path]
    assert all(path.endswith(".txt") for path in camera_downloads)
    assert not any(path.endswith(".mp4") for path in plan.materialization_paths)
    assert all(path.endswith(".mp4") for path in plan.planned_camera_media_paths)
    assert not any("_00." in path for path in plan.materialization_paths)


def test_missing_exact_tactile_baseline_is_retained_as_technical_failure() -> None:
    module = _module()
    files = [
        item
        for item in _complete_files(module)
        if "brics-odroid_tactilel_left/median_" not in item.path
    ]
    plan = module.build_unit_plan(_unit(), files)

    assert plan.status == "technical_failure"
    assert "tactile_baseline_count:brics-odroid_tactilel_left:0" in (
        plan.technical_failures
    )
    assert len(plan.tactile_recordings) == 1


def test_manifest_rejects_any_confirmation_path() -> None:
    module = _module()
    plan = module.build_unit_plan(_unit(), _complete_files(module))
    selection = SimpleNamespace(
        protocol_id="deform360-official-hub-visuotactile-v1",
        snapshot_id="b" * 64,
        source_sha256="c" * 64,
        selection_artifact_sha256="d" * 64,
        dataset_revision="1" * 40,
        calibration_units=(_unit(),),
        confirmation_units=(SimpleNamespace(object_id="999-confirmation"),),
    )
    visual = SimpleNamespace(artifact_id="e" * 64)

    changed = replace(plan, metadata_path="raw/999-confirmation/metadata.json")
    with pytest.raises(ValueError, match="confirmation paths"):
        module.build_manifest(
            selection=selection,
            visual_provider=visual,
            plans=(changed,),
            implementation_revision="2" * 40,
            processing_revision="3" * 40,
            opened_payloads=False,
        )


def test_materialize_paths_hashes_exact_downloaded_bytes(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "dataset"

    def fake_download(**arguments: Any) -> str:
        path = root / arguments["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(arguments["filename"].encode("utf-8"))
        return str(path)

    records = module.materialize_paths(
        ("raw/001-object/metadata.json", "raw/001-object/timestamps.txt"),
        revision="1" * 40,
        dataset_root=root,
        token=None,
        workers=2,
        download_file=fake_download,
    )

    assert [item["path"] for item in records] == [
        "raw/001-object/metadata.json",
        "raw/001-object/timestamps.txt",
    ]
    for item in records:
        expected = hashlib.sha256(item["path"].encode("utf-8")).hexdigest()
        assert item["status"] == "downloaded"
        assert item["sha256"] == expected


def test_materialize_paths_retains_download_failures(tmp_path: Path) -> None:
    module = _module()

    def failing_download(**arguments: Any) -> str:
        raise OSError(arguments["filename"])

    records = module.materialize_paths(
        ("raw/001-object/metadata.json",),
        revision="1" * 40,
        dataset_root=tmp_path / "dataset",
        token=None,
        workers=1,
        download_file=failing_download,
    )

    assert records == (
        {
            "path": "raw/001-object/metadata.json",
            "status": "failed",
            "error_type": "OSError",
            "error": "raw/001-object/metadata.json",
        },
    )


def test_hub_paths_fail_closed() -> None:
    module = _module()
    with pytest.raises(ValueError, match="canonical"):
        module.HubFile(path="raw/object/../confirmation/file.npy")
    with pytest.raises(ValueError, match="canonical"):
        module.HubFile(path="./raw/object/file.npy")


def test_cli_has_no_confirmation_or_target_input_surface() -> None:
    module = _module()
    destinations = {action.dest for action in module.build_parser()._actions}

    assert "confirmation_root" not in destinations
    assert "confirmation_object" not in destinations
    assert "target" not in destinations
    assert {
        "selection_lock",
        "visual_provider_lock",
        "dataset_root",
        "open_calibration_payloads",
    }.issubset(destinations)


def test_unpaired_camera_sidecar_is_retained_as_technical_failure() -> None:
    module = _module()
    files = [
        item
        for item in _complete_files(module)
        if not item.path.endswith("brics-odroid-001_cam0/capture_00.txt")
    ]
    plan = module.build_unit_plan(_unit(), files)

    assert plan.status == "technical_failure"
    assert (
        "camera_pairing:brics-odroid-001_cam0:missing_timestamp:capture_00"
        in plan.technical_failures
    )
    assert not any(
        camera == "brics-odroid-001_cam0" for camera, _, _ in plan.camera_recordings
    )
