"""Contracts for plan-derived runner storage admission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.check_deform360_runner_capacity import (
    CapacityContractError,
    FilesystemSnapshot,
    build_capacity_report,
    load_planned_files,
    main,
)


def _write_plan(path: Path, rows: list[tuple[str, int]]) -> None:
    path.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_id": "calibration-object",
                        "selected_files": [
                            {"path": file_path, "size": size}
                            for file_path, size in rows
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    data = tmp_path / "data"
    processed = tmp_path / "processed"
    cache = tmp_path / "cache"
    for path in (data, processed, cache):
        path.mkdir()
    return data, processed, cache


def test_capacity_report_groups_workloads_on_one_filesystem(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    _write_plan(plan, [("raw/object/camera.mp4", 100), ("raw/object/times.txt", 20)])
    data, processed, cache = _roots(tmp_path)

    def probe(_: Path) -> FilesystemSnapshot:
        return FilesystemSnapshot(device=7, available_bytes=1_000)

    report = build_capacity_report(
        plan_path=plan,
        data_root=data,
        processed_root=processed,
        cache_root=cache,
        reserve_bytes=100,
        processed_multiplier=2.0,
        filesystem_probe=probe,
    )

    assert report["planned_source_bytes"] == 120
    assert report["missing_download_bytes"] == 120
    assert report["estimated_cache_download_bytes"] == 120
    assert report["estimated_processed_bytes"] == 240
    assert report["filesystems"] == [
        {
            "device": 7,
            "roles": ["cache", "data", "processed"],
            "available_bytes": 1_000,
            "estimated_workload_bytes": 480,
            "reserve_bytes": 100,
            "required_available_bytes": 580,
            "passed": True,
        }
    ]
    assert report["passed"] is True
    assert report["information_boundary"]["file_contents_opened"] is False


def test_exact_existing_files_reduce_download_and_cache_capacity(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.json"
    _write_plan(plan, [("raw/object/camera.mp4", 100), ("raw/object/times.txt", 20)])
    data, processed, cache = _roots(tmp_path)
    existing = data / "raw/object/camera.mp4"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"x" * 100)

    snapshots = {
        data: FilesystemSnapshot(device=1, available_bytes=500),
        cache: FilesystemSnapshot(device=2, available_bytes=500),
        processed: FilesystemSnapshot(device=3, available_bytes=500),
    }

    report = build_capacity_report(
        plan_path=plan,
        data_root=data,
        processed_root=processed,
        cache_root=cache,
        reserve_bytes=50,
        processed_multiplier=1.0,
        filesystem_probe=lambda path: snapshots[path],
    )

    assert report["present_exact_file_count"] == 1
    assert report["present_exact_bytes"] == 100
    assert report["missing_file_count"] == 1
    assert report["missing_download_bytes"] == 20
    required = {
        tuple(item["roles"]): item["required_available_bytes"]
        for item in report["filesystems"]
    }
    assert required[("data",)] == 70
    assert required[("cache",)] == 70
    assert required[("processed",)] == 170


def test_capacity_report_uses_conservative_free_space_for_shared_device(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.json"
    _write_plan(plan, [("raw/object/file.bin", 10)])
    data, processed, cache = _roots(tmp_path)
    snapshots = iter(
        [
            FilesystemSnapshot(device=5, available_bytes=1_000),
            FilesystemSnapshot(device=5, available_bytes=900),
            FilesystemSnapshot(device=5, available_bytes=950),
        ]
    )

    report = build_capacity_report(
        plan_path=plan,
        data_root=data,
        processed_root=processed,
        cache_root=cache,
        reserve_bytes=0,
        processed_multiplier=0.0,
        filesystem_probe=lambda _: next(snapshots),
    )

    assert report["filesystems"][0]["available_bytes"] == 900
    assert report["passed"] is True


def test_plan_rejects_nonportable_and_conflicting_paths(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    _write_plan(plan, [("../escape", 1)])
    with pytest.raises(CapacityContractError, match="escapes"):
        load_planned_files(plan)

    _write_plan(plan, [("raw/file.bin", 1), ("raw/file.bin", 2)])
    with pytest.raises(CapacityContractError, match="conflicting sizes"):
        load_planned_files(plan)


def test_capacity_cli_writes_a_failing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tmp_path / "plan.json"
    _write_plan(plan, [("raw/object/file.bin", 100)])
    data, processed, cache = _roots(tmp_path)
    output = tmp_path / "capacity.json"

    monkeypatch.setattr(
        "scripts.ci.check_deform360_runner_capacity._default_filesystem_probe",
        lambda _: FilesystemSnapshot(device=9, available_bytes=1),
    )
    status = main(
        [
            "--plan",
            str(plan),
            "--data-root",
            str(data),
            "--processed-root",
            str(processed),
            "--cache-root",
            str(cache),
            "--reserve-bytes",
            "0",
            "--processed-multiplier",
            "0",
            "--output",
            str(output),
        ]
    )

    assert status == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["filesystems"][0]["required_available_bytes"] == 200
