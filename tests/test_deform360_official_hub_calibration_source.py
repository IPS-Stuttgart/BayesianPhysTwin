from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.science.deform360_calibration_source.contracts import (
    DATASET_REVISION,
    PROCESSING_REVISION,
    PROTOCOL_ID,
    CalibrationUnit,
    load_protocol,
    load_units,
    summary_gate,
)
from scripts.science.deform360_calibration_source.download import download_one
from scripts.science.deform360_calibration_source.planning import (
    build_plan,
    repository_files,
    select_object_files,
    verify_plan,
)

PROTOCOL = Path("protocols/deform360_official_hub_calibration_source_v1.json")
SELECTION = Path(
    "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)
PROVIDER = Path(
    "protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider/"
    "visual-provider-lock.json"
)


def _entry(path: str, *, payload: bytes | None = None) -> SimpleNamespace:
    value = payload if payload is not None else path.encode("utf-8")
    return SimpleNamespace(
        path=path,
        type="file",
        blob_id=hashlib.sha1(value, usedforsecurity=False).hexdigest(),
        size=len(value),
        lfs=SimpleNamespace(sha256=hashlib.sha256(value).hexdigest()),
    )


def _object_entries(
    object_id: str,
    *,
    cameras: int = 10,
    tactile: int = 2,
) -> list[SimpleNamespace]:
    prefix = f"raw/{object_id}"
    rows = [
        _entry(f"{prefix}/metadata.json"),
        _entry(f"{prefix}/calibration_refined/intrinsics.npy"),
        _entry(f"{prefix}/calibration_refined/extrinsics.npy"),
        _entry(f"{prefix}/calibration_refined/dist.npy"),
    ]
    for camera_index in range(cameras):
        camera = f"brics-odroid-{camera_index + 1:03d}_cam0"
        for episode in range(10):
            stem = f"{camera}_{episode:02d}"
            rows.extend(
                (
                    _entry(f"{prefix}/{camera}/{stem}.mp4"),
                    _entry(f"{prefix}/{camera}/{stem}.txt"),
                )
            )
    for sensor_index in range(tactile):
        sensor = f"brics-odroid_tactilel_sensor{sensor_index}"
        rows.append(_entry(f"{prefix}/{sensor}/median_release.npy"))
        for episode in range(10):
            stem = f"{sensor}_{episode:02d}"
            rows.extend(
                (
                    _entry(f"{prefix}/{sensor}/{stem}.npy"),
                    _entry(f"{prefix}/{sensor}/{stem}.txt"),
                )
            )
    return rows


class _Api:
    def __init__(self, entries_by_object: dict[str, list[SimpleNamespace]]) -> None:
        self.entries_by_object = entries_by_object
        self.calls: list[dict[str, object]] = []

    def list_repo_tree(self, **kwargs: object):
        self.calls.append(dict(kwargs))
        object_id = Path(str(kwargs["path_in_repo"])).name
        return iter(self.entries_by_object[object_id])


def _all_entries() -> dict[str, list[SimpleNamespace]]:
    units, _confirmations = load_units(SELECTION)
    return {unit.object_id: _object_entries(unit.object_id) for unit in units}


def test_protocol_digest_and_locked_boundaries() -> None:
    protocol = load_protocol(PROTOCOL)

    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["dataset"]["revision"] == DATASET_REVISION
    assert protocol["processing"]["revision"] == PROCESSING_REVISION
    assert protocol["information_boundary"]["confirmation_payloads_opened"] is False


def test_selected_episode_file_plan_uses_exact_sorted_recording() -> None:
    unit = CalibrationUnit(
        object_id="201-example",
        episode_id=7,
        stratum="sheet",
        metadata_path="raw/201-example/metadata.json",
        metadata_sha256="a" * 64,
    )
    records = repository_files(
        _object_entries(unit.object_id),
        prefix=f"raw/{unit.object_id}/",
    )

    row = select_object_files(records, unit=unit)

    assert row["status"] == "planned"
    paths = {record["path"] for record in row["selected_files"]}
    assert any(path.endswith("_07.mp4") for path in paths)
    assert any(path.endswith("_07.txt") for path in paths)
    assert not any(path.endswith("_06.mp4") for path in paths)
    assert not any(path.endswith("_08.mp4") for path in paths)
    assert len(row["camera_streams"]) == 10
    assert len(row["tactile_streams"]) == 2


def test_names_only_plan_excludes_every_confirmation_object(tmp_path: Path) -> None:
    api = _Api(_all_entries())
    output = tmp_path / "plan.json"

    plan = build_plan(
        protocol_path=PROTOCOL,
        selection_path=SELECTION,
        provider_path=PROVIDER,
        output_path=output,
        api=api,
    )

    assert plan["gate"]["support_passed"] is True
    assert plan["gate"]["supported_object_count"] == 10
    assert plan["information_boundary"]["calibration_payloads_opened"] is False
    _units, confirmations = load_units(SELECTION)
    paths = {
        record["path"] for row in plan["objects"] for record in row["selected_files"]
    }
    assert not any(
        path.startswith(f"raw/{object_id}/")
        for path in paths
        for object_id in confirmations
    )
    assert all(call["revision"] == DATASET_REVISION for call in api.calls)
    assert json.loads(output.read_text(encoding="utf-8")) == plan


def test_names_only_support_gate_retains_failures_without_replacement(
    tmp_path: Path,
) -> None:
    entries = _all_entries()
    units, _confirmations = load_units(SELECTION)
    sheet = [unit for unit in units if unit.stratum == "sheet"]
    for unit in sheet[:2]:
        entries[unit.object_id] = _object_entries(unit.object_id, cameras=4)

    plan = build_plan(
        protocol_path=PROTOCOL,
        selection_path=SELECTION,
        provider_path=PROVIDER,
        output_path=tmp_path / "plan.json",
        api=_Api(entries),
    )

    assert plan["gate"]["support_passed"] is False
    assert plan["gate"]["supported_object_count"] == 8
    assert plan["gate"]["supported_by_stratum"]["sheet"] == 3
    failures = [
        row
        for row in plan["objects"]
        if row["status"] == "unsupported_without_replacement"
    ]
    assert {row["object_id"] for row in failures} == {
        unit.object_id for unit in sheet[:2]
    }


def test_plan_digest_tampering_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    plan = build_plan(
        protocol_path=PROTOCOL,
        selection_path=SELECTION,
        provider_path=PROVIDER,
        output_path=output,
        api=_Api(_all_entries()),
    )
    plan["objects"][0]["episode_id"] = 99
    output.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="plan digest changed"):
        verify_plan(
            output,
            protocol_path=PROTOCOL,
            selection_path=SELECTION,
            provider_path=PROVIDER,
        )


def test_download_one_binds_lfs_sha256(tmp_path: Path) -> None:
    relative = "raw/201-example/metadata.json"
    payload = b"locked-calibration-bytes"
    record = {
        "path": relative,
        "size": len(payload),
        "blob_id": "b" * 40,
        "lfs_sha256": hashlib.sha256(payload).hexdigest(),
    }

    def download(**kwargs: object) -> str:
        assert kwargs["revision"] == DATASET_REVISION
        path = tmp_path / str(kwargs["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)

    result = download_one(record=record, root=tmp_path, hub_download=download)

    assert result["downloaded_sha256"] == record["lfs_sha256"]
    assert result["downloaded_size"] == len(payload)


def test_download_one_rejects_byte_drift(tmp_path: Path) -> None:
    record = {
        "path": "raw/201-example/metadata.json",
        "size": 5,
        "blob_id": "b" * 40,
        "lfs_sha256": hashlib.sha256(b"right").hexdigest(),
    }

    def download(**kwargs: object) -> str:
        path = tmp_path / str(kwargs["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wrong")
        return str(path)

    with pytest.raises(ValueError, match="LFS digest changed"):
        download_one(record=record, root=tmp_path, hub_download=download)


def test_gate_uses_independent_object_and_stratum_counts() -> None:
    rows = [
        {"object_id": f"sheet-{index}", "stratum": "sheet", "status": "ok"}
        for index in range(4)
    ] + [
        {
            "object_id": f"volumetric-{index}",
            "stratum": "volumetric",
            "status": "ok",
        }
        for index in range(4)
    ]

    gate = summary_gate(rows, status="ok")

    assert gate == {
        "supported_object_count": 8,
        "supported_by_stratum": {"sheet": 4, "volumetric": 4},
        "minimum_supported_objects": 8,
        "minimum_supported_per_stratum": 4,
        "support_passed": True,
    }
