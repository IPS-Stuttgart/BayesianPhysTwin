from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.science.acquire_deform360_covariance_target_v1 import (
    build_file_plan,
    verify_file_plan,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_LOCK = (
    _REPOSITORY
    / "protocols"
    / "locks"
    / "deform360_covariance_only_target_acquisition_v1.json"
)
_SELECTION = (
    _REPOSITORY
    / "results"
    / "science"
    / "deform360_covariance_only_target_v1"
    / "target_roster_v1_2.json"
)


def _entry(path: str, *, size: int = 17) -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        type="file",
        blob_id="blob",
        size=size,
        lfs=None,
    )


def _raw_entries(object_id: str) -> tuple[SimpleNamespace, ...]:
    prefix = f"raw/{object_id}"
    rows = [
        _entry(f"{prefix}/metadata.json"),
        _entry(f"{prefix}/calibration_refined/dist.npy"),
        _entry(f"{prefix}/calibration_refined/extrinsics.npy"),
        _entry(f"{prefix}/calibration_refined/intrinsics.npy"),
    ]
    for camera_index in range(8):
        stream = f"brics-odroid-{camera_index + 1:03d}_cam0"
        for episode_id in range(10):
            stem = f"{stream}_{1_700_000_000_000_000 + episode_id}"
            rows.append(_entry(f"{prefix}/{stream}/{stem}.mp4"))
            rows.append(_entry(f"{prefix}/{stream}/{stem}.txt"))
    stream = "brics-odroid_tactilel_left"
    for episode_id in range(10):
        stem = f"{stream}_{1_700_000_000_000_000 + episode_id}"
        rows.append(_entry(f"{prefix}/{stream}/{stem}.npy"))
        rows.append(_entry(f"{prefix}/{stream}/{stem}.txt"))
    rows.append(_entry(f"{prefix}/{stream}/median_1699999999999999.npy"))
    return tuple(rows)


class _FakeApi:
    def repo_info(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(sha="f804696d7a133908c7497ffdab43819d879b5cbc")

    def list_repo_tree(self, **kwargs: object) -> tuple[SimpleNamespace, ...]:
        path = str(kwargs["path_in_repo"])
        if path.startswith("processed/"):
            return ()
        object_id = path.removeprefix("raw/")
        return _raw_entries(object_id)


def test_exact_file_plan_keeps_all_24_targets_without_payload_access(
    tmp_path: Path,
) -> None:
    output = tmp_path / "plan.json"
    plan = build_file_plan(
        repository=_REPOSITORY,
        lock_path=_LOCK,
        selection_path=_SELECTION,
        output_path=output,
        api=_FakeApi(),
        implementation_revision="a" * 40,
    )

    assert plan["summary"]["locked_target_count"] == 24
    assert plan["summary"]["ordinary_raw_plan_count"] == 24
    assert plan["summary"]["retained_raw_plan_failure_count"] == 0
    assert plan["summary"]["exact_processed_annotation_count"] == 0
    assert plan["summary"]["selected_file_count"] == 24 * 23
    assert all(len(row["camera_streams"]) == 8 for row in plan["objects"])
    assert all(len(row["tactile_streams"]) == 1 for row in plan["objects"])
    assert plan["information_boundary"]["payload_bytes_opened"] is False
    assert plan["information_boundary"]["target_outcomes_opened"] is False
    verified = verify_file_plan(
        repository=_REPOSITORY,
        lock_path=_LOCK,
        selection_path=_SELECTION,
        plan_path=output,
    )
    assert verified["plan_sha256"] == plan["plan_sha256"]


def test_file_plan_tampering_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    build_file_plan(
        repository=_REPOSITORY,
        lock_path=_LOCK,
        selection_path=_SELECTION,
        output_path=output,
        api=_FakeApi(),
        implementation_revision="a" * 40,
    )
    value = json.loads(output.read_text())
    value["objects"][0]["episode_id"] += 1
    output.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="file-plan digest changed"):
        verify_file_plan(
            repository=_REPOSITORY,
            lock_path=_LOCK,
            selection_path=_SELECTION,
            plan_path=output,
        )
