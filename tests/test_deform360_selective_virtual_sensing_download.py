from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_selective_virtual_sensing_download import (
    build_selective_download_manifest,
    download_selective_virtual_sensing_panel,
    selective_virtual_sensing_download_plan,
    validate_selective_download_root,
)


PROTOCOL = (
    Path(__file__).parents[1]
    / "configs"
    / "sota"
    / "deform360_selective_virtual_sensing_v1.json"
)


def _write_fixture_panel(root: Path, object_ids: tuple[str, ...]) -> None:
    for object_id in object_ids:
        object_root = root / "raw" / object_id
        object_root.mkdir(parents=True)
        (object_root / "metadata.json").write_text(
            json.dumps(
                {
                    "object": object_id,
                    "sam_prompt": "fixture",
                    "sequences": {str(index): {} for index in range(10)},
                }
            ),
            encoding="utf-8",
        )


def test_plan_is_exactly_the_locked_twelve_object_panel() -> None:
    plan = selective_virtual_sensing_download_plan(PROTOCOL)

    assert len(plan.object_ids) == 12
    assert plan.object_ids[:4] == (
        "005-thread",
        "069-jump-rope",
        "071-climbing-rope",
        "077-hemp-rope",
    )
    assert plan.allow_patterns == tuple(
        f"raw/{object_id}/*" for object_id in plan.object_ids
    )
    assert plan.ignore_patterns == ("*.flac", "*.wav")


def test_root_validation_rejects_any_unlocked_object(tmp_path: Path) -> None:
    plan = selective_virtual_sensing_download_plan(PROTOCOL)
    (tmp_path / "raw" / "001-rope").mkdir(parents=True)

    with pytest.raises(ValueError, match="unlocked objects"):
        validate_selective_download_root(
            tmp_path, plan=plan, require_complete=False
        )


def test_download_uses_exact_request_and_seals_metadata(tmp_path: Path) -> None:
    calls = []
    plan = selective_virtual_sensing_download_plan(PROTOCOL)

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        _write_fixture_panel(tmp_path, plan.object_ids)
        return str(tmp_path)

    manifest = download_selective_virtual_sensing_panel(
        PROTOCOL,
        tmp_path,
        max_workers=3,
        snapshot_download=fake_snapshot_download,
    )

    assert calls == [
        {
            "repo_id": plan.repository,
            "repo_type": "dataset",
            "revision": plan.revision,
            "local_dir": str(tmp_path.resolve()),
            "allow_patterns": list(plan.allow_patterns),
            "ignore_patterns": list(plan.ignore_patterns),
            "max_workers": 3,
        }
    ]
    assert manifest["object_count"] == 12
    assert manifest["information_boundary"]["target_metrics_opened"] is False
    assert len(manifest["manifest_sha256"]) == 64
    assert [row["object_id"] for row in manifest["objects"]] == list(
        plan.object_ids
    )


def test_manifest_rejects_metadata_episode_inventory_change(tmp_path: Path) -> None:
    plan = selective_virtual_sensing_download_plan(PROTOCOL)
    _write_fixture_panel(tmp_path, plan.object_ids)
    metadata = tmp_path / "raw" / plan.object_ids[0] / "metadata.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    del payload["sequences"]["9"]
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="episode inventory changed"):
        build_selective_download_manifest(tmp_path, plan=plan)
