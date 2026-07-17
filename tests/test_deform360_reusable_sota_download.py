from __future__ import annotations

from pathlib import Path

import pytest

from causal4d_public.deform360_reusable_sota_download import (
    development_download_plan,
    download_development_panel,
    validate_development_root,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/causal4d_public/deform360_reusable_sota_v1.json"


def test_development_download_plan_is_pinned_and_excludes_targets() -> None:
    plan = development_download_plan(PROTOCOL)
    assert plan.revision == "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
    assert len(plan.object_ids) == 12
    assert all("068-nylon-rope" not in pattern for pattern in plan.allow_patterns)
    assert plan.ignore_patterns == ("*.flac", "*.wav")


def test_development_root_rejects_confirmatory_object(tmp_path: Path) -> None:
    (tmp_path / "raw/068-nylon-rope").mkdir(parents=True)
    with pytest.raises(ValueError, match="confirmatory objects"):
        validate_development_root(tmp_path, protocol_path=PROTOCOL)


def test_download_is_pinned_and_manifested(tmp_path: Path) -> None:
    observed: dict = {}

    def fake_download(**kwargs) -> str:
        observed.update(kwargs)
        for pattern in kwargs["allow_patterns"]:
            object_id = pattern.split("/")[1]
            root = Path(kwargs["local_dir"]) / "raw" / object_id
            root.mkdir(parents=True)
            (root / "metadata.json").write_text(
                '{"object":"' + object_id + '"}\n', encoding="utf-8"
            )
        return kwargs["local_dir"]

    manifest = download_development_panel(
        PROTOCOL,
        tmp_path,
        max_workers=3,
        snapshot_download=fake_download,
    )
    assert observed["revision"] == "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
    assert observed["max_workers"] == 3
    assert len(observed["allow_patterns"]) == 12
    assert len(manifest["objects"]) == 12
    assert len(manifest["manifest_sha256"]) == 64
