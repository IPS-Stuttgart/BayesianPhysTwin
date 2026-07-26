from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from causal4d_public.deform360_released_warp_readout import (
    load_released_warp_readout_protocol,
    released_pcd_manifest,
    validate_released_warp_readout_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "configs"
    / "causal4d_public"
    / "deform360_released_warp_readout_source_v1.json"
)


def test_canonical_released_warp_readout_protocol_passes() -> None:
    payload = load_released_warp_readout_protocol(PROTOCOL)

    result = validate_released_warp_readout_protocol(payload)

    assert result["passed"] is True
    assert result["episode_ids"] == [0, 3, 4, 5, 8]
    assert result["pcd_file_count"] == 75


def test_released_warp_readout_protocol_is_checksum_locked() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    mutated = deepcopy(payload)
    mutated["config"]["transfer_gate"][
        "minimum_episode_chamfer_wins"
    ] = 2

    with pytest.raises(ValueError, match="checksum"):
        validate_released_warp_readout_protocol(mutated)


def test_released_pcd_manifest_is_path_and_content_bound(tmp_path: Path) -> None:
    payload = {
        "config": {
            "episodes": [
                {
                    "episode_id": 4,
                    "matched_origin_frame": 10,
                    "evaluation_frames": [12, 14],
                }
            ]
        }
    }
    frame_root = tmp_path / "episode_4" / "pcd_clean"
    frame_root.mkdir(parents=True)
    (frame_root / "000010.npz").write_bytes(b"origin")
    (frame_root / "000012.npz").write_bytes(b"future-1")
    (frame_root / "000014.npz").write_bytes(b"future-2")

    first = released_pcd_manifest(tmp_path, payload)
    (frame_root / "000014.npz").write_bytes(b"changed")
    second = released_pcd_manifest(tmp_path, payload)

    assert first["file_count"] == 3
    assert first["sha256"] != second["sha256"]
    assert [record["relative_path"] for record in first["records"]] == [
        "episode_4/pcd_clean/000010.npz",
        "episode_4/pcd_clean/000012.npz",
        "episode_4/pcd_clean/000014.npz",
    ]
