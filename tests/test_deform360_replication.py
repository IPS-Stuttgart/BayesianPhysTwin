from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from causal4d_public.deform360_replication import (
    load_deform360_replication_protocol,
    validate_deform360_replication_protocol,
)


def _protocol_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "deform360_replication_v1.json"
    )


def test_replication_cohort_and_targets_are_metadata_locked() -> None:
    payload = load_deform360_replication_protocol(_protocol_path())
    result = validate_deform360_replication_protocol(payload)
    assert result["passed"] is True
    assert result["selected_objects"] == [
        "002-rope-silk",
        "081-stripe-rope",
        "085-scarf-cloth",
        "083-blanket-cloth",
        "092-squirrel",
        "170-spider",
    ]
    assert result["target_unimanual_count"] == 3
    assert result["target_bimanual_count"] == 3
    assert [record["target_episode_id"] for record in payload["config"]["cohort"]] == [
        1,
        5,
        2,
        7,
        1,
        6,
    ]


def test_replication_checksum_rejects_target_change() -> None:
    payload = json.loads(_protocol_path().read_text(encoding="utf-8"))
    mutated = copy.deepcopy(payload)
    mutated["config"]["cohort"][0]["target_episode_id"] = 0
    with pytest.raises(ValueError, match="checksum"):
        validate_deform360_replication_protocol(mutated)


def test_replication_forbids_exhausted_pilot_target() -> None:
    payload = load_deform360_replication_protocol(_protocol_path())
    gate = payload["config"]["gates"]["official_warp_feasibility"]
    boundary = payload["config"]["information_boundary"]
    assert gate["allowed_source_episode_ids"] == [0, 3, 4, 5, 8]
    assert 6 in gate["forbidden_episode_ids"]
    assert boundary["pilot_001_target_episode_6_reused_for_selection"] is False
    assert boundary["full_tactile_oracle_post_seal_only"] is True
