from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_causal_response_direct_depth_reserve_prediction_v14 import (
    expected_physical_prelock,
    load_v14_composite_physical_custody,
)

ROOT = Path(__file__).resolve().parents[1]
CUSTODY = (
    ROOT / "configs/sota/"
    "deform360_causal_response_direct_depth_v14_composite_physical_custody_v1.json"
)


def test_composite_physical_custody_separates_selected_ranks() -> None:
    custody = load_v14_composite_physical_custody(CUSTODY, repository=ROOT)
    original = expected_physical_prelock(
        custody,
        queue_rank=14,
        repository=ROOT,
    )
    reserve = expected_physical_prelock(
        custody,
        queue_rank=15,
        repository=ROOT,
    )

    assert original != reserve
    assert original[1] == (
        "a96f5b62471b26693b11fc98a819df2ef288a71f4f664b04e158646ab8e1fc02"
    )
    assert reserve[1] == (
        "cf11aed78360da67524e668141ff5a4e7350fe48e2033aa09ed0d1a9ee9d7fe9"
    )
    with pytest.raises(ValueError, match="outside composite custody"):
        expected_physical_prelock(
            custody,
            queue_rank=13,
            repository=ROOT,
        )


def test_composite_physical_custody_rejects_rank_mutation(
    tmp_path: Path,
) -> None:
    payload = json.loads(CUSTODY.read_text(encoding="utf-8"))
    payload["rank_contract"]["reserve_selected_ranks"].append(19)
    changed = tmp_path / "changed-custody.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity or checksum"):
        load_v14_composite_physical_custody(changed, repository=ROOT)
