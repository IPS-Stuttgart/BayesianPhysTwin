from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_causal_response_direct_depth_reserve_source_finalizer_v14 import (
    expected_admission_prelock,
    load_v14_composite_admission_custody,
    load_v14_reserve_source_finalizer_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
CUSTODY = (
    ROOT / "configs/sota/"
    "deform360_causal_response_direct_depth_v14_composite_admission_custody_v1.json"
)
FINALIZER = (
    ROOT / "configs/sota/"
    "deform360_causal_response_direct_depth_v14_reserve_source_finalizer_v1.json"
)


def test_composite_admission_custody_separates_rank_scopes() -> None:
    custody = load_v14_composite_admission_custody(CUSTODY, repository=ROOT)
    original = expected_admission_prelock(custody, queue_rank=14)
    reserve = expected_admission_prelock(custody, queue_rank=15)

    assert original != reserve
    assert original[0] == (
        "ccb5ac22ed87695c0d21562902d15ffe60e30218012a601c4bbbc9f593aa99cf"
    )
    assert reserve[0] == (
        "9b4532cff75dc33226188a1f3dffb73e7b797dadf99bee25441e3582eb0ca8a2"
    )
    with pytest.raises(ValueError, match="outside composite custody"):
        expected_admission_prelock(custody, queue_rank=19)


def test_reserve_source_finalizer_binds_composite_custody() -> None:
    custody = load_v14_composite_admission_custody(CUSTODY, repository=ROOT)
    finalizer = load_v14_reserve_source_finalizer_protocol(
        FINALIZER,
        repository=ROOT,
        composite_custody_path=CUSTODY,
    )

    assert (
        finalizer["parent_artifacts"]["admission_prelock"]["semantic_sha256"]
        == custody["config_sha256"]
    )
    assert finalizer["trigger"]["admitted_source_count"] == 12
    assert finalizer["trigger"]["source_outcome_read"] is False


def test_composite_admission_custody_rejects_rank_mutation(
    tmp_path: Path,
) -> None:
    payload = json.loads(CUSTODY.read_text(encoding="utf-8"))
    payload["rank_contract"]["final_queue_rank"] = 19
    changed = tmp_path / "changed-custody.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity or checksum"):
        load_v14_composite_admission_custody(changed, repository=ROOT)
