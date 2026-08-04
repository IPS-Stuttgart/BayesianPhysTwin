import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_instance_freshness import (
    FRESHNESS_AUDIT_FILE_SHA256,
    FRESHNESS_AUDIT_SHA256,
    SELECTED_ZIP_SHA256,
    build_instance_freshness_audit,
    freshness_audit_sha256,
    public_take_ids,
    validate_instance_freshness_audit,
)

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "configs" / "sota" / "pokeflex_fresh12_exclusion_audit_v1.json"
FROZEN = ROOT / "configs" / "sota" / "pokeflex_instance_fresh12_exclusion_audit_v2.json"


def test_public_inventory_reconstructs_all_released_archives() -> None:
    take_ids = public_take_ids()

    assert len(take_ids) == 116
    assert len(set(take_ids)) == 116


def test_second_freshness_audit_is_deterministic_and_disjoint() -> None:
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    audit = build_instance_freshness_audit(
        previous,
        locked_at_utc="2026-08-04T17:54:44Z",
    )
    validation = validate_instance_freshness_audit(audit)

    assert validation["passed"] is True
    assert tuple(validation["target_take_ids"]) == tuple(sorted(SELECTED_ZIP_SHA256))
    assert audit["prior_exposure_audit"]["excluded_take_count"] == 96
    assert audit["eligibility"]["eligible_take_count"] == 20
    assert set(audit["selection"]["take_ids"]).isdisjoint(
        audit["prior_exposure_audit"]["take_ids"]
    )


def test_frozen_freshness_bytes_match_the_builder() -> None:
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    rebuilt = build_instance_freshness_audit(
        previous,
        locked_at_utc="2026-08-04T17:54:44Z",
    )

    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == (
        FRESHNESS_AUDIT_FILE_SHA256
    )
    assert frozen["audit_sha256"] == FRESHNESS_AUDIT_SHA256
    assert frozen == rebuilt


def test_freshness_validator_rejects_a_self_consistent_eligibility_mutation() -> None:
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    audit = build_instance_freshness_audit(
        previous,
        locked_at_utc="2026-08-04T17:54:44Z",
    )
    audit["eligibility"]["take_ids"][0] = "3dPrintedBunny_T1"
    audit["audit_sha256"] = freshness_audit_sha256(audit)

    with pytest.raises(ValueError, match="eligible take was exposed"):
        validate_instance_freshness_audit(audit)
