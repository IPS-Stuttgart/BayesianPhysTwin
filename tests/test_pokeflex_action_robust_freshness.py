import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_action_robust_freshness import (
    FRESHNESS_AUDIT_FILE_SHA256,
    FRESHNESS_AUDIT_SHA256,
    SELECTED_ZIP_SHA256,
    build_action_robust_freshness_audit,
    freshness_audit_sha256,
    validate_action_robust_freshness_audit,
)

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = (
    ROOT / "configs" / "sota" / "pokeflex_instance_fresh12_exclusion_audit_v2.json"
)
FROZEN = (
    ROOT
    / "configs"
    / "sota"
    / "pokeflex_action_robust_fresh6_exclusion_audit_v3.json"
)


def test_action_robust_freshness_selects_salted_third_panel() -> None:
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    audit = build_action_robust_freshness_audit(
        previous,
        locked_at_utc="2026-08-04T20:13:40Z",
    )
    validation = validate_action_robust_freshness_audit(
        audit,
        bind_registered_digest=False,
    )

    assert validation["target_take_ids"] == tuple(sorted(SELECTED_ZIP_SHA256))
    assert audit["eligibility"]["eligible_take_count"] == 8
    assert audit["eligibility"]["eligible_object_count"] == 6
    assert audit["prior_exposure_audit"]["excluded_take_count"] == 108
    assert audit["post_v2_exact_exposure_scan"][
        "gpuserver6000_recent_unregistered_exact_matches"
    ] == []


def test_action_robust_freshness_rejects_exposure_or_selection_change() -> None:
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    audit = build_action_robust_freshness_audit(
        previous,
        locked_at_utc="2026-08-04T20:13:40Z",
    )
    exposed = deepcopy(audit)
    exposed["post_v2_exact_exposure_scan"][
        "gpuserver4090_recent_unregistered_exact_matches"
    ] = ["redacted"]
    with pytest.raises(ValueError, match="checksum"):
        validate_action_robust_freshness_audit(
            exposed,
            bind_registered_digest=False,
        )

    selected = deepcopy(audit)
    selected["selection"]["take_ids"][0] = "Pillow_T4"
    selected["audit_sha256"] = freshness_audit_sha256(selected)
    with pytest.raises(ValueError, match="selection"):
        validate_action_robust_freshness_audit(
            selected,
            bind_registered_digest=False,
        )


def test_frozen_action_robust_freshness_audit_is_exact() -> None:
    payload = json.loads(FROZEN.read_text(encoding="utf-8"))
    validation = validate_action_robust_freshness_audit(payload)

    assert payload["audit_sha256"] == FRESHNESS_AUDIT_SHA256
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == (
        FRESHNESS_AUDIT_FILE_SHA256
    )
    assert validation["target_take_ids"] == tuple(sorted(SELECTED_ZIP_SHA256))
