import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_action_robust_final_freshness import (
    FRESHNESS_AUDIT_FILE_SHA256,
    FRESHNESS_AUDIT_SHA256,
    SELECTED_ZIP_SHA256,
    build_final_freshness_audit,
    freshness_audit_sha256,
    validate_final_freshness_audit,
)

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = (
    ROOT
    / "configs"
    / "sota"
    / "pokeflex_action_robust_fresh6_exclusion_audit_v3.json"
)
FROZEN = (
    ROOT
    / "configs"
    / "sota"
    / "pokeflex_action_robust_fresh2_exclusion_audit_v5.json"
)


def _built() -> dict[str, object]:
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    return build_final_freshness_audit(
        previous,
        locked_at_utc="2026-08-05T00:00:00Z",
    )


def test_final_freshness_is_exhaustive_public_complement() -> None:
    audit = _built()
    validation = validate_final_freshness_audit(
        audit,
        bind_registered_digest=False,
    )

    assert validation["target_take_ids"] == ("Pillow_T4", "PlushDice_T3")
    assert audit["prior_exposure_audit"]["excluded_take_count"] == 114
    assert audit["eligibility"]["eligible_take_count"] == 2
    assert audit["selection"]["take_ids"] == sorted(SELECTED_ZIP_SHA256)


def test_final_freshness_rejects_resigned_selection_or_exposure() -> None:
    audit = _built()
    selected = deepcopy(audit)
    selected["selection"]["take_ids"].reverse()
    selected["audit_sha256"] = freshness_audit_sha256(selected)
    with pytest.raises(ValueError, match="selection"):
        validate_final_freshness_audit(
            selected,
            bind_registered_digest=False,
        )

    exposed = deepcopy(audit)
    exposed["post_v3_exact_exposure_scan"][
        "gpuserver4090_unregistered_outcome_matches"
    ] = ["redacted"]
    exposed["audit_sha256"] = freshness_audit_sha256(exposed)
    with pytest.raises(ValueError, match="previously exposed"):
        validate_final_freshness_audit(
            exposed,
            bind_registered_digest=False,
        )


def test_frozen_final_freshness_audit_is_exact() -> None:
    payload = json.loads(FROZEN.read_text(encoding="utf-8"))
    validation = validate_final_freshness_audit(payload)

    assert payload["audit_sha256"] == FRESHNESS_AUDIT_SHA256
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == (
        FRESHNESS_AUDIT_FILE_SHA256
    )
    assert validation["target_take_ids"] == ("Pillow_T4", "PlushDice_T3")
